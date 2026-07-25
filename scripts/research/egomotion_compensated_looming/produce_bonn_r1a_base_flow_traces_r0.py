#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


FX = 542.822841
FY = 542.576870
CX = 315.593520
CY = 237.756098
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_pixel_sha256(image: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def average_hash(gray: np.ndarray) -> str:
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    bits = resized >= float(resized.mean())
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def decode_rgb(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise ValueError("unexpected Bonn RGB decode")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return bgr, gray


def farneback(
    previous: np.ndarray, current: np.ndarray, parameters: dict[str, Any]
) -> np.ndarray:
    return cv2.calcOpticalFlowFarneback(
        previous,
        current,
        None,
        float(parameters["pyr_scale"]),
        int(parameters["levels"]),
        int(parameters["winsize"]),
        int(parameters["iterations"]),
        int(parameters["poly_n"]),
        float(parameters["poly_sigma"]),
        int(parameters["flags"]),
    )


def spatial_arrays() -> dict[str, np.ndarray]:
    y, x = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]
    dx = x.astype(np.float32) - np.float32(CX)
    dy = y.astype(np.float32) - np.float32(CY)
    radius = np.sqrt(dx * dx + dy * dy)
    radial_x = dx / np.maximum(radius, 1e-6)
    radial_y = dy / np.maximum(radius, 1e-6)
    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "radius": radius,
        "radial_x": radial_x,
        "radial_y": radial_y,
    }


def grid_summaries(
    radial_rate: np.ndarray,
    valid: np.ndarray,
    grid_columns: int,
    grid_rows: int,
    quantile: float,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for row in range(grid_rows):
        y0 = row * IMAGE_HEIGHT // grid_rows
        y1 = (row + 1) * IMAGE_HEIGHT // grid_rows
        for column in range(grid_columns):
            x0 = column * IMAGE_WIDTH // grid_columns
            x1 = (column + 1) * IMAGE_WIDTH // grid_columns
            cell_valid = valid[y0:y1, x0:x1]
            values = radial_rate[y0:y1, x0:x1][cell_valid]
            summaries.append(
                {
                    "grid_row": row,
                    "grid_column": column,
                    "valid_pixel_count": int(len(values)),
                    "valid_fraction": float(cell_valid.mean()),
                    "positive_radial_rate_q90_per_second": (
                        float(np.quantile(np.maximum(values, 0.0), quantile))
                        if len(values)
                        else None
                    ),
                    "signed_radial_rate_median_per_second": (
                        float(np.median(values)) if len(values) else None
                    ),
                }
            )
    return summaries


def summarize_pair(
    forward: np.ndarray,
    backward: np.ndarray,
    delta_seconds: float,
    contract: dict[str, Any],
    spatial: dict[str, np.ndarray],
) -> dict[str, Any]:
    quality = contract["flow_producer"]["forward_backward_quality"]
    spatial_contract = contract["spatial_contract"]
    map_x = spatial["x"] + forward[:, :, 0]
    map_y = spatial["y"] + forward[:, :, 1]
    backward_at_forward = cv2.remap(
        backward,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    roundtrip = np.linalg.norm(forward + backward_at_forward, axis=2)
    border = int(spatial_contract["border_exclusion_pixels"])
    radius_min = float(
        spatial_contract["principal_point_exclusion_radius_pixels"]
    )
    inside = (
        (map_x >= 0.0)
        & (map_x < IMAGE_WIDTH - 1)
        & (map_y >= 0.0)
        & (map_y < IMAGE_HEIGHT - 1)
    )
    spatial_mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)
    spatial_mask[
        border : IMAGE_HEIGHT - border,
        border : IMAGE_WIDTH - border,
    ] = True
    spatial_mask &= spatial["radius"] >= radius_min
    valid = (
        spatial_mask
        & inside
        & np.isfinite(roundtrip)
        & (
            roundtrip
            <= float(quality["maximum_roundtrip_error_pixels"])
        )
    )
    common_support_fraction = float(valid.sum() / spatial_mask.sum())
    if (
        common_support_fraction
        < float(quality["minimum_common_support_fraction"])
    ):
        return {
            "eligible": False,
            "evaluated": False,
            "abstained": True,
            "abstention_reason": (
                "FLOW_FORWARD_BACKWARD_COMMON_SUPPORT_BELOW_FROZEN_0_50"
            ),
            "common_support_fraction": common_support_fraction,
        }
    magnitude_rate = np.linalg.norm(forward, axis=2) / delta_seconds
    radial_rate = (
        forward[:, :, 0] * spatial["radial_x"]
        + forward[:, :, 1] * spatial["radial_y"]
    ) / (delta_seconds * np.maximum(spatial["radius"], radius_min))
    quantile = float(spatial_contract["continuous_summary_quantile"])
    magnitude_values = magnitude_rate[valid]
    radial_values = radial_rate[valid]
    grid_columns, grid_rows = spatial_contract["roi_grid"]
    return {
        "eligible": True,
        "evaluated": True,
        "abstained": False,
        "abstention_reason": None,
        "common_support_fraction": common_support_fraction,
        "forward_backward_roundtrip_error_median_pixels": float(
            np.median(roundtrip[valid])
        ),
        "RAW_FLOW_ENERGY": {
            "q90_flow_magnitude_pixels_per_second": float(
                np.quantile(magnitude_values, quantile)
            )
        },
        "BBOX_LOG_AREA_GROWTH": {
            "evaluated": False,
            "abstained": True,
            "abstention_reason": (
                "BONN_STATIC_SURFACE_HAS_NO_FROZEN_TARGET_BBOX"
            ),
        },
        "UNCOMPENSATED_LOCAL_RADIAL_EXPANSION": {
            "q90_positive_radial_rate_per_second": float(
                np.quantile(np.maximum(radial_values, 0.0), quantile)
            ),
            "q90_signed_radial_rate_per_second": float(
                np.quantile(radial_values, quantile)
            ),
            "median_signed_radial_rate_per_second": float(
                np.median(radial_values)
            ),
            "grid": grid_summaries(
                radial_rate,
                valid,
                int(grid_columns),
                int(grid_rows),
                quantile,
            ),
        },
    }


def produce_sequence(
    sequence: dict[str, Any],
    archive_dir: Path,
    contract: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    archive_path = archive_dir / sequence["archive_filename"]
    if sha256(archive_path) != sequence["archive_sha256"]:
        raise ValueError("discovery archive SHA-256 mismatch")
    parameters = contract["flow_producer"]["parameters"]
    spatial = spatial_arrays()
    traces: list[dict[str, Any]] = []
    frame_identities: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        previous_member: str | None = None
        previous_gray: np.ndarray | None = None
        for pair in sequence["pairs"]:
            if not pair["eligible"]:
                traces.append(
                    {
                        **pair,
                        "session_id": sequence["session_id"],
                        "evaluated": False,
                    }
                )
                continue
            if previous_member != pair["previous_rgb_member"]:
                previous_bgr, previous_gray = decode_rgb(
                    archive.read(pair["previous_rgb_member"])
                )
                frame_identities[pair["previous_rgb_member"]] = {
                    "member": pair["previous_rgb_member"],
                    "decoded_pixel_sha256": decoded_pixel_sha256(previous_bgr),
                    "average_hash_8x8": average_hash(previous_gray),
                }
            assert previous_gray is not None
            current_bgr, current_gray = decode_rgb(
                archive.read(pair["current_rgb_member"])
            )
            frame_identities[pair["current_rgb_member"]] = {
                "member": pair["current_rgb_member"],
                "decoded_pixel_sha256": decoded_pixel_sha256(current_bgr),
                "average_hash_8x8": average_hash(current_gray),
            }
            forward = farneback(previous_gray, current_gray, parameters)
            backward = farneback(current_gray, previous_gray, parameters)
            traces.append(
                {
                    "unit_id": pair["unit_id"],
                    "source_family": "BONN_RGBD_DYNAMIC",
                    "capture_cluster_id": sequence["capture_cluster_id"],
                    "session_id": sequence["session_id"],
                    "previous_timestamp": pair["previous_timestamp"],
                    "current_timestamp": pair["current_timestamp"],
                    "delta_seconds": pair["delta_seconds"],
                    **summarize_pair(
                        forward,
                        backward,
                        pair["delta_seconds"],
                        contract,
                        spatial,
                    ),
                    "truth_pose_depth_cell_or_outcome_read": False,
                }
            )
            previous_member = pair["current_rgb_member"]
            previous_gray = current_gray
    return {
        "session_id": sequence["session_id"],
        "frame_identities": list(frame_identities.values()),
        "trace_count": len(traces),
        "evaluated_trace_count": sum(
            bool(item.get("evaluated")) for item in traces
        ),
    }, traces


def produce(
    pair_manifest: dict[str, Any],
    contract: dict[str, Any],
    archive_dir: Path,
) -> dict[str, Any]:
    if cv2.__version__ != "4.13.0":
        raise ValueError(f"unexpected OpenCV version: {cv2.__version__}")
    sequences: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for sequence in pair_manifest["sequences"]:
        sequence_receipt, sequence_traces = produce_sequence(
            sequence, archive_dir, contract
        )
        sequences.append(sequence_receipt)
        traces.extend(sequence_traces)
    evaluated = [item for item in traces if item.get("evaluated")]
    return {
        "schema_version": "bonn_r1a_base_flow_traces_r0",
        "goal_id": "EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R1",
        "source_family": "BONN_RGBD_DYNAMIC",
        "producer_namespace": "R1A_BASE_RGB_INTRINSICS_ONLY",
        "frozen_input_receipts": {
            "pair_manifest_sha256": None,
            "signal_contract_sha256": None,
        },
        "producer": {
            "opencv_version": cv2.__version__,
            "algorithm": "calcOpticalFlowFarneback",
            "parameters": contract["flow_producer"]["parameters"],
        },
        "sequences": sequences,
        "traces": traces,
        "counts": {
            "pair_count": len(traces),
            "evaluated_pair_count": len(evaluated),
            "abstained_pair_count": len(traces) - len(evaluated),
            "unique_rgb_member_decode_count": sum(
                len(item["frame_identities"]) for item in sequences
            ),
        },
        "namespace_firewall": {
            "rgb_read": True,
            "intrinsics_read": True,
            "pose_read": False,
            "depth_read": False,
            "truth_ledger_read": False,
            "cell_label_read": False,
            "old_frame_or_outcome_read": False,
            "validation_or_holdout_read": False,
        },
        "claim_effect": {
            "Bonn_C2": "BASE_ARM_TRACES_AVAILABLE_ORACLE_NOT_JOINED",
            "Bonn_C1": "ABSTAIN_NO_PURE_ROTATION_DISCOVERY_WINDOW",
            "algorithm_result": "NOT_YET_EVALUATED_AGAINST_TRUTH",
        },
        "terminal": "BONN_R1A_BASE_FLOW_TRACES_FROZEN",
        "status": "VALID",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest", required=True, type=Path)
    parser.add_argument("--signal-contract", required=True, type=Path)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pair_manifest = json.loads(
        args.pair_manifest.read_text(encoding="utf-8")
    )
    contract = json.loads(args.signal_contract.read_text(encoding="utf-8"))
    receipt = produce(pair_manifest, contract, args.archive_dir)
    receipt["frozen_input_receipts"].update(
        {
            "pair_manifest_sha256": sha256(args.pair_manifest),
            "signal_contract_sha256": sha256(args.signal_contract),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "terminal": receipt["terminal"],
                **receipt["counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
