#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parent))
import produce_bonn_r1a_base_flow_traces_r0 as base  # noqa: E402


FX = 542.822841
FY = 542.576870
CX = 315.593520
CY = 237.756098
DEPTH_SCALE = 5000.0
MAX_POSE_JOIN_DELTA_SECONDS = 0.040
MAX_DEPTH_JOIN_DELTA_SECONDS = 0.040
MAX_DEPTH_METERS = 10.0
T_ROS = np.asarray(
    [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
T_M = np.asarray(
    [
        [1.0157, 0.1828, -0.2389, 0.0113],
        [0.0009, -0.8431, -0.6413, -0.0098],
        [-0.3009, 0.6147, -0.8085, 0.0111],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(data: bytes, columns: int) -> list[list[str]]:
    result = [
        line.split()
        for line in data.decode("utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    if not result or any(len(row) != columns for row in result):
        raise ValueError("invalid source index")
    times = [float(row[0]) for row in result]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("timestamps must be strictly increasing")
    return result


def nearest(
    rows_: list[list[str]],
    times: list[float],
    timestamp: float,
    maximum_delta_seconds: float,
) -> tuple[list[str], float] | None:
    index = bisect.bisect_left(times, timestamp)
    candidates = rows_[max(0, index - 1) : min(len(rows_), index + 1)]
    selected = min(candidates, key=lambda row: abs(float(row[0]) - timestamp))
    delta = abs(float(selected[0]) - timestamp)
    return (
        (selected, delta)
        if delta <= maximum_delta_seconds
        else None
    )


def quaternion_xyzw_rotation(values: np.ndarray) -> np.ndarray:
    x, y, z, w = values / np.linalg.norm(values)
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def pose_matrix(row: list[str]) -> np.ndarray:
    values = np.asarray([float(value) for value in row[1:]], dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_xyzw_rotation(values[3:7])
    matrix[:3, 3] = values[:3]
    return matrix


def map_from_camera(pose: np.ndarray) -> np.ndarray:
    return np.linalg.inv(T_ROS) @ pose @ T_ROS @ T_M


def closest_rotation(linear: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(linear)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return rotation


def relative_current_from_previous(
    previous_pose: np.ndarray, current_pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    map_from_previous = map_from_camera(previous_pose)
    map_from_current = map_from_camera(current_pose)
    full = np.linalg.inv(map_from_current) @ map_from_previous
    previous_rotation = closest_rotation(map_from_previous[:3, :3])
    current_rotation = closest_rotation(map_from_current[:3, :3])
    rotation = current_rotation.T @ previous_rotation
    return rotation, full


def rotational_flow(
    rotation_current_from_previous: np.ndarray,
    spatial: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    x = spatial["x"].reshape(-1).astype(np.float64)
    y = spatial["y"].reshape(-1).astype(np.float64)
    rays = np.vstack(((x - CX) / FX, (y - CY) / FY, np.ones_like(x)))
    rotated = rotation_current_from_previous @ rays
    z = rotated[2]
    u = FX * rotated[0] / np.maximum(z, 1e-12) + CX
    v = FY * rotated[1] / np.maximum(z, 1e-12) + CY
    flow = np.column_stack((u - x, v - y)).reshape(
        base.IMAGE_HEIGHT, base.IMAGE_WIDTH, 2
    )
    valid = (
        (z > 0.0)
        & (u >= 0.0)
        & (u < base.IMAGE_WIDTH - 1)
        & (v >= 0.0)
        & (v < base.IMAGE_HEIGHT - 1)
    ).reshape(base.IMAGE_HEIGHT, base.IMAGE_WIDTH)
    return flow.astype(np.float32), valid


def flow_quality_mask(
    forward: np.ndarray,
    backward: np.ndarray,
    contract: dict[str, Any],
    spatial: dict[str, np.ndarray],
) -> tuple[np.ndarray, float]:
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
    spatial_mask = np.zeros(
        (base.IMAGE_HEIGHT, base.IMAGE_WIDTH), dtype=bool
    )
    spatial_mask[
        border : base.IMAGE_HEIGHT - border,
        border : base.IMAGE_WIDTH - border,
    ] = True
    spatial_mask &= spatial["radius"] >= radius_min
    inside = (
        (map_x >= 0.0)
        & (map_x < base.IMAGE_WIDTH - 1)
        & (map_y >= 0.0)
        & (map_y < base.IMAGE_HEIGHT - 1)
    )
    valid = (
        spatial_mask
        & inside
        & np.isfinite(roundtrip)
        & (
            roundtrip
            <= float(quality["maximum_roundtrip_error_pixels"])
        )
    )
    return valid, float(valid.sum() / spatial_mask.sum())


def radial_summary(
    residual_flow: np.ndarray,
    valid: np.ndarray,
    delta_seconds: float,
    contract: dict[str, Any],
    spatial: dict[str, np.ndarray],
) -> dict[str, float]:
    radius_min = float(
        contract["spatial_contract"][
            "principal_point_exclusion_radius_pixels"
        ]
    )
    quantile = float(
        contract["spatial_contract"]["continuous_summary_quantile"]
    )
    radial_rate = (
        residual_flow[:, :, 0] * spatial["radial_x"]
        + residual_flow[:, :, 1] * spatial["radial_y"]
    ) / (delta_seconds * np.maximum(spatial["radius"], radius_min))
    values = radial_rate[valid]
    magnitude = np.linalg.norm(residual_flow, axis=2)[valid] / delta_seconds
    return {
        "q90_positive_radial_rate_per_second": float(
            np.quantile(np.maximum(values, 0.0), quantile)
        ),
        "q90_signed_radial_rate_per_second": float(
            np.quantile(values, quantile)
        ),
        "median_signed_radial_rate_per_second": float(np.median(values)),
        "q90_residual_flow_magnitude_pixels_per_second": float(
            np.quantile(magnitude, quantile)
        ),
    }


def decode_depth(data: bytes) -> np.ndarray:
    depth = np.asarray(Image.open(io.BytesIO(data)), dtype=np.uint16)
    if depth.shape != (base.IMAGE_HEIGHT, base.IMAGE_WIDTH):
        raise ValueError("unexpected Bonn depth shape")
    return depth.astype(np.float64) / DEPTH_SCALE


def rigid_flow(
    depth: np.ndarray,
    current_from_previous: np.ndarray,
    spatial: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    z = depth.reshape(-1)
    x = spatial["x"].reshape(-1).astype(np.float64)
    y = spatial["y"].reshape(-1).astype(np.float64)
    points = np.vstack(((x - CX) * z / FX, (y - CY) * z / FY, z))
    transformed = (
        current_from_previous[:3, :3] @ points
        + current_from_previous[:3, 3:4]
    )
    target_z = transformed[2]
    u = FX * transformed[0] / np.maximum(target_z, 1e-12) + CX
    v = FY * transformed[1] / np.maximum(target_z, 1e-12) + CY
    flow = np.column_stack((u - x, v - y)).reshape(
        base.IMAGE_HEIGHT, base.IMAGE_WIDTH, 2
    )
    valid = (
        (z > 0.0)
        & (z <= MAX_DEPTH_METERS)
        & (target_z > 0.0)
        & (u >= 0.0)
        & (u < base.IMAGE_WIDTH - 1)
        & (v >= 0.0)
        & (v < base.IMAGE_HEIGHT - 1)
    ).reshape(base.IMAGE_HEIGHT, base.IMAGE_WIDTH)
    return flow.astype(np.float32), valid


def verify_base(
    recomputed: dict[str, Any], frozen: dict[str, Any]
) -> None:
    checks = [
        (
            recomputed["RAW_FLOW_ENERGY"][
                "q90_flow_magnitude_pixels_per_second"
            ],
            frozen["RAW_FLOW_ENERGY"][
                "q90_flow_magnitude_pixels_per_second"
            ],
        ),
        (
            recomputed["UNCOMPENSATED_LOCAL_RADIAL_EXPANSION"][
                "q90_positive_radial_rate_per_second"
            ],
            frozen["UNCOMPENSATED_LOCAL_RADIAL_EXPANSION"][
                "q90_positive_radial_rate_per_second"
            ],
        ),
    ]
    if any(not np.isclose(left, right, rtol=0.0, atol=1e-12) for left, right in checks):
        raise ValueError("recomputed base trace differs from frozen trace")


def produce(
    pair_manifest: dict[str, Any],
    contract: dict[str, Any],
    base_receipt: dict[str, Any],
    archive_dir: Path,
) -> dict[str, Any]:
    frozen_base = {
        item["unit_id"]: item for item in base_receipt["traces"]
    }
    parameters = contract["flow_producer"]["parameters"]
    spatial = base.spatial_arrays()
    traces: list[dict[str, Any]] = []
    depth_decode_count = 0
    rgb_decode_count = 0
    for sequence in pair_manifest["sequences"]:
        archive_path = archive_dir / sequence["archive_filename"]
        if sha256(archive_path) != sequence["archive_sha256"]:
            raise ValueError("discovery archive SHA-256 mismatch")
        prefix = f"{sequence['session_id']}/"
        with zipfile.ZipFile(archive_path) as archive:
            poses = rows(archive.read(f"{prefix}groundtruth.txt"), 8)
            depths = rows(archive.read(f"{prefix}depth.txt"), 2)
            pose_times = [float(row[0]) for row in poses]
            depth_times = [float(row[0]) for row in depths]
            previous_member: str | None = None
            previous_gray: np.ndarray | None = None
            for pair in sequence["pairs"]:
                frozen = frozen_base[pair["unit_id"]]
                if previous_member != pair["previous_rgb_member"]:
                    _, previous_gray = base.decode_rgb(
                        archive.read(pair["previous_rgb_member"])
                    )
                    rgb_decode_count += 1
                assert previous_gray is not None
                _, current_gray = base.decode_rgb(
                    archive.read(pair["current_rgb_member"])
                )
                rgb_decode_count += 1
                forward = base.farneback(
                    previous_gray, current_gray, parameters
                )
                backward = base.farneback(
                    current_gray, previous_gray, parameters
                )
                recomputed = base.summarize_pair(
                    forward,
                    backward,
                    pair["delta_seconds"],
                    contract,
                    spatial,
                )
                verify_base(recomputed, frozen)
                previous_pose_match = nearest(
                    poses,
                    pose_times,
                    pair["previous_timestamp"],
                    MAX_POSE_JOIN_DELTA_SECONDS,
                )
                current_pose_match = nearest(
                    poses,
                    pose_times,
                    pair["current_timestamp"],
                    MAX_POSE_JOIN_DELTA_SECONDS,
                )
                if previous_pose_match is None or current_pose_match is None:
                    traces.append(
                        {
                            "unit_id": pair["unit_id"],
                            "session_id": sequence["session_id"],
                            "eligible": False,
                            "evaluated": False,
                            "abstained": True,
                            "abstention_reason": (
                                "POSE_JOIN_EXCEEDS_FROZEN_40MS_HARD_CAP"
                            ),
                        }
                    )
                    previous_member = pair["current_rgb_member"]
                    previous_gray = current_gray
                    continue
                previous_pose_row, previous_pose_delta = previous_pose_match
                current_pose_row, current_pose_delta = current_pose_match
                rotation, full = relative_current_from_previous(
                    pose_matrix(previous_pose_row),
                    pose_matrix(current_pose_row),
                )
                rotation_prediction, rotation_valid = rotational_flow(
                    rotation, spatial
                )
                quality_valid, base_support = flow_quality_mask(
                    forward, backward, contract, spatial
                )
                oracle_valid = quality_valid & rotation_valid
                oracle_support = float(
                    oracle_valid.sum() / max(quality_valid.sum(), 1)
                )
                minimum_support = float(
                    contract["flow_producer"][
                        "forward_backward_quality"
                    ]["minimum_common_support_fraction"]
                )
                if oracle_support < minimum_support:
                    traces.append(
                        {
                            "unit_id": pair["unit_id"],
                            "session_id": sequence["session_id"],
                            "eligible": False,
                            "evaluated": False,
                            "abstained": True,
                            "abstention_reason": (
                                "ORACLE_ROTATION_PROJECTION_SUPPORT_BELOW_"
                                "FROZEN_0_50"
                            ),
                            "base_common_support_fraction": base_support,
                            "oracle_relative_support_fraction": oracle_support,
                        }
                    )
                    previous_member = pair["current_rgb_member"]
                    previous_gray = current_gray
                    continue
                oracle_residual = forward - rotation_prediction
                oracle_summary = radial_summary(
                    oracle_residual,
                    oracle_valid,
                    pair["delta_seconds"],
                    contract,
                    spatial,
                )

                depth_match = nearest(
                    depths,
                    depth_times,
                    pair["previous_timestamp"],
                    MAX_DEPTH_JOIN_DELTA_SECONDS,
                )
                full_diagnostic: dict[str, Any]
                if depth_match is None:
                    full_diagnostic = {
                        "evaluated": False,
                        "abstained": True,
                        "abstention_reason": (
                            "DEPTH_JOIN_EXCEEDS_FROZEN_40MS_HARD_CAP"
                        ),
                    }
                else:
                    depth_row, depth_delta = depth_match
                    depth = decode_depth(archive.read(f"{prefix}{depth_row[1]}"))
                    depth_decode_count += 1
                    rigid_prediction, rigid_valid = rigid_flow(
                        depth, full, spatial
                    )
                    full_valid = quality_valid & rigid_valid
                    full_support = float(
                        full_valid.sum() / max(quality_valid.sum(), 1)
                    )
                    if full_support < minimum_support:
                        full_diagnostic = {
                            "evaluated": False,
                            "abstained": True,
                            "abstention_reason": (
                                "FULL_6DOF_DEPTH_SUPPORT_BELOW_FROZEN_0_50"
                            ),
                            "relative_support_fraction": full_support,
                            "depth_join_delta_seconds": depth_delta,
                        }
                    else:
                        full_diagnostic = {
                            "evaluated": True,
                            "abstained": False,
                            "abstention_reason": None,
                            "relative_support_fraction": full_support,
                            "depth_join_delta_seconds": depth_delta,
                            **radial_summary(
                                forward - rigid_prediction,
                                full_valid,
                                pair["delta_seconds"],
                                contract,
                                spatial,
                            ),
                        }
                traces.append(
                    {
                        "unit_id": pair["unit_id"],
                        "source_family": "BONN_RGBD_DYNAMIC",
                        "capture_cluster_id": sequence[
                            "capture_cluster_id"
                        ],
                        "session_id": sequence["session_id"],
                        "previous_timestamp": pair["previous_timestamp"],
                        "current_timestamp": pair["current_timestamp"],
                        "delta_seconds": pair["delta_seconds"],
                        "eligible": True,
                        "evaluated": True,
                        "abstained": False,
                        "abstention_reason": None,
                        "previous_pose_join_delta_seconds": (
                            previous_pose_delta
                        ),
                        "current_pose_join_delta_seconds": current_pose_delta,
                        "base_common_support_fraction": base_support,
                        "oracle_relative_support_fraction": oracle_support,
                        "ORACLE_ROTATION_COMPENSATION": oracle_summary,
                        "FULL_6DOF_RESIDUAL_DIAGNOSTIC": full_diagnostic,
                        "closing_truth_or_outcome_read": False,
                    }
                )
                previous_member = pair["current_rgb_member"]
                previous_gray = current_gray
    evaluated = [item for item in traces if item.get("evaluated")]
    full_evaluated = [
        item
        for item in evaluated
        if item["FULL_6DOF_RESIDUAL_DIAGNOSTIC"]["evaluated"]
    ]
    return {
        "schema_version": "bonn_r1a_oracle_flow_traces_r0",
        "goal_id": "EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R1",
        "source_family": "BONN_RGBD_DYNAMIC",
        "producer_namespace": "R1A_ORACLE_ORIENTATION_AND_FULL_6DOF_DIAGNOSTIC",
        "frozen_input_receipts": {
            "pair_manifest_sha256": None,
            "signal_contract_sha256": None,
            "base_flow_trace_sha256": None,
        },
        "traces": traces,
        "counts": {
            "pair_count": len(traces),
            "oracle_evaluated_pair_count": len(evaluated),
            "oracle_abstained_pair_count": len(traces) - len(evaluated),
            "full_6dof_evaluated_pair_count": len(full_evaluated),
            "rgb_member_decode_count": rgb_decode_count,
            "depth_member_decode_count": depth_decode_count,
            "frozen_base_trace_reverification_count": len(traces),
        },
        "namespace_firewall": {
            "base_trace_was_hash_frozen_before_oracle": True,
            "orientation_truth_read": True,
            "full_pose_truth_read": True,
            "source_depth_read_for_6dof_diagnostic": True,
            "closing_truth_ledger_read": False,
            "cell_label_or_outcome_read": False,
            "old_frame_or_outcome_read": False,
            "validation_or_holdout_read": False,
            "deployable_rotation_estimator_run": False,
        },
        "claim_effect": {
            "Bonn_C2": "ORACLE_TRACE_AVAILABLE_TRUTH_JOIN_PENDING",
            "Bonn_C1": "ABSTAIN_NO_PURE_ROTATION_DISCOVERY_WINDOW",
            "algorithm_result": "NOT_YET_EVALUATED_AGAINST_TRUTH",
        },
        "terminal": "BONN_R1A_ORACLE_FLOW_TRACES_FROZEN_TRUTH_JOIN_PENDING",
        "status": "VALID",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest", required=True, type=Path)
    parser.add_argument("--signal-contract", required=True, type=Path)
    parser.add_argument("--base-traces", required=True, type=Path)
    parser.add_argument("--archive-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pair_manifest = json.loads(
        args.pair_manifest.read_text(encoding="utf-8")
    )
    contract = json.loads(args.signal_contract.read_text(encoding="utf-8"))
    base_receipt = json.loads(
        args.base_traces.read_text(encoding="utf-8")
    )
    receipt = produce(
        pair_manifest, contract, base_receipt, args.archive_dir
    )
    receipt["frozen_input_receipts"].update(
        {
            "pair_manifest_sha256": sha256(args.pair_manifest),
            "signal_contract_sha256": sha256(args.signal_contract),
            "base_flow_trace_sha256": sha256(args.base_traces),
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
