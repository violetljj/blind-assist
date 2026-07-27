"""Geometry-only producer for the frozen EVIMO2 Flea3 sanity_ll role."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pickle
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.lib.format as npformat

from scripts.research.egomotion_compensated_looming.pb_h1_role_proxy.geometry import (
    summarize_translation_induced_geometry,
    translation_induced_geometry,
)


PROTOCOL_ID = "RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R0"
ARCHIVE_SHA256 = "b6a45becaf2d750df9f5f91bcb5c1d61d2f4942c419da8f22dd6ac40981945a8"
WINDOW_SECONDS = 10.0
MAX_DT_SECONDS = 0.100
MIN_COVERAGE = 0.80
MIN_EVALUABLE_PAIRS = 8
MIN_SIGNED_RADIAL = 0.05
MIN_POSITIVE_FRACTION = 0.75


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        allowed = {
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"FORBIDDEN_PICKLE_GLOBAL:{module}.{name}")
        return allowed[(module, name)]


def load_metadata(info_path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    with zipfile.ZipFile(info_path) as archive:
        raw = archive.read("meta.npy")
    stream = io.BytesIO(raw)
    version = npformat.read_magic(stream)
    shape, _fortran, dtype = npformat._read_array_header(stream, version)
    if shape != () or dtype != np.dtype("O"):
        raise ValueError("EVIMO_META_NPY_SHAPE_OR_DTYPE")
    meta_array = _RestrictedNumpyUnpickler(stream).load()
    metadata = meta_array.item()
    with np.load(info_path, allow_pickle=False) as info:
        intrinsic = np.asarray(info["K"], dtype=np.float64)
        distortion = np.asarray(info["D"], dtype=np.float64)
    if metadata.get("meta", {}).get("dist_model") != "radtan":
        raise ValueError("EVIMO_DISTORTION_MODEL")
    return metadata, intrinsic, distortion


def quaternion_rotation_wxyz(value: dict[str, Any]) -> np.ndarray:
    q = np.asarray(
        [value["x"], value["y"], value["z"], value["w"]],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or abs(norm - 1.0) > 0.001:
        raise ValueError("EVIMO_QUATERNION")
    x, y, z, w = q / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def frame_pose(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pose = frame["cam"]["pos"]
    translation = pose["t"]
    center_world = np.asarray(
        [translation["x"], translation["y"], translation["z"]],
        dtype=np.float64,
    )
    return center_world, quaternion_rotation_wxyz(pose["q"])


def depth_samples(
    depth_mm: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if depth_mm.shape != (1552, 2080) or depth_mm.dtype != np.uint16:
        raise ValueError("EVIMO_DEPTH_SHAPE_OR_DTYPE")
    ys = np.arange(0, depth_mm.shape[0], 8, dtype=np.int64)
    xs = np.arange(0, depth_mm.shape[1], 8, dtype=np.int64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    distorted = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float64)
    rectified = cv2.undistortPoints(
        distorted.reshape(-1, 1, 2),
        intrinsic,
        distortion,
        P=intrinsic,
    ).reshape(-1, 2)
    depth_m = depth_mm[grid_y, grid_x].ravel().astype(np.float64) / 1000.0
    return rectified, depth_m


def relative_pose(
    previous: tuple[np.ndarray, np.ndarray],
    current: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    previous_center, previous_rotation = previous
    current_center, current_rotation = current
    return (
        current_rotation.T @ previous_rotation,
        current_rotation.T @ (previous_center - current_center),
    )


def canonical_json_sha(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluate_sequence(
    info_path: Path,
    depth_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata, intrinsic, distortion = load_metadata(info_path)
    frames = metadata["frames"]
    timestamps = [float(frame["ts"]) for frame in frames]
    ids = [int(frame["id"]) for frame in frames]
    if (
        len(frames) < 2
        or any(a >= b for a, b in zip(timestamps, timestamps[1:]))
        or ids != list(range(len(ids)))
    ):
        raise ValueError("EVIMO_FRAME_ORDER_OR_IDENTITY")
    start = timestamps[0]
    end = start + WINDOW_SECONDS
    selected = [frame for frame in frames if start <= float(frame["ts"]) < end]
    pair_records: list[dict[str, Any]] = []
    with np.load(depth_path, allow_pickle=False) as depths:
        expected_keys = {f"depth_{int(frame['id']):010d}" for frame in frames}
        actual_keys = set(depths.files)
        if not actual_keys.issubset(expected_keys):
            raise ValueError("EVIMO_DEPTH_EXTRA_KEY_IDENTITY")
        for previous, current in zip(selected, selected[1:]):
            previous_ts = float(previous["ts"])
            current_ts = float(current["ts"])
            dt_s = current_ts - previous_ts
            record: dict[str, Any] = {
                "sequence_id": info_path.parent.name,
                "window_start_s": start,
                "window_end_s": end,
                "previous_frame_id": int(previous["id"]),
                "current_frame_id": int(current["id"]),
                "previous_timestamp_s": previous_ts,
                "current_timestamp_s": current_ts,
                "dt_s": dt_s,
            }
            if not (0.0 < dt_s <= MAX_DT_SECONDS):
                record.update({"evaluable": False, "reason": "PAIR_DT_OUT_OF_RANGE"})
                pair_records.append(record)
                continue
            if (
                "cam" not in previous
                or "pos" not in previous["cam"]
                or "cam" not in current
                or "pos" not in current["cam"]
            ):
                record.update({"evaluable": False, "reason": "POSE_MISSING"})
                pair_records.append(record)
                continue
            depth_key = f"depth_{int(previous['id']):010d}"
            if depth_key not in actual_keys:
                record.update({"evaluable": False, "reason": "DEPTH_MEMBER_MISSING"})
                pair_records.append(record)
                continue
            pixels, depth_m = depth_samples(
                np.asarray(depths[depth_key]),
                intrinsic,
                distortion,
            )
            rotation, translation = relative_pose(
                frame_pose(previous),
                frame_pose(current),
            )
            summary = summarize_translation_induced_geometry(
                translation_induced_geometry(
                    pixels,
                    depth_m,
                    intrinsic,
                    rotation,
                    translation,
                    dt_s,
                    image_size_wh=(2080, 1552),
                    minimum_radius_px=8.0,
                    zbuffer=True,
                )
            )
            record.update(summary)
            if not summary["evaluable"]:
                record["reason"] = "NO_VALID_GEOMETRY_SAMPLES"
            pair_records.append(record)
    evaluable = [record for record in pair_records if record["evaluable"]]
    coverage = len(evaluable) / len(pair_records) if pair_records else 0.0
    signed = (
        float(
            np.median(
                [record["median_signed_radial_expansion_per_s"] for record in evaluable]
            )
        )
        if evaluable
        else None
    )
    positive = (
        float(
            np.median(
                [record["radial_expansion_positive_fraction"] for record in evaluable]
            )
        )
        if evaluable
        else None
    )
    parallax = (
        float(
            np.median(
                [record["q90_time_normalized_parallax_rad_per_s"] for record in evaluable]
            )
        )
        if evaluable
        else None
    )
    admitted = bool(
        coverage >= MIN_COVERAGE
        and len(evaluable) >= MIN_EVALUABLE_PAIRS
        and signed is not None
        and signed >= MIN_SIGNED_RADIAL
        and positive is not None
        and positive >= MIN_POSITIVE_FRACTION
    )
    window = {
        "sequence_id": info_path.parent.name,
        "window_start_s": start,
        "window_end_s": end,
        "candidate_pair_count": len(pair_records),
        "evaluable_pair_count": len(evaluable),
        "candidate_pair_coverage": coverage,
        "median_signed_radial_expansion_per_s": signed,
        "median_radial_expansion_positive_fraction": positive,
        "median_q90_time_normalized_parallax_rad_per_s": parallax,
        "admitted": admitted,
        "content_identity": (
            f"sha256:{ARCHIVE_SHA256}#{info_path.parent.name}"
            f"[{start:.9f},{end:.9f})"
        ),
    }
    return window, pair_records


def run(source_root: Path, output_dir: Path, bindings: dict[str, Path]) -> dict[str, Any]:
    binding_hashes = {name: sha256_file(path) for name, path in bindings.items()}
    if binding_hashes["archive"] != ARCHIVE_SHA256:
        raise ValueError("ARCHIVE_SHA256_MISMATCH")
    info_paths = sorted(source_root.rglob("dataset_info.npz"))
    if len(info_paths) != 13:
        raise ValueError("EVIMO_SEQUENCE_COUNT")
    windows: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    source_members: list[dict[str, Any]] = []
    for info_path in info_paths:
        depth_path = info_path.with_name("dataset_depth.npz")
        if not depth_path.is_file():
            raise ValueError("EVIMO_DEPTH_MEMBER_MISSING")
        window, sequence_pairs = evaluate_sequence(info_path, depth_path)
        windows.append(window)
        pairs.extend(sequence_pairs)
        source_members.append(
            {
                "sequence_id": info_path.parent.name,
                "info_sha256": sha256_file(info_path),
                "depth_sha256": sha256_file(depth_path),
            }
        )
    admitted_windows = [window for window in windows if window["admitted"]]
    terminal = (
        "REAL_POSITIVE_APPROACH_ROLE_ADMITTED / VALID"
        if admitted_windows
        else "HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "pair_ledger.jsonl"
    with ledger_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in pairs:
            stream.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    result = {
        "schema_version": "rcle.real_positive_approach_role.result.v1",
        "protocol_id": PROTOCOL_ID,
        "terminal": terminal,
        "authority": "DATA_ROLE_ONLY_RGB_ALGORITHM_NOT_AUTHORIZED",
        "source_identity": {
            "candidate_id": "EVIMO2_V2_FLEA3_SANITY_LL",
            "archive_bytes": bindings["archive"].stat().st_size,
            "archive_sha256": binding_hashes["archive"],
            "source_members": source_members,
        },
        "access": {
            "read_kinds": ["dataset_info.npz", "dataset_depth.npz"],
            "rgb_pixels_read": False,
            "events_read": False,
            "masks_read": False,
            "algorithm_outcome_read": False,
            "replacement_source_count": 0,
        },
        "identity_and_independence": {
            "independence_group": "EVIMO2_V2_FLEA3_SANITY_LL_CAPTURE_FAMILY",
            "ancestry": [
                "EVIMO2_V2_OFFICIAL",
                "EVIMO2_V2_FLEA3",
                "EVIMO2_V2_FLEA3_SANITY_LL",
                PROTOCOL_ID,
            ],
            "non_overlap_groups": [
                "TUM_FR2_RPY_SINGLE_SEQUENCE_3a35b799",
                "BONN_FROZEN_COHORT_513b770d",
                "RCLE_PHASE_A_SYNTHETIC_GENERATOR_FAMILY",
            ],
            "reuse_policy": "GEOMETRY_SELECTED_REAL_APPROACH_CANARY_ONLY; NEVER_CONFIRMATION; MAY_REMAIN_CANARY_OR_REGRESSION",
            "future_confirmation_exclusion": "EVIMO2_V2_FLEA3_SANITY_LL_CAPTURE_FAMILY",
        },
        "frozen_gates": {
            "window_seconds": WINDOW_SECONDS,
            "maximum_pair_dt_s": MAX_DT_SECONDS,
            "candidate_pair_coverage_min": MIN_COVERAGE,
            "minimum_evaluable_pairs": MIN_EVALUABLE_PAIRS,
            "median_signed_radial_expansion_per_s_min": MIN_SIGNED_RADIAL,
            "median_positive_fraction_min": MIN_POSITIVE_FRACTION,
        },
        "windows": windows,
        "admitted_window_count": len(admitted_windows),
        "admitted_content_identities": [
            window["content_identity"] for window in admitted_windows
        ],
        "pair_record_count": len(pairs),
        "pair_ledger_sha256": sha256_file(ledger_path),
        "bindings": binding_hashes,
        "algorithm_implementation_or_execution_authorized": False,
    }
    result["result_payload_sha256"] = canonical_json_sha(result)
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    receipt = {
        "schema_version": "rcle.real_positive_approach_role.receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "terminal": terminal,
        "result_sha256": sha256_file(result_path),
        "pair_ledger_sha256": sha256_file(ledger_path),
        "archive_sha256": ARCHIVE_SHA256,
        "algorithm_outcome_read": False,
        "replacement_source_count": 0,
    }
    (output_dir / "receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--claim", required=True, type=Path)
    parser.add_argument("--source-resolution", required=True, type=Path)
    args = parser.parse_args()
    result = run(
        args.source_root,
        args.output_dir,
        {
            "archive": args.archive,
            "contract": args.contract,
            "claim": args.claim,
            "source_resolution": args.source_resolution,
        },
    )
    print(
        json.dumps(
            {
                "terminal": result["terminal"],
                "admitted_window_count": result["admitted_window_count"],
                "pair_record_count": result["pair_record_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
