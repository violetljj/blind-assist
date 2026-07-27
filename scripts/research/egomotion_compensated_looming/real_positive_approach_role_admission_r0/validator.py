"""Independent source replay for real positive approach role admission R0."""

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

from scripts.research.egomotion_compensated_looming.real_data_geometry_canary_r0.validator import (
    _independent_geometry,
)


PROTOCOL_ID = "RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R0"
ARCHIVE_SHA256 = "b6a45becaf2d750df9f5f91bcb5c1d61d2f4942c419da8f22dd6ac40981945a8"
WINDOW_SECONDS = 10.0
MAX_DT_SECONDS = 0.100
MIN_COVERAGE = 0.80
MIN_EVALUABLE_PAIRS = 8
MIN_SIGNED_RADIAL = 0.05
MIN_POSITIVE_FRACTION = 0.75
GEOMETRY_CONFIG = {
    "image_size_wh": [2080, 1552],
    "minimum_radius_px": 8.0,
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SafeNumpyObject(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        table = {
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        value = table.get((module, name))
        if value is None:
            raise pickle.UnpicklingError(f"FORBIDDEN_PICKLE_GLOBAL:{module}.{name}")
        return value


def source_info(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        payload = archive.read("meta.npy")
    stream = io.BytesIO(payload)
    version = npformat.read_magic(stream)
    shape, _order, dtype = npformat._read_array_header(stream, version)
    if shape != () or dtype != np.dtype("O"):
        raise ValueError("META_CONTAINER")
    metadata = SafeNumpyObject(stream).load().item()
    with np.load(path, allow_pickle=False) as values:
        intrinsic = np.array(values["K"], dtype=np.float64, copy=True)
        distortion = np.array(values["D"], dtype=np.float64, copy=True)
    return metadata, intrinsic, distortion


def rotation_from_wxyz(q: dict[str, Any]) -> np.ndarray:
    x, y, z, w = map(float, (q["x"], q["y"], q["z"], q["w"]))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if abs(norm - 1.0) > 0.001:
        raise ValueError("QUATERNION_NORM")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    native = frame["cam"]["pos"]
    t = native["t"]
    center = np.array([t["x"], t["y"], t["z"]], dtype=np.float64)
    return center, rotation_from_wxyz(native["q"])


def motion(
    before: tuple[np.ndarray, np.ndarray],
    after: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    center_before, rotation_before = before
    center_after, rotation_after = after
    return (
        rotation_after.T.dot(rotation_before),
        rotation_after.T.dot(center_before - center_after),
    )


def samples(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if depth.shape != (1552, 2080) or depth.dtype != np.uint16:
        raise ValueError("DEPTH_FORMAT")
    yy, xx = np.mgrid[0:1552:8, 0:2080:8]
    distorted = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
    rectified = cv2.undistortPoints(
        distorted[:, None, :],
        intrinsic,
        distortion,
        P=intrinsic,
    )[:, 0, :]
    depth_m = depth[yy, xx].reshape(-1).astype(np.float64) * 0.001
    return rectified, depth_m


def replay_sequence(
    info_path: Path,
    depth_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata, intrinsic, distortion = source_info(info_path)
    frames = metadata["frames"]
    timestamps = [float(frame["ts"]) for frame in frames]
    identifiers = [int(frame["id"]) for frame in frames]
    if (
        identifiers != list(range(len(identifiers)))
        or any(left >= right for left, right in zip(timestamps, timestamps[1:]))
    ):
        raise ValueError("FRAME_IDENTITY")
    window_start = timestamps[0]
    window_end = window_start + WINDOW_SECONDS
    window_frames = [
        frame for frame in frames if window_start <= float(frame["ts"]) < window_end
    ]
    records: list[dict[str, Any]] = []
    with np.load(depth_path, allow_pickle=False) as depths:
        expected = {f"depth_{identifier:010d}" for identifier in identifiers}
        actual = set(depths.files)
        if not actual.issubset(expected):
            raise ValueError("EXTRA_DEPTH_KEY")
        for before, after in zip(window_frames, window_frames[1:]):
            before_ts, after_ts = float(before["ts"]), float(after["ts"])
            dt = after_ts - before_ts
            record: dict[str, Any] = {
                "sequence_id": info_path.parent.name,
                "window_start_s": window_start,
                "window_end_s": window_end,
                "previous_frame_id": int(before["id"]),
                "current_frame_id": int(after["id"]),
                "previous_timestamp_s": before_ts,
                "current_timestamp_s": after_ts,
                "dt_s": dt,
            }
            if not 0.0 < dt <= MAX_DT_SECONDS:
                record.update(evaluable=False, reason="PAIR_DT_OUT_OF_RANGE")
                records.append(record)
                continue
            if (
                "cam" not in before
                or "pos" not in before["cam"]
                or "cam" not in after
                or "pos" not in after["cam"]
            ):
                record.update(evaluable=False, reason="POSE_MISSING")
                records.append(record)
                continue
            key = f"depth_{int(before['id']):010d}"
            if key not in actual:
                record.update(evaluable=False, reason="DEPTH_MEMBER_MISSING")
                records.append(record)
                continue
            pixels, depths_m = samples(
                np.asarray(depths[key]),
                intrinsic,
                distortion,
            )
            relative_rotation, relative_translation = motion(pose(before), pose(after))
            try:
                geometry, valid_fraction = _independent_geometry(
                    pixels,
                    depths_m,
                    intrinsic,
                    relative_rotation,
                    relative_translation,
                    dt,
                    GEOMETRY_CONFIG,
                )
            except ValueError as error:
                if str(error) != "PB_H1_NO_VISIBLE_DEPTH_SUPPORT":
                    raise
                record.update(
                    {
                        "evaluable": False,
                        "source_count": int(pixels.shape[0]),
                        "valid_count": 0,
                        "valid_fraction": 0.0,
                        "raw_translation_speed_m_s": float(
                            np.linalg.norm(relative_translation) / dt
                        ),
                        "reason": "NO_VALID_GEOMETRY_SAMPLES",
                    }
                )
                records.append(record)
                continue
            source_count = int(pixels.shape[0])
            record.update(
                {
                    "evaluable": True,
                    "source_count": source_count,
                    "valid_count": int(round(valid_fraction * source_count)),
                    "valid_fraction": valid_fraction,
                    **geometry,
                }
            )
            records.append(record)
    evaluable = [record for record in records if record["evaluable"]]
    coverage = len(evaluable) / len(records) if records else 0.0

    def median(field: str) -> float | None:
        return (
            float(np.median([record[field] for record in evaluable]))
            if evaluable
            else None
        )

    signed = median("median_signed_radial_expansion_per_s")
    positive = median("radial_expansion_positive_fraction")
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
        "window_start_s": window_start,
        "window_end_s": window_end,
        "candidate_pair_count": len(records),
        "evaluable_pair_count": len(evaluable),
        "candidate_pair_coverage": coverage,
        "median_signed_radial_expansion_per_s": signed,
        "median_radial_expansion_positive_fraction": positive,
        "median_q90_time_normalized_parallax_rad_per_s": median(
            "q90_time_normalized_parallax_rad_per_s"
        ),
        "admitted": admitted,
        "content_identity": (
            f"sha256:{ARCHIVE_SHA256}#{info_path.parent.name}"
            f"[{window_start:.9f},{window_end:.9f})"
        ),
    }
    return window, records


def close(left: Any, right: Any, *, tolerance: float = 1e-11) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        if left is None or right is None:
            return left is right
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            close(left[key], right[key], tolerance=tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            close(a, b, tolerance=tolerance) for a, b in zip(left, right)
        )
    return left == right


def validate(
    source_root: Path,
    formal_dir: Path,
    archive_path: Path,
    contract_path: Path,
    claim_path: Path,
    source_resolution_path: Path,
) -> dict[str, Any]:
    result_path = formal_dir / "result.json"
    receipt_path = formal_dir / "receipt.json"
    ledger_path = formal_dir / "pair_ledger.jsonl"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    produced_pairs = [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    source_files = sorted(source_root.rglob("*"))
    forbidden_source_files = [
        path.as_posix()
        for path in source_files
        if path.is_file() and path.name not in {"dataset_info.npz", "dataset_depth.npz"}
    ]
    replay_windows: list[dict[str, Any]] = []
    replay_pairs: list[dict[str, Any]] = []
    for info_path in sorted(source_root.rglob("dataset_info.npz")):
        window, pairs = replay_sequence(info_path, info_path.with_name("dataset_depth.npz"))
        replay_windows.append(window)
        replay_pairs.extend(pairs)
    pair_mismatch = sum(
        not close(produced, replayed)
        for produced, replayed in zip(produced_pairs, replay_pairs)
    ) + abs(len(produced_pairs) - len(replay_pairs))
    window_mismatch = sum(
        not close(produced, replayed)
        for produced, replayed in zip(result["windows"], replay_windows)
    ) + abs(len(result["windows"]) - len(replay_windows))
    admitted = [window for window in replay_windows if window["admitted"]]
    expected_terminal = (
        "REAL_POSITIVE_APPROACH_ROLE_ADMITTED / VALID"
        if admitted
        else "HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID"
    )
    binding_mismatches = []
    current_bindings = {
        "archive": file_hash(archive_path),
        "contract": file_hash(contract_path),
        "claim": file_hash(claim_path),
        "source_resolution": file_hash(source_resolution_path),
    }
    for name, observed in current_bindings.items():
        if result["bindings"].get(name) != observed:
            binding_mismatches.append(name)
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    access_violations = []
    if forbidden_source_files:
        access_violations.append("FORBIDDEN_EXTRACTED_SOURCE_FILE")
    if result["access"] != {
        "read_kinds": ["dataset_info.npz", "dataset_depth.npz"],
        "rgb_pixels_read": False,
        "events_read": False,
        "masks_read": False,
        "algorithm_outcome_read": False,
        "replacement_source_count": 0,
    }:
        access_violations.append("RESULT_ACCESS_DECLARATION")
    if claim["source_access_started_before_claim"] is not False:
        access_violations.append("PRECLAIM_ACCESS")
    independence = result["identity_and_independence"]
    overlap = (
        independence["independence_group"] in independence["non_overlap_groups"]
        or independence["future_confirmation_exclusion"]
        != "EVIMO2_V2_FLEA3_SANITY_LL_CAPTURE_FAMILY"
    )
    errors = []
    if file_hash(archive_path) != ARCHIVE_SHA256:
        errors.append("ARCHIVE_SHA256")
    if receipt["result_sha256"] != file_hash(result_path):
        errors.append("RECEIPT_RESULT_SHA256")
    if receipt["pair_ledger_sha256"] != file_hash(ledger_path):
        errors.append("RECEIPT_PAIR_LEDGER_SHA256")
    if pair_mismatch:
        errors.append("PAIR_REPLAY_MISMATCH")
    if window_mismatch:
        errors.append("WINDOW_REPLAY_MISMATCH")
    if result["terminal"] != expected_terminal or receipt["terminal"] != expected_terminal:
        errors.append("TERMINAL_MISMATCH")
    if binding_mismatches:
        errors.append("BINDING_MISMATCH")
    if access_violations:
        errors.append("ACCESS_VIOLATION")
    if overlap:
        errors.append("ANCESTRY_OR_CONFIRMATION_OVERLAP")
    validation = {
        "schema_version": "rcle.real_positive_approach_role.validation.v1",
        "protocol_id": PROTOCOL_ID,
        "validation_terminal": "VALID" if not errors else "INVALID",
        "scientific_terminal": expected_terminal,
        "producer_pair_record_count": len(produced_pairs),
        "validator_pair_record_count": len(replay_pairs),
        "pair_replay_mismatch_count": pair_mismatch,
        "window_replay_mismatch_count": window_mismatch,
        "binding_mismatches": binding_mismatches,
        "forbidden_extracted_source_files": forbidden_source_files,
        "access_violations": access_violations,
        "ancestry_overlap_violation": overlap,
        "algorithm_outcome_read": False,
        "replacement_source_count": 0,
        "admitted_window_count": len(admitted),
        "errors": errors,
        "independent_geometry_implementation": (
            "real_data_geometry_canary_r0.validator._independent_geometry"
        ),
        "producer_imported": False,
    }
    validation_path = formal_dir / "validation.json"
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--formal-dir", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--claim", required=True, type=Path)
    parser.add_argument("--source-resolution", required=True, type=Path)
    args = parser.parse_args()
    validation = validate(
        args.source_root,
        args.formal_dir,
        args.archive,
        args.contract,
        args.claim,
        args.source_resolution,
    )
    print(json.dumps(validation, ensure_ascii=False, sort_keys=True))
    return 0 if validation["validation_terminal"] == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(main())
