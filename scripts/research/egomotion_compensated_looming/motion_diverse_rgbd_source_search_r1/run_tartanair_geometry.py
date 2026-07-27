"""Run the unchanged strict geometry gate on one frozen TartanAir window."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.egomotion_compensated_looming.motion_diverse_rgbd_geometry_admission_r0.run_eth3d_geometry import (
    longest,
)
from scripts.research.egomotion_compensated_looming.motion_diverse_rgbd_geometry_admission_r0.template import (
    decimal_median,
)
from scripts.research.egomotion_compensated_looming.real_positive_approach_role_admission_r2_cid_sims import (
    producer as geometry,
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_exclusive(path: Path, value: Any, *, jsonl: bool = False) -> str:
    if jsonl:
        payload = b"".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            for row in value
        )
    else:
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def read_camera(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as camera:
        pose = np.asarray(camera["camera_pose"], dtype=np.float64)
        intrinsic = np.asarray(camera["camera_intrinsics"], dtype=np.float64)
    if pose.shape != (4, 4) or intrinsic.shape != (3, 3):
        raise ValueError("TARTANAIR_CAMERA_FORMAT")
    if not np.all(np.isfinite(pose)) or not np.all(np.isfinite(intrinsic)):
        raise ValueError("TARTANAIR_CAMERA_NONFINITE")
    return pose, intrinsic


def pair_worker(task: tuple[Any, ...]) -> dict[str, Any]:
    base, depth_path, intrinsic, previous_pose, current_pose, dt = task
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        limits = None
    else:
        limits = threadpool_limits(limits=1)
        limits.__enter__()
    try:
        try:
            depth = np.load(depth_path, allow_pickle=False)
            if depth.ndim != 2 or depth.dtype not in (np.float32, np.float64):
                raise ValueError("TARTANAIR_DEPTH_FORMAT")
            height, width = depth.shape
            yy, xx = np.mgrid[0:height:8, 0:width:8]
            sampled = np.asarray(depth[yy, xx], dtype=np.float64).reshape(-1)
            valid = np.isfinite(sampled) & (sampled > 0.0)
            pixels = np.column_stack((xx.reshape(-1)[valid], yy.reshape(-1)[valid])).astype(np.float64)
            depth_m = sampled[valid]
            previous_rotation = previous_pose[:3, :3]
            current_rotation = current_pose[:3, :3]
            rotation = current_rotation.T @ previous_rotation
            translation = current_rotation.T @ (previous_pose[:3, 3] - current_pose[:3, 3])
            summary = geometry.summarize_translation_induced_geometry(
                geometry.translation_induced_geometry(
                    pixels,
                    depth_m,
                    intrinsic,
                    rotation,
                    translation,
                    dt,
                    image_size_wh=(width, height),
                    minimum_radius_px=8.0,
                    zbuffer=True,
                )
            )
        except (OSError, ValueError, KeyError) as error:
            return {
                **base,
                "geometry_evaluable": False,
                "geometry_abstention_reason": str(error),
                "geometry_band": None,
            }
        if summary.get("evaluable") is not True:
            return {
                **base,
                "geometry_evaluable": False,
                "geometry_abstention_reason": "NO_VALID_GEOMETRY_SAMPLES",
                "geometry_band": None,
            }
        signed = float(summary["median_signed_radial_expansion_per_s"])
        band = (
            "BELOW_TRIGGER_REFERENCE"
            if signed < 0.01
            else "WEAK_POSITIVE_RADIAL"
            if signed < 0.05
            else "POSITIVE_APPROACH_GEOMETRY"
        )
        return {
            **base,
            "geometry_evaluable": True,
            "geometry_abstention_reason": None,
            "geometry_band": band,
            "geometry_signed_radial_expansion_per_s": signed,
            "geometry_radial_expansion_positive_fraction": float(summary["radial_expansion_positive_fraction"]),
            "geometry_q90_time_normalized_parallax_rad_per_s": float(summary["q90_time_normalized_parallax_rad_per_s"]),
        }
    finally:
        if limits is not None:
            limits.__exit__(None, None, None)


def summarize(window: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = int(window["pair_count"])
    evaluable = [row for row in rows if row["geometry_evaluable"] is True]
    counts = Counter(row["geometry_band"] for row in evaluable)
    abstentions = Counter(
        row["geometry_abstention_reason"] for row in rows if row["geometry_evaluable"] is not True
    )
    positive_count, positive_duration = longest(rows, "POSITIVE_APPROACH_GEOMETRY")
    below_count, below_duration = longest(rows, "BELOW_TRIGGER_REFERENCE")
    coverage = Decimal(len(evaluable)) / Decimal(denominator)
    positive_fraction = Decimal(counts["POSITIVE_APPROACH_GEOMETRY"]) / Decimal(denominator)
    below_fraction = Decimal(counts["BELOW_TRIGGER_REFERENCE"]) / Decimal(denominator)
    positive_ok = coverage >= Decimal("0.8") and positive_fraction >= Decimal("0.8") and positive_duration >= Decimal("5")
    below_ok = coverage >= Decimal("0.8") and below_fraction >= Decimal("0.8") and below_duration >= Decimal("5")
    if positive_ok and below_ok:
        raise ValueError("TARTANAIR_ROLE_OVERLAP")
    values = [Decimal(str(row["geometry_signed_radial_expansion_per_s"])) for row in evaluable]
    return {
        "window_id": window["window_id"],
        "source_id": "TARTANAIR_JAPANESEALLEY_HARD",
        "source_kind": "SYNTHETIC_DEVELOPMENT_ANCHOR",
        "candidate_pair_count": denominator,
        "geometry_evaluable_pair_count": len(evaluable),
        "geometry_pair_coverage_fixed_denominator": float(coverage),
        "geometry_abstention_reasons": dict(sorted(abstentions.items())),
        "geometry_band_counts": dict(sorted(counts.items())),
        "positive_fraction_fixed_denominator": float(positive_fraction),
        "below_fraction_fixed_denominator": float(below_fraction),
        "longest_positive_run_pair_count": positive_count,
        "longest_positive_run_duration_s": float(positive_duration),
        "longest_below_run_pair_count": below_count,
        "longest_below_run_duration_s": float(below_duration),
        "median_signed_radial_expansion_per_s": float(decimal_median(values)) if values else None,
        "role": (
            "POSITIVE_APPROACH_WINDOW"
            if positive_ok
            else "BELOW_TRIGGER_REFERENCE_WINDOW"
            if below_ok
            else "AMBIGUOUS_OR_INELIGIBLE"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--implementation-lock", type=Path, required=True)
    parser.add_argument("--extract-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers != 8:
        raise ValueError("WORKERS_MUST_EQUAL_8")
    repo = Path(__file__).resolve().parents[4]
    amendment_path = args.amendment.resolve()
    manifest_path = args.extract_manifest.resolve()
    amendment = load(amendment_path)
    manifest = load(manifest_path)
    lock = load(args.implementation_lock.resolve())
    if lock.get("status") != "FROZEN_BEFORE_TARTANAIR_GEOMETRY":
        raise ValueError("IMPLEMENTATION_LOCK_STATUS")
    for entry in lock["files"]:
        if sha(repo / entry["path"]) != entry["sha256"]:
            raise ValueError(f"IMPLEMENTATION_LOCK_DRIFT:{entry['path']}")
    if lock["local_inputs"] != {
        "amendment_sha256": sha(amendment_path),
        "extract_manifest_sha256": sha(manifest_path),
    }:
        raise ValueError("IMPLEMENTATION_LOCK_LOCAL_INPUT")
    if manifest["amendment_sha256"] != sha(amendment_path) or manifest["rgb_bytes_accessed"] != 0:
        raise ValueError("TARTANAIR_MANIFEST_BOUNDARY")
    root = Path(manifest["output_root"]).resolve()
    for record in manifest["members"]:
        path = root / record["relative_path"]
        if len(path.read_bytes()) != int(record["bytes"]) or sha(path) != record["sha256"]:
            raise ValueError(f"TARTANAIR_MEMBER_IDENTITY:{record['relative_path']}")
    window = manifest["window"]
    poses = {}
    intrinsics = {}
    for frame_id in window["frame_ids"]:
        poses[frame_id], intrinsics[frame_id] = read_camera(root / "pose" / f"{frame_id}.npz")
    all_rows = []
    tasks = []
    for pair_index, (previous_id, current_id) in enumerate(zip(window["frame_ids"], window["frame_ids"][1:])):
        if not np.allclose(intrinsics[previous_id], intrinsics[current_id], rtol=1e-12, atol=1e-15):
            raise ValueError("TARTANAIR_INTRINSIC_DRIFT")
        base = {
            "window_id": window["window_id"],
            "pair_index": pair_index,
            "previous_frame_id": previous_id,
            "current_frame_id": current_id,
            "previous_timestamp_s": pair_index / 10.0,
            "current_timestamp_s": (pair_index + 1) / 10.0,
            "dt_s": 0.1,
        }
        tasks.append(
            (
                base,
                str(root / "depth" / f"{previous_id}.npy"),
                intrinsics[previous_id],
                poses[previous_id],
                poses[current_id],
                0.1,
            )
        )
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        all_rows = list(executor.map(pair_worker, tasks))
    all_rows.sort(key=lambda row: row["pair_index"])
    if len(all_rows) != int(window["pair_count"]):
        raise ValueError("TARTANAIR_PAIR_COUNT")
    summary = summarize(window, all_rows)
    output = args.output_dir.resolve()
    ledger_sha = write_exclusive(output / "geometry_pair_ledger.jsonl", all_rows, jsonl=True)
    result = {
        "protocol_id": amendment["protocol_id"],
        "amendment_id": amendment["amendment_id"],
        "window_summary": summary,
        "terminal": (
            "TARTANAIR_SYNTHETIC_POSITIVE_ANCHOR_ADMITTED"
            if summary["role"] == "POSITIVE_APPROACH_WINDOW"
            else "ADVANCE_FROZEN_TARTANAIR_POSE_QUEUE"
        ),
        "workers": args.workers,
        "geometry_pair_ledger_sha256": ledger_sha,
        "rgb_bytes_accessed": 0,
        "algorithm_changed": False,
        "new_real_holdout_authority": False,
    }
    write_exclusive(output / "result.json", result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
