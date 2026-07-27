"""Read-only host throughput pilot on an already-burned TUM-format RGB-D root.

This utility refuses sofa_3 and emits benchmark telemetry to stdout only.  It
does not evaluate admission gates and cannot write a scientific result.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from scripts.research.egomotion_compensated_looming.real_positive_approach_role_admission_r1 import (
    producer,
)
from scripts.research.egomotion_compensated_looming.tum_fr2_rpy_geometry_audit.audit import (
    associate_unique_nearest,
    parse_index,
    parse_poses_with_diagnostics,
)


PILOT_SOURCE_IDS = {
    "cables_1": "ETH3D_SLAM_CABLES_1_CAPTURE_FAMILY",
    "rgbd_dataset_freiburg2_rpy": "TUM_FR2_RPY_SINGLE_SEQUENCE_3a35b799",
}


def _rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None
    return int(psutil.Process().memory_info().rss)


def run_pilot(root: Path, *, pair_limit: int, workers: int = 1) -> dict[str, Any]:
    lexical = root.as_posix().lower()
    if "sofa_3" in lexical:
        raise ValueError("R1_PILOT_SOFA_3_FORBIDDEN")
    source_id = PILOT_SOURCE_IDS.get(root.name.lower())
    if source_id is None:
        raise ValueError("R1_PILOT_REQUIRES_FROZEN_BURNED_ROOT")
    if workers < 1 or pair_limit < 1:
        raise ValueError("R1_PILOT_POSITIVE_LIMITS")
    controls = {
        name: root / name
        for name in ("groundtruth.txt", "depth.txt")
    }
    calibration_path = root / "calibration.txt"
    associated_path = root / "associated.txt"
    rgb_path = root / "rgb.txt"
    if source_id == "TUM_FR2_RPY_SINGLE_SEQUENCE_3a35b799":
        controls["rgb.txt"] = rgb_path
    else:
        controls["calibration.txt"] = calibration_path
        controls["associated.txt"] = associated_path
    if not all(path.is_file() for path in controls.values()):
        raise ValueError("R1_PILOT_CONTROL_FILE_MISSING")
    intrinsic = (
        producer._parse_intrinsic(calibration_path.read_bytes())
        if calibration_path.is_file()
        else np.asarray(
            ((520.908620, 0.0, 325.141442), (0.0, 521.007327, 249.701764), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
    )
    if source_id == "TUM_FR2_RPY_SINGLE_SEQUENCE_3a35b799":
        tum_poses, _ = parse_poses_with_diagnostics(controls["groundtruth.txt"])
        poses = [
            producer.PoseRow(
                row.timestamp,
                row.center_world_m,
                row.quaternion_xyzw,
            )
            for row in tum_poses
        ]
    else:
        poses = producer._parse_poses(controls["groundtruth.txt"].read_bytes())
    depth_index = producer._parse_depth_index(controls["depth.txt"].read_bytes())
    if associated_path.is_file():
        associated = producer._parse_associated(associated_path.read_bytes())
    else:
        rgb_rows = parse_index(rgb_path)
        depth_rows = parse_index(controls["depth.txt"])
        matches = associate_unique_nearest(rgb_rows, depth_rows)
        associated = [
            producer.AssociatedRow(
                rgb_rows[rgb_index].timestamp,
                rgb_rows[rgb_index].relative_path,
                depth_rows[depth_index_value].timestamp,
                depth_rows[depth_index_value].relative_path,
            )
            for rgb_index, depth_index_value in sorted(matches.items())
        ]
    associated_depth = [
        (row.depth_timestamp, row.depth_path) for row in associated
    ]
    if source_id == "ETH3D_SLAM_CABLES_1_CAPTURE_FAMILY":
        if associated_depth != depth_index:
            raise ValueError("R1_PILOT_DEPTH_IDENTITY")
    elif not set(associated_depth).issubset(set(depth_index)):
        raise ValueError("R1_PILOT_DEPTH_IDENTITY")
    pairs = list(zip(associated, associated[1:]))[:pair_limit]
    tasks = []
    for source_order, (before, after) in enumerate(pairs):
        dt = float(after.depth_timestamp - before.depth_timestamp)
        record: dict[str, Any] = {
            "source_order": source_order,
            "previous_depth_timestamp_s": float(before.depth_timestamp),
            "current_depth_timestamp_s": float(after.depth_timestamp),
            "dt_s": dt,
        }
        if not 0.0 < dt <= producer.MAX_DT_SECONDS:
            record.update(evaluable=False, reason="PAIR_DT_OUT_OF_RANGE")
            tasks.append((source_order, record, None))
            continue
        try:
            before_pose = producer._interpolate_pose(poses, before.depth_timestamp)
            after_pose = producer._interpolate_pose(poses, after.depth_timestamp)
        except ValueError as error:
            record.update(evaluable=False, reason=str(error))
            tasks.append((source_order, record, None))
            continue
        depth_path = root / before.depth_path
        if not depth_path.is_file():
            raise ValueError("R1_PILOT_DEPTH_MISSING")
        worker_task = (
            record,
            depth_path.read_bytes(),
            intrinsic,
            before_pose,
            after_pose,
            dt,
        )
        tasks.append((source_order, record, worker_task))

    worker_tasks = [task for _, _, task in tasks if task is not None]
    started = time.perf_counter()
    rss_before = _rss_bytes()
    if workers == 1:
        stream = map(producer._pair_worker, worker_tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        stream = executor.map(producer._pair_worker, worker_tasks)
    evaluated = []
    progress = []
    try:
        for completed, (record, image_size) in enumerate(stream, start=1):
            evaluated.append((record, image_size))
            elapsed = max(time.perf_counter() - started, 1e-9)
            progress.append(
                {
                    "completed_units": completed,
                    "total_units": len(worker_tasks),
                    "throughput_pairs_s": completed / elapsed,
                    "eta_seconds": (len(worker_tasks) - completed)
                    / (completed / elapsed),
                }
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    evaluated_by_order = {
        int(record["source_order"]): record for record, _ in evaluated
    }
    summaries = []
    for source_order, record, worker_task in tasks:
        value = record if worker_task is None else evaluated_by_order[source_order]
        summaries.append(value)
    elapsed = max(time.perf_counter() - started, 1e-9)
    rss_after = _rss_bytes()
    deterministic_payload = {
        "pilot_source_id": source_id,
        "pair_limit": pair_limit,
        "summaries": summaries,
    }
    summary_hash = hashlib.sha256(
        json.dumps(
            deterministic_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "rcle.host_pair_pilot.v1",
        "protocol_id": producer.PROTOCOL_ID,
        "pilot_only": True,
        "scientific_result_written": False,
        "admission_gate_evaluated": False,
        "source_id": source_id,
        "pair_limit": pair_limit,
        "workers": workers,
        "worker_opencv_threads": 1,
        "worker_blas_threads": 1,
        "source_order_preserved": True,
        "wall_seconds": elapsed,
        "throughput_pairs_s": len(pairs) / elapsed,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_delta_bytes": (
            None
            if rss_before is None or rss_after is None
            else rss_after - rss_before
        ),
        "progress": progress,
        "deterministic_summary_sha256": summary_hash,
        "processed_pair_count": len(pairs),
        "evaluable_pair_count": sum(bool(row.get("evaluable")) for row in summaries),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    report = run_pilot(args.root, pair_limit=args.pairs, workers=args.workers)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
