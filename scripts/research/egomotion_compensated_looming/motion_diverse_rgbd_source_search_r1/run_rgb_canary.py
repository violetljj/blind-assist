"""Run the unchanged RGB pair algorithm on the frozen four-window cohort."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from scripts.research.egomotion_compensated_looming.rgb_algorithm_development_canary_cid_sims_r0 import (
    producer as rgb_core,
)
from scripts.research.egomotion_compensated_looming.real_positive_approach_role_admission_r2_cid_sims import (
    producer as pose_core,
)
from scripts.research.egomotion_compensated_looming.real_data_geometry_canary_r0 import (
    producer as tum_pose_core,
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


def matrix_homography(previous: np.ndarray, current: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    current_from_previous = current[:3, :3].T @ previous[:3, :3]
    return intrinsic @ current_from_previous @ np.linalg.inv(intrinsic)


def longest_trigger(rows: list[dict[str, Any]]) -> tuple[int, float]:
    best_count = count = 0
    best_duration = 0.0
    start = None
    for row in rows:
        if row["evaluable"] is True and row["trigger"] is True:
            if count == 0:
                start = float(row["previous_timestamp_s"])
            count += 1
            duration = float(row["current_timestamp_s"]) - float(start)
            if count > best_count or (count == best_count and duration > best_duration):
                best_count, best_duration = count, duration
        else:
            count = 0
            start = None
    return best_count, best_duration


def evaluate_task(task: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = load(Path(task["protocol_path"]))
    kind = task["kind"]
    records = task["members"]
    root = Path(task["rgb_root"])
    if kind == "ETH3D":
        records = sorted(records, key=lambda row: Decimal(Path(row["path"]).stem))
        timestamps = [Decimal(Path(row["path"]).stem) for row in records]
        paths = [root.joinpath(*Path(row["path"]).parts) for row in records]
        poses = pose_core._parse_poses(Path(task["groundtruth"]).read_bytes())
        intrinsic = pose_core._parse_intrinsic(Path(task["calibration"]).read_bytes())
        homographies = []
        for left, right in zip(timestamps, timestamps[1:]):
            try:
                homographies.append(rgb_core._homography(poses, left, right, intrinsic))
            except ValueError as error:
                if str(error) not in {"R2_POSE_NOT_BRACKETED", "R2_POSE_BRACKET_TOO_WIDE"}:
                    raise
                homographies.append(None)
    elif kind == "TARTANAIR":
        records = sorted(records, key=lambda row: int(row["frame_id"]))
        timestamps = [Decimal(index) / Decimal("10") for index in range(len(records))]
        paths = [root / row["relative_path"] for row in records]
        pose_root = Path(task["pose_root"])
        matrices = []
        intrinsics = []
        for row in records:
            with np.load(pose_root / "pose" / f"{row['frame_id']}.npz", allow_pickle=False) as camera:
                matrices.append(np.asarray(camera["camera_pose"], dtype=np.float64))
                intrinsics.append(np.asarray(camera["camera_intrinsics"], dtype=np.float64))
        if any(not np.allclose(intrinsics[0], value, rtol=1e-12, atol=1e-15) for value in intrinsics[1:]):
            raise ValueError("TARTANAIR_RGB_INTRINSIC_DRIFT")
        homographies = [
            matrix_homography(left, right, intrinsics[0])
            for left, right in zip(matrices, matrices[1:])
        ]
    elif kind == "TUM":
        records = sorted(records, key=lambda row: Decimal(row["timestamp_s"]))
        timestamps = [Decimal(row["timestamp_s"]) for row in records]
        paths = [root / row["relative_path"] for row in records]
        poses = tum_pose_core._parse_poses(Path(task["groundtruth"]).read_bytes())
        intrinsic = np.asarray(
            ((525.0, 0.0, 319.5), (0.0, 525.0, 239.5), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        homographies = []
        for left, right in zip(timestamps, timestamps[1:]):
            try:
                previous_pose = tum_pose_core._interpolate(poses, left, Decimal("0.050"))
                current_pose = tum_pose_core._interpolate(poses, right, Decimal("0.050"))
                rotation, _, _ = tum_pose_core._relative(
                    previous_pose, current_pose, float(right - left)
                )
                homographies.append(intrinsic @ rotation @ np.linalg.inv(intrinsic))
            except ValueError as error:
                if str(error) not in {"POSE_NOT_BRACKETED", "POSE_BRACKET_GT_0P050_S"}:
                    raise
                homographies.append(None)
    else:
        raise ValueError(f"UNKNOWN_SOURCE_ADAPTER:{kind}")
    if any(not (Decimal("0") < right - left <= Decimal("0.1")) for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"RGB_PAIR_DT:{task['window_id']}")
    frames = [rgb_core._decode_gray(path) for path in paths]
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError(f"RGB_SHAPE_DRIFT:{task['window_id']}")
    cv2.setRNGSeed(20260727 + int(task["seed_offset"]))
    cv2.setNumThreads(1)
    state = rgb_core.PairState()
    rows = []
    started = time.perf_counter()
    for pair_index, (left, right, homography) in enumerate(
        zip(timestamps, timestamps[1:], homographies)
    ):
        if homography is None:
            state = rgb_core.PairState()
            rows.append(
                {
                    "pair_index": pair_index,
                    "previous_timestamp_s": float(left),
                    "current_timestamp_s": float(right),
                    "dt_s": float(right - left),
                    "evaluable": False,
                    "reason": "POSE_NOT_BRACKETED_OR_TOO_WIDE",
                    "trigger": False,
                    "window_id": task["window_id"],
                    "role": task["role"],
                }
            )
            continue
        row = rgb_core._evaluate_pair(
            pair_index,
            frames[pair_index],
            frames[pair_index + 1],
            left,
            right,
            homography,
            protocol,
            state,
        )
        row.update(window_id=task["window_id"], role=task["role"])
        rows.append(row)
        if (pair_index + 1) % 50 == 0 or pair_index + 1 == len(frames) - 1:
            print(
                f"window={task['window_id']} completed={pair_index + 1}/{len(frames) - 1}",
                flush=True,
            )
    evaluable = [row for row in rows if row["evaluable"] is True]
    triggered = [row for row in evaluable if row["trigger"] is True]
    abstentions = Counter(str(row["reason"]) for row in rows if row["evaluable"] is not True)
    longest_count, longest_duration = longest_trigger(rows)
    summary = {
        "window_id": task["window_id"],
        "role": task["role"],
        "source_kind": task["source_kind"],
        "candidate_pair_count": len(rows),
        "evaluable_pair_count": len(evaluable),
        "pair_coverage": len(evaluable) / len(rows),
        "abstention_count": len(rows) - len(evaluable),
        "abstention_reasons": dict(sorted(abstentions.items())),
        "median_compensated_expansion_per_s": (
            float(np.median([row["compensated_expansion_median_per_s"] for row in evaluable]))
            if evaluable
            else None
        ),
        "trigger_threshold_per_s": 0.01,
        "trigger_count": len(triggered),
        "trigger_coverage_fixed_denominator": len(triggered) / len(rows),
        "longest_consecutive_trigger_pair_count": longest_count,
        "longest_consecutive_trigger_duration_s": longest_duration,
        "runtime_s": time.perf_counter() - started,
    }
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary", type=Path, required=True)
    parser.add_argument("--adapter-amendment", type=Path, required=True)
    parser.add_argument("--implementation-lock", type=Path, required=True)
    parser.add_argument("--rgb-manifest", type=Path, required=True)
    parser.add_argument("--eth3d-metadata-root", type=Path, required=True)
    parser.add_argument("--tartanair-pose-root", type=Path, required=True)
    parser.add_argument("--tum-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers != 8:
        raise ValueError("WORKERS_MUST_EQUAL_8")
    repo = Path(__file__).resolve().parents[4]
    canary_path = args.canary.resolve()
    amendment_path = args.adapter_amendment.resolve()
    manifest_path = args.rgb_manifest.resolve()
    canary = load(canary_path)
    amendment = load(amendment_path)
    manifest = load(manifest_path)
    lock = load(args.implementation_lock.resolve())
    if canary["status"] != "FROZEN_BEFORE_RGB_ALGORITHM_OUTCOME":
        raise ValueError("RGB_CANARY_NOT_FROZEN")
    if amendment["status"] != "FROZEN_BEFORE_REEXECUTED_RGB_ALGORITHM_OUTCOME":
        raise ValueError("RGB_ADAPTER_AMENDMENT_NOT_FROZEN")
    if amendment["original_canary"]["sha256"] != sha(canary_path):
        raise ValueError("RGB_ADAPTER_AMENDMENT_CANARY_IDENTITY")
    failure_path = repo / amendment["failure_receipt"]["path"]
    if amendment["failure_receipt"]["sha256"] != sha(failure_path):
        raise ValueError("RGB_ADAPTER_FAILURE_RECEIPT_IDENTITY")
    if canary["inputs"]["rgb_manifest"]["sha256"] != sha(manifest_path):
        raise ValueError("RGB_MANIFEST_CANARY_IDENTITY")
    if lock.get("status") != "FROZEN_BEFORE_RGB_ALGORITHM_OUTCOME":
        raise ValueError("RGB_IMPLEMENTATION_LOCK_STATUS")
    if lock["local_inputs"] != {
        "adapter_amendment_sha256": sha(amendment_path),
        "canary_sha256": sha(canary_path),
        "rgb_manifest_sha256": sha(manifest_path),
    }:
        raise ValueError("RGB_IMPLEMENTATION_LOCK_INPUT")
    for entry in lock["files"]:
        if sha(repo / entry["path"]) != entry["sha256"]:
            raise ValueError(f"RGB_IMPLEMENTATION_LOCK_DRIFT:{entry['path']}")
    for window in manifest["windows"]:
        root = Path(window["rgb_root"])
        for record in window["members"]:
            relative = record.get("path", record.get("relative_path"))
            path = root.joinpath(*Path(relative).parts) if "path" in record else root / relative
            if path.stat().st_size != int(record["bytes"]) or sha(path) != record["sha256"]:
                raise ValueError(f"RGB_MEMBER_IDENTITY:{window['window_id']}:{relative}")
    protocol_path = repo / canary["unchanged_algorithm"]["protocol_path"]
    windows = {row["window_id"]: row for row in manifest["windows"]}
    eth_root = args.eth3d_metadata_root.resolve()
    tum_root = args.tum_root.resolve()
    tasks = [
        {
            **windows["desk_changing_1@4065.364250422"],
            "kind": "ETH3D",
            "groundtruth": str(eth_root / "groundtruth.txt"),
            "calibration": str(eth_root / "calibration.txt"),
            "protocol_path": str(protocol_path),
            "seed_offset": 0,
        },
        {
            **windows["japanesealley/Hard/P002@000260"],
            "kind": "TARTANAIR",
            "pose_root": str(args.tartanair_pose_root.resolve()),
            "protocol_path": str(protocol_path),
            "seed_offset": 1,
        },
        {
            **windows["TUM_RGBD_FR2_RPY@2"],
            "kind": "TUM",
            "groundtruth": str(tum_root / "groundtruth.txt"),
            "protocol_path": str(protocol_path),
            "seed_offset": 2,
        },
        {
            **windows["TUM_RGBD_FR2_RPY@7"],
            "kind": "TUM",
            "groundtruth": str(tum_root / "groundtruth.txt"),
            "protocol_path": str(protocol_path),
            "seed_offset": 3,
        },
    ]
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("RGB_OUTPUT_DIRECTORY_EXISTS")
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        evaluated = list(executor.map(evaluate_task, tasks))
    summaries = [item[0] for item in evaluated]
    rows = [row for item in evaluated for row in item[1]]
    coverage_ok = all(summary["pair_coverage"] >= 0.8 for summary in summaries)
    positive = [row for row in summaries if row["role"] == "POSITIVE_APPROACH_WINDOW"]
    below = [row for row in summaries if row["role"] == "BELOW_TRIGGER_REFERENCE_WINDOW"]
    if len(positive) != 2 or len(below) != 2:
        raise ValueError("RGB_ROLE_COUNT")
    role_aggregates = {}
    for name, selected in (("positive", positive), ("below_reference", below)):
        role_aggregates[name] = {
            "median_compensated_expansion_per_s": (
                float(np.median([row["median_compensated_expansion_per_s"] for row in selected]))
                if coverage_ok
                else None
            ),
            "median_trigger_coverage_fixed_denominator": (
                float(np.median([row["trigger_coverage_fixed_denominator"] for row in selected]))
                if coverage_ok
                else None
            ),
        }
    direction = bool(
        coverage_ok
        and role_aggregates["positive"]["median_compensated_expansion_per_s"]
        > role_aggregates["below_reference"]["median_compensated_expansion_per_s"]
        and role_aggregates["positive"]["median_trigger_coverage_fixed_denominator"]
        > role_aggregates["below_reference"]["median_trigger_coverage_fixed_denominator"]
    )
    terminal = (
        "DEVELOPMENT_SIGNAL_DIRECTION_SUPPORTED / VALID"
        if direction
        else "DEVELOPMENT_SIGNAL_DIRECTION_NOT_SUPPORTED / VALID"
        if coverage_ok
        else "NOT_EVALUABLE / VALID"
    )
    ledger_sha = write_exclusive(output / "rgb_pair_ledger.jsonl", rows, jsonl=True)
    result = {
        "schema": "rcle.motion_diverse_rgbd.source_search.rgb_development_canary_result.v1",
        "protocol_id": canary["protocol_id"],
        "canary_id": canary["canary_id"],
        "adapter_amendment_sha256": sha(amendment_path),
        "terminal": terminal,
        "workers": args.workers,
        "window_summaries": summaries,
        "role_aggregates": role_aggregates,
        "all_window_coverage_pass": coverage_ok,
        "direction_supported": direction,
        "rgb_pair_ledger_sha256": ledger_sha,
        "threshold_tuned": False,
        "algorithm_changed": False,
        "window_substitution": False,
        "authority": {
            "development_cohort": True,
            "all_real_cross_source_holdout": False,
            "performance": False,
            "android": False,
            "product": False,
        },
    }
    write_exclusive(output / "result.json", result)
    print(json.dumps({"terminal": terminal, "role_aggregates": role_aggregates}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
