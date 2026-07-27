"""Attribute false triggers on the frozen below-reference development windows."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.research.egomotion_compensated_looming.rgb_algorithm_development_canary_cid_sims_r0 import (
    producer as rgb_core,
)
from scripts.research.egomotion_compensated_looming.real_data_geometry_canary_r0 import (
    producer as tum_pose_core,
)


THRESHOLD = 0.01
BELOW_WINDOWS = (
    ("TUM_RGBD_FR2_RPY@2", 2, 20260729),
    ("TUM_RGBD_FR2_RPY@7", 7, 20260730),
)


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL_OBJECT_REQUIRED:{path}")
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_exclusive(path: Path, value: Any, *, jsonl: bool = False) -> str:
    if jsonl:
        payload = b"".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
            for row in value
        )
    else:
        payload = (
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
            + b"\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def _verify_bound_input(repo: Path, entry: dict[str, Any]) -> Path:
    path = repo / entry["path"]
    if not path.is_file() or sha256(path) != entry["sha256"]:
        raise ValueError(f"BOUND_INPUT_IDENTITY:{entry['path']}")
    return path


def _evaluate_baseline_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    protocol = load_object(Path(task["protocol_path"]))
    records = sorted(task["members"], key=lambda row: Decimal(row["timestamp_s"]))
    root = Path(task["rgb_root"])
    timestamps = [Decimal(row["timestamp_s"]) for row in records]
    paths = [root / row["relative_path"] for row in records]
    poses = tum_pose_core._parse_poses(Path(task["groundtruth"]).read_bytes())
    intrinsic = np.asarray(
        ((525.0, 0.0, 319.5), (0.0, 525.0, 239.5), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    homographies: list[np.ndarray | None] = []
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
    if any(
        not (Decimal("0") < right - left <= Decimal("0.1"))
        for left, right in zip(timestamps, timestamps[1:])
    ):
        raise ValueError(f"RGB_PAIR_DT:{task['window_id']}")
    frames = [rgb_core._decode_gray(path) for path in paths]
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError(f"RGB_SHAPE_DRIFT:{task['window_id']}")
    cv2.setRNGSeed(int(task["rng_seed"]))
    cv2.setNumThreads(1)
    rows: list[dict[str, Any]] = []
    for pair_index, (left, right, homography) in enumerate(
        zip(timestamps, timestamps[1:], homographies)
    ):
        if homography is None:
            row = {
                "pair_index": pair_index,
                "previous_timestamp_s": float(left),
                "current_timestamp_s": float(right),
                "dt_s": float(right - left),
                "evaluable": False,
                "reason": "POSE_NOT_BRACKETED_OR_TOO_WIDE",
                "trigger": False,
            }
        else:
            row = rgb_core._evaluate_pair(
                0,
                frames[pair_index],
                frames[pair_index + 1],
                left,
                right,
                homography,
                protocol,
                rgb_core.PairState(),
            )
            row["pair_index"] = pair_index
        row.update(
            window_id=task["window_id"],
            role="BELOW_TRIGGER_REFERENCE_WINDOW",
            counterfactual="BASELINE_ONLY_PAIR_STATE_RESET",
        )
        rows.append(row)
        if (pair_index + 1) % 50 == 0 or pair_index + 1 == len(frames) - 1:
            print(
                f"baseline window={task['window_id']} completed={pair_index + 1}/{len(frames) - 1}",
                flush=True,
            )
    return rows


def classify_trigger(
    old: dict[str, Any], baseline: dict[str, Any], geometry: dict[str, Any]
) -> str:
    if old.get("trigger") is not True:
        return "OLD_NOT_TRIGGERED"
    if float(geometry["median_signed_radial_expansion_per_s"]) >= THRESHOLD:
        return "GEOMETRY_AT_OR_ABOVE_THRESHOLD"
    if baseline.get("evaluable") is not True:
        return "SUPPORT_MANAGER_ENABLED_EVALUABILITY"
    if baseline.get("trigger") is not True:
        return "SUPPORT_MANAGER_INDUCED_TRIGGER"
    if float(baseline["raw_expansion_median_per_s"]) <= THRESHOLD:
        return "ROTATION_COMPENSATION_THRESHOLD_CROSSING"
    return "BASELINE_LOCAL_FLOW_THRESHOLD_CROSSING"


def _longest_trigger(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    best_count = count = 0
    best_duration = 0.0
    start: float | None = None
    for row in rows:
        active = bool(row[field])
        if active:
            if count == 0:
                start = float(row["previous_timestamp_s"])
            count += 1
            duration = float(row["current_timestamp_s"]) - float(start)
            if count > best_count or (count == best_count and duration > best_duration):
                best_count, best_duration = count, duration
        else:
            count = 0
            start = None
    return {"pair_count": best_count, "duration_s": best_duration}


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    category_counts = Counter(
        row["attribution"] for row in rows if row["old_trigger"] is True
    )
    false_category_counts = Counter(
        row["attribution"]
        for row in rows
        if row["old_trigger"] is True and row["geometry_below_threshold"] is True
    )
    transitions = Counter(
        (bool(row["old_trigger"]), bool(row["baseline_trigger"])) for row in rows
    )
    return {
        "candidate_pair_count": len(rows),
        "old_trigger_count": sum(row["old_trigger"] is True for row in rows),
        "baseline_trigger_count": sum(row["baseline_trigger"] is True for row in rows),
        "geometry_below_old_trigger_count": sum(
            row["old_trigger"] is True and row["geometry_below_threshold"] is True
            for row in rows
        ),
        "old_trigger_attribution_counts": dict(sorted(category_counts.items())),
        "geometry_below_attribution_counts": dict(sorted(false_category_counts.items())),
        "managed_baseline_trigger_transitions": {
            f"old_{str(old).lower()}_baseline_{str(base).lower()}": count
            for (old, base), count in sorted(transitions.items())
        },
        "manager_active_old_trigger_count": sum(
            row["old_trigger"] is True and row["support_manager_active"] is True
            for row in rows
        ),
        "old_longest_trigger_run": _longest_trigger(rows, "old_trigger"),
        "baseline_longest_trigger_run": _longest_trigger(rows, "baseline_trigger"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--tum-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers != 2:
        raise ValueError("WORKERS_MUST_EQUAL_2")
    repo = Path(__file__).resolve().parents[4]
    contract_path = args.contract.resolve()
    contract = load_object(contract_path)
    if contract["protocol_id"] != "RCLE_LOW_REFERENCE_FALSE_TRIGGER_ATTRIBUTION_R0":
        raise ValueError("PROTOCOL_ID")
    bound = {
        name: _verify_bound_input(repo, entry)
        for name, entry in contract["inputs"].items()
    }
    manifest = load_object(bound["rgb_manifest"])
    old_result = load_object(bound["old_result"])
    old_rows = read_jsonl(bound["old_pair_ledger"])
    geometry_audit = load_object(bound["tum_geometry_audit"])
    if len(old_rows) != 967 or old_result["rgb_pair_ledger_sha256"] != sha256(
        bound["old_pair_ledger"]
    ):
        raise ValueError("OLD_PAIR_DENOMINATOR_OR_RESULT_IDENTITY")
    windows = {row["window_id"]: row for row in manifest["windows"]}
    tum_root = args.tum_root.resolve()
    if not (tum_root / "groundtruth.txt").is_file():
        raise ValueError("TUM_GROUNDTRUTH_MISSING")
    tasks = []
    for window_id, _, rng_seed in BELOW_WINDOWS:
        window = windows[window_id]
        for record in window["members"]:
            path = Path(window["rgb_root"]) / record["relative_path"]
            if (
                not path.is_file()
                or path.stat().st_size != int(record["bytes"])
                or sha256(path) != record["sha256"]
            ):
                raise ValueError(f"RGB_MEMBER_IDENTITY:{window_id}:{record['relative_path']}")
        tasks.append(
            {
                **window,
                "groundtruth": str(tum_root / "groundtruth.txt"),
                "protocol_path": str(bound["algorithm_protocol"]),
                "rng_seed": rng_seed,
            }
        )
    with ProcessPoolExecutor(max_workers=2) as executor:
        baseline_by_window = list(executor.map(_evaluate_baseline_task, tasks))
    baseline_rows = [row for group in baseline_by_window for row in group]
    if len(baseline_rows) != 598:
        raise ValueError("BASELINE_PAIR_DENOMINATOR")
    old_by_window = {
        window_id: [row for row in old_rows if row["window_id"] == window_id]
        for window_id, _, _ in BELOW_WINDOWS
    }
    attribution_rows: list[dict[str, Any]] = []
    for (window_id, window_index, _), baseline_group in zip(
        BELOW_WINDOWS, baseline_by_window
    ):
        old_group = old_by_window[window_id]
        geometry_group = geometry_audit["windows"][window_index]["pairs"]
        if not (len(old_group) == len(baseline_group) == len(geometry_group) == 299):
            raise ValueError(f"WINDOW_PAIR_DENOMINATOR:{window_id}")
        for old, baseline, geometry in zip(old_group, baseline_group, geometry_group):
            if (
                old["pair_index"] != baseline["pair_index"]
                or not math.isclose(
                    float(old["current_timestamp_s"]),
                    float(baseline["current_timestamp_s"]),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or not math.isclose(
                    float(old["current_timestamp_s"]),
                    float(geometry["current_rgb_timestamp"]),
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            ):
                raise ValueError(f"PAIR_IDENTITY:{window_id}:{old['pair_index']}")
            manager = old["support_manager"]
            attribution_rows.append(
                {
                    "window_id": window_id,
                    "pair_index": old["pair_index"],
                    "previous_timestamp_s": old["previous_timestamp_s"],
                    "current_timestamp_s": old["current_timestamp_s"],
                    "dt_s": old["dt_s"],
                    "old_evaluable": old["evaluable"],
                    "baseline_evaluable": baseline["evaluable"],
                    "old_trigger": old["trigger"],
                    "baseline_trigger": baseline["trigger"],
                    "old_raw_expansion_per_s": old.get("raw_expansion_median_per_s"),
                    "old_compensated_expansion_per_s": old.get(
                        "compensated_expansion_median_per_s"
                    ),
                    "baseline_raw_expansion_per_s": baseline.get(
                        "raw_expansion_median_per_s"
                    ),
                    "baseline_compensated_expansion_per_s": baseline.get(
                        "compensated_expansion_median_per_s"
                    ),
                    "geometry_signed_radial_expansion_per_s": geometry[
                        "median_signed_radial_expansion_per_s"
                    ],
                    "geometry_angular_rate_rad_s": geometry["angular_rate_rad_s"],
                    "geometry_parallax_q90_rad_per_s": geometry[
                        "q90_time_normalized_parallax_rad_per_s"
                    ],
                    "geometry_below_threshold": float(
                        geometry["median_signed_radial_expansion_per_s"]
                    )
                    < THRESHOLD,
                    "support_manager_active": bool(
                        manager["activated_cell_indices"]
                    ),
                    "support_manager_supplement_count": manager[
                        "spatial_supplement_count"
                    ],
                    "attribution": classify_trigger(old, baseline, geometry),
                }
            )
    per_window = {
        window_id: _summary(
            [row for row in attribution_rows if row["window_id"] == window_id]
        )
        for window_id, _, _ in BELOW_WINDOWS
    }
    aggregate = _summary(attribution_rows)
    false_counts = aggregate["geometry_below_attribution_counts"]
    named_total = sum(
        false_counts.get(name, 0)
        for name in (
            "SUPPORT_MANAGER_ENABLED_EVALUABILITY",
            "SUPPORT_MANAGER_INDUCED_TRIGGER",
            "ROTATION_COMPENSATION_THRESHOLD_CROSSING",
            "BASELINE_LOCAL_FLOW_THRESHOLD_CROSSING",
        )
    )
    dominant = (
        max(false_counts, key=false_counts.get) if false_counts else "NOT_EVALUABLE"
    )
    result = {
        "schema": "rcle.low_reference_false_trigger.attribution_result.v1",
        "protocol_id": contract["protocol_id"],
        "execution_validity": "VALID",
        "scientific_outcome": "IMPLEMENTATION_NOT_READY",
        "old_result_sha256": sha256(bound["old_result"]),
        "old_pair_ledger_sha256": sha256(bound["old_pair_ledger"]),
        "baseline_pair_ledger_sha256": None,
        "attribution_ledger_sha256": None,
        "threshold_per_s": THRESHOLD,
        "per_window": per_window,
        "aggregate": aggregate,
        "dominant_geometry_below_attribution": dominant,
        "geometry_below_named_attribution_denominator": named_total,
        "authority": {
            "development": True,
            "confirmation": False,
            "android": False,
            "product": False,
            "safety": False,
        },
    }
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("OUTPUT_DIRECTORY_EXISTS")
    baseline_sha = write_exclusive(
        output / "baseline_pair_ledger.jsonl", baseline_rows, jsonl=True
    )
    attribution_sha = write_exclusive(
        output / "attribution_ledger.jsonl", attribution_rows, jsonl=True
    )
    result["baseline_pair_ledger_sha256"] = baseline_sha
    result["attribution_ledger_sha256"] = attribution_sha
    write_exclusive(output / "result.json", result)
    print(
        json.dumps(
            {
                "dominant": dominant,
                "false_counts": false_counts,
                "old_triggers": aggregate["old_trigger_count"],
                "baseline_triggers": aggregate["baseline_trigger_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
