#!/usr/bin/env python3
"""Route stale MIL alerts through a causal static-semantic event-exit guard.

The guard never uses the held-out label. It closes a static surface-material
event only when the immediately preceding episode from the same source contains
a frozen prompt-free semantic detection, the current episode contains none,
the time/index gap is bounded, and the independent source-isolated COCO
trajectory probe also reports no current dynamic hazard. Otherwise the original
MIL decision is preserved.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import run_public_silver_frozen_feature_probe as common
import run_public_silver_prompt_free_semantic_probe as semantic
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_silver_semantic_exit_router_v1"


def verify_report(path: Path) -> dict[str, Any]:
    path = path.resolve()
    mil.reject_independent_direction(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"report sidecar is missing: {path}")
    expected = sidecar.read_text(encoding="ascii").strip().lower()
    actual = common.sha256_file(path)
    if expected != actual:
        raise ValueError(f"report sidecar mismatch: {path}")
    return common.load_json(path)


def prediction_map_from_folds(report: dict[str, Any], key: str) -> dict[str, int]:
    evaluation = report.get(key)
    if not isinstance(evaluation, dict) or not isinstance(evaluation.get("folds"), list):
        raise ValueError(f"report has no fold evaluation: {key}")
    predictions: dict[str, int] = {}
    for fold in evaluation["folds"]:
        episode_ids = fold.get("held_out_episode_ids")
        values = fold.get("predicted")
        if not isinstance(episode_ids, list) or not isinstance(values, list) or len(episode_ids) != len(values):
            raise ValueError(f"misaligned fold predictions: {key}")
        for episode_id, prediction in zip(episode_ids, values):
            if episode_id in predictions or prediction not in (0, 1):
                raise ValueError(f"invalid or duplicate fold prediction: {episode_id}")
            predictions[episode_id] = int(prediction)
    return predictions


def surface_detection_count(summary: dict[str, Any]) -> int:
    counts = summary.get("semantic_class_counts")
    if not isinstance(counts, dict):
        raise ValueError("semantic summary lacks class counts")
    surface_names = semantic.SEMANTIC_GROUPS["surface_material"]
    return int(sum(int(value) for name, value in counts.items() if name in surface_names))


def episode_time_bounds(episode: dict[str, Any]) -> dict[str, int | None]:
    source = common.load_json(Path(episode["source_path"]))
    source_frames = {
        frame.get("frame_index"): frame
        for frame in source.get("frames", [])
        if isinstance(frame, dict) and isinstance(frame.get("frame_index"), int)
    }
    indices = [int(frame["frame_index"]) for frame in episode["frames"]]
    bound_frames = [source_frames.get(index, {}) for index in indices]
    timestamps = [
        int(frame["source_timestamp_ms"])
        for frame in bound_frames
        if isinstance(frame.get("source_timestamp_ms"), (int, float))
    ]
    return {
        "start_manifest_index": min(indices),
        "end_manifest_index": max(indices),
        "start_timestamp_ms": min(timestamps) if len(timestamps) == len(indices) else None,
        "end_timestamp_ms": max(timestamps) if len(timestamps) == len(indices) else None,
    }


def bounded_gap(previous: dict[str, Any], current: dict[str, Any], *, max_gap_ms: int, max_manifest_gap: int) -> tuple[bool, dict[str, Any]]:
    previous_end_ms = previous.get("end_timestamp_ms")
    current_start_ms = current.get("start_timestamp_ms")
    if isinstance(previous_end_ms, int) and isinstance(current_start_ms, int):
        gap = current_start_ms - previous_end_ms
        return 0 <= gap <= max_gap_ms, {"gap_kind": "source_timestamp_ms", "gap_value": gap}
    gap = int(current["start_manifest_index"]) - int(previous["end_manifest_index"])
    return 0 <= gap <= max_manifest_gap, {"gap_kind": "manifest_frame_index", "gap_value": gap}


def find_exit_candidates(
    episodes: Sequence[dict[str, Any]],
    semantic_counts: dict[str, int],
    trajectory_predictions: dict[str, int],
    *,
    max_gap_ms: int,
    max_manifest_gap: int,
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        timing = episode_time_bounds(episode)
        by_source.setdefault(episode["source_id"], []).append({
            "episode_id": episode["episode_id"],
            "source_id": episode["source_id"],
            **timing,
        })
    candidates: list[dict[str, Any]] = []
    for source_id, rows in by_source.items():
        ordered = sorted(rows, key=lambda row: (row["start_manifest_index"], row["episode_id"]))
        for previous, current in zip(ordered, ordered[1:]):
            previous_count = semantic_counts.get(previous["episode_id"], 0)
            current_count = semantic_counts.get(current["episode_id"], 0)
            gap_ok, gap = bounded_gap(
                previous,
                current,
                max_gap_ms=max_gap_ms,
                max_manifest_gap=max_manifest_gap,
            )
            dynamic_clear = trajectory_predictions.get(current["episode_id"]) == 0
            if previous_count > 0 and current_count == 0 and gap_ok and dynamic_clear:
                candidates.append({
                    "source_id": source_id,
                    "previous_episode_id": previous["episode_id"],
                    "episode_id": current["episode_id"],
                    "previous_surface_detection_count": previous_count,
                    "current_surface_detection_count": current_count,
                    "trajectory_current_hazard_prediction": 0,
                    **gap,
                })
    return candidates


def routed_metrics(
    labels: np.ndarray,
    episode_ids: Sequence[str],
    baseline_predictions: np.ndarray,
    exit_episode_ids: set[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    predictions = np.asarray(baseline_predictions, dtype=np.int64).copy()
    for index, episode_id in enumerate(episode_ids):
        if episode_id in exit_episode_ids:
            predictions[index] = 0
    return predictions, common.binary_metrics(labels, predictions)


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_root = args.package_root.resolve()
    mil.reject_independent_direction(package_root)
    episodes, excluded = common.load_episode_specs(package_root)
    episode_ids = [episode["episode_id"] for episode in episodes]
    labels = np.asarray([episode["label"] for episode in episodes], dtype=np.int64)
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode IDs must be unique")

    mil_report = verify_report(args.mil_report)
    trajectory_report = verify_report(args.trajectory_report)
    semantic_report = verify_report(args.semantic_report)
    report_roots = []
    for report in (mil_report, trajectory_report, semantic_report):
        value = report.get("package_root")
        if not isinstance(value, str):
            raise ValueError("router input has no package root")
        report_roots.append(Path(value).resolve())
    if any(root != package_root for root in report_roots):
        raise ValueError("router inputs must bind the same package root")

    trajectory_predictions = prediction_map_from_folds(trajectory_report, "evaluation")
    semantic_summaries = semantic_report.get("episode_semantic_summaries")
    if not isinstance(semantic_summaries, list):
        raise ValueError("semantic report lacks episode summaries")
    semantic_counts = {
        summary["episode_id"]: surface_detection_count(summary)
        for summary in semantic_summaries
    }
    if set(trajectory_predictions) != set(episode_ids) or set(semantic_counts) != set(episode_ids):
        raise ValueError("router inputs do not cover exactly the real episode population")

    candidates = find_exit_candidates(
        episodes,
        semantic_counts,
        trajectory_predictions,
        max_gap_ms=args.max_gap_ms,
        max_manifest_gap=args.max_manifest_gap,
    )
    exit_episode_ids = {candidate["episode_id"] for candidate in candidates}
    runs: list[dict[str, Any]] = []
    for source_run in mil_report.get("runs", []):
        profiles = source_run.get("episode_profiles")
        if not isinstance(profiles, list):
            raise ValueError("MIL report run lacks episode profiles")
        profile_by_id = {profile["episode_id"]: profile for profile in profiles}
        if set(profile_by_id) != set(episode_ids):
            raise ValueError("MIL report run does not align with real episodes")
        baseline_predictions = np.asarray([
            int(float(profile_by_id[episode_id]["episode_probability"]) >= 0.5)
            for episode_id in episode_ids
        ], dtype=np.int64)
        routed, metrics = routed_metrics(labels, episode_ids, baseline_predictions, exit_episode_ids)
        runs.append({
            "seed": source_run["seed"],
            "baseline_metrics": common.binary_metrics(labels, baseline_predictions),
            "routed_metrics": metrics,
            "changed_episode_ids": [
                episode_ids[index]
                for index in range(len(episode_ids))
                if baseline_predictions[index] != routed[index]
            ],
            "changed_expected": [
                int(labels[index])
                for index in range(len(episode_ids))
                if baseline_predictions[index] != routed[index]
            ],
            "routed_predictions": routed.tolist(),
        })
    if not runs:
        raise ValueError("MIL report contains no runs")
    baseline_values = [run["baseline_metrics"]["balanced_accuracy"] for run in runs]
    routed_values = [run["routed_metrics"]["balanced_accuracy"] for run in runs]
    non_degrading = all(after >= before for before, after in zip(baseline_values, routed_values))
    exit_correct_all_runs = all(
        all(run["routed_predictions"][episode_ids.index(episode_id)] == labels[episode_ids.index(episode_id)] for episode_id in exit_episode_ids)
        for run in runs
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_root": str(package_root),
        "episode_count": len(episodes),
        "excluded_abstain_count": len(excluded),
        "input_reports": {
            "mil": {"path": str(args.mil_report.resolve()), "sha256": common.sha256_file(args.mil_report)},
            "trajectory": {"path": str(args.trajectory_report.resolve()), "sha256": common.sha256_file(args.trajectory_report)},
            "semantic": {"path": str(args.semantic_report.resolve()), "sha256": common.sha256_file(args.semantic_report)},
        },
        "router_contract": {
            "causal": True,
            "surface_material_classes": sorted(semantic.SEMANTIC_GROUPS["surface_material"]),
            "max_gap_ms": args.max_gap_ms,
            "max_manifest_gap": args.max_manifest_gap,
            "required_previous_surface_detection": True,
            "required_current_surface_absence": True,
            "required_current_trajectory_no_hazard": True,
            "held_out_label_consumed_by_router": False,
            "learned_router_parameters": 0,
        },
        "exit_candidates": candidates,
        "runs": runs,
        "summary": {
            "run_count": len(runs),
            "baseline_balanced_accuracy_values": baseline_values,
            "routed_balanced_accuracy_values": routed_values,
            "baseline_balanced_accuracy_median": float(np.median(baseline_values)),
            "routed_balanced_accuracy_median": float(np.median(routed_values)),
            "routed_balanced_accuracy_min": float(min(routed_values)),
            "all_runs_non_degrading": non_degrading,
            "exit_candidates_correct_in_all_runs": exit_correct_all_runs,
            "passed": bool(exit_episode_ids and non_degrading and exit_correct_all_runs and min(routed_values) >= 0.70),
        },
        "evidence_limit": "Tiny provisional source-isolated set; validates a causal routing prototype, not production lifecycle accuracy.",
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_model_replacement_authorized": False,
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--mil-report", type=Path, required=True)
    parser.add_argument("--trajectory-report", type=Path, required=True)
    parser.add_argument("--semantic-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-gap-ms", type=int, default=5000)
    parser.add_argument("--max-manifest-gap", type=int, default=3)
    args = parser.parse_args()
    if args.max_gap_ms <= 0 or args.max_manifest_gap <= 0:
        parser.error("gap limits must be positive")
    return args


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **report["summary"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
