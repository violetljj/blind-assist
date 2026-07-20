#!/usr/bin/env python3
"""Test a multi-group static-risk exit router without changing the v1 router.

This experiment reuses the frozen MIL, trajectory, and prompt-free semantic
reports. It closes a stale static event only after every selected semantic risk
group is absent in the next causal episode and the trajectory probe is clear.
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
import run_public_silver_semantic_exit_router as base


SCHEMA = "blindassist_public_silver_risk_profile_exit_router_v1"


def selected_detection_count(summary: dict[str, Any], selected_groups: Sequence[str]) -> int:
    counts = summary.get("semantic_class_counts")
    if not isinstance(counts, dict):
        raise ValueError("semantic summary lacks class counts")
    names: set[str] = set()
    for group in selected_groups:
        if group not in semantic.SEMANTIC_GROUPS:
            raise ValueError(f"unknown semantic group: {group}")
        names.update(semantic.SEMANTIC_GROUPS[group])
    return int(sum(int(value) for name, value in counts.items() if name in names))


def generic_candidates(
    episodes: Sequence[dict[str, Any]],
    semantic_counts: dict[str, int],
    trajectory_predictions: dict[str, int],
    *,
    max_gap_ms: int,
    max_manifest_gap: int,
) -> list[dict[str, Any]]:
    candidates = base.find_exit_candidates(
        episodes,
        semantic_counts,
        trajectory_predictions,
        max_gap_ms=max_gap_ms,
        max_manifest_gap=max_manifest_gap,
    )
    for candidate in candidates:
        candidate["previous_risk_profile_detection_count"] = candidate.pop(
            "previous_surface_detection_count"
        )
        candidate["current_risk_profile_detection_count"] = candidate.pop(
            "current_surface_detection_count"
        )
    return candidates


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_root = args.package_root.resolve()
    mil.reject_independent_direction(package_root)
    episodes, excluded = common.load_episode_specs(package_root)
    episode_ids = [episode["episode_id"] for episode in episodes]
    labels = np.asarray([episode["label"] for episode in episodes], dtype=np.int64)
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("episode IDs must be unique")

    mil_report = base.verify_report(args.mil_report)
    trajectory_report = base.verify_report(args.trajectory_report)
    semantic_report = base.verify_report(args.semantic_report)
    for report in (mil_report, trajectory_report, semantic_report):
        value = report.get("package_root")
        if not isinstance(value, str) or Path(value).resolve() != package_root:
            raise ValueError("router inputs must bind the same package root")

    trajectory_predictions = base.prediction_map_from_folds(trajectory_report, "evaluation")
    semantic_summaries = semantic_report.get("episode_semantic_summaries")
    if not isinstance(semantic_summaries, list):
        raise ValueError("semantic report lacks episode summaries")
    semantic_counts = {
        summary["episode_id"]: selected_detection_count(summary, args.semantic_groups)
        for summary in semantic_summaries
    }
    if set(trajectory_predictions) != set(episode_ids) or set(semantic_counts) != set(episode_ids):
        raise ValueError("router inputs do not cover exactly the real episode population")

    candidates = generic_candidates(
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
        routed, metrics = base.routed_metrics(labels, episode_ids, baseline_predictions, exit_episode_ids)
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
        all(
            run["routed_predictions"][episode_ids.index(episode_id)]
            == labels[episode_ids.index(episode_id)]
            for episode_id in exit_episode_ids
        )
        for run in runs
    )
    selected_classes = sorted({
        name
        for group in args.semantic_groups
        for name in semantic.SEMANTIC_GROUPS[group]
    })
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
            "selected_semantic_groups": list(args.semantic_groups),
            "selected_semantic_classes": selected_classes,
            "max_gap_ms": args.max_gap_ms,
            "max_manifest_gap": args.max_manifest_gap,
            "required_previous_selected_risk_detection": True,
            "required_current_selected_risk_absence": True,
            "required_current_trajectory_no_hazard": True,
            "held_out_label_consumed_by_router": False,
            "learned_router_parameters": 0,
            "base_v1_router_modified": False,
        },
        "exit_candidates": candidates,
        "runs": runs,
        "summary": {
            "run_count": len(runs),
            "exit_candidate_count": len(candidates),
            "baseline_balanced_accuracy_values": baseline_values,
            "routed_balanced_accuracy_values": routed_values,
            "baseline_balanced_accuracy_median": float(np.median(baseline_values)),
            "routed_balanced_accuracy_median": float(np.median(routed_values)),
            "routed_balanced_accuracy_min": float(min(routed_values)),
            "all_runs_non_degrading": non_degrading,
            "exit_candidates_correct_in_all_runs": exit_correct_all_runs,
            "passed": bool(exit_episode_ids and non_degrading and exit_correct_all_runs and min(routed_values) >= 0.70),
        },
        "evidence_limit": "Tiny provisional source-isolated package; risk-profile prototype only, not production lifecycle accuracy.",
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
    parser.add_argument(
        "--semantic-groups",
        nargs="+",
        choices=sorted(semantic.SEMANTIC_GROUPS),
        default=["surface_material", "barrier_structure"],
    )
    parser.add_argument("--max-gap-ms", type=int, default=5000)
    parser.add_argument("--max-manifest-gap", type=int, default=3)
    args = parser.parse_args()
    if args.max_gap_ms <= 0 or args.max_manifest_gap <= 0:
        parser.error("gap limits must be positive")
    if len(set(args.semantic_groups)) != len(args.semantic_groups):
        parser.error("semantic groups must be unique")
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
