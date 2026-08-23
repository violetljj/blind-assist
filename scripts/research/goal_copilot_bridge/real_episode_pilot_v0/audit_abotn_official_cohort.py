"""Aggregate frozen ABotN episode audits without new model calls."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "blindassist_abotn_official_cohort_audit_v0"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _outcome_class(audit: dict[str, Any]) -> str:
    metric = audit["metric_outcome"]
    if metric["false_arrival"]:
        return "FALSE_ARRIVAL"
    if metric["episode_completion"]:
        return "METRIC_GOAL_SUCCESS"
    if metric["instruction_attributable_progress_m"] > 0:
        return "PARTIAL_METRIC_PROGRESS_NO_GOAL_SUCCESS"
    if audit["provider_behavior"]["instruction_count"] == 0:
        return "CURRENT_FRAME_RELIABILITY_BEFORE_INSTRUCTION"
    return "NO_POSITIVE_METRIC_PROGRESS_AFTER_INSTRUCTION"


def summarize_audits(cohort: dict[str, Any], audits: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected = [row["episode_id"] for row in cohort["selection"]["tasks"]]
    observed = [row["episode_id"] for row in audits]
    if observed != expected:
        raise ValueError("episode audit order/identity does not match frozen cohort")
    if any(row["truth_boundaries"]["selection_accuracy"] !=
           "NOT_EVALUABLE_FUNCTIONAL_PIXEL_REGION_MISSING" for row in audits):
        raise ValueError("functional selection truth boundary drift")
    if any(row["truth_boundaries"]["lost_after_visible"] !=
           "NOT_EVALUABLE_NO_FUNCTIONAL_PIXEL_VISIBILITY_TRUTH" for row in audits):
        raise ValueError("LOST authority boundary drift")
    classes = [_outcome_class(row) for row in audits]
    counts = Counter(classes)
    episode_count = len(audits)
    reliability_limited = sum(
        row["provider_behavior"]["reliability_drop_after_provider_commitment"]
        or row["provider_behavior"]["instruction_count"] == 0
        for row in audits
    )
    dominant = (
        "CURRENT_FRAME_RELIABILITY_LIMITATION"
        if reliability_limited > episode_count / 2
        else "NO_SINGLE_SUPPORTED_DOMINANT_FAILURE"
    )
    return {
        "terminal": "ABOTN_OFFICIAL_FRESH_COHORT_EVALUATED",
        "episode_count": episode_count,
        "outcome_class_counts": dict(sorted(counts.items())),
        "metric_goal_success_count": counts["METRIC_GOAL_SUCCESS"],
        "metric_goal_success_rate": counts["METRIC_GOAL_SUCCESS"] / episode_count,
        "false_arrival_count": counts["FALSE_ARRIVAL"],
        "positive_instruction_progress_episode_count": sum(
            row["metric_outcome"]["instruction_attributable_progress_m"] > 0 for row in audits
        ),
        "current_frame_reliability_limited_episode_count": reliability_limited,
        "supported_dominance": dominant,
        "functional_selection_truth_coverage": {
            "strong_or_usable": 0,
            "unknown_or_not_evaluable": episode_count,
        },
        "selection_accuracy": "NOT_EVALUABLE_FUNCTIONAL_PIXEL_REGION_MISSING",
        "lost_after_visible": "NOT_EVALUABLE_NO_FUNCTIONAL_PIXEL_VISIBILITY_TRUTH",
        "p1_authorized": False,
        "claim_ceiling": cohort["truth_and_claim_boundary"]["claim_ceiling"],
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    cohort_path = args.cohort_freeze.resolve()
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if cohort.get("terminal") != "ABOTN_OFFICIAL_FRESH_COHORT_FROZEN":
        raise ValueError("cohort is not frozen")
    audit_paths = [path.resolve() for path in args.episode_audit]
    audits = [json.loads(path.read_text(encoding="utf-8")) for path in audit_paths]
    summary = summarize_audits(cohort, audits)
    return {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **summary,
        "inputs": {
            "cohort_freeze_sha256": _sha256(cohort_path),
            "episode_audits": [
                {"episode_id": row["episode_id"], "path": str(path), "sha256": _sha256(path)}
                for path, row in zip(audit_paths, audits)
            ],
        },
        "episode_results": [
            {
                "episode_id": row["episode_id"],
                "outcome_class": _outcome_class(row),
                "terminal_distance_to_goal_m": row["metric_outcome"]["terminal_distance_to_goal_m"],
                "instruction_attributable_progress_m": row["metric_outcome"]["instruction_attributable_progress_m"],
                "provider_observation_calls": row["execution"]["provider_observation_calls"],
            }
            for row in audits
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-freeze", type=Path, required=True)
    parser.add_argument("--episode-audit", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("cohort audit already exists")
    result = audit(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps({
        "terminal": result["terminal"],
        "metric_goal_success_count": result["metric_goal_success_count"],
        "supported_dominance": result["supported_dominance"],
        "selection_accuracy": result["selection_accuracy"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
