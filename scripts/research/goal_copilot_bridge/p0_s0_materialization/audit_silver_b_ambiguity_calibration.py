"""Compare a consumed-development baseline with a P0-D1 calibration canary."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer
from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_silver_b_brain_baseline as runner


class CalibrationAuditError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationAuditError(message)


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def _decision_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    decisions = report.get("raw_decisions")
    _require(isinstance(decisions, list), "report raw_decisions missing")
    result = {str(item["episode_id"]): item for item in decisions}
    _require(len(result) == len(decisions), "duplicate decision episode_id")
    return result


def _evaluation_map(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    episodes = report.get("frozen_evaluator", {}).get("episodes")
    _require(isinstance(episodes, list), "frozen evaluator episodes missing")
    return {str(item["episode_id"]): item for item in episodes}


def audit(
    cohort: Mapping[str, Any],
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    cohort_sha = str(cohort.get("report_sha256"))
    _require(baseline.get("cohort_report_sha256") == cohort_sha, "baseline cohort mismatch")
    _require(candidate.get("cohort_report_sha256") == cohort_sha, "candidate cohort mismatch")
    _require(baseline.get("policy_id") == runner.POLICY_ID, "unexpected baseline policy")
    _require(
        candidate.get("policy_id") in {runner.CALIBRATION_POLICY_ID, runner.CALIBRATION_POLICY_V2_ID},
        "unexpected calibration policy",
    )
    for field in ("executable", "executable_sha256", "cli_version", "model", "reasoning_effort"):
        _require(baseline["provider"].get(field) == candidate["provider"].get(field), f"provider drift: {field}")

    episodes = cohort.get("episodes")
    _require(isinstance(episodes, list) and episodes, "cohort episodes missing")
    baseline_decisions = _decision_map(baseline)
    candidate_decisions = _decision_map(candidate)
    expected_ids = {str(item["episode_id"]) for item in episodes}
    _require(set(baseline_decisions) == expected_ids, "baseline episode set mismatch")
    _require(set(candidate_decisions) == expected_ids, "candidate episode set mismatch")

    by_resolution: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    ambiguous_by_parent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for episode in episodes:
        resolution = str(episode["evaluator_episode"]["goal_reference_resolution"])
        by_resolution[resolution].append(episode)
        if resolution == "AMBIGUOUS":
            parent = str(episode["evaluator_episode"]["goal_spec"]["target_name"])
            ambiguous_by_parent[parent].append(episode)

    def unsupported(decisions: Mapping[str, Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> int:
        return sum(decisions[str(row["episode_id"])]["action"] == "SELECT" for row in rows)

    ambiguous = by_resolution["AMBIGUOUS"]
    unique = by_resolution["UNIQUE"]
    resolvable = unique + by_resolution["SET_VALUED"]
    parent_rows = []
    for parent in sorted(ambiguous_by_parent):
        rows = ambiguous_by_parent[parent]
        before = unsupported(baseline_decisions, rows)
        after = unsupported(candidate_decisions, rows)
        parent_rows.append({
            "venue_parent": parent,
            "episode_count": len(rows),
            "baseline_unsupported_commits": before,
            "candidate_unsupported_commits": after,
            "baseline_rate": before / len(rows),
            "candidate_rate": after / len(rows),
        })

    baseline_eval = _evaluation_map(baseline)
    candidate_eval = _evaluation_map(candidate)
    baseline_correct_ids = {
        episode_id for episode_id, row in baseline_eval.items()
        if row["end_to_end"]["outcome"] == "CORRECT_GROUNDING"
    }
    retained = sum(
        candidate_eval[episode_id]["end_to_end"]["outcome"] == "CORRECT_GROUNDING"
        for episode_id in baseline_correct_ids
    )
    unique_candidate_refusals = sum(
        candidate_decisions[str(row["episode_id"])]["action"] != "SELECT" for row in unique
    )
    resolvable_candidate_refusals = sum(
        candidate_decisions[str(row["episode_id"])]["action"] != "SELECT" for row in resolvable
    )

    baseline_macro = sum(row["baseline_rate"] for row in parent_rows) / len(parent_rows)
    candidate_macro = sum(row["candidate_rate"] for row in parent_rows) / len(parent_rows)
    support_pair_counts: dict[str, int] = defaultdict(int)
    for row in candidate_decisions.values():
        if "place_support" in row and "entrance_relation_support" in row:
            key = f'{row["place_support"]}/{row["entrance_relation_support"]}/{row["action"]}'
            support_pair_counts[key] += 1
    report = {
        "schema_version": 1,
        "status": "CONSUMED_DEVELOPMENT_MECHANISM_CANARY_COMPLETE",
        "data_role": "CONSUMED_DEVELOPMENT_NOT_PARENT_DISJOINT_CONFIRMATION",
        "cohort_report_sha256": cohort_sha,
        "baseline_report_sha256": baseline["report_sha256"],
        "candidate_report_sha256": candidate["report_sha256"],
        "candidate_policy_id": candidate["policy_id"],
        "provider": candidate["provider"],
        "metrics": {
            "unsupported_commit_rate": {
                "baseline": _rate(unsupported(baseline_decisions, ambiguous), len(ambiguous)),
                "candidate": _rate(unsupported(candidate_decisions, ambiguous), len(ambiguous)),
            },
            "venue_parent_macro_unsupported_commit_rate": {
                "baseline": baseline_macro,
                "candidate": candidate_macro,
                "parent_count": len(parent_rows),
            },
            "baseline_correct_grounding_retention": _rate(retained, len(baseline_correct_ids)),
            "unnecessary_unique_refusal_rate": _rate(unique_candidate_refusals, len(unique)),
            "resolvable_refusal_rate": _rate(resolvable_candidate_refusals, len(resolvable)),
        },
        "venue_parent_breakdown": parent_rows,
        "candidate_support_pair_action_counts": dict(sorted(support_pair_counts.items())),
        "claim_ceiling": "CONSUMED_SILVER_B_DEVELOPMENT_MECHANICS_ONLY_NO_GENERALIZATION_OR_SCIENTIFIC_VERDICT",
        "next_required_evidence": "NEW_25_TO_40_EPISODE_VENUE_PARENT_DISJOINT_DEVELOPMENT_SLICE",
    }
    report["report_sha256"] = materializer.content_sha256(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    inputs = [json.loads(path.read_text(encoding="utf-8")) for path in (args.cohort, args.baseline, args.candidate)]
    report = audit(*inputs)
    materializer.write_json(args.output, report)
    print(json.dumps({"status": report["status"], "metrics": report["metrics"], "report_sha256": report["report_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
