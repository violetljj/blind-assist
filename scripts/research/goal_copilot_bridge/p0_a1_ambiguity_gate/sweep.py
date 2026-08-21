"""Run the single frozen P0-A1 ambiguity feature sweep."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer


class SweepError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SweepError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else None,
    }


def _center(region: Mapping[str, Any]) -> tuple[float, float]:
    return (
        (float(region["x_min"]) + float(region["x_max"])) / 2,
        (float(region["y_min"]) + float(region["y_max"])) / 2,
    )


def _features(episode: Mapping[str, Any], decision: Mapping[str, Any]) -> dict[str, float | None]:
    candidates = sorted(episode["candidates"], key=lambda row: int(row["provider_rank"]))
    scores = [float(row["proposal_score"]) for row in candidates]
    top1 = scores[0] if scores else None
    top2 = scores[1] if len(scores) > 1 else None
    selected_ids = {str(value) for value in decision.get("selected_candidate_ids", [])}
    selected = [row for row in candidates if str(row["candidate_id"]) in selected_ids]
    selected_row = min(selected, key=lambda row: int(row["provider_rank"])) if selected else None
    other_scores = [float(row["proposal_score"]) for row in candidates if row is not selected_row]
    centers = [_center(row["region"]) for row in candidates]
    if centers:
        mean_x = sum(value[0] for value in centers) / len(centers)
        mean_y = sum(value[1] for value in centers) / len(centers)
        dispersion = sum(math.hypot(x - mean_x, y - mean_y) for x, y in centers) / len(centers)
    else:
        dispersion = None
    return {
        "brain_confidence": float(decision["confidence"]) if decision.get("confidence") is not None else None,
        "detector_top1_score": top1,
        "detector_top1_margin": top1 - top2 if top1 is not None and top2 is not None else top1,
        "detector_candidate_count": float(len(candidates)),
        "detector_near_tie_count_005": float(sum(score >= top1 - 0.05 for score in scores)) if top1 is not None else 0.0,
        "selected_candidate_rank": float(selected_row["provider_rank"]) if selected_row else None,
        "selected_score_margin": (
            float(selected_row["proposal_score"]) - max(other_scores)
            if selected_row is not None and other_scores
            else (float(selected_row["proposal_score"]) if selected_row is not None else None)
        ),
        "candidate_center_dispersion": dispersion,
    }


def _load_rows(repo_root: Path, protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for source in protocol["inputs"]:
        cohort_path = repo_root / source["cohort_path"]
        baseline_path = repo_root / source["baseline_path"]
        _require(_file_sha256(cohort_path) == source["cohort_file_sha256"], "cohort file hash drift")
        _require(_file_sha256(baseline_path) == source["baseline_file_sha256"], "baseline file hash drift")
        cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        _require(cohort["report_sha256"] == source["cohort_report_sha256"], "cohort report hash drift")
        _require(baseline["report_sha256"] == source["baseline_report_sha256"], "baseline report hash drift")
        _require(baseline["cohort_report_sha256"] == cohort["report_sha256"], "baseline cohort mismatch")
        decisions = {str(row["episode_id"]): row for row in baseline["raw_decisions"]}
        evaluations = {str(row["episode_id"]): row for row in baseline["frozen_evaluator"]["episodes"]}
        _require(len(decisions) == len(cohort["episodes"]), "decision count mismatch")
        for episode in cohort["episodes"]:
            episode_id = str(episode["episode_id"])
            _require(episode_id not in seen_ids, "duplicate episode across sources")
            seen_ids.add(episode_id)
            decision = decisions[episode_id]
            evaluation = evaluations[episode_id]
            resolution = str(episode["evaluator_episode"]["goal_reference_resolution"])
            _require(resolution in {"UNIQUE", "SET_VALUED", "AMBIGUOUS"}, "unexpected truth class")
            rows.append({
                "episode_id": episode_id,
                "venue_parent_id": str(episode["evaluator_episode"]["goal_spec"]["target_name"]),
                "truth": "AMBIGUOUS" if resolution == "AMBIGUOUS" else "RESOLVABLE",
                "resolution": resolution,
                "baseline_select": decision["action"] == "SELECT",
                "baseline_correct": evaluation["end_to_end"]["outcome"] == "CORRECT_GROUNDING",
                "features": _features(episode, decision),
            })
    return rows


def _thresholds(values: Sequence[float]) -> list[float]:
    unique = sorted(set(values))
    if len(unique) <= 12:
        return unique
    result = []
    for index in range(11):
        rank = round(index * (len(unique) - 1) / 10)
        result.append(unique[rank])
    return sorted(set(result))


def _metrics(rows: Sequence[Mapping[str, Any]], committed: set[str]) -> dict[str, Any]:
    ambiguous = [row for row in rows if row["truth"] == "AMBIGUOUS"]
    resolvable = [row for row in rows if row["truth"] == "RESOLVABLE"]
    ambiguous_commits = sum(row["episode_id"] in committed for row in ambiguous)
    resolvable_commits = [row for row in resolvable if row["episode_id"] in committed]
    correct_commits = sum(row["baseline_correct"] for row in resolvable_commits)
    parent_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ambiguous:
        parent_groups[str(row["venue_parent_id"])].append(row)
    macro = sum(
        sum(row["episode_id"] in committed for row in group) / len(group)
        for group in parent_groups.values()
    ) / len(parent_groups)
    return {
        "ambiguous_false_commit_rate_episode_micro": _rate(ambiguous_commits, len(ambiguous)),
        "ambiguous_false_commit_rate_venue_parent_macro": {
            "parent_count": len(parent_groups),
            "value": macro,
        },
        "resolvable_commit_coverage_episode_micro": _rate(len(resolvable_commits), len(resolvable)),
        "committed_resolvable_correctness_episode_micro": _rate(correct_commits, len(resolvable_commits)),
    }


def _passes(value: float | None, threshold: float, direction: str) -> bool:
    if value is None:
        return False
    return value >= threshold if direction == "min" else value <= threshold


def _candidate(
    rows: Sequence[Mapping[str, Any]],
    conditions: Sequence[tuple[str, str, float]],
) -> dict[str, Any]:
    committed = {
        str(row["episode_id"])
        for row in rows
        if row["baseline_select"] and all(
            _passes(row["features"][feature], threshold, direction)
            for feature, direction, threshold in conditions
        )
    }
    return {
        "complexity": len(conditions),
        "conditions": [
            {"feature": feature, "direction": direction, "threshold": threshold}
            for feature, direction, threshold in conditions
        ],
        "metrics": _metrics(rows, committed),
    }


def _value(candidate: Mapping[str, Any], metric: str) -> float:
    value = candidate["metrics"][metric]["value"]
    return float(value) if value is not None else -1.0


def _frontier(candidates: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    representatives: dict[tuple[float, float, float], Mapping[str, Any]] = {}
    for candidate in candidates:
        key = (
            _value(candidate, "ambiguous_false_commit_rate_episode_micro"),
            _value(candidate, "resolvable_commit_coverage_episode_micro"),
            _value(candidate, "committed_resolvable_correctness_episode_micro"),
        )
        incumbent = representatives.get(key)
        if incumbent is None or (
            candidate["complexity"], json.dumps(candidate["conditions"], sort_keys=True)
        ) < (
            incumbent["complexity"], json.dumps(incumbent["conditions"], sort_keys=True)
        ):
            representatives[key] = candidate
    candidates = list(representatives.values())
    result = []
    for candidate in candidates:
        amb = _value(candidate, "ambiguous_false_commit_rate_episode_micro")
        coverage = _value(candidate, "resolvable_commit_coverage_episode_micro")
        correctness = _value(candidate, "committed_resolvable_correctness_episode_micro")
        dominated = any(
            other is not candidate
            and _value(other, "ambiguous_false_commit_rate_episode_micro") <= amb
            and _value(other, "resolvable_commit_coverage_episode_micro") >= coverage
            and _value(other, "committed_resolvable_correctness_episode_micro") >= correctness
            and (
                _value(other, "ambiguous_false_commit_rate_episode_micro") < amb
                or _value(other, "resolvable_commit_coverage_episode_micro") > coverage
                or _value(other, "committed_resolvable_correctness_episode_micro") > correctness
            )
            for other in candidates
        )
        if not dominated:
            result.append(candidate)
    return sorted(result, key=lambda row: (_value(row, "ambiguous_false_commit_rate_episode_micro"), -_value(row, "resolvable_commit_coverage_episode_micro"), row["complexity"]))


def run(repo_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    _require(protocol["status"] == "FROZEN_BEFORE_FEATURE_SWEEP", "protocol is not frozen")
    _require(protocol["sweep"]["maximum_sweeps"] == 1, "protocol must allow exactly one sweep")
    rows = _load_rows(repo_root, protocol)
    directions = protocol["runtime_surface"]["features"]
    grids = {
        feature: _thresholds([
            float(row["features"][feature])
            for row in rows
            if row["baseline_select"] and row["features"][feature] is not None
        ])
        for feature in directions
    }
    candidates = []
    for feature, direction in directions.items():
        for threshold in grids[feature]:
            candidates.append(_candidate(rows, [(feature, direction, threshold)]))
    for left, right in itertools.combinations(directions, 2):
        for left_threshold, right_threshold in itertools.product(grids[left], grids[right]):
            candidates.append(_candidate(rows, [
                (left, directions[left], left_threshold),
                (right, directions[right], right_threshold),
            ]))

    baseline = {
        "complexity": 0,
        "conditions": [],
        "metrics": _metrics(rows, {str(row["episode_id"]) for row in rows if row["baseline_select"]}),
    }
    always_abstain = {"complexity": 0, "conditions": [], "metrics": _metrics(rows, set())}
    baseline_amb = _value(baseline, "ambiguous_false_commit_rate_episode_micro")
    baseline_macro = _value(baseline, "ambiguous_false_commit_rate_venue_parent_macro")
    baseline_coverage = _value(baseline, "resolvable_commit_coverage_episode_micro")
    baseline_correctness = _value(baseline, "committed_resolvable_correctness_episode_micro")
    gates = protocol["terminal_rule"]
    clear_gate = gates["CLEAR_SIGNAL_COMPACT_POLICY_NEXT"]
    weak_gate = gates["WEAK_SIGNAL_SKY_ELIGIBLE"]

    clear = [candidate for candidate in candidates if (
        baseline_amb - _value(candidate, "ambiguous_false_commit_rate_episode_micro") >= clear_gate["minimum_ambiguous_micro_absolute_reduction"]
        and baseline_macro - _value(candidate, "ambiguous_false_commit_rate_venue_parent_macro") >= clear_gate["minimum_ambiguous_parent_macro_absolute_reduction"]
        and baseline_coverage - _value(candidate, "resolvable_commit_coverage_episode_micro") <= clear_gate["maximum_resolvable_coverage_absolute_loss"]
        and baseline_correctness - _value(candidate, "committed_resolvable_correctness_episode_micro") <= clear_gate["maximum_committed_correctness_absolute_loss"]
    )]
    weak = [candidate for candidate in candidates if (
        baseline_amb - _value(candidate, "ambiguous_false_commit_rate_episode_micro") >= weak_gate["minimum_ambiguous_micro_absolute_reduction"]
        and _value(candidate, "resolvable_commit_coverage_episode_micro") >= baseline_coverage * weak_gate["minimum_resolvable_coverage_fraction_of_baseline"]
        and baseline_correctness - _value(candidate, "committed_resolvable_correctness_episode_micro") <= weak_gate["maximum_committed_correctness_absolute_loss"]
    )]

    def selection_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            -_value(candidate, "resolvable_commit_coverage_episode_micro"),
            _value(candidate, "ambiguous_false_commit_rate_episode_micro"),
            -_value(candidate, "committed_resolvable_correctness_episode_micro"),
            candidate["complexity"],
            json.dumps(candidate["conditions"], sort_keys=True),
        )

    if clear:
        terminal = "CLEAR_SIGNAL_COMPACT_POLICY_NEXT"
        selected = min(clear, key=selection_key)
    elif weak:
        terminal = "WEAK_SIGNAL_SKY_ELIGIBLE"
        selected = min(weak, key=selection_key)
    else:
        terminal = "AMBIGUITY_NOT_IDENTIFIABLE_FROM_CURRENT_RUNTIME_REPRESENTATION"
        selected = None

    report = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": materializer.content_sha256(protocol),
        "status": "ONE_FROZEN_FEATURE_SWEEP_COMPLETE",
        "terminal": terminal,
        "data_role": protocol["data_role"],
        "population": {
            "episode_count": len(rows),
            "venue_parent_count": len({row["venue_parent_id"] for row in rows}),
            "ambiguous_episode_count": sum(row["truth"] == "AMBIGUOUS" for row in rows),
            "ambiguous_parent_count": len({row["venue_parent_id"] for row in rows if row["truth"] == "AMBIGUOUS"}),
            "resolvable_episode_count": sum(row["truth"] == "RESOLVABLE" for row in rows),
            "resolvable_parent_count": len({row["venue_parent_id"] for row in rows if row["truth"] == "RESOLVABLE"}),
        },
        "feature_threshold_counts": {feature: len(values) for feature, values in grids.items()},
        "candidate_rule_count": len(candidates),
        "baselines": {"terra_ungated": baseline, "always_abstain": always_abstain},
        "qualifying_rule_counts": {"clear": len(clear), "weak_including_clear": len(weak)},
        "selected_rule": selected,
        "safety_coverage_frontier": _frontier(candidates),
        "unavailable_features_not_imputed": protocol["runtime_surface"]["unavailable_not_imputed"],
        "next_step": {
            "CLEAR_SIGNAL_COMPACT_POLICY_NEXT": "Run P0-A2 compact ambiguity policy discovery without changing representation.",
            "WEAK_SIGNAL_SKY_ELIGIBLE": "A small expression search may be justified; P0-A1 admits no rule.",
            "AMBIGUITY_NOT_IDENTIFIABLE_FROM_CURRENT_RUNTIME_REPRESENTATION": "Stop classifier and threshold work; move to active perception, persistence, or multi-frame evidence.",
        }[terminal],
        "claim_ceiling": protocol["claim_ceiling"],
    }
    report["report_sha256"] = materializer.content_sha256(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    report = run(args.repo_root.resolve(), protocol)
    materializer.write_json(args.output, report)
    print(json.dumps({
        "status": report["status"],
        "terminal": report["terminal"],
        "population": report["population"],
        "candidate_rule_count": report["candidate_rule_count"],
        "selected_rule": report["selected_rule"],
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
