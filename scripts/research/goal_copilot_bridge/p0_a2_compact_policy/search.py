"""Deterministically enumerate the frozen P0-A2 compact policy DSL."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts.research.goal_copilot_bridge.p0_a1_ambiguity_gate import sweep as a1
from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer


class SearchError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SearchError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_without_report_sha(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("report_sha256", None)
    return materializer.content_sha256(payload)


def _predicate(feature: str, direction: str, threshold: float, mask: int) -> dict[str, Any]:
    return {
        "expression": {"op": "predicate", "feature": feature, "direction": direction, "threshold": threshold},
        "mask": mask,
        "complexity": 1,
    }


def _expression_key(expression: Mapping[str, Any]) -> str:
    return json.dumps(expression, sort_keys=True, separators=(",", ":"))


def _combine(op: str, children: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ordered = sorted(children, key=lambda row: _expression_key(row["expression"]))
    if op == "and":
        mask = ordered[0]["mask"]
        for child in ordered[1:]:
            mask &= child["mask"]
    else:
        mask = 0
        for child in ordered:
            mask |= child["mask"]
    return {
        "expression": {"op": op, "args": [child["expression"] for child in ordered]},
        "mask": mask,
        "complexity": sum(int(child["complexity"]) for child in ordered),
    }


def _parent_behavior(rows: Sequence[Mapping[str, Any]], committed: set[str]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["truth"] == "AMBIGUOUS":
            groups[str(row["venue_parent_id"])].append(row)
    per_parent = []
    bins = {"ZERO": 0, "GT_0_LE_0_25": 0, "GT_0_25_LE_0_50": 0, "GT_0_50": 0}
    for parent in sorted(groups):
        group = groups[parent]
        commits = sum(str(row["episode_id"]) in committed for row in group)
        rate = commits / len(group)
        if rate == 0:
            bucket = "ZERO"
        elif rate <= 0.25:
            bucket = "GT_0_LE_0_25"
        elif rate <= 0.50:
            bucket = "GT_0_25_LE_0_50"
        else:
            bucket = "GT_0_50"
        bins[bucket] += 1
        per_parent.append({
            "venue_parent_id": parent,
            "false_commits": commits,
            "ambiguous_episodes": len(group),
            "rate": rate,
            "bin": bucket,
        })
    worst_rate = max(row["rate"] for row in per_parent)
    return {
        "bins": bins,
        "worst_rate": worst_rate,
        "worst_parent_ids": [row["venue_parent_id"] for row in per_parent if row["rate"] == worst_rate],
        "per_parent": per_parent,
    }


def _policy_report(
    rows: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    committed = {
        str(row["episode_id"])
        for index, row in enumerate(rows)
        if policy["mask"] & (1 << index)
    }
    return {
        "expression": policy["expression"],
        "complexity": policy["complexity"],
        "metrics": a1._metrics(rows, committed),
        "worst_parent_behavior": _parent_behavior(rows, committed),
    }


def _metric(policy: Mapping[str, Any], name: str) -> float:
    value = policy["metrics"][name]["value"]
    return float(value) if value is not None else -1.0


def _objective_key(policy: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _metric(policy, "ambiguous_false_commit_rate_venue_parent_macro"),
        _metric(policy, "ambiguous_false_commit_rate_episode_micro"),
        float(policy["worst_parent_behavior"]["worst_rate"]),
        int(policy["complexity"]),
        _expression_key(policy["expression"]),
    )


def _decide_terminal(
    incumbent: Mapping[str, Any],
    best_hard: Mapping[str, Any],
    best_relaxed: Mapping[str, Any] | None,
    minimum_gain: float,
) -> str:
    incumbent_macro = _metric(incumbent, "ambiguous_false_commit_rate_venue_parent_macro")
    if incumbent_macro - _metric(best_hard, "ambiguous_false_commit_rate_venue_parent_macro") >= minimum_gain - 1e-12:
        return "CLEAR_COMPACT_POLICY_IMPROVEMENT"
    if best_relaxed is not None and incumbent_macro - _metric(best_relaxed, "ambiguous_false_commit_rate_venue_parent_macro") >= minimum_gain - 1e-12:
        return "COMPLEXITY_ONLY_BUYS_ABSTENTION"
    return "A1_COMPACT_RULE_RETAINED_NO_MEANINGFUL_COMPLEXITY_GAIN"


def _forms_for_three(
    left: Mapping[str, Any], middle: Mapping[str, Any], right: Mapping[str, Any]
) -> list[dict[str, Any]]:
    leaves = [left, middle, right]
    forms = [_combine("and", leaves), _combine("or", leaves)]
    for singleton_index in range(3):
        singleton = leaves[singleton_index]
        pair = [leaf for index, leaf in enumerate(leaves) if index != singleton_index]
        forms.append(_combine("or", [_combine("and", pair), singleton]))
        forms.append(_combine("and", [_combine("or", pair), singleton]))
    unique = {_expression_key(form["expression"]): form for form in forms}
    return [unique[key] for key in sorted(unique)]


def run(repo_root: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    _require(protocol["status"] == "FROZEN_BEFORE_DETERMINISTIC_SEARCH", "protocol is not frozen")
    binding = protocol["a1_binding"]
    a1_protocol_path = repo_root / binding["protocol_path"]
    a1_result_path = repo_root / binding["result_path"]
    _require(_file_sha256(a1_protocol_path) == binding["protocol_file_sha256"], "A1 protocol file hash drift")
    _require(_file_sha256(a1_result_path) == binding["result_file_sha256"], "A1 result file hash drift")
    a1_protocol = json.loads(a1_protocol_path.read_text(encoding="utf-8"))
    a1_result = json.loads(a1_result_path.read_text(encoding="utf-8"))
    _require(materializer.content_sha256(a1_protocol) == binding["protocol_content_sha256"], "A1 protocol content drift")
    _require(_content_without_report_sha(a1_result) == binding["result_content_sha256"], "A1 result content drift")
    rows = a1._load_rows(repo_root, a1_protocol)
    directions = a1_protocol["runtime_surface"]["features"]
    base_mask = sum((1 << index) for index, row in enumerate(rows) if row["baseline_select"])

    predicates_by_feature: dict[str, list[dict[str, Any]]] = {}
    for feature, direction in directions.items():
        thresholds = a1._thresholds([
            float(row["features"][feature])
            for row in rows
            if row["baseline_select"] and row["features"][feature] is not None
        ])
        predicates = []
        for threshold in thresholds:
            mask = 0
            for index, row in enumerate(rows):
                if row["baseline_select"] and a1._passes(row["features"][feature], threshold, direction):
                    mask |= 1 << index
            predicates.append(_predicate(feature, direction, threshold, mask & base_mask))
        predicates_by_feature[feature] = predicates

    syntactic_count = 0
    representatives: dict[int, dict[str, Any]] = {}

    def consider(policy: dict[str, Any]) -> None:
        nonlocal syntactic_count
        syntactic_count += 1
        incumbent = representatives.get(policy["mask"])
        if incumbent is None or (
            policy["complexity"], _expression_key(policy["expression"])
        ) < (
            incumbent["complexity"], _expression_key(incumbent["expression"])
        ):
            representatives[policy["mask"]] = policy

    features = sorted(predicates_by_feature)
    for feature in features:
        for predicate in predicates_by_feature[feature]:
            consider(predicate)
    for left_feature, right_feature in itertools.combinations(features, 2):
        for left, right in itertools.product(predicates_by_feature[left_feature], predicates_by_feature[right_feature]):
            consider(_combine("and", [left, right]))
            consider(_combine("or", [left, right]))
    for feature_names in itertools.combinations(features, 3):
        grids = [predicates_by_feature[name] for name in feature_names]
        for left, middle, right in itertools.product(*grids):
            for policy in _forms_for_three(left, middle, right):
                consider(policy)

    policies = [_policy_report(rows, policy) for policy in representatives.values()]
    constraints = protocol["objective"]["hard_constraints"]
    hard = [policy for policy in policies if (
        abs(_metric(policy, "resolvable_commit_coverage_episode_micro") - constraints["resolvable_commit_coverage"]) <= 1e-12
        and _metric(policy, "committed_resolvable_correctness_episode_micro") >= constraints["minimum_committed_resolvable_correctness"] - 1e-12
        and policy["complexity"] <= constraints["maximum_predicates"]
    )]
    _require(hard, "incumbent disappeared from the hard-feasible search space")
    best_hard = min(hard, key=_objective_key)
    relaxed = [policy for policy in policies if (
        0.65 <= _metric(policy, "resolvable_commit_coverage_episode_micro") < 1.0
        and _metric(policy, "committed_resolvable_correctness_episode_micro") >= constraints["minimum_committed_resolvable_correctness"] - 1e-12
    )]
    best_relaxed = min(relaxed, key=_objective_key) if relaxed else None

    incumbent_spec = protocol["incumbent"]
    incumbent_conditions = [
        (row["feature"], row["direction"], float(row["threshold"]))
        for row in incumbent_spec["predicates"]
    ]
    incumbent_mask = base_mask
    incumbent_children = []
    for feature, direction, threshold in incumbent_conditions:
        predicate = next(
            row for row in predicates_by_feature[feature]
            if row["expression"]["direction"] == direction and abs(row["expression"]["threshold"] - threshold) <= 1e-15
        )
        incumbent_mask &= predicate["mask"]
        incumbent_children.append(predicate)
    incumbent_policy = _policy_report(rows, _combine("and", incumbent_children))
    _require(abs(_metric(incumbent_policy, "ambiguous_false_commit_rate_venue_parent_macro") - incumbent_spec["ambiguous_parent_macro_false_commit"]) <= 1e-12, "A1 incumbent macro mismatch")
    _require(abs(_metric(incumbent_policy, "resolvable_commit_coverage_episode_micro") - incumbent_spec["resolvable_commit_coverage"]) <= 1e-12, "A1 incumbent coverage mismatch")

    minimum_gain = float(protocol["objective"]["minimum_meaningful_parent_macro_absolute_improvement"])
    terminal = _decide_terminal(incumbent_policy, best_hard, best_relaxed, minimum_gain)
    selected = best_hard if terminal == "CLEAR_COMPACT_POLICY_IMPROVEMENT" else incumbent_policy
    incumbent_macro = _metric(incumbent_policy, "ambiguous_false_commit_rate_venue_parent_macro")
    report = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": materializer.content_sha256(protocol),
        "status": "DETERMINISTIC_COMPACT_POLICY_SEARCH_COMPLETE",
        "terminal": terminal,
        "population": a1_result["population"],
        "search": {
            "feature_count": len(features),
            "predicate_threshold_counts": {name: len(rows_) for name, rows_ in predicates_by_feature.items()},
            "syntactic_policy_count": syntactic_count,
            "unique_commit_behavior_count": len(representatives),
            "hard_feasible_behavior_count": len(hard),
            "relaxed_behavior_count": len(relaxed),
        },
        "incumbent": incumbent_policy,
        "best_hard_feasible": best_hard,
        "best_relaxed_diagnostic": best_relaxed,
        "selected_policy": selected,
        "parent_macro_absolute_improvement_over_a1": incumbent_macro - _metric(best_hard, "ambiguous_false_commit_rate_venue_parent_macro"),
        "minimum_meaningful_improvement": minimum_gain,
        "next_step": {
            "CLEAR_COMPACT_POLICY_IMPROVEMENT": "Freeze this Development winner, then design one limited fresh scientific confirmation.",
            "COMPLEXITY_ONLY_BUYS_ABSTENTION": "Retain A1; do not exchange coverage for a lower false-commit rate.",
            "A1_COMPACT_RULE_RETAINED_NO_MEANINGFUL_COMPLEXITY_GAIN": "Retain A1; the third predicate did not buy meaningful compact-policy value.",
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
        "search": report["search"],
        "parent_macro_absolute_improvement_over_a1": report["parent_macro_absolute_improvement_over_a1"],
        "selected_policy": report["selected_policy"],
        "report_sha256": report["report_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
