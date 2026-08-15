#!/usr/bin/env python3
"""Apply the pre-frozen SATOM-R0 Real E0 winner rule without retuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .bonn import sha256_file


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _metric(row: dict[str, Any], name: str) -> float:
    value = row.get(name)
    if value is None:
        raise RuntimeError(f"required metric unavailable: {name}")
    return float(value)


def _check_global(
    candidate: dict[str, Any], comparator: dict[str, Any], tolerances: dict[str, Any]
) -> list[dict[str, Any]]:
    checks = []
    for scope, limits in tolerances.items():
        left, right = candidate[scope], comparator[scope]
        definitions = (
            ("coverage", ">=", _metric(right, "coverage") + float(limits["coverage_floor_delta"])),
            ("false_clear", "<=", _metric(right, "false_clear") + float(limits["false_clear_ceiling_delta"])),
            ("false_block", "<=", _metric(right, "false_block") + float(limits["false_block_ceiling_delta"])),
            ("clearance_mae_m", "<=", _metric(right, "clearance_mae_m") + float(limits["clearance_mae_m_ceiling_delta"])),
        )
        for metric, operator, threshold in definitions:
            value = _metric(left, metric)
            passed = value >= threshold if operator == ">=" else value <= threshold
            checks.append(
                {"scope": scope, "metric": metric, "operator": operator, "candidate": value,
                 "comparator": _metric(right, metric), "threshold": threshold, "passed": passed}
            )
    return checks


def _check_matched(candidate: dict[str, Any], comparator: dict[str, Any], rule: dict[str, Any]) -> list[dict[str, Any]]:
    left = candidate["matched_coverage"]["across_parents"]["0.70"]
    right = comparator["matched_coverage"]["across_parents"]["0.70"]
    _require(left.get("available") is True and right.get("available") is True, "matched 0.70 coverage unavailable")
    limits = (
        ("false_clear", float(rule["parent_macro_false_clear_ceiling_delta"])),
        ("false_block", float(rule["parent_macro_false_block_ceiling_delta"])),
    )
    checks = []
    for metric, delta in limits:
        value = _metric(left["parent_macro"], metric)
        comparison = _metric(right["parent_macro"], metric)
        checks.append(
            {"scope": "matched_coverage_0.70_parent_macro", "metric": metric, "operator": "<=",
             "candidate": value, "comparator": comparison, "threshold": comparison + delta,
             "passed": value <= comparison + delta}
        )
    return checks


def _check_gain(candidate: dict[str, Any], comparator: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    checks = []
    mapping = {
        "parent_macro_coverage_improvement_min": ("coverage", 1.0),
        "parent_macro_false_clear_improvement_min": ("false_clear", -1.0),
        "parent_macro_clearance_mae_m_improvement_min": ("clearance_mae_m", -1.0),
        "parent_macro_calibration_error_improvement_min": ("calibration_error", -1.0),
    }
    for key, minimum in rule.items():
        metric, direction = mapping[key]
        left = _metric(candidate["parent_macro"], metric)
        right = _metric(comparator["parent_macro"], metric)
        improvement = (left - right) * direction
        checks.append(
            {"scope": "parent_macro", "metric": metric, "candidate": left, "comparator": right,
             "improvement": improvement, "minimum": float(minimum), "passed": improvement >= float(minimum)}
        )
    return {"rule": "any", "passed": any(row["passed"] for row in checks), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--validity-amendment", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock_path = args.lock.resolve()
    amendment_path = args.validity_amendment.resolve()
    manifest_path, result_path = args.manifest.resolve(), args.result.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    _require(manifest["execution_lock"]["sha256"] == sha256_file(lock_path), "manifest/lock SHA drift")
    _require(amendment["original_lock"]["sha256"] == sha256_file(lock_path), "amendment/lock SHA drift")
    _require(
        manifest["execution_validity_amendment"]["sha256"] == sha256_file(amendment_path),
        "manifest/amendment SHA drift",
    )
    _require(result["evidence_role"] == lock["evidence_role"], "result evidence role drift")
    arms = lock["arms"]
    primary_name = arms["primary"]
    _require(primary_name == "satom_round_robin", "primary arm drift")
    candidate = result["arms"][primary_name]
    winner = lock["winner_rule"]
    comparisons = []
    for name in arms["required_comparators"]:
        comparator = result["arms"][name]
        global_checks = _check_global(candidate, comparator, winner["global_no_regret_tolerances"])
        matched_checks = _check_matched(candidate, comparator, winner["matched_coverage_0_70_no_regret_tolerances"])
        gain = _check_gain(candidate, comparator, winner["meaningful_gain_any_of"][name])
        passed = all(row["passed"] for row in global_checks + matched_checks) and gain["passed"]
        comparisons.append(
            {"comparator": name, "passed": passed, "global_no_regret": global_checks,
             "matched_coverage_no_regret": matched_checks, "meaningful_gain": gain}
        )
    passed = all(row["passed"] for row in comparisons)
    assessment = {
        "schema": "blindassist.satom_r0.bonn_real_e0_assessment.v1",
        "lock_id": lock["lock_id"],
        "execution_lock_sha256": sha256_file(lock_path),
        "execution_validity_amendment_sha256": sha256_file(amendment_path),
        "manifest_sha256": sha256_file(manifest_path),
        "result_sha256": sha256_file(result_path),
        "primary": primary_name,
        "passed": passed,
        "status": winner["pass_status"] if passed else winner["fail_status"],
        "comparisons": comparisons,
        "exploratory_arms_not_used_for_winner": arms["exploratory_only"],
        "negative_controls_not_used_for_winner": arms["negative_controls"],
        "claim_ceiling": lock["claim_ceiling"],
        "next_step": lock["next_step_if_pass"] if passed else lock["next_step_if_fail"],
        "default_app_changed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _require(not args.output.exists(), f"assessment already exists: {args.output}")
    args.output.write_text(json.dumps(assessment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": assessment["status"], "passed": passed}, indent=2))


if __name__ == "__main__":
    main()
