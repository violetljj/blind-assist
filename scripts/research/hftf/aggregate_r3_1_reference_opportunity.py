#!/usr/bin/env python3
"""Aggregate the frozen HFTF R3.1 reference-only opportunity screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = (
    "blindassist_hftf_stage_b_reference_only_opportunity_qualification_r3_1"
)
PLAN_SCHEMA = "blindassist_hftf_r3_1_inventory_candidate_plan"
SOURCE_SCHEMA = (
    "blindassist_hftf_stage_b_reference_opportunity_source_result_r3_1"
)
OUTPUT_SCHEMA = (
    "blindassist_hftf_stage_b_reference_opportunity_cohort_result_r3_1"
)
SOURCE_QUALIFIED = "R3_1_SOURCE_REFERENCE_OPPORTUNITY_QUALIFIED"
SOURCE_REJECTED = "R3_1_SOURCE_REFERENCE_OPPORTUNITY_REJECTED"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_reference_only(report: dict[str, Any]) -> None:
    assertions = report.get("reference_only_assertions", {})
    expected = {
        "reference_grid_computed": True,
        "candidate_grid_computed": False,
        "angular_baseline_computed": False,
        "arm_metric_or_delta_computed": False,
    }
    if assertions != expected:
        raise ValueError(
            "Source report does not preserve the reference-only firewall"
        )
    if report.get("arm_outcome_authorized") is not False:
        raise ValueError("Source report authorizes an arm outcome")


def aggregate(
    protocol_path: Path,
    ledger_path: Path,
    plan_path: Path,
    report_paths: list[Path],
) -> dict[str, Any]:
    protocol = _load(protocol_path)
    ledger = _load(ledger_path)
    plan = _load(plan_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status")
        != "FROZEN_QUALIFICATION_ONLY_ARM_OUTCOME_PROHIBITED"
    ):
        raise ValueError("R3.1 qualification protocol is not frozen")
    protocol_sha = _sha256(protocol_path)
    ledger_sha = _sha256(ledger_path)
    plan_sha = _sha256(plan_path)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("terminal")
        != "R3_1_INVENTORY_CANDIDATE_PLAN_READY"
        or plan.get("protocol_sha256") != protocol_sha
        or plan.get("burn_ledger_sha256") != ledger_sha
    ):
        raise ValueError("Inventory plan binding mismatch")

    source_pool = protocol["source_pool"]
    budget = int(source_pool["maximum_inventory_eligible_sessions_to_screen"])
    required = int(source_pool["required_qualified_sessions"])
    candidates = plan.get("inventory_candidates", [])
    if len(candidates) != budget:
        raise ValueError("Inventory plan does not contain the frozen budget")
    expected_by_rank = {
        int(item["inventory_eligible_rank"]): str(item["session_id"])
        for item in candidates
    }
    if sorted(expected_by_rank) != list(range(1, budget + 1)):
        raise ValueError("Inventory plan ranks are not contiguous")
    if not report_paths or len(report_paths) > budget:
        raise ValueError("Invalid R3.1 report count")

    rows: list[dict[str, Any]] = []
    seen_ranks: set[int] = set()
    seen_sessions: set[str] = set()
    qualified_rows: list[dict[str, Any]] = []
    rejection_check_counts: Counter[str] = Counter()
    authority_failures = 0
    incomplete_geometry = 0
    ground_reports = 0
    ground_total_risk_cells = 0
    ground_nonzero_sessions: list[str] = []
    obstacle_gate_pass_counts: Counter[str] = Counter()

    for report_path in report_paths:
        report = _load(report_path)
        if report.get("schema") != SOURCE_SCHEMA:
            raise ValueError(f"Unexpected source schema: {report_path}")
        rank = int(report["inventory_eligible_rank"])
        session_id = str(report["source_session_id"])
        if rank in seen_ranks or session_id in seen_sessions:
            raise ValueError("Duplicate R3.1 rank or session")
        if expected_by_rank.get(rank) != session_id:
            raise ValueError(f"Report does not match inventory rank {rank}")
        if (
            report.get("protocol_sha256") != protocol_sha
            or report.get("burn_ledger_sha256") != ledger_sha
            or report.get("inventory_plan_sha256") != plan_sha
        ):
            raise ValueError(f"Source binding mismatch at rank {rank}")
        _assert_reference_only(report)

        qualified = report.get("qualified")
        terminal = report.get("terminal")
        if qualified is True and terminal != SOURCE_QUALIFIED:
            raise ValueError(f"Qualified terminal mismatch at rank {rank}")
        if qualified is False and terminal != SOURCE_REJECTED:
            raise ValueError(f"Rejected terminal mismatch at rank {rank}")
        if qualified not in (True, False):
            raise ValueError(f"Missing qualification decision at rank {rank}")

        checks = report.get("checks", {})
        for name, passed in checks.items():
            if passed is False:
                rejection_check_counts[str(name)] += 1
        authority_ok = report.get("authority_validation", {}).get("ok")
        if authority_ok is False:
            authority_failures += 1
        if report.get("missing_geometry_bindings"):
            incomplete_geometry += 1

        ground = report.get("reference_ground")
        if isinstance(ground, dict):
            ground_reports += 1
            risk_cells = int(ground["risk_cells"])
            ground_total_risk_cells += risk_cells
            if risk_cells > 0:
                ground_nonzero_sessions.append(session_id)
        for name in (
            "obstacle_known_coverage_each_height",
            "obstacle_primary_positive_each_height",
            "obstacle_primary_negative_each_height",
            "obstacle_all_sensitivity_thresholds_have_micro_opportunity",
        ):
            if checks.get(name) is True:
                obstacle_gate_pass_counts[name] += 1

        row = {
            "inventory_eligible_rank": rank,
            "source_session_id": session_id,
            "report_path": str(report_path.resolve()),
            "report_sha256": _sha256(report_path),
            "terminal": terminal,
            "qualified": qualified,
            "authority_ok": authority_ok,
            "missing_geometry_bindings": bool(
                report.get("missing_geometry_bindings")
            ),
            "ground_risk_cells": (
                int(ground["risk_cells"]) if isinstance(ground, dict) else None
            ),
        }
        rows.append(row)
        if qualified:
            qualified_rows.append(row)
        seen_ranks.add(rank)
        seen_sessions.add(session_id)

    rows.sort(key=lambda item: item["inventory_eligible_rank"])
    actual_ranks = [item["inventory_eligible_rank"] for item in rows]
    if actual_ranks != list(range(1, len(rows) + 1)):
        raise ValueError("Reports do not follow the frozen lexicographic prefix")

    if len(qualified_rows) >= required:
        fourth_rank = qualified_rows[required - 1]["inventory_eligible_rank"]
        if len(rows) != fourth_rank:
            raise ValueError("Screening continued after the required cohort")
        terminal = protocol["qualification_terminal"]["success"]
        selected = qualified_rows[:required]
    else:
        if len(rows) != budget:
            raise ValueError("Budget-exhausted terminal requires all reports")
        terminal = protocol["qualification_terminal"]["budget_exhausted"]
        selected = []

    return {
        "schema": OUTPUT_SCHEMA,
        "terminal": terminal,
        "workflow_profile": protocol["workflow_profile"],
        "claim_population": protocol["claim_population"],
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": protocol_sha,
        "burn_ledger_path": str(ledger_path.resolve()),
        "burn_ledger_sha256": ledger_sha,
        "inventory_plan_path": str(plan_path.resolve()),
        "inventory_plan_sha256": plan_sha,
        "screened_session_count": len(rows),
        "maximum_screening_budget": budget,
        "required_qualified_sessions": required,
        "qualified_session_count": len(qualified_rows),
        "selected_qualified_sessions": selected,
        "source_reports": rows,
        "diagnostic_summary": {
            "authority_failure_count": authority_failures,
            "incomplete_geometry_binding_count": incomplete_geometry,
            "reference_ground_report_count": ground_reports,
            "reference_ground_total_risk_cells": ground_total_risk_cells,
            "reference_ground_nonzero_risk_session_count": len(
                ground_nonzero_sessions
            ),
            "reference_ground_nonzero_risk_session_ids": (
                ground_nonzero_sessions
            ),
            "failed_check_counts": dict(sorted(rejection_check_counts.items())),
            "obstacle_gate_pass_counts": dict(
                sorted(obstacle_gate_pass_counts.items())
            ),
        },
        "reference_only_assertions": {
            "all_reports_reference_only": True,
            "candidate_grid_computed": False,
            "angular_baseline_computed": False,
            "arm_metric_or_delta_computed": False,
        },
        "arm_outcome_authorized": False,
        "future_stage_c_authorized": False,
        "student_training_authorized": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--burn-ledger", type=Path, required=True)
    parser.add_argument("--inventory-plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = aggregate(
            args.protocol,
            args.burn_ledger,
            args.inventory_plan,
            args.report,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except Exception as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 1
    print(
        json.dumps(
            {
                "terminal": result["terminal"],
                "qualified_session_count": result[
                    "qualified_session_count"
                ],
                "output": str(args.output.resolve()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
