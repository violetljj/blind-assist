#!/usr/bin/env python3
"""Lock the first four R4 obstacle-qualified SANPO sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = "blindassist_hftf_stage_b_split_source_validation_r4"
LEDGER_SCHEMA = "blindassist_hftf_r4_source_pool_burn_ledger"
PLAN_SCHEMA = "blindassist_hftf_r4_obstacle_inventory_candidate_plan"
REPORT_SCHEMA = (
    "blindassist_hftf_r4_obstacle_reference_opportunity_source_result"
)
SCHEMA = "blindassist_hftf_r4_obstacle_opportunity_cohort_lock"
QUALIFIED = "R4_SOURCE_OBSTACLE_REFERENCE_OPPORTUNITY_QUALIFIED"
COHORT_QUALIFIED = "R4_OBSTACLE_OPPORTUNITY_COHORT_QUALIFIED"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_firewall(report: dict[str, Any]) -> None:
    expected = {
        "obstacle_reference_grid_computed": True,
        "ground_reference_computed": False,
        "candidate_grid_computed": False,
        "angular_baseline_computed": False,
        "arm_metric_or_delta_computed": False,
    }
    if report.get("reference_only_assertions") != expected:
        raise ValueError("R4 qualification report violates arm firewall")
    if report.get("arm_outcome_authorized") is not False:
        raise ValueError("R4 qualification report authorizes arm outcome")


def lock(
    protocol_path: Path,
    ledger_path: Path,
    inventory_plan_path: Path,
    report_paths: list[Path],
) -> dict[str, Any]:
    protocol = _load(protocol_path)
    ledger = _load(ledger_path)
    plan = _load(inventory_plan_path)
    protocol_sha = _sha256(protocol_path)
    ledger_sha = _sha256(ledger_path)
    plan_sha = _sha256(inventory_plan_path)
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("status") != "FROZEN_BEFORE_R4_OUTCOME"
    ):
        raise ValueError("R4 protocol is not frozen")
    if (
        ledger.get("schema") != LEDGER_SCHEMA
        or ledger.get("status") != "FROZEN_BEFORE_R4_OUTCOME"
    ):
        raise ValueError("R4 burn ledger is not frozen")
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("terminal")
        != "R4_OBSTACLE_INVENTORY_CANDIDATE_PLAN_READY"
        or plan.get("protocol_sha256") != protocol_sha
        or plan.get("burn_ledger_sha256") != ledger_sha
        or any(
            plan.get(name) is not False
            for name in (
                "reference_outcome_read",
                "ground_outcome_read",
                "candidate_outcome_read",
                "baseline_outcome_read",
            )
        )
    ):
        raise ValueError("R4 inventory plan binding mismatch")
    source = protocol["obstacle_source_role"]
    required = int(source["required_qualified_sessions"])
    budget = int(source["maximum_inventory_eligible_sessions_to_screen"])
    if not report_paths or len(report_paths) > budget:
        raise ValueError("Invalid R4 qualification report count")
    expected_by_rank = {
        int(item["inventory_eligible_rank"]): str(item["session_id"])
        for item in plan["inventory_candidates"]
    }

    reports: list[dict[str, Any]] = []
    qualified: list[dict[str, Any]] = []
    seen_sessions: set[str] = set()
    seen_ranks: set[int] = set()
    for report_path in report_paths:
        report = _load(report_path)
        if report.get("schema") != REPORT_SCHEMA:
            raise ValueError("Unexpected R4 qualification report schema")
        rank = int(report["inventory_eligible_rank"])
        session_id = str(report["source_session_id"])
        if rank in seen_ranks or session_id in seen_sessions:
            raise ValueError("Duplicate R4 rank or session")
        if expected_by_rank.get(rank) != session_id:
            raise ValueError(f"R4 report/plan mismatch at rank {rank}")
        if (
            report.get("protocol_sha256") != protocol_sha
            or report.get("burn_ledger_sha256") != ledger_sha
            or report.get("inventory_plan_sha256") != plan_sha
        ):
            raise ValueError(f"R4 report binding mismatch at rank {rank}")
        if report.get("qualified") is True:
            if report.get("terminal") != QUALIFIED:
                raise ValueError("R4 qualified terminal mismatch")
            _validate_firewall(report)
            authority_path = Path(str(report["authority_report_path"]))
            if _sha256(authority_path) != report["authority_report_sha256"]:
                raise ValueError("R4 authority report changed after qualification")
            item = {
                "inventory_eligible_rank": rank,
                "source_session_id": session_id,
                "qualification_report_path": str(report_path.resolve()),
                "qualification_report_sha256": _sha256(report_path),
                "authority_report_path": str(authority_path.resolve()),
                "authority_report_sha256": report[
                    "authority_report_sha256"
                ],
                "manifest_sha256": report["manifest_sha256"],
                "dataset_spec_sha256": report["dataset_spec_sha256"],
                "camera_poses_sha256": report["camera_poses_sha256"],
            }
            qualified.append(item)
        elif report.get("qualified") is not False:
            raise ValueError("R4 qualification decision is missing")
        reports.append(
            {
                "inventory_eligible_rank": rank,
                "source_session_id": session_id,
                "qualification_report_path": str(report_path.resolve()),
                "qualification_report_sha256": _sha256(report_path),
                "qualified": bool(report["qualified"]),
            }
        )
        seen_ranks.add(rank)
        seen_sessions.add(session_id)

    reports.sort(key=lambda item: item["inventory_eligible_rank"])
    qualified.sort(key=lambda item: item["inventory_eligible_rank"])
    if [item["inventory_eligible_rank"] for item in reports] != list(
        range(1, len(reports) + 1)
    ):
        raise ValueError("R4 reports are not a contiguous inventory prefix")
    if len(qualified) != required:
        raise ValueError("R4 lock requires exactly four qualified sources")
    if qualified[-1]["inventory_eligible_rank"] != len(reports):
        raise ValueError("R4 screening did not stop at fourth qualification")

    return {
        "schema": SCHEMA,
        "terminal": COHORT_QUALIFIED,
        "workflow_profile": protocol["workflow_profile"],
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": protocol_sha,
        "burn_ledger_path": str(ledger_path.resolve()),
        "burn_ledger_sha256": ledger_sha,
        "inventory_plan_path": str(inventory_plan_path.resolve()),
        "inventory_plan_sha256": plan_sha,
        "screened_report_count": len(reports),
        "screened_reports": reports,
        "required_session_count": required,
        "required_sessions": qualified,
        "selection_firewall": {
            "obstacle_reference_used": True,
            "ground_reference_used": False,
            "candidate_used": False,
            "angular_baseline_used": False,
            "arm_metric_or_delta_used": False,
        },
        "formal_arm_outcome_authorized": True,
        "joint_terminal_decided": False,
        "stage_c_execution_authorized": False,
        "student_training_authorized": False,
        "research_mainline_changed": False,
        "default_app_changed": False,
    }


def _require_artifacts_output(path: Path) -> Path:
    artifacts_root = (
        Path(__file__).resolve().parents[3] / "artifacts.local"
    ).resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError(
            f"Output must stay under {artifacts_root}: {resolved}"
        ) from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--burn-ledger", type=Path, required=True)
    parser.add_argument("--inventory-plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = _require_artifacts_output(args.output)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite report: {output}")
        result = lock(
            args.protocol.resolve(),
            args.burn_ledger.resolve(),
            args.inventory_plan.resolve(),
            [path.resolve() for path in args.report],
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(
            json.dumps(
                {
                    "terminal": result["terminal"],
                    "required_session_count": result[
                        "required_session_count"
                    ],
                    "output": str(output),
                }
            )
        )
        return 0
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
