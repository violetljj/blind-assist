from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aggregate_r3_1_reference_opportunity import aggregate


def _write(root: Path, name: str, value: dict[str, object]) -> Path:
    path = root / f"{name}.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _source(
    rank: int,
    session: str,
    protocol_sha: str,
    ledger_sha: str,
    plan_sha: str,
    *,
    qualified: bool = False,
    ground_risk: int = 0,
) -> dict[str, object]:
    return {
        "schema": (
            "blindassist_hftf_stage_b_reference_opportunity_source_result_r3_1"
        ),
        "inventory_eligible_rank": rank,
        "source_session_id": session,
        "protocol_sha256": protocol_sha,
        "burn_ledger_sha256": ledger_sha,
        "inventory_plan_sha256": plan_sha,
        "terminal": (
            "R3_1_SOURCE_REFERENCE_OPPORTUNITY_QUALIFIED"
            if qualified
            else "R3_1_SOURCE_REFERENCE_OPPORTUNITY_REJECTED"
        ),
        "qualified": qualified,
        "authority_validation": {"ok": True},
        "reference_only_assertions": {
            "reference_grid_computed": True,
            "candidate_grid_computed": False,
            "angular_baseline_computed": False,
            "arm_metric_or_delta_computed": False,
        },
        "reference_ground": {"risk_cells": ground_risk},
        "checks": {
            "authority": True,
            "ground_reference_risk_cells": ground_risk >= 5,
        },
        "arm_outcome_authorized": False,
    }


class R31ReferenceOpportunityAggregationTest(unittest.TestCase):
    def _fixture(
        self, root: Path, *, budget: int = 4, required: int = 2
    ) -> tuple[Path, Path, Path, str, str, str]:
        protocol = _write(
            root,
            "protocol",
            {
                "schema": (
                    "blindassist_hftf_stage_b_reference_only_opportunity_"
                    "qualification_r3_1"
                ),
                "status": "FROZEN_QUALIFICATION_ONLY_ARM_OUTCOME_PROHIBITED",
                "workflow_profile": "DEVELOPMENT_STANDARD",
                "claim_population": "challenge_only",
                "source_pool": {
                    "maximum_inventory_eligible_sessions_to_screen": budget,
                    "required_qualified_sessions": required,
                },
                "qualification_terminal": {
                    "success": "R3_1_REFERENCE_OPPORTUNITY_COHORT_QUALIFIED",
                    "budget_exhausted": (
                        "R3_1_REFERENCE_OPPORTUNITY_COHORT_NOT_EVALUABLE"
                    ),
                },
            },
        )
        ledger = _write(root, "ledger", {"frozen": True})
        import hashlib

        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        plan = _write(
            root,
            "plan",
            {
                "schema": "blindassist_hftf_r3_1_inventory_candidate_plan",
                "terminal": "R3_1_INVENTORY_CANDIDATE_PLAN_READY",
                "protocol_sha256": sha(protocol),
                "burn_ledger_sha256": sha(ledger),
                "inventory_candidates": [
                    {
                        "inventory_eligible_rank": rank,
                        "session_id": f"session-{rank}",
                    }
                    for rank in range(1, budget + 1)
                ],
            },
        )
        return protocol, ledger, plan, sha(protocol), sha(ledger), sha(plan)

    def test_budget_exhaustion_closes_not_evaluable(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol, ledger, plan, p_sha, l_sha, plan_sha = self._fixture(
                root
            )
            reports = [
                _write(
                    root,
                    f"report-{rank}",
                    _source(
                        rank,
                        f"session-{rank}",
                        p_sha,
                        l_sha,
                        plan_sha,
                    ),
                )
                for rank in range(1, 5)
            ]
            result = aggregate(protocol, ledger, plan, reports)
        self.assertEqual(
            result["terminal"],
            "R3_1_REFERENCE_OPPORTUNITY_COHORT_NOT_EVALUABLE",
        )
        self.assertEqual(result["qualified_session_count"], 0)
        self.assertEqual(
            result["diagnostic_summary"][
                "reference_ground_nonzero_risk_session_count"
            ],
            0,
        )

    def test_success_must_stop_at_required_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol, ledger, plan, p_sha, l_sha, plan_sha = self._fixture(
                root
            )
            reports = [
                _write(
                    root,
                    f"report-{rank}",
                    _source(
                        rank,
                        f"session-{rank}",
                        p_sha,
                        l_sha,
                        plan_sha,
                        qualified=rank in (1, 3),
                    ),
                )
                for rank in range(1, 4)
            ]
            result = aggregate(protocol, ledger, plan, reports)
        self.assertEqual(
            result["terminal"],
            "R3_1_REFERENCE_OPPORTUNITY_COHORT_QUALIFIED",
        )
        self.assertEqual(
            [item["inventory_eligible_rank"] for item in result[
                "selected_qualified_sessions"
            ]],
            [1, 3],
        )

    def test_rejects_candidate_metric_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol, ledger, plan, p_sha, l_sha, plan_sha = self._fixture(
                root, budget=1, required=1
            )
            value = _source(
                1, "session-1", p_sha, l_sha, plan_sha, qualified=True
            )
            value["reference_only_assertions"]["candidate_grid_computed"] = True
            report = _write(root, "report", value)
            with self.assertRaisesRegex(ValueError, "reference-only"):
                aggregate(protocol, ledger, plan, [report])


if __name__ == "__main__":
    unittest.main()
