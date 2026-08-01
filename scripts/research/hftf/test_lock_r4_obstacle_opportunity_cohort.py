from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lock_r4_obstacle_opportunity_cohort import lock


def _write(root: Path, name: str, value: object) -> Path:
    path = root / f"{name}.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class R4ObstacleOpportunityCohortLockTest(unittest.TestCase):
    def test_first_four_qualified_sources_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            protocol = _write(
                root,
                "protocol",
                {
                    "schema": (
                        "blindassist_hftf_stage_b_split_source_validation_r4"
                    ),
                    "status": "FROZEN_BEFORE_R4_OUTCOME",
                    "workflow_profile": "DEVELOPMENT_STANDARD",
                    "obstacle_source_role": {
                        "required_qualified_sessions": 4,
                        "maximum_inventory_eligible_sessions_to_screen": 12,
                    },
                },
            )
            ledger = _write(
                root,
                "ledger",
                {
                    "schema": (
                        "blindassist_hftf_r4_source_pool_burn_ledger"
                    ),
                    "status": "FROZEN_BEFORE_R4_OUTCOME",
                },
            )
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            plan = _write(
                root,
                "plan",
                {
                    "schema": (
                        "blindassist_hftf_r4_obstacle_inventory_candidate_plan"
                    ),
                    "terminal": (
                        "R4_OBSTACLE_INVENTORY_CANDIDATE_PLAN_READY"
                    ),
                    "protocol_sha256": sha(protocol),
                    "burn_ledger_sha256": sha(ledger),
                    "inventory_candidates": [
                        {
                            "inventory_eligible_rank": rank,
                            "session_id": f"session-{rank}",
                        }
                        for rank in range(1, 13)
                    ],
                    "reference_outcome_read": False,
                    "ground_outcome_read": False,
                    "candidate_outcome_read": False,
                    "baseline_outcome_read": False,
                },
            )
            authority = _write(root, "authority", {"ok": True})
            reports = []
            for rank in range(1, 5):
                reports.append(
                    _write(
                        root,
                        f"report-{rank}",
                        {
                            "schema": (
                                "blindassist_hftf_r4_obstacle_reference_"
                                "opportunity_source_result"
                            ),
                            "inventory_eligible_rank": rank,
                            "source_session_id": f"session-{rank}",
                            "protocol_sha256": sha(protocol),
                            "burn_ledger_sha256": sha(ledger),
                            "inventory_plan_sha256": sha(plan),
                            "qualified": True,
                            "terminal": (
                                "R4_SOURCE_OBSTACLE_REFERENCE_"
                                "OPPORTUNITY_QUALIFIED"
                            ),
                            "reference_only_assertions": {
                                "obstacle_reference_grid_computed": True,
                                "ground_reference_computed": False,
                                "candidate_grid_computed": False,
                                "angular_baseline_computed": False,
                                "arm_metric_or_delta_computed": False,
                            },
                            "arm_outcome_authorized": False,
                            "authority_report_path": str(authority),
                            "authority_report_sha256": sha(authority),
                            "manifest_sha256": str(rank) * 64,
                            "dataset_spec_sha256": "a" * 64,
                            "camera_poses_sha256": "b" * 64,
                        },
                    )
                )
            result = lock(protocol, ledger, plan, reports)
        self.assertEqual(
            "R4_OBSTACLE_OPPORTUNITY_COHORT_QUALIFIED",
            result["terminal"],
        )
        self.assertEqual(4, len(result["required_sessions"]))
        self.assertTrue(result["formal_arm_outcome_authorized"])

    def test_arm_firewall_leakage_is_rejected(self) -> None:
        # Directly exercise the public validator through a minimally malformed
        # report set by reusing the successful fixture logic above is verbose;
        # the exact firewall dictionary is covered by the lock's success path.
        from lock_r4_obstacle_opportunity_cohort import _validate_firewall

        with self.assertRaisesRegex(ValueError, "arm firewall"):
            _validate_firewall(
                {
                    "reference_only_assertions": {
                        "candidate_grid_computed": True
                    },
                    "arm_outcome_authorized": False,
                }
            )


if __name__ == "__main__":
    unittest.main()
