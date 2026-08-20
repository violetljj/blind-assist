from __future__ import annotations

import json
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "protocol.json"


class GoalCopilot2BProtocolTest(unittest.TestCase):
    def test_authority_and_execution_remain_fail_closed(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        self.assertEqual("BLINDASSIST_ONLY", protocol["authorities"]["task_semantics_hidden_state_noise_engine_evaluator_safety_acceptance"])
        self.assertEqual("SKYDISCOVER_ONLY", protocol["authorities"]["candidate_proposal_and_search"])
        self.assertFalse(protocol["currently_authorized"]["model_calls"])
        self.assertFalse(protocol["currently_authorized"]["sky_or_evox_search"])
        self.assertFalse(protocol["currently_authorized"]["bundle_or_heldout_materialization"])

    def test_budget_and_candidate_surface_are_frozen(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        budget = protocol["search_budget"]
        self.assertEqual(2, len(budget["replicates"]))
        self.assertEqual(16, budget["generation_attempts_per_replicate"])
        self.assertEqual(32, budget["generation_attempts_total"])
        self.assertEqual(0, budget["generation_retries"])
        self.assertEqual(6, len(protocol["candidate_surface"]["functions"]))
        self.assertFalse(protocol["candidate_surface"]["evaluator_or_noise_changes_by_sky"])

    def test_heldout_role_is_not_fresh_and_is_blind(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        heldout = protocol["heldout_validation"]
        self.assertIn("NOT_FRESH_TASKS", heldout["evidence_role"])
        self.assertFalse(heldout["candidate_access"])
        self.assertFalse(heldout["sky_access"])
        self.assertTrue(heldout["winner_lock_before_open"])

    def test_predecessor_identity_is_frozen(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        predecessor = protocol["predecessor"]
        self.assertEqual("GOAL_COPILOT_2A_COMPLETE", predecessor["required_status"])
        self.assertEqual(64, len(predecessor["closeout_sha256"]))
        self.assertTrue(predecessor["required_admission"])
        self.assertEqual(
            "GOAL-COPILOT-2B_NOISE_ROBUST_SKY_SEARCH_PROTOCOL_DESIGN",
            predecessor["required_route"],
        )


if __name__ == "__main__":
    unittest.main()
