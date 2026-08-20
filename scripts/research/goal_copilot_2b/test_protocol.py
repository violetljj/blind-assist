from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.research.goal_copilot_2b.export_bundle import export_bundle, verify_bundle
from scripts.research.goal_copilot_2b.heldout_crypto import create_once, unseal
from scripts.research.goal_copilot_2b.search_evaluator import evaluate, evaluate_matrix

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

    def test_frozen_winner_search_calibration(self) -> None:
        winner = HERE.parent / "goal_copilot_2a" / "frozen_gc1_winner.py"
        matrix = evaluate_matrix(winner)
        self.assertEqual(12, matrix["CLEAN"]["metrics"]["completion_count"])
        self.assertEqual(0, matrix["COMBINED_MODERATE"]["metrics"]["completion_count"])
        result = evaluate(str(winner))
        self.assertEqual(0, result["validity"])
        self.assertEqual(0, result["combined_moderate_completion_count"])

    def test_bundle_is_integral_and_contains_no_heldout_material(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bundle = export_bundle(Path(temporary))
            manifest = verify_bundle(bundle)
            self.assertFalse(manifest["heldout_material_exported"])
            self.assertEqual(9, len(json.loads((bundle / "protocol.json").read_text())["search_development_conditions"]))

    def test_heldout_encryption_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            envelope_path, manifest_path, key_path = create_once(Path(temporary) / "heldout")
            payload = unseal(
                json.loads(envelope_path.read_text(encoding="utf-8")),
                key_path.read_text(encoding="ascii").strip(),
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual([8627, 11939], manifest["schedule_seeds"])
            self.assertEqual(2, len(payload["schedules"]))
            self.assertTrue(manifest["not_fresh_task_evidence"])


if __name__ == "__main__":
    unittest.main()
