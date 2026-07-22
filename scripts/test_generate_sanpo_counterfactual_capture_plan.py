from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("capture_plan", SCRIPTS / "generate_sanpo_counterfactual_capture_plan.py")
assert SPEC and SPEC.loader
capture_plan = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture_plan)


class CounterfactualCapturePlanTest(unittest.TestCase):
    def test_generates_empty_pair_slots_with_lifecycle_templates(self) -> None:
        config = {
            "schema": "blindassist_sanpo_counterfactual_episode_collection_v1",
            "design": {"matched_pairs_per_session_scene": 1},
            "episode_duration_policy": {"minimum_duration_ms": 10000, "maximum_duration_ms": 20000},
            "sessions": [{"session_id": "s1"}],
            "scenes": [{"scene_id": "step_curb", "positive_contract": "enters corridor", "matched_negative_contract": "outside corridor"}],
            "matrix_contract": {"matched_pair_members_must_share_capture_context": ["location", "device"]},
        }
        plan = capture_plan.build_capture_plan(config)
        self.assertEqual(2, plan["episode_slot_count"])
        self.assertFalse(plan["training_eligible"])
        positive, negative = plan["slots"]
        self.assertEqual("awaiting_autonomous_acquisition", positive["status"])
        self.assertFalse(positive["human_operator_required"])
        self.assertTrue(positive["human_fallback_forbidden"])
        self.assertEqual("approach_alertable_clear", positive["risk_profile_template"]["lifecycle"])
        self.assertEqual("no_alert", negative["risk_profile_template"]["lifecycle"])

    def test_route_conditioned_pilot_is_exact_and_non_evidentiary(self) -> None:
        config = {
            "schema": "blindassist_sanpo_counterfactual_episode_collection_v1",
            "contract_id": "route-truth-v1",
            "design": {
                "matched_pairs_per_session_scene": 2,
                "pilot_before_full_matrix": {
                    "contract_id": "route-pilot-v1",
                    "origin_scope": "pipeline_audit_pilot_capture",
                    "session_count": 1,
                    "matched_pairs_per_scene": 1,
                    "episode_count": 4,
                    "authority": "collection-pipeline-audit-only",
                },
            },
            "episode_duration_policy": {"minimum_duration_ms": 10000, "maximum_duration_ms": 20000},
            "sessions": [{"session_id": "s1"}, {"session_id": "s2"}],
            "scenes": [
                {"scene_id": "step_curb", "positive_contract": "enters route", "matched_negative_contract": "outside route"},
                {"scene_id": "route_obstacle", "positive_contract": "blocks route", "matched_negative_contract": "clear route"},
            ],
            "matrix_contract": {"matched_pair_members_must_share_capture_context": ["location", "device"]},
        }

        plan = capture_plan.build_capture_plan(config, pilot=True)

        self.assertEqual("pilot_autonomous_acquisition_plan_only", plan["status"])
        self.assertEqual("route-pilot-v1", plan["contract_id"])
        self.assertEqual("route-truth-v1", plan["source_truth_contract_id"])
        self.assertEqual({"pipeline_audit_pilot_capture"}, {row["origin_scope"] for row in plan["slots"]})
        self.assertEqual(4, plan["episode_slot_count"])
        self.assertEqual(2, plan["matched_pair_slot_count"])
        self.assertEqual({"s1"}, {row["session_id"] for row in plan["slots"]})
        self.assertFalse(plan["route_conditioned_truth_eligible"])
        self.assertFalse(plan["u0_evaluation_eligible"])
        self.assertIn("capture_frame_ledger", plan["slots"][0]["evidence_requirements"])


if __name__ == "__main__":
    unittest.main()
