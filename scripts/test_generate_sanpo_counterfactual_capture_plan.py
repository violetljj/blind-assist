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
        self.assertEqual("not_captured", positive["status"])
        self.assertEqual("approach_alertable_clear", positive["risk_profile_template"]["lifecycle"])
        self.assertEqual("no_alert", negative["risk_profile_template"]["lifecycle"])


if __name__ == "__main__":
    unittest.main()
