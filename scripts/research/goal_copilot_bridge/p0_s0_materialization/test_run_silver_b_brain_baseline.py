from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p0_grounding import p0_evaluator
from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_silver_b_brain_baseline as baseline
from scripts.research.goal_copilot_bridge.p0_s0_materialization import silver_b_brain_cohort
from scripts.research.goal_copilot_bridge.p0_s0_materialization.test_silver_b_brain_cohort import annotation, source_inputs


class SilverBBrainBaselineAdapterTest(unittest.TestCase):
    def _episode(self, resolution: str) -> dict:
        silver, metadata, receipt = source_inputs()
        reviewed = annotation("silver-b-f1", resolution)
        cohort = silver_b_brain_cohort.build_brain_cohort(
            silver,
            metadata,
            receipt,
            {
                "schema_version": 1,
                "annotation_authority": "INDEPENDENT_FRAME_VISUAL_REVIEW_SILVER_B",
                "episodes": [reviewed],
            },
        )
        return cohort["episodes"][0]

    def test_selected_candidate_translates_to_frozen_valid_output(self) -> None:
        episode = self._episode("UNIQUE")
        candidate_id = episode["candidates"][0]["candidate_id"]
        raw = {
            "episode_id": episode["episode_id"],
            "action": "SELECT",
            "selected_candidate_ids": [candidate_id],
            "confidence": 0.8,
            "rationale": "The candidate matches the goal.",
        }
        output = baseline._frozen_output(episode, raw)
        p0_evaluator.validate_output(output, episode["evaluator_episode"])
        result = p0_evaluator.evaluate_episode(episode["evaluator_episode"], output)
        self.assertEqual("CORRECT_GROUNDING", result["end_to_end"]["outcome"])

    def test_ambiguous_action_preserves_fail_closed_semantics(self) -> None:
        episode = self._episode("AMBIGUOUS")
        raw = {
            "episode_id": episode["episode_id"],
            "action": "AMBIGUOUS",
            "selected_candidate_ids": [],
            "confidence": 0.7,
            "rationale": "The goal does not establish a referent.",
        }
        output = baseline._frozen_output(episode, raw)
        result = p0_evaluator.evaluate_episode(episode["evaluator_episode"], output)
        self.assertTrue(result["valid_system_output"])
        self.assertEqual("CORRECT_AMBIGUITY", result["end_to_end"]["outcome"])

    def test_model_prompt_uses_opaque_case_id_and_remaps_only_after_response(self) -> None:
        episode = self._episode("UNIQUE")
        prompt = baseline._prompt([("case-001", episode)])
        self.assertIn("case-001", prompt)
        self.assertNotIn(episode["episode_id"], prompt)
        candidate_id = episode["candidates"][0]["candidate_id"]
        decisions = baseline._validate_raw(
            {
                "decisions": [{
                    "episode_id": "case-001",
                    "action": "SELECT",
                    "selected_candidate_ids": [candidate_id],
                    "confidence": 0.8,
                    "rationale": "Visible entrance.",
                }]
            },
            [("case-001", episode)],
        )
        self.assertEqual("case-001", decisions[0]["model_case_id"])
        self.assertEqual(episode["episode_id"], decisions[0]["episode_id"])


if __name__ == "__main__":
    unittest.main()
