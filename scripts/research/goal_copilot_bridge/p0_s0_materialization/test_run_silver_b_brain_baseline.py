from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

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

    def test_calibration_prompt_separates_place_identity_from_entrance_relation(self) -> None:
        episode = self._episode("AMBIGUOUS")
        prompt = baseline._prompt([("case-001", episode)], baseline.CALIBRATION_POLICY_ID)
        self.assertIn("place-identity evidence", prompt)
        self.assertIn("entrance-relation evidence", prompt)
        self.assertIn("cannot by itself establish entrance relation", prompt)
        self.assertNotIn(episode["episode_id"], prompt)

    def test_two_level_prompt_allows_direct_or_cumulative_evidence(self) -> None:
        episode = self._episode("UNIQUE")
        prompt = baseline._prompt([("case-001", episode)], baseline.CALIBRATION_POLICY_V2_ID)
        self.assertIn("substitutable and cumulative, not a checklist", prompt)
        self.assertIn("one direct targeted cue", prompt)
        self.assertIn("multiple independent medium cues", prompt)
        schema = baseline._schema(baseline.CALIBRATION_POLICY_V2_ID)
        required = schema["properties"]["decisions"]["items"]["required"]
        self.assertIn("place_support", required)
        self.assertIn("entrance_relation_support", required)

    def test_two_level_policy_rejects_select_without_strong_relation_support(self) -> None:
        episode = self._episode("UNIQUE")
        candidate_id = episode["candidates"][0]["candidate_id"]
        raw = {"decisions": [{
            "episode_id": "case-001",
            "action": "SELECT",
            "selected_candidate_ids": [candidate_id],
            "confidence": 0.8,
            "rationale": "Place is visible but relation is weak.",
            "place_support": "STRONG",
            "entrance_relation_support": "WEAK",
        }]}
        with self.assertRaisesRegex(baseline.BrainRunError, "SELECT requires strong"):
            baseline._validate_raw(raw, [("case-001", episode)], baseline.CALIBRATION_POLICY_V2_ID)

    def test_candidate_zoom_representation_preserves_scene_and_adds_tiles(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (320, 180), (120, 140, 160)).save(source)
            episode = {
                "goal_text": "the left door",
                "image_path": str(source),
                "candidates": [
                    {
                        "candidate_id": "candidate-1",
                        "category_label": "door",
                        "provider_rank": 1,
                        "region": {
                            "frame_id": "frame-1",
                            "x_min": 0.05,
                            "y_min": 0.1,
                            "x_max": 0.25,
                            "y_max": 0.8,
                        },
                    },
                    {
                        "candidate_id": "candidate-2",
                        "category_label": "door",
                        "provider_rank": 2,
                        "region": {
                            "frame_id": "frame-1",
                            "x_min": 0.65,
                            "y_min": 0.1,
                            "x_max": 0.9,
                            "y_max": 0.8,
                        },
                    },
                ],
                "evaluator_episode": {"observation_window": {"frame_ids": ["frame-1"]}},
            }
            output = root / "zoom.jpg"
            baseline._render_input_with_candidate_zoom(episode, "case-001", output)
            with Image.open(output) as rendered:
                self.assertGreaterEqual(rendered.width, 960)
                self.assertGreater(rendered.height, 138 + 180)
            prompt = baseline._prompt_with_candidate_zoom([("case-001", episode)])
            self.assertIn(baseline.CANDIDATE_ZOOM_REPRESENTATION_ID, prompt)
            self.assertIn("full scene for spatial and relational context", prompt)


if __name__ == "__main__":
    unittest.main()
