from __future__ import annotations

import json
from pathlib import Path
import unittest

from .annotation import make_annotation
from .baseline import run_baseline
from .evaluate import evaluate
from .public_real_mining import mine_prospective


def public_manifest():
    return {
        "schema_version": "blindassist_real_episode_public_manifest_v0",
        "episodes": [{
            "episode_id": "ep-001",
            "goal_contract": {"goal_contract": {"reference_mode": "UNIQUE"}},
            "observations": [
                {"observation_id": "o1", "timestamp_ms": 0},
                {"observation_id": "o2", "timestamp_ms": 250},
            ],
        }],
    }


def config():
    return json.loads((Path(__file__).parent / "baseline_config.json").read_text(encoding="utf-8"))


class RealEpisodePilotTest(unittest.TestCase):
    def test_annotation_starts_truth_unknown(self):
        annotation = make_annotation(public_manifest())
        self.assertTrue(annotation["private_evaluator_only"])
        self.assertEqual("UNKNOWN", annotation["episodes"][0]["observations"][0]["target_visibility"])

    def test_baseline_contests_without_selection_authority_and_handoffs_near(self):
        provider = {"observations": [
            {"observation_id": "o1", "candidate_cardinality": "SET_VALUED", "selection_authorized": False, "candidates": [
                {"candidate_id": "a", "rank": 1, "x_center_fraction": 0.2, "range_m": 3.0}
            ]},
            {"observation_id": "o2", "candidate_cardinality": "UNIQUE", "selection_authorized": True, "candidates": [
                {"candidate_id": "b", "rank": 1, "x_center_fraction": 0.5, "range_m": 1.2}
            ]},
        ]}
        result = run_baseline(public_manifest(), provider, config())
        self.assertEqual("CONTESTED", result["predictions"][0]["decision_state"])
        self.assertEqual("HANDOFF_READY", result["predictions"][1]["decision_state"])
        self.assertFalse(result["identity_persistence"])

    def test_evaluator_keeps_conditioned_denominators_and_attribution(self):
        annotation = make_annotation(public_manifest())
        episode = annotation["episodes"][0]
        episode["observations"][0].update({"target_visibility": "VISIBLE", "legal_candidate_ids": ["a"], "allowed_decision_states": ["FOUND"], "range_truth": "RANGE_FAR"})
        episode["observations"][1].update({"target_visibility": "NOT_VISIBLE", "allowed_decision_states": ["LOST"]})
        prediction = {
            "schema_version": "blindassist_real_episode_baseline_prediction_v0",
            "predictions": [
                {"observation_id": "o1", "candidate_ids": ["a"], "selected_referent": "a", "decision_state": "FOUND", "command": "GUIDE_LEFT", "range_bucket": "RANGE_FAR", "confident_spoken_guidance": True},
                {"observation_id": "o2", "candidate_ids": [], "selected_referent": None, "decision_state": "LOST", "command": None, "range_bucket": "RANGE_UNKNOWN", "confident_spoken_guidance": False},
            ],
        }
        result = evaluate(annotation, prediction)
        recall = result["observation_metrics"]["proposal_recall_at_k_given_visible"]["1"]
        self.assertEqual(1, recall["eligible"])
        self.assertEqual(1.0, recall["rate"])
        self.assertEqual(2, result["observation_metrics"]["unconditional_target_visibility"]["eligible"])

    def test_unknown_truth_never_becomes_wrong_referent(self):
        annotation = make_annotation(public_manifest())
        prediction = {
            "schema_version": "blindassist_real_episode_baseline_prediction_v0",
            "predictions": [
                {"observation_id": "o1", "candidate_ids": ["a"], "selected_referent": "a", "decision_state": "FOUND", "command": "GUIDE_LEFT", "range_bucket": "RANGE_UNKNOWN", "confident_spoken_guidance": True},
                {"observation_id": "o2", "candidate_ids": [], "selected_referent": None, "decision_state": "ABSTAIN", "command": None, "range_bucket": "RANGE_UNKNOWN", "confident_spoken_guidance": False},
            ],
        }
        result = evaluate(annotation, prediction)
        metric = result["observation_metrics"]["confident_wrong_referent_guidance"]
        self.assertEqual(0, metric["eligible"])
        self.assertEqual(1, metric["unknown"])
        self.assertIsNone(metric["rate"])

    def test_prospective_miner_enforces_precedence_and_keeps_set_valued_goal(self):
        roster = {
            "schema_version": "blindassist_public_goal_roster_v0",
            "precedence": {
                "mapillary_metadata_accessed": False,
                "mapillary_pixels_accessed": False,
                "model_outputs_created": False,
                "evaluator_truth_created": False,
            },
            "goals": [{
                "goal_id": "g1", "goal_type": "NAMED_BUILDING_ENTRANCE", "target_name": "venue",
                "public_entrance_candidates": [
                    {"candidate_id": "a", "coordinates": [0.0, 0.0]},
                    {"candidate_id": "b", "coordinates": [0.0, 0.0]},
                ],
            }],
        }
        metadata = {
            "schema_version": "blindassist_mapillary_sequence_metadata_v0",
            "images": [
                {"image_id": "i1", "sequence_id": "s", "captured_at_ms": 0, "coordinates": [-0.00027, 0.0], "computed_compass_angle": 90.0},
                {"image_id": "i2", "sequence_id": "s", "captured_at_ms": 1000, "coordinates": [-0.00018, 0.0], "computed_compass_angle": 90.0},
                {"image_id": "i3", "sequence_id": "s", "captured_at_ms": 2000, "coordinates": [-0.00009, 0.0], "computed_compass_angle": 90.0},
            ],
        }
        result = mine_prospective(roster, metadata)
        self.assertEqual(1, result["episode_count"])
        self.assertEqual("SET_VALUED", result["episodes"][0]["goal_contract"]["goal_contract"]["reference_mode"])
        self.assertTrue(result["goal_before_mapillary_metadata_and_pixels"])

    def test_prospective_miner_rejects_post_pixel_goal_roster(self):
        roster = {
            "schema_version": "blindassist_public_goal_roster_v0",
            "precedence": {
                "mapillary_metadata_accessed": True,
                "mapillary_pixels_accessed": False,
                "model_outputs_created": False,
                "evaluator_truth_created": False,
            },
            "goals": [],
        }
        with self.assertRaises(ValueError):
            mine_prospective(roster, {"schema_version": "blindassist_mapillary_sequence_metadata_v0", "images": []})

if __name__ == "__main__":
    unittest.main()
