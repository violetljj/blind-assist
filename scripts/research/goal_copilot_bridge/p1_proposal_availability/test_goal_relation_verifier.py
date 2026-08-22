from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p1_proposal_availability.evaluate_goal_relation_verifier import evaluate
from scripts.research.goal_copilot_bridge.p1_proposal_availability.run_goal_relation_verifier import rerank


class GoalRelationVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.public = {
            "schema_version": "blindassist_p1_pa3_public_input_v1",
            "cases": [{"case_id": "c1", "goal_contract": {"goal_type": "LEFTMOST_BUILDING_ENTRANCE", "reference_mode": "UNIQUE"}}],
        }
        self.prediction = {
            "schema_version": "blindassist_p1_pa3_prediction_v1",
            "public_input_sha256": "public",
            "cases": [{"case_id": "c1", "candidates": [
                {"rank": 1, "bbox_xyxy": [50, 0, 70, 20], "proposal_score": 0.9},
                {"rank": 2, "bbox_xyxy": [10, 0, 30, 20], "proposal_score": 0.8},
                {"rank": 3, "bbox_xyxy": [80, 0, 90, 10], "proposal_score": 0.1},
            ]}],
        }
        self.manifest = {
            "schema_version": "blindassist_p1_goal_relation_verifier_manifest_v1",
            "protocol_id": "test",
            "goal_type": "LEFTMOST_BUILDING_ENTRANCE",
            "reference_mode": "UNIQUE",
            "verifier": {"kind": "DETERMINISTIC_PUBLIC_GOAL_RELATION", "private_truth_access": False, "threshold_sweep": False, "configuration_sweep": False},
            "claim_ceiling": "test only",
        }

    def test_leftmost_rerank_improves_unique_target(self) -> None:
        relation = rerank(self.public, self.prediction, self.manifest, public_sha256="public", prediction_sha256="prediction", manifest_sha256="manifest")
        self.assertEqual(2, relation["cases"][0]["candidates"][0]["original_rank"])
        private = {
            "schema_version": "blindassist_p1_pa3_private_eval_v1",
            "public_input_sha256": "public",
            "cases": [{
                "case_id": "c1",
                "target_visibility": "VISIBLE",
                "reference_mode": "UNIQUE",
                "legal_target_bboxes_xyxy": [[10, 0, 30, 20]],
                "same_class_distractor_bboxes_xyxy": [[50, 0, 70, 20]],
            }],
        }
        result = evaluate(self.public, private, self.prediction, relation, self.manifest, public_sha256="public", private_sha256="private", prediction_sha256="prediction", relation_sha256="relation", manifest_sha256="manifest")
        self.assertEqual(0.0, result["baseline_top1_accuracy"])
        self.assertEqual(1.0, result["relation_top1_accuracy"])
        self.assertEqual(1, result["contrastive_abc_case_count"])

    def test_rejects_prediction_public_mismatch(self) -> None:
        self.prediction["public_input_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "prediction/public binding mismatch"):
            rerank(self.public, self.prediction, self.manifest, public_sha256="public", prediction_sha256="prediction", manifest_sha256="manifest")


if __name__ == "__main__":
    unittest.main()
