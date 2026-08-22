from __future__ import annotations

import unittest

import numpy as np

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import SUNRGBD_PROTOCOL_ID, _config, _confusion, decision_for, region_depth_median
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_sunrgbd_door_depth import _door_targets


DECISION = {
    "baseline_bbox_height_threshold": 0.55,
    "depth_gate_requires_predicted_region_depth_lte_m": 2.0,
}


class CompletionNearnessTest(unittest.TestCase):
    def test_sunrgbd_provider_score_contract_is_accepted(self) -> None:
        manifest = {
            "schema_version": "blindassist_completion_nearness_experiment_manifest_v1",
            "protocol_id": SUNRGBD_PROTOCOL_ID,
            "created_before_dataset_payload_access": True,
            "created_before_private_truth_access": True,
            "threshold_model_prompt_or_pool_sweep": False,
            "retry_or_replay_authorized": False,
            "frozen_provider": {
                "proposal": "YOLOE-26n-seg text prompt door",
                "metric_depth": "Depth Anything V2 metric Hypersim ViT-S ONNX",
                "metric_depth_input_shape": [1, 3, 518, 686],
                "bounded_pool_size": 10,
                "selection_rule": "PROVIDER_SCORE_TOP1",
                "selected_region_aggregation": "median of finite positive depth in bbox inset 20 percent per side",
            },
            "frozen_decision": {"interaction_range_m": 2.0, "baseline_bbox_height_threshold": 0.55, "target_hit_iou_threshold": 0.30},
        }
        provider, _ = _config(manifest)
        self.assertEqual("PROVIDER_SCORE_TOP1", provider["selection_rule"])

    def test_set_valued_exact_door_regions_keep_independent_depth(self) -> None:
        labels = np.zeros((40, 40), dtype=np.uint8)
        labels[0:20, :] = 1
        labels[20:40, :] = 2
        depth = np.full((40, 40), 3.0, dtype=np.float32)
        depth[0:20, :] = 1.25
        rule = {"minimum_connected_region_pixels": 800, "minimum_valid_depth_fraction_in_region": 0.5, "valid_depth_range_m": [0.4, 8.0]}
        targets = _door_targets(labels, np.asarray(["door", "door"], dtype=object), depth, rule)
        self.assertEqual(2, len(targets))
        self.assertAlmostEqual(1.25, targets[0]["target_depth_median_m"])
        self.assertAlmostEqual(3.0, targets[1]["target_depth_median_m"])

    def test_depth_gate_is_independent_of_bbox_height(self) -> None:
        short_centered = {"bbox_xyxy": [250, 200, 390, 450]}
        self.assertEqual((False, True), decision_for(short_centered, 1.5, 640, 480, DECISION))
        tall_centered = {"bbox_xyxy": [100, 10, 540, 470]}
        self.assertEqual((True, False), decision_for(tall_centered, 3.0, 640, 480, DECISION))

    def test_region_depth_uses_inset_median(self) -> None:
        depth = np.full((10, 10), 4.0, dtype=np.float32)
        depth[2:8, 2:8] = 1.25
        self.assertAlmostEqual(1.25, region_depth_median(depth, [0, 0, 100, 100], 100, 100))

    def test_confusion_counts_false_completion(self) -> None:
        rows = [
            {"truth_positive": True, "decision": True},
            {"truth_positive": False, "decision": True},
            {"truth_positive": True, "decision": False},
            {"truth_positive": False, "decision": False},
        ]
        result = _confusion(rows, "decision")
        self.assertEqual({"tp": 1, "fp": 1, "fn": 1, "tn": 1, "precision": 0.5, "recall": 0.5}, result)


if __name__ == "__main__":
    unittest.main()
