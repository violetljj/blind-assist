from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
MODULE = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE))

import evaluate_r12d_model as evaluator
import r12d_contract
import train_r12d_detector_matrix as trainer


MATRIX = ROOT / "configs/ustrf_crosscam_small_target_detector_r12d_matrix_v1.json"


class R12dControlledResearchTest(unittest.TestCase):
    def replay(self) -> dict:
        return {
            "association_iou_at_least": 0.1,
            "association_center_distance_frame_diagonal_at_most": 0.12,
            "association_area_ratio_min": 0.33,
            "association_area_ratio_max": 3.0,
            "association_ambiguity_margin": 0.05,
            "clear_after_consecutive_misses": 2,
        }

    def source(self) -> dict:
        return {
            "event_id": "e1", "source_id": "s1", "alertable_start_ms": 0,
            "primary_anchor_timestamp_ms": 0, "detector_label_allowlist": ["traffic cone"],
            "route_polygon_xy_norm": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            "target_anchors": [
                {"timestamp_ms": 0, "visibility": "visible", "bbox_xyxy_norm": [0.4, 0.4, 0.6, 0.8]},
                {"timestamp_ms": 500, "visibility": "visible", "bbox_xyxy_norm": [0.4, 0.4, 0.6, 0.8]},
            ],
        }

    def detection(self, left: float = 40.0) -> dict:
        return {"class_id": 0, "label": "traffic cone", "confidence": 0.9,
                "bbox_xyxy_px": [left, 40.0, left + 20.0, 80.0]}

    def test_live_matrix_freezes_paired_strides_and_three_seeds(self) -> None:
        matrix = r12d_contract.validate_matrix(MATRIX, ROOT)
        self.assertEqual([2026072201, 2026072202, 2026072203], matrix["training"]["seeds"])
        self.assertEqual([4, 8, 16, 32], matrix["paired_arms"][0]["expected_strides"])
        self.assertFalse(matrix["authority"]["r13_inventory_read_authorized"])

    def test_truth_blind_alert_generation_never_requires_expected_class(self) -> None:
        frames = [
            {"timestamp_ms": 0, "width": 100, "height": 100, "detections": [self.detection()]},
            {"timestamp_ms": 500, "width": 100, "height": 100, "detections": [self.detection()]},
        ]
        trace = evaluator.generate_truth_blind_trace(self.source(), frames, self.replay(), set(r12d_contract.CLASSES))
        self.assertTrue(trace["event_hit"])
        self.assertEqual(1, trace["delivered_alert_count"])
        self.assertNotIn("expected_class", trace)

    def test_unassigned_route_inside_is_pressure_not_delivered_alert(self) -> None:
        frames = [
            {"timestamp_ms": 0, "width": 100, "height": 100,
             "detections": [self.detection(), self.detection(left=5.0)]},
            {"timestamp_ms": 500, "width": 100, "height": 100,
             "detections": [self.detection(), self.detection(left=5.0)]},
        ]
        trace = evaluator.generate_truth_blind_trace(self.source(), frames, self.replay(), set(r12d_contract.CLASSES))
        self.assertEqual(1, trace["delivered_alert_count"])
        self.assertEqual(0, trace["cooccurrence_triggered_target_event_count"])
        self.assertGreater(trace["unassigned_route_inside_pressure_count"], 0)

    def test_never_activated_clearance_is_censored_not_observed(self) -> None:
        source = self.source(); source["known_not_visible_from_ms"] = 500
        frames = [
            {"timestamp_ms": 0, "width": 100, "height": 100, "detections": []},
            {"timestamp_ms": 500, "width": 100, "height": 100, "detections": []},
        ]
        trace = evaluator.generate_truth_blind_trace(source, frames, self.replay(), set(r12d_contract.CLASSES))
        self.assertFalse(trace["clearance_observable"])
        self.assertIsNone(trace["target_exit_clearance_delay_ms"])
        self.assertEqual(10_000, trace["target_exit_clearance_censored_ms"])

    def test_shared_backbone_key_parser_excludes_neck(self) -> None:
        self.assertEqual(10, trainer.layer_index("model.10.cv1.conv.weight"))
        self.assertEqual(11, trainer.layer_index("model.11.conv.weight"))
        self.assertIsNone(trainer.layer_index("ema.updates"))

    def test_model_family_image_sizes_remain_frozen(self) -> None:
        matrix = r12d_contract.validate_matrix(MATRIX, ROOT)
        self.assertEqual(640, evaluator.inference_matrix_for_model(matrix, "yolo", 640)["frozen_inference"]["image_size"])
        self.assertEqual(768, evaluator.inference_matrix_for_model(matrix, "yoloe_tflite", None)["frozen_inference"]["image_size"])
        with self.assertRaisesRegex(ValueError, "paired R1.2d arm image size"):
            evaluator.inference_matrix_for_model(matrix, "yolo", 768)


if __name__ == "__main__":
    unittest.main()
