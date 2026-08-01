"""Unit tests for the evidence-bound instance correspondence core."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.research.dual_loop_segmentation_instance_correspondence.correspondence import (
    ABSTAIN,
    MATCH,
    NO_MATCH,
    CorrespondenceThresholds,
    annotate_frame,
    bbox_iou,
    class_compatibility,
    depth_consistency,
    mask_box_metrics,
    score_pair,
    warp_box,
    warp_mask,
)


class GeometryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mask = np.zeros((32, 32), dtype=bool)
        self.mask[10:20, 10:20] = True
        self.component = {
            "component_id": "component-0",
            "semantic_class": "obstacle",
            "mask": self.mask,
            "temporal_track_id": "component-track-0",
        }
        self.detection = {
            "detection_id": "detection-0",
            "label": "person",
            "bbox_xyxy": [9, 9, 21, 21],
            "temporal_track_id": "detection-track-0",
        }

    def test_mask_box_metrics_and_bbox_iou_are_deterministic(self) -> None:
        metrics = mask_box_metrics(self.mask, self.detection["bbox_xyxy"])
        self.assertAlmostEqual(metrics["component_coverage"], 1.0)
        self.assertGreater(metrics["mask_iou"], 0.6)
        self.assertGreater(bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]), 0.1)

    def test_affine_helpers_propagate_analysis_geometry(self) -> None:
        matrix = [[1, 0, 2], [0, 1, 1]]
        warped = warp_mask(self.mask, matrix)
        self.assertTrue(warped[11, 12])
        self.assertEqual(warp_box([9, 9, 21, 21], matrix), (11.0, 10.0, 23.0, 22.0))


class EvidenceStateTest(unittest.TestCase):
    def setUp(self) -> None:
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 10:20] = True
        self.component = {
            "component_id": "component-0",
            "semantic_class": "obstacle",
            "mask": mask,
            "temporal_track_id": "component-track-0",
        }
        self.matching_detection = {
            "detection_id": "detection-0",
            "label": "person",
            "bbox_xyxy": [9, 9, 21, 21],
            "temporal_track_id": "detection-track-0",
        }
        self.mapping = {
            "yolo_label_to_semantic": {"person": "obstacle", "curb": "boundary_step_curb"},
            "segmentation_aliases": {
                "obstacle": ["person", "obstacle"],
                "boundary_step_curb": ["curb"],
            },
        }

    def test_match_uses_class_and_geometry(self) -> None:
        row = score_pair(self.component, self.matching_detection, class_mapping=self.mapping)
        self.assertEqual(row["state"], MATCH)
        self.assertIn("depth_consistency", row["missing_evidence"])
        self.assertIn("optical_flow", row["missing_evidence"])

    def test_incompatible_class_is_no_match(self) -> None:
        detection = dict(self.matching_detection, label="curb")
        row = score_pair(self.component, detection, class_mapping=self.mapping)
        self.assertEqual(row["state"], NO_MATCH)
        self.assertEqual(row["state_reason"], "CLASS_INCOMPATIBLE")

    def test_missing_class_and_weak_geometry_abstain(self) -> None:
        detection = {
            "detection_id": "detection-unknown",
            "label": "unknown-object",
            "bbox_xyxy": [0, 0, 3, 3],
        }
        row = score_pair(self.component, detection, class_mapping=self.mapping)
        self.assertEqual(row["state"], ABSTAIN)

    def test_depth_and_flow_evidence_are_reported(self) -> None:
        row = score_pair(
            self.component,
            self.matching_detection,
            class_mapping=self.mapping,
            component_depth={"depth_cluster_id": "c1", "median_depth": 2.0},
            detection_depth={"depth_cluster_id": "c1", "median_depth": 2.1},
            temporal_continuity=1.0,
            flow_evidence={"component_iou": 0.8, "detection_iou": 0.9},
        )
        self.assertEqual(row["state"], MATCH)
        self.assertEqual(row["depth_consistency"]["state"], "CONSISTENT")
        self.assertNotIn("depth_consistency", row["missing_evidence"])
        self.assertNotIn("optical_flow", row["missing_evidence"])

    def test_depth_conflict_with_weak_geometry_is_no_match(self) -> None:
        detection = dict(self.matching_detection, bbox_xyxy=[0, 0, 4, 4])
        row = score_pair(
            self.component,
            detection,
            class_mapping=self.mapping,
            component_depth={"depth_cluster_id": "c1", "median_depth": 1.0},
            detection_depth={"depth_cluster_id": "c2", "median_depth": 4.0},
        )
        self.assertEqual(row["state"], NO_MATCH)

    def test_depth_unknown_is_not_negative(self) -> None:
        result = depth_consistency({"depth_cluster_id": "c1"}, None)
        self.assertEqual(result["state"], "UNKNOWN")
        self.assertIsNone(result["score"])


class AssignmentTest(unittest.TestCase):
    def _component(self, component_id: str, x: int) -> dict[str, object]:
        mask = np.zeros((24, 24), dtype=bool)
        mask[8:16, x : x + 8] = True
        return {"component_id": component_id, "semantic_class": "obstacle", "mask": mask}

    def test_one_to_one_conflict_abstains_when_scores_tie(self) -> None:
        components = [self._component("c0", 8), self._component("c1", 8)]
        detections = [{"detection_id": "d0", "label": "person", "bbox_xyxy": [7, 7, 17, 17]}]
        result = annotate_frame(
            components,
            detections,
            class_mapping={"yolo_label_to_semantic": {"person": "obstacle"}},
        )
        self.assertEqual(len(result["component_rows"]), 2)
        self.assertTrue(all(row["state"] == ABSTAIN for row in result["component_rows"]))
        self.assertTrue(all(row["state"] == ABSTAIN for row in result["pair_rows"]))

    def test_class_compatibility_reports_unknown_without_guessing(self) -> None:
        result = class_compatibility("obstacle", {"label": "mystery"}, {})
        self.assertEqual(result["state"], "UNKNOWN")
        self.assertIsNone(result["score"])


if __name__ == "__main__":
    unittest.main()
