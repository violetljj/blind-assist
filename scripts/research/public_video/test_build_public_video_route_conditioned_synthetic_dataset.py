#!/usr/bin/env python3

import unittest
from pathlib import Path

import build_public_video_route_conditioned_synthetic_dataset as subject


class RouteConditionedSyntheticDatasetTest(unittest.TestCase):
    def test_overlap_fraction_uses_normalized_route_points(self) -> None:
        points = [[0.2, 0.8], [0.5, 0.8], [0.8, 0.8]]
        self.assertEqual(1 / 3, subject.point_bbox_overlap_fraction(points, [40, 70, 60, 90], 100, 100))

    def test_overlap_excludes_right_and_bottom_edges(self) -> None:
        points = [[0.6, 0.8], [0.5, 0.9], [0.5, 0.8]]
        self.assertEqual(1 / 3, subject.point_bbox_overlap_fraction(points, [40, 70, 60, 90], 100, 100))

    def test_lifecycle_requires_consecutive_states(self) -> None:
        self.assertTrue(subject.lifecycle_open([False, True, True], 2))
        self.assertFalse(subject.lifecycle_open([True, False, True], 2))

    def test_route_waypoints_preserve_template_alignment(self) -> None:
        template = {"fixed_templates": {"y_norm": [0.9, 0.8, 0.7], "LEFT_x_norm": [0.4, 0.3, 0.2]}}
        self.assertEqual([[0.4, 0.9], [0.3, 0.8], [0.2, 0.7]], subject.route_waypoints(template, "LEFT"))

    def test_independent_direction_path_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            subject.reject_independent_direction(Path("artifacts.local/secondary-corridor-causal/out"))

    def test_original_contract_keeps_legacy_class_only_for_audit(self) -> None:
        contract, annotation_class = subject.resolve_contract(Path("configs/public_video_route_conditioned_synthetic_contract_r812.json").resolve())
        self.assertEqual("r812_train_only_route_conditioned_obstruction_triplets_v1", contract["contract_id"])
        self.assertEqual("static_obstacle", annotation_class)

    def test_factorial_contract_expands_both_assets_without_changing_class(self) -> None:
        contract, annotation_class = subject.resolve_contract(Path("configs/public_video_route_conditioned_synthetic_factorial_contract_r814.json").resolve())
        self.assertEqual("inserted_temporary_obstacle", annotation_class)
        self.assertTrue(all(row["asset_names"] == ["barricade", "sand_pile"] for row in contract["parents"]))

    def test_factorial_reexecution_preserves_design(self) -> None:
        contract, annotation_class = subject.resolve_contract(Path("configs/public_video_route_conditioned_synthetic_factorial_contract_r814a.json").resolve())
        self.assertEqual("r814a_cross_source_two_obstacle_family_factorial_reexecution_v1", contract["contract_id"])
        self.assertEqual("inserted_temporary_obstacle", annotation_class)
        self.assertTrue(all(row["asset_names"] == ["barricade", "sand_pile"] for row in contract["parents"]))


if __name__ == "__main__":
    unittest.main()
