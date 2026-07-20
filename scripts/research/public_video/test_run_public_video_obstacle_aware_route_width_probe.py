#!/usr/bin/env python3
"""Pure tests for obstacle-aware route-width geometry."""

from __future__ import annotations

import unittest

import numpy as np

import run_public_video_obstacle_aware_route_width_probe as subject


class ObstacleAwareRouteWidthProbeTest(unittest.TestCase):
    def test_chromatic_detection_expands_by_object_height(self) -> None:
        detection = {
            "features": {
                "center_x_norm": 0.5,
                "width_norm": 0.1,
                "height_norm": 0.2,
                "bottom_y_norm": 0.8,
                "high_saturation_fraction": 0.4,
                "dark_fraction": 0.1,
            }
        }
        mask = subject.obstacle_mask_from_detections([detection], (100, 100))
        self.assertTrue(mask[70, 30])
        self.assertTrue(mask[70, 69])
        self.assertFalse(mask[50, 50])

    def test_dark_false_detection_is_ignored(self) -> None:
        detection = {
            "features": {
                "center_x_norm": 0.5,
                "width_norm": 0.1,
                "height_norm": 0.2,
                "bottom_y_norm": 0.8,
                "high_saturation_fraction": 0.1,
                "dark_fraction": 0.4,
            }
        }
        self.assertEqual(int(subject.obstacle_mask_from_detections([detection], (64, 64)).sum()), 0)

    def test_full_open_map_has_route(self) -> None:
        result = subject.widest_route(np.ones((64, 64), dtype=bool), np.zeros((64, 64), dtype=bool))
        self.assertTrue(result["path_found"])
        self.assertGreater(result["route_radius_norm"], 0.2)

    def test_complete_horizontal_barrier_blocks_route(self) -> None:
        obstacle = np.zeros((64, 64), dtype=bool)
        obstacle[38:42, :] = True
        result = subject.widest_route(np.ones((64, 64), dtype=bool), obstacle)
        self.assertFalse(result["path_found"])
        self.assertEqual(result["route_width_norm"], 0.0)

    def test_lateral_barrier_preserves_wider_route_than_center_split(self) -> None:
        walkable = np.ones((64, 64), dtype=bool)
        lateral = np.zeros((64, 64), dtype=bool)
        lateral[36:44, 42:] = True
        center = np.zeros((64, 64), dtype=bool)
        center[36:44, 19:45] = True
        lateral_result = subject.widest_route(walkable, lateral)
        center_result = subject.widest_route(walkable, center)
        self.assertGreater(lateral_result["route_radius_norm"], center_result["route_radius_norm"])

    def test_soft_route_margin_remains_connected_without_argmax_threshold(self) -> None:
        walkable = np.full((64, 64), 0.7, dtype=np.float32)
        obstacle = np.zeros((64, 64), dtype=bool)
        result = subject.widest_soft_route(walkable, obstacle)
        self.assertTrue(result["path_found"])
        self.assertGreater(result["route_radius_norm"], 0.1)

    def test_soft_center_barrier_reduces_margin_more_than_lateral_barrier(self) -> None:
        walkable = np.full((64, 64), 0.8, dtype=np.float32)
        lateral = np.zeros((64, 64), dtype=bool)
        lateral[36:44, 42:] = True
        center = np.zeros((64, 64), dtype=bool)
        center[36:44, 19:45] = True
        lateral_result = subject.widest_soft_route(walkable, lateral)
        center_result = subject.widest_soft_route(walkable, center)
        self.assertGreater(lateral_result["route_radius_norm"], center_result["route_radius_norm"])

    def test_adaptive_centerline_distance_field_ignores_far_lateral_obstacle(self) -> None:
        walkable = np.zeros((64, 64), dtype=np.float32)
        walkable[:, 28:36] = 1.0
        lateral = np.zeros((64, 64), dtype=bool)
        lateral[30:38, 50:] = True
        center = np.zeros((64, 64), dtype=bool)
        center[30:38, 28:36] = True
        lateral_result = subject.adaptive_path_obstacle_clearance(walkable, lateral)
        center_result = subject.adaptive_path_obstacle_clearance(walkable, center)
        self.assertGreater(lateral_result["route_radius_norm"], center_result["route_radius_norm"])


if __name__ == "__main__":
    unittest.main()
