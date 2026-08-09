from __future__ import annotations

import unittest

import numpy as np

from scripts.research.assistive_geometry_cbf.audit_grid_support import (
    evaluate_frame_metrics,
    evaluate_route_gate,
    ground_axes,
    select_parent_frames,
)


POLICY = {
    "frame_gate": {
        "minimum_in_grid_point_count": 80,
        "minimum_observed_cell_count": 64,
        "minimum_ground_cell_count": 32,
        "minimum_ground_cells_per_forward_quartile": 4,
        "minimum_observed_cells_per_lateral_third": 8,
    },
    "route_gate": {
        "minimum_evaluable_frames_per_passing_parent": 32,
        "minimum_passing_parents": 12,
        "minimum_total_evaluable_frames": 640,
        "minimum_evaluable_frames_per_orientation": 128,
    },
}


class AssistiveGeometryCbfGridSupportTest(unittest.TestCase):
    def test_ground_axes_match_upright_camera_convention(self) -> None:
        heading, lateral = ground_axes(np.asarray([0.0, -1.0, 0.0]))
        self.assertTrue(np.allclose([0.0, 0.0, 1.0], heading))
        self.assertTrue(np.allclose([1.0, 0.0, 0.0], lateral))

    def test_frame_gate_passes_at_every_frozen_boundary(self) -> None:
        metrics = {
            "ground_plane_valid": True,
            "finite_geometry_contract": True,
            "in_grid_point_count": 80,
            "observed_cell_count": 64,
            "ground_cell_count": 32,
            "obstacle_cell_count": 0,
            "ground_cells_by_forward_quartile": [4, 4, 4, 4],
            "observed_cells_by_lateral_third": [8, 8, 8],
        }
        self.assertEqual((True, []), evaluate_frame_metrics(metrics, POLICY))
        metrics["ground_cells_by_forward_quartile"][3] = 3
        evaluable, reasons = evaluate_frame_metrics(metrics, POLICY)
        self.assertFalse(evaluable)
        self.assertIn("UNKNOWN_LONGITUDINAL_GROUND_SUPPORT", reasons)

    def test_unknown_ground_is_never_counted_as_evaluable(self) -> None:
        metrics = {
            "ground_plane_valid": False,
            "finite_geometry_contract": False,
            "in_grid_point_count": 1000,
            "observed_cell_count": 500,
            "ground_cell_count": 100,
            "obstacle_cell_count": 50,
            "ground_cells_by_forward_quartile": [20, 20, 20, 20],
            "observed_cells_by_lateral_third": [20, 20, 20],
        }
        evaluable, reasons = evaluate_frame_metrics(metrics, POLICY)
        self.assertFalse(evaluable)
        self.assertEqual(["UNKNOWN_GROUND_PLANE", "UNKNOWN_GEOMETRY_CONTRACT"], reasons)

    def test_route_gate_is_parent_and_orientation_bounded(self) -> None:
        parents = {f"p{index}": 40 for index in range(16)}
        qualified, metrics = evaluate_route_gate(
            parents,
            {"portrait": 320, "landscape": 320},
            POLICY,
        )
        self.assertTrue(qualified)
        self.assertEqual(16, metrics["passing_parent_count"])
        parents["p0"] = 0
        parents["p1"] = 0
        parents["p2"] = 0
        parents["p3"] = 0
        parents["p4"] = 0
        qualified, _ = evaluate_route_gate(
            parents,
            {"portrait": 220, "landscape": 220},
            POLICY,
        )
        self.assertFalse(qualified)

    def test_selection_is_deterministic_and_parent_ordered(self) -> None:
        frames = [
            {"video_id": parent, "frame_stem": f"{parent}-{index}"}
            for parent in ("b", "a")
            for index in range(10)
        ]
        selected = select_parent_frames(frames, ["a", "b"], 4)
        self.assertEqual(
            ["a-0", "a-3", "a-6", "a-9", "b-0", "b-3", "b-6", "b-9"],
            [frame["frame_stem"] for frame in selected],
        )


if __name__ == "__main__":
    unittest.main()
