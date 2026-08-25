"""Focused tests for the V1-F portal-interior teacher."""

import unittest

import numpy as np

from .observation import CameraIntrinsics
from .portal_interior_teacher import infer_portal_interior
from .two_view_observation import SourceCameraPose


class PortalInteriorTeacherTest(unittest.TestCase):
    def test_anchor_plane_and_cross_view_free_component(self) -> None:
        intr = CameraIntrinsics(160, 120, 120.0, 120.0, 80.0, 60.0)
        depth = np.full((120, 160), 2.0, dtype=np.float32)
        depth[22:108, 58:112] = 4.0
        pose = SourceCameraPose((0.0, 0.0, 0.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        prediction = infer_portal_interior((depth, depth.copy()), (pose, pose), intr, (24, 42, 42, 60))
        self.assertIsNotNone(prediction.support_plane)
        self.assertIsNotNone(prediction.width_m)
        self.assertGreater(prediction.confidence, 0.5)
        self.assertTrue(prediction.views[0].component_mask.any())
        self.assertGreater(float(prediction.views[0].soft_mask.max()), 0.9)
        self.assertEqual(len(prediction.derived_boundary_lines), 2)

    def test_mesh_hole_is_unknown_not_free(self) -> None:
        intr = CameraIntrinsics(120, 96, 100.0, 100.0, 60.0, 48.0)
        depth = np.full((96, 120), 2.0, dtype=np.float32)
        depth[20:82, 48:78] = np.inf
        pose = SourceCameraPose((0.0, 0.0, 0.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
        prediction = infer_portal_interior((depth, depth.copy()), (pose, pose), intr, (15, 35, 33, 53))
        self.assertIsNone(prediction.width_m)
        self.assertEqual(prediction.diagnostics["failure"], "CROSS_VIEW_PORTAL_COMPONENT_MISSING")


if __name__ == "__main__":
    unittest.main()
