"""Focused tests for the V1-E0 mesh-only teacher field."""

import unittest

import numpy as np

from .privileged_geometry_teacher import geometry_fields, heatmap_line_candidates


class PrivilegedGeometryTeacherTest(unittest.TestCase):
    def test_signed_jump_assigns_left_and_right_roles(self) -> None:
        depth = np.full((96, 128), 2.0, dtype=np.float32)
        depth[12:88, 35:91] = 4.0
        normals = np.zeros((96, 128, 3), dtype=np.float32)
        normals[..., 2] = 1.0
        signed, valid, heatmap, diagnostics = geometry_fields(depth, normals)
        left = heatmap_line_candidates(heatmap[0])
        right = heatmap_line_candidates(heatmap[1])
        self.assertGreater(diagnostics["label_valid_fraction"], 0.0)
        self.assertTrue(valid.any())
        self.assertGreater(float(signed[:, 35].max()), 0.0)
        self.assertLess(float(signed[:, 91].min()), 0.0)
        self.assertLess(abs(left[0].x_at(48) - 35), 4.0)
        self.assertLess(abs(right[0].x_at(48) - 91), 4.0)

    def test_mesh_holes_are_ignored(self) -> None:
        depth = np.full((64, 96), 2.0, dtype=np.float32)
        depth[:, 42:54] = np.inf
        normals = np.zeros((64, 96, 3), dtype=np.float32)
        normals[..., 2] = 1.0
        signed, valid, heatmap, _ = geometry_fields(depth, normals)
        self.assertFalse(valid[:, 40:56].any())
        self.assertEqual(float(np.max(np.abs(signed[:, 40:56]))), 0.0)
        self.assertEqual(float(np.max(heatmap[:, :, 40:56])), 0.0)


if __name__ == "__main__":
    unittest.main()
