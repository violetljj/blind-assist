"""Focused implementation tests for the V1-D parallax field."""

from __future__ import annotations

import unittest

import numpy as np

from .active_parallax_observation import forward_backward_mask, parallax_boundary_roles, rotational_flow
from .two_view_observation import SourceCameraPose


class ActiveParallaxObservationTest(unittest.TestCase):
    def test_identity_rotation_has_zero_flow(self) -> None:
        pose = SourceCameraPose((0.0, 0.0, 0.0), tuple(tuple(float(v) for v in row) for row in np.eye(3)))
        intrinsic = np.asarray([[100.0, 0.0, 31.5], [0.0, 100.0, 23.5], [0.0, 0.0, 1.0]])
        flow = rotational_flow(48, 64, intrinsic, pose, pose)
        self.assertLess(float(np.max(np.abs(flow))), 1e-5)

    def test_forward_backward_consistency_rejects_mismatch(self) -> None:
        forward = np.zeros((16, 20, 2), dtype=np.float32)
        backward = np.zeros_like(forward)
        forward[..., 0] = 1.0
        backward[..., 0] = -1.0
        valid, _ = forward_backward_mask(forward, backward)
        self.assertTrue(np.all(valid[:, :-1]))
        backward[8, 9, 0] = 5.0
        valid, _ = forward_backward_mask(forward, backward)
        self.assertFalse(bool(valid[8, 8]))

    def test_vertical_accumulation_finds_two_parallax_steps(self) -> None:
        height, width = 96, 128
        residual = np.zeros((height, width, 2), dtype=np.float32)
        residual[:, 28:92, 0] = 4.0
        valid = np.ones((height, width), dtype=bool)
        left, right, diagnostics = parallax_boundary_roles(residual, valid, (55, 40, 73, 58))
        self.assertTrue(any(abs(line.x_at(height * 0.5) - 28) <= 2 for line in left))
        self.assertTrue(any(abs(line.x_at(height * 0.5) - 92) <= 2 for line in right))
        self.assertEqual(len(diagnostics["left_candidate_x_px"]), 8)
        self.assertEqual(len(diagnostics["right_candidate_x_px"]), 8)


if __name__ == "__main__":
    unittest.main()
