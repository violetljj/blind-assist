#!/usr/bin/env python3

import unittest

import numpy as np

from evaluate_ag_st_multiteacher_boundary_gate import (
    depth_boundary_seed,
    teacher_boundary_consensus,
)


class MultiTeacherBoundaryGateTest(unittest.TestCase):
    def test_point_plane_seed_rejects_plane_and_keeps_depth_step(self) -> None:
        intrinsics = np.asarray(
            [[120.0, 0.0, 11.5], [0.0, 120.0, 11.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        plane = np.full((24, 24), 2.0, dtype=np.float32)
        valid = np.ones_like(plane, dtype=np.bool_)
        self.assertEqual(0, int(np.sum(depth_boundary_seed(plane, valid, intrinsics))))
        stepped = plane.copy()
        stepped[:, 12:] = 2.8
        seed = depth_boundary_seed(stepped, valid, intrinsics)
        self.assertGreater(int(np.sum(seed[:, 10:14])), 0)

    def test_consensus_requires_nearby_teacher_and_quality(self) -> None:
        primary = np.zeros((9, 9), dtype=np.bool_)
        secondary = np.zeros_like(primary)
        quality = np.ones_like(primary)
        primary[4, 2] = True
        secondary[4, 4] = True
        consensus = teacher_boundary_consensus(primary, secondary, quality)
        self.assertTrue(consensus[4, 2])
        self.assertTrue(consensus[4, 4])
        quality[4, 2] = False
        consensus = teacher_boundary_consensus(primary, secondary, quality)
        self.assertFalse(consensus[4, 2])


if __name__ == "__main__":
    unittest.main()
