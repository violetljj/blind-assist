#!/usr/bin/env python3
"""Focused mechanics tests for formal source-native F1 label materialization."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from materialize_ag_r2_f1_source_native_labels import (  # noqa: E402
    persistent_height_modes,
    source_geometric_factors,
    verify_orientation_projection,
)


class SourceNativeMaterializerMechanicsTest(unittest.TestCase):
    def test_clockwise_intrinsics_preserve_projection(self) -> None:
        source = np.asarray(
            [[520.9, 0.0, 325.1], [0.0, 521.0, 249.7], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        output = np.asarray(
            [[521.0, 0.0, 229.3], [0.0, 520.9, 325.1], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.assertTrue(verify_orientation_projection(source, output, "PORTRAIT_ROT90_CLOCKWISE"))

    def test_persistent_support_identity_selects_lowest_mode(self) -> None:
        rng = np.random.default_rng(7)
        values = []
        for _ in range(3):
            floor = rng.normal(0.0, 0.008, size=900)
            table = rng.normal(0.75, 0.008, size=1200)
            values.append(np.concatenate((floor, table)))
        modes = persistent_height_modes(values)
        self.assertGreaterEqual(len(modes), 2)
        self.assertAlmostEqual(modes[0]["world_height_m"], 0.0, delta=0.03)

    def test_missing_gravity_keeps_support_and_boundary_unknown(self) -> None:
        depth = np.full((12, 16), 2.0, dtype=np.float32)
        valid = np.ones_like(depth, dtype=np.bool_)
        intrinsics = np.asarray(
            [[120.0, 0.0, 7.5], [0.0, 120.0, 5.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        factors = source_geometric_factors(
            depth,
            valid,
            intrinsics,
            np.eye(4, dtype=np.float64),
            None,
            None,
        )
        self.assertFalse(bool(factors["support_plane_valid"]))
        self.assertEqual(int(np.sum(factors["support_truth_valid_hw"])), 0)
        self.assertEqual(int(np.sum(factors["evidence_truth_valid_hw"])), 0)
        self.assertTrue(np.isnan(factors["boundary_distance_px_hw"]).all())

    def test_continuous_sloped_plane_has_no_boundary_seed(self) -> None:
        height, width = 40, 48
        intrinsics = np.asarray(
            [[160.0, 0.0, (width - 1) / 2], [0.0, 160.0, (height - 1) / 2], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        columns = np.arange(width, dtype=np.float64)[None, :]
        depth = (2.0 + 0.004 * (columns - (width - 1) / 2)).repeat(height, axis=0).astype(np.float32)
        valid = np.ones((height, width), dtype=np.bool_)
        pose = np.eye(4, dtype=np.float64)
        pose[1, 3] = 1.2
        identity = {
            "world_up_unit": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            "support_world_height_m": 0.0,
            "median_absolute_residual_m": 0.01,
        }
        factors = source_geometric_factors(
            depth,
            valid,
            intrinsics,
            pose,
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            identity,
        )
        self.assertEqual(int(np.sum(factors["boundary_seed_diagnostic_hw"])), 0)


if __name__ == "__main__":
    unittest.main()
