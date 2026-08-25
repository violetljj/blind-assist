#!/usr/bin/env python3

import tempfile
from pathlib import Path
import unittest

import numpy as np

from grail_natural_3d_m0 import (
    SceneGrid,
    _line_clear,
    estimate_floor_z,
    load_binary_ply_vertices,
)


class GrailNatural3DM0Tests(unittest.TestCase):
    def test_floor_estimator_prefers_dense_lower_plane(self) -> None:
        floor = np.column_stack((np.linspace(0, 2, 200), np.zeros(200), np.zeros(200)))
        clutter = np.column_stack((np.zeros(50), np.ones(50), np.linspace(0.2, 2.0, 50)))
        self.assertAlmostEqual(estimate_floor_z(np.vstack((floor, clutter))), 0.0, delta=0.04)

    def test_binary_ply_loader_rejects_wrong_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.ply"
            path.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n")
            with self.assertRaisesRegex(ValueError, "unsupported PLY"):
                load_binary_ply_vertices(path)

    def test_visibility_uses_raw_occupancy_not_clearance_mask(self) -> None:
        free = np.zeros((1, 5), dtype=bool)
        occupied = np.zeros((1, 5), dtype=bool)
        component = np.zeros((1, 5), dtype=np.int32)
        grid = SceneGrid("fixture", 0.0, 1.0, 0.0, 0.0, free, occupied, component, 0)

        self.assertTrue(_line_clear(grid, (0.0, 0.0), (4.0, 0.0)))
        occupied[0, 2] = True
        self.assertFalse(_line_clear(grid, (0.0, 0.0), (4.0, 0.0)))

    def test_zero_component_never_counts_background_as_reachable(self) -> None:
        free = np.zeros((2, 2), dtype=bool)
        occupied = np.zeros((2, 2), dtype=bool)
        component = np.zeros((2, 2), dtype=np.int32)
        grid = SceneGrid("fixture", 0.0, 1.0, 0.0, 0.0, free, occupied, component, 0)

        self.assertFalse(grid.reachable(0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
