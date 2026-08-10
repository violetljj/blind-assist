#!/usr/bin/env python3

import unittest

import numpy as np

from run_ag_st_analytic_support_boundary_canary import (
    FLOOR_ID,
    TABLE_ID,
    camera_to_world_look_at,
    render_floor_and_table,
)


class AnalyticSupportBoundaryCanaryTest(unittest.TestCase):
    def test_renderer_contains_floor_table_and_exact_boundary(self) -> None:
        pose = camera_to_world_look_at(
            np.asarray([0.0, -1.6, 1.5]),
            np.asarray([0.0, 0.0, 0.68]),
        )
        rendered = render_floor_and_table(pose)
        self.assertGreater(int(np.sum(rendered["surface_id"] == FLOOR_ID)), 100)
        self.assertGreater(int(np.sum(rendered["surface_id"] == TABLE_ID)), 100)
        self.assertGreater(int(np.sum(rendered["boundary"])), 20)
        self.assertTrue(np.all(rendered["depth_m"][rendered["valid"]] > 0.0))
        self.assertAlmostEqual(1.0, np.linalg.det(pose[:3, :3]), places=6)


if __name__ == "__main__":
    unittest.main()
