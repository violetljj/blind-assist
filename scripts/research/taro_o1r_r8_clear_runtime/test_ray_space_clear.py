from __future__ import annotations

import unittest

import numpy as np

from scripts.research.taro_o1r_r8_clear_runtime import ray_space_clear as ray


class RaySpaceClearTests(unittest.TestCase):
    def setUp(self) -> None:
        plane = {"evaluable": True, "normal_camera_xyz": [0.0, -1.0, 0.0], "camera_height_m": 1.5}
        self.query = ray.build_truth_queries("frame", plane)[4]
        self.matrix = [[250.0, 0.0, 960.0], [0.0, 250.0, 720.0], [0.0, 0.0, 1.0]]

    def test_far_observations_prove_clear(self) -> None:
        depth = np.full((1440, 1920), 5.0, dtype=np.float64)
        label = ray.ray_query_label(depth, self.matrix, self.query)
        self.assertEqual(label["state"], "CLEAR_OBSERVED")
        self.assertEqual(label["blocked_anchor_count"], 0)

    def test_near_observations_prove_occupied(self) -> None:
        depth = np.full((1440, 1920), 0.3, dtype=np.float64)
        label = ray.ray_query_label(depth, self.matrix, self.query)
        self.assertEqual(label["state"], "OCCUPIED_OBSERVED")

    def test_missing_observations_remain_unknown(self) -> None:
        depth = np.zeros((1440, 1920), dtype=np.float64)
        label = ray.ray_query_label(depth, self.matrix, self.query)
        self.assertEqual(label["state"], "UNKNOWN")

    def test_truth_queries_ignore_source_availability(self) -> None:
        plane = {"evaluable": True, "normal_camera_xyz": [0.0, -1.0, 0.0], "camera_height_m": 1.5}
        queries = ray.build_truth_queries("frame", plane)
        self.assertEqual(len(queries), 9)
        self.assertNotIn("query_receipt", queries[0])


if __name__ == "__main__":
    unittest.main()
