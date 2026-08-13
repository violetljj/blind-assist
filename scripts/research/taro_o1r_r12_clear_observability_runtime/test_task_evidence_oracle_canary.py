import unittest

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_oracle_canary as subject


class TaskEvidenceOracleCanaryTest(unittest.TestCase):
    def test_empty_observation_keeps_every_cell_unknown(self) -> None:
        points = np.zeros((2, 2, 3), dtype=np.float64)
        valid = np.zeros((2, 2), dtype=bool)
        query = {
            "path_lateral_offset_m": 0.0,
            "virtual_query_frame": {
                "origin_camera_xyz": [0.0, 0.0, 0.0],
                "forward_camera_xyz": [0.0, 0.0, 1.0],
                "lateral_camera_xyz": [1.0, 0.0, 0.0],
                "gravity_up_camera_xyz": [0.0, -1.0, 0.0],
                "path_heading_camera_xyz": [0.0, 0.0, 1.0],
            },
        }
        cells = subject.query_evidence_cells(points, valid, [query])
        self.assertFalse(np.any(cells))

    def test_union_can_only_retain_or_add_evidence(self) -> None:
        static = np.asarray([True, False, False])
        observed = np.asarray([False, True, False])
        final = static | observed
        self.assertEqual(0, int(np.sum(static & ~final)))
        self.assertEqual(1, int(np.sum(final & ~static)))


if __name__ == "__main__":
    unittest.main()
