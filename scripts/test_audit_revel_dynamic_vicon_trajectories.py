import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).with_name("audit_revel_dynamic_vicon_trajectories.py")
SPEC = importlib.util.spec_from_file_location("revel_vicon", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class RevelDynamicViconTrajectoryTest(unittest.TestCase):
    def test_nearest_indices_selects_left_and_right_neighbours(self):
        result = MODULE._nearest_indices(np.asarray([1, 8, 19]), np.asarray([0, 10, 20]))
        np.testing.assert_array_equal(result, np.asarray([0, 1, 2]))

    def test_rotation_matrix_identity(self):
        import torch
        matrix = MODULE._rotation_matrix(torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
        self.assertTrue(torch.allclose(matrix, torch.eye(3).reshape(1, 3, 3)))

    def test_stats_empty_is_explicit(self):
        import torch
        self.assertEqual({"count": 0}, MODULE._stats(torch.empty(0)))


if __name__ == "__main__":
    unittest.main()
