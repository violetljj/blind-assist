import unittest

import numpy as np

from .task_boundary_field_experiment import MODEL_HEIGHT, MODEL_WIDTH, _select_peaks, boundary_targets, synthetic_anchor


class TaskBoundaryFieldTest(unittest.TestCase):
    def test_boundary_targets_follow_component_edges(self):
        mask = np.zeros((MODEL_HEIGHT, MODEL_WIDTH), dtype=np.uint8)
        mask[30:160, 44:181] = 1
        targets, columns = boundary_targets(mask)
        self.assertEqual(columns, (44.0, 180.0))
        self.assertEqual(tuple(targets.shape), (2, MODEL_HEIGHT, MODEL_WIDTH))
        self.assertGreater(float(targets[0, 80, 44]), 0.99)
        self.assertGreater(float(targets[1, 80, 180]), 0.99)

    def test_anchor_is_deterministic_and_nonempty(self):
        first = synthetic_anchor((40, 20, 180, 170), "sample")
        second = synthetic_anchor((40, 20, 180, 170), "sample")
        self.assertTrue(np.array_equal(first, second))
        self.assertGreater(int(np.count_nonzero(first)), 0)

    def test_peak_selection_applies_local_suppression(self):
        profile = np.zeros(64, dtype=np.float32)
        profile[10], profile[12], profile[40] = 1.0, 0.9, 0.8
        peaks = _select_peaks(profile, top_k=2)
        self.assertEqual(peaks, [10, 40])


if __name__ == "__main__":
    unittest.main()
