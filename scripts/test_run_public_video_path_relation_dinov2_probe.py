import unittest

import numpy as np

import run_public_video_path_relation_dinov2_probe as probe


class PathRelationDinoProbeTest(unittest.TestCase):
    def test_leave_one_pair_out_uses_other_pairs_only(self) -> None:
        rows = [
            {"pair_id": "a", "delta": np.array([1.0, 0.0])},
            {"pair_id": "b", "delta": np.array([0.9, 0.1])},
            {"pair_id": "c", "delta": np.array([0.8, -0.1])},
        ]
        folds = probe.leave_one_pair_out(rows)
        self.assertEqual(3, len(folds))
        self.assertTrue(all(row["ordered"] for row in folds))


if __name__ == "__main__":
    unittest.main()
