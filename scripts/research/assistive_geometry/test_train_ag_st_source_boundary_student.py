#!/usr/bin/env python3

import unittest

import numpy as np

from train_ag_st_source_boundary_student import average_precision, deterministic_split


class SourceBoundaryStudentTest(unittest.TestCase):
    def test_split_is_source_stratified_and_disjoint(self) -> None:
        rows = []
        for source, count in (("arkitscenes", 16), ("tum_rgbd", 7), ("icl_exact", 1)):
            for index in range(count):
                rows.append({"source": source, "parent_id": f"p{index}"})
        first = deterministic_split(rows)
        second = deterministic_split(list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual(18, len(first["fit"]))
        self.assertEqual(3, len(first["selection"]))
        self.assertEqual(3, len(first["canary"]))
        self.assertFalse(set(first["fit"]) & set(first["canary"]))

    def test_average_precision_rewards_correct_ranking(self) -> None:
        target = np.asarray([0, 1, 0, 1], dtype=np.bool_)
        good = average_precision(target, np.asarray([0.1, 0.9, 0.2, 0.8]))
        bad = average_precision(target, np.asarray([0.9, 0.1, 0.8, 0.2]))
        self.assertEqual(1.0, good)
        self.assertGreater(good, bad)


if __name__ == "__main__":
    unittest.main()
