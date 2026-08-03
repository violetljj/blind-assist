#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from prepare_external_rgb_metric_depth_manifest import (
    sample_source_indices,
    select_largest_person,
    torso_roi_from_person_box,
)


class ExternalRgbMetricDepthManifestTest(unittest.TestCase):
    def test_timestamp_sampling_approximates_requested_fps(self) -> None:
        selected = sample_source_indices(15.0, 10.0, 150)
        self.assertEqual(len(selected), 100)
        self.assertEqual(selected[:6], [0, 2, 3, 5, 6, 8])
        self.assertEqual(
            sample_source_indices(10.0, 30.0, 5), [0, 1, 2, 3, 4]
        )

    def test_selects_largest_person_then_score(self) -> None:
        boxes = np.asarray(
            [[0, 0, 10, 20], [0, 0, 20, 20], [0, 0, 20, 20]],
            dtype=np.float32,
        )
        scores = np.asarray([0.9, 0.7, 0.8], dtype=np.float32)
        selected = select_largest_person(boxes, scores)
        self.assertIsNotNone(selected)
        box, score = selected
        np.testing.assert_array_equal(box, boxes[2])
        self.assertAlmostEqual(score, 0.8, places=6)

    def test_derives_bounded_central_torso_roi(self) -> None:
        roi = torso_roi_from_person_box(
            [100, 50, 300, 450], (480, 640, 3)
        )
        self.assertEqual(roi, [150, 122, 250, 310])
        with self.assertRaises(ValueError):
            torso_roi_from_person_box([1, 1, 1, 2], (10, 10, 3))


if __name__ == "__main__":
    unittest.main()
