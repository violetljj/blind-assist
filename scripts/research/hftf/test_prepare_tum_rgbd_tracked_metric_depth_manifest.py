#!/usr/bin/env python3

from __future__ import annotations

import unittest

import numpy as np

from prepare_tum_rgbd_tracked_metric_depth_manifest import (
    complete_track_ids,
    semantic_torso_roi,
)


class TumRgbdTrackedMetricDepthManifestTest(unittest.TestCase):
    def test_pose_torso_uses_shoulders_and_hips_only(self) -> None:
        points = np.zeros((17, 2), dtype=np.float32)
        confidence = np.ones(17, dtype=np.float32)
        points[[5, 6, 11, 12]] = [[10, 20], [30, 20], [12, 50], [28, 50]]
        self.assertEqual(
            semantic_torso_roi(points, confidence, (60, 40, 3), 0.5),
            [10, 20, 30, 50],
        )
        confidence[11] = 0.49
        self.assertIsNone(
            semantic_torso_roi(points, confidence, (60, 40, 3), 0.5)
        )

    def test_admits_only_complete_truth_valid_tracks(self) -> None:
        tracks = {
            1: [
                {"frame_index": index, "truth_admissible": True}
                for index in range(3)
            ],
            2: [
                {"frame_index": index, "truth_admissible": index != 1}
                for index in range(3)
            ],
            3: [
                {"frame_index": index, "truth_admissible": True}
                for index in range(2)
            ],
        }
        self.assertEqual(complete_track_ids(tracks, 3), [1])


if __name__ == "__main__":
    unittest.main()
