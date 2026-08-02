from __future__ import annotations

import unittest

from inventory_sanpo_frame_spans import (
    _contiguous_runs,
    _frame_index,
    _summarize_media,
)


class InventorySanpoFrameSpansTest(unittest.TestCase):
    def test_frame_kind_and_index(self) -> None:
        self.assertEqual(
            _frame_index({"name": "x/video_frames/000007.png"}),
            ("rgb", 7),
        )
        self.assertEqual(
            _frame_index({"name": "x/depth_maps/000008.float16.gz"}),
            ("depth", 8),
        )
        self.assertEqual(
            _frame_index({"name": "x/segmentation_masks/000009.png"}),
            ("mask", 9),
        )
        self.assertIsNone(_frame_index({"name": "x/camera_poses.csv"}))

    def test_contiguous_runs_keep_gaps_explicit(self) -> None:
        self.assertEqual(
            _contiguous_runs([5, 6, 7, 11, 13, 14]),
            [
                {"start": 5, "end": 7, "count": 3},
                {"start": 11, "end": 11, "count": 1},
                {"start": 13, "end": 14, "count": 2},
            ],
        )

    def test_complete_frames_require_all_three_modalities(self) -> None:
        objects = [
            {"name": f"x/video_frames/{index:06d}.png", "size": 1}
            for index in (0, 1, 2)
        ]
        objects += [
            {"name": f"x/depth_maps/{index:06d}.float16.gz", "size": 1}
            for index in (0, 1, 2)
        ]
        objects += [
            {"name": "x/segmentation_masks/000000.png", "size": 1},
            {"name": "x/segmentation_masks/000002.png", "size": 1},
        ]
        summary = _summarize_media(objects)
        self.assertEqual(summary["complete_frame_indices"], [0, 2])
        self.assertEqual(summary["complete_frame_runs"], [
            {"start": 0, "end": 0, "count": 1},
            {"start": 2, "end": 2, "count": 1},
        ])
        self.assertEqual(summary["complete_frame_count"], 2)


if __name__ == "__main__":
    unittest.main()
