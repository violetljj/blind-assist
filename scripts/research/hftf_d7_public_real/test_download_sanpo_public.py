from __future__ import annotations

import unittest

from download_sanpo_public import ContractError, _frame_index, _nominal_time_ns


class DownloadSanpoPublicTest(unittest.TestCase):
    def test_frame_index_mapping(self) -> None:
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

    def test_explicit_fps_is_required_for_timestamp_derivation(self) -> None:
        self.assertIsNone(_nominal_time_ns(15, None))
        self.assertEqual(_nominal_time_ns(15, 15.0), 1_000_000_000)
        with self.assertRaises(ContractError):
            _nominal_time_ns(1, 0.0)


if __name__ == "__main__":
    unittest.main()
