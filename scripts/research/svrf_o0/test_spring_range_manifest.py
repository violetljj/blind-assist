from __future__ import annotations

import unittest

from scripts.research.svrf_o0.build_spring_range_manifest import parent_from_member, visibility


class SvrfO0SpringRangeManifestTest(unittest.TestCase):
    def test_parent_identity_is_source_native_sequence(self) -> None:
        self.assertEqual(parent_from_member("spring/train/0018/frame_left/frame_left_0001.png"), "0018")
        self.assertIsNone(parent_from_member("spring/test/0018/frame_left/frame_left_0001.png"))

    def test_candidate_truth_visibility_is_archive_frozen(self) -> None:
        self.assertEqual(visibility("train_frame_left.zip"), (True, False))
        for archive in ("train_cam_data.zip", "train_disp1_left.zip", "train_flow_FW_left.zip", "train_maps.zip"):
            self.assertEqual(visibility(archive), (False, True))


if __name__ == "__main__":
    unittest.main()
