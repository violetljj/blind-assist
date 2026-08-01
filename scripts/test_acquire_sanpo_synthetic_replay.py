from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import acquire_sanpo_synthetic_replay as replay


class SanpoSyntheticReplayTest(unittest.TestCase):
    def test_selects_only_rgb_mask_depth_aligned_indices(self) -> None:
        objects = lambda values: {value: {"name": f"{value:06d}.png"} for value in values}
        selected = replay.select_aligned_indices(
            objects([0, 1, 2, 3]), objects([0, 1, 2]), objects([0, 2, 3]),
            source_fps=20.0, target_fps=20.0, start_frame=0, frame_count=2,
        )
        self.assertEqual([0, 2], selected)

    def test_rejects_target_rate_above_source_or_missing_full_bundle(self) -> None:
        objects = lambda values: {value: {"name": f"{value:06d}.png"} for value in values}
        with self.assertRaisesRegex(replay.ReplayError, "cannot exceed"):
            replay.select_aligned_indices(objects([0]), objects([0]), objects([0]), source_fps=10, target_fps=11, start_frame=0, frame_count=1)
        with self.assertRaisesRegex(replay.ReplayError, "aligned frames"):
            replay.select_aligned_indices(objects([0]), objects([0]), objects([]), source_fps=10, target_fps=10, start_frame=0, frame_count=1)

    def test_camera_metadata_requires_a_complete_published_intrinsic_contract(self) -> None:
        description = {"session_camera_location": ["camera_chest"], "session_camera_details": [{"fps": 20.0, "left_camera_params": {"fx": 1, "fy": 1, "cx": 1, "cy": 1, "image_width": 2, "image_height": 3}}]}
        fps, dimensions = replay.camera_metadata(description, "camera_chest", "left")
        self.assertEqual(20.0, fps)
        self.assertEqual(2, dimensions["image_width"])
        del description["session_camera_details"][0]["left_camera_params"]["fx"]
        with self.assertRaisesRegex(replay.ReplayError, "missing fx"):
            replay.camera_metadata(description, "camera_chest", "left")

    def test_split_contract_is_explicit_and_heldout_is_not_training(self) -> None:
        train = replay.split_contract("train")
        test = replay.split_contract("test")
        self.assertTrue(train["pretraining_candidate"])
        self.assertFalse(
            train["synthetic_heldout_evaluation_candidate"]
        )
        self.assertFalse(test["pretraining_candidate"])
        self.assertTrue(
            test["synthetic_heldout_evaluation_candidate"]
        )
        self.assertTrue(
            test["split_object_name"].endswith(
                "/test_session_ids.txt"
            )
        )
        with self.assertRaisesRegex(replay.ReplayError, "one of"):
            replay.split_contract("dev")


if __name__ == "__main__":
    unittest.main()
