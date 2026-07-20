#!/usr/bin/env python3
"""Pure tests for the synthetic-mask dense teacher probe."""

from __future__ import annotations

import unittest

import numpy as np

import run_public_silver_synthetic_mask_teacher_probe as subject


class PublicSilverSyntheticMaskTeacherProbeTest(unittest.TestCase):
    def test_small_mask_keeps_at_least_one_patch(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[31:33, 31:33] = 255
        selected = subject.mask_to_patch_grid(mask, height=4, width=4)
        self.assertEqual((4, 4), selected.shape)
        self.assertGreaterEqual(int(selected.sum()), 1)

    def test_frame_teacher_vector_increases_for_lower_corridor_peak(self) -> None:
        baseline = np.zeros((8, 8), dtype=np.float64)
        obstacle = baseline.copy()
        obstacle[5:, 3:5] = 3.0
        clear_vector = subject.frame_teacher_vector(baseline)
        obstacle_vector = subject.frame_teacher_vector(obstacle)
        self.assertEqual(clear_vector.shape, obstacle_vector.shape)
        self.assertGreater(float(obstacle_vector[6 + 3]), float(clear_vector[6 + 3]))

    def test_episode_vector_preserves_terminal_recovery(self) -> None:
        values = np.asarray([
            [0.0, 0.0],
            [2.0, 1.0],
            [0.2, 0.1],
        ])
        vector = subject.episode_teacher_vector(values)
        dimension = values.shape[1]
        terminal = vector[2 * dimension:3 * dimension]
        last_minus_first = vector[3 * dimension:4 * dimension]
        self.assertTrue(np.allclose(values[-1], terminal))
        self.assertTrue(np.allclose(values[-1] - values[0], last_minus_first))

    def test_patch_teacher_separates_matched_positive_and_clear_features(self) -> None:
        positive_path = "positive"
        negative_path = "negative"
        feature_maps = {
            positive_path: np.ones((2, 2, 3), dtype=np.float64),
            negative_path: -np.ones((2, 2, 3), dtype=np.float64),
        }
        import cv2
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            mask_path = Path(directory) / "mask.png"
            cv2.imwrite(str(mask_path), np.full((8, 8), 255, dtype=np.uint8))
            fitted = subject.fit_patch_teacher(
                [{
                    "source_id": "synthetic",
                    "parent_source_id": "parent",
                    "positive_path": positive_path,
                    "negative_path": negative_path,
                    "mask_path": str(mask_path),
                }],
                feature_maps,
                ridge=1.0,
            )
        positive_score = subject.teacher_score_map(
            feature_maps[positive_path],
            fitted["kernel"],
            fitted["bias"],
        )
        negative_score = subject.teacher_score_map(
            feature_maps[negative_path],
            fitted["kernel"],
            fitted["bias"],
        )
        self.assertGreater(float(positive_score.mean()), float(negative_score.mean()))


if __name__ == "__main__":
    unittest.main()
