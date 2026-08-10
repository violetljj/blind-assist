from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ag_st_tum_rgbd import (  # noqa: E402
    DEFAULT_TUM_COHORT_MANIFEST,
    TumIndexRow,
    interpolate_camera_to_world,
    load_tum_cohort,
    load_tum_role_payloads,
    pair_rgb_depth_unique,
    parse_tum_poses,
)


class AgStTumRgbdTest(unittest.TestCase):
    def test_manifest_has_four_fit_three_evaluation_and_no_overlap(self) -> None:
        _, fit = load_tum_cohort(DEFAULT_TUM_COHORT_MANIFEST, "fit")
        _, evaluation = load_tum_cohort(DEFAULT_TUM_COHORT_MANIFEST, "evaluation")
        fit_ids = {str(row["parent_id"]) for row in fit}
        evaluation_ids = {str(row["parent_id"]) for row in evaluation}
        self.assertEqual((4, 3), (len(fit_ids), len(evaluation_ids)))
        self.assertFalse(fit_ids & evaluation_ids)
        self.assertTrue(all(len(row["rgb_row_indices_zero_based"]) == 3 for row in fit))
        self.assertTrue(
            all(len(row["rgb_row_indices_zero_based"]) == 3 for row in evaluation)
        )

    def test_pairing_is_one_to_one_and_prefers_smallest_delta(self) -> None:
        rgb = [
            TumIndexRow(0, 1.000, "rgb/0.png"),
            TumIndexRow(1, 1.010, "rgb/1.png"),
        ]
        depth = [
            TumIndexRow(0, 1.009, "depth/0.png"),
            TumIndexRow(1, 1.019, "depth/1.png"),
        ]
        paired = pair_rgb_depth_unique(rgb, depth)
        self.assertEqual({0: 1, 1: 0}, {key: value.row_index for key, value in paired.items()})
        self.assertEqual(2, len({value.row_index for value in paired.values()}))

    def test_pose_interpolation_preserves_metric_translation_and_rotation(self) -> None:
        rows = parse_tum_poses(
            "1.0 0 0 0 0 0 0 1\n"
            "1.1 1 0 0 0 0 0.7071067811865476 0.7071067811865476\n"
        )
        pose, gap = interpolate_camera_to_world(rows, 1.05)
        self.assertAlmostEqual(gap, 0.1)
        np.testing.assert_allclose(pose[:3, 3], [0.5, 0.0, 0.0], atol=1e-9)
        expected = np.asarray(
            [[2**-0.5, -(2**-0.5), 0.0], [2**-0.5, 2**-0.5, 0.0], [0.0, 0.0, 1.0]]
        )
        np.testing.assert_allclose(pose[:3, :3], expected, atol=1e-8)

    def test_real_selected_payloads_have_pose_receipts(self) -> None:
        fit, _ = load_tum_role_payloads(DEFAULT_TUM_COHORT_MANIFEST, "fit")
        evaluation, _ = load_tum_role_payloads(
            DEFAULT_TUM_COHORT_MANIFEST, "evaluation"
        )
        self.assertEqual((12, 9), (len(fit), len(evaluation)))
        for payload in fit + evaluation:
            self.assertEqual((4, 4), payload.camera_to_world.shape)
            self.assertLessEqual(payload.pose_bracketing_gap_seconds, 0.10)
            np.testing.assert_allclose(
                payload.camera_to_world[3], [0.0, 0.0, 0.0, 1.0], atol=0.0
            )


if __name__ == "__main__":
    unittest.main()
