from __future__ import annotations

import csv
import unittest
from pathlib import Path

import numpy as np

from scripts.research.egomotion_compensated_looming.ecological_response_discovery_r0 import (
    runner,
)


class EcologicalResponseDiscoveryR0Test(unittest.TestCase):
    def test_quaternion_identity_rotation(self) -> None:
        rotation = runner.quaternion_rotation_xyzw(
            np.asarray((0.0, 0.0, 0.0, 1.0))
        )
        np.testing.assert_allclose(rotation, np.eye(3), rtol=0.0, atol=1e-15)

    def test_slerp_halfway_about_z(self) -> None:
        left = np.asarray((0.0, 0.0, 0.0, 1.0))
        right = np.asarray((0.0, 0.0, 1.0, 0.0))
        midpoint = runner.slerp_xyzw(left, right, 0.5)
        rotation = runner.quaternion_rotation_xyzw(midpoint)
        np.testing.assert_allclose(
            rotation @ np.asarray((1.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
            rtol=0.0,
            atol=1e-12,
        )

    def test_pair_geometry_identity(self) -> None:
        pose = (
            np.asarray((0.0, 0.0, 0.0)),
            np.asarray((0.0, 0.0, 0.0, 1.0)),
        )
        homography, angular_speed, translation_speed = runner.pair_geometry(
            pose, pose, 0.02
        )
        np.testing.assert_allclose(homography, np.eye(3), rtol=0.0, atol=1e-12)
        self.assertEqual(angular_speed, 0.0)
        self.assertEqual(translation_speed, 0.0)

    def test_three_pair_confirmation_resets(self) -> None:
        row = {"evaluable": True, "raw_expansion_median_per_s": 0.02}
        streak = 0
        for expected in (False, False, True):
            streak = runner.update_confirmation(row, streak, "raw")
            self.assertEqual(row["raw_three_pair_trigger"], expected)
        row = {"evaluable": False, "raw_expansion_median_per_s": None}
        streak = runner.update_confirmation(row, streak, "raw")
        self.assertEqual(streak, 0)
        self.assertFalse(row["raw_three_pair_trigger"])

    def test_method_summary_uses_fixed_denominator(self) -> None:
        rows = [
            {
                "raw_expansion_median_per_s": 0.02,
                "raw_three_pair_trigger": False,
                "previous_timestamp_s": 0.0,
                "current_timestamp_s": 0.1,
            },
            {
                "raw_expansion_median_per_s": None,
                "raw_three_pair_trigger": False,
                "previous_timestamp_s": 0.1,
                "current_timestamp_s": 0.2,
            },
        ]
        summary = runner.method_summary(
            rows, "raw_expansion_median_per_s", "raw_three_pair_trigger"
        )
        self.assertEqual(summary["evaluable_value_count"], 1)
        self.assertEqual(summary["positive_pair_fraction_fixed_denominator"], 0.5)

    def test_scaled_rotation_homography_scales_translation_terms(self) -> None:
        homography = np.array(
            [[1.0, 0.02, 10.0], [-0.01, 1.0, 20.0], [0.0, 0.0, 1.0]]
        )
        scale = 0.5
        scale_matrix = np.diag([scale, scale, 1.0])
        scaled = scale_matrix @ homography @ np.linalg.inv(scale_matrix)
        np.testing.assert_allclose(
            scaled,
            np.array(
                [
                    [1.0, 0.02, 5.0],
                    [-0.01, 1.0, 10.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            rtol=0.0,
            atol=1e-15,
        )

    def test_active_capability_map_is_exact_ten_columns(self) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        path = (
            repo_root
            / "docs/research/rcle/"
            "RCLE_ACTIVE_DATA_CAPABILITY_MAP_R1_2026-07-28.csv"
        )
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected = [
            "dataset_id",
            "sequence_id",
            "scene_motion",
            "available_modalities",
            "observation_unit",
            "access_cost",
            "outcome_access_state",
            "assigned_role",
            "claim_ceiling",
            "notes",
        ]
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            self.assertEqual(next(reader), expected)
            self.assertTrue(all(len(row) == len(expected) for row in reader))
        self.assertTrue(rows)
        allowed_states = {
            "CONTENT_INSPECTED",
            "OUTPUT_INSPECTED",
            "TUNED_ON",
            "SEALED_UNSEEN",
        }
        self.assertTrue(
            all(row["outcome_access_state"] in allowed_states for row in rows)
        )

    def test_current_activates_discovery_without_rewriting_old_terminal(
        self,
    ) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        current = (
            repo_root / "docs/research/rcle/README.md"
        ).read_text(encoding="utf-8")
        self.assertIn("ECOLOGICAL_DISCOVERY_ACTIVE", current)
        self.assertIn("RGB_SEGMENT_CONFIRMATION_R1_NOT_EVALUABLE", current)
        self.assertIn("SEALED EVALUATION: NOT_YET_ALLOCATED", current)


if __name__ == "__main__":
    unittest.main()
