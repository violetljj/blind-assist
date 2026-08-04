import json
import unittest
from pathlib import Path

import numpy as np
from run_offline_stress_r0 import (
    DEFAULT_PROTOCOL,
    add_clean_delta,
    apply_scenario,
    build_scenarios,
    is_accepted_bad,
    local_multiplier,
    mask_ground_roi,
    sha256,
    summarize_scenario,
)
from sealed_student import MODEL_ID, SealedScaleStudent

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL = REPO_ROOT / "configs/hftf/camera_conditioned_scale_student_r0_model.json"


class OfflineStressProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))

    def test_protocol_is_frozen_and_binds_sealed_no_fit_model(self) -> None:
        self.assertEqual(
            "FROZEN_BEFORE_OFFLINE_STRESS_EXECUTION", self.protocol["status"]
        )
        self.assertFalse(
            self.protocol["sealed_student"]["fit_or_update_entrypoint_allowed"]
        )
        self.assertEqual(MODEL_ID, self.protocol["sealed_student"]["model_id"])
        self.assertEqual(
            self.protocol["sealed_student"]["receipt_sha256"], sha256(MODEL)
        )
        student = SealedScaleStudent.load(MODEL)
        self.assertEqual(MODEL_ID, self.protocol["sealed_student"]["model_id"])
        self.assertEqual(10, len(student.feature_names))

    def test_scenario_grid_is_fixed_unique_and_canaries_exclude_effect_truth(self) -> None:
        scenarios = build_scenarios(self.protocol)
        self.assertEqual(47, len(scenarios))
        self.assertEqual(47, len({row["id"] for row in scenarios}))
        self.assertEqual("clean", scenarios[0]["id"])
        canaries = [row for row in scenarios if row["family"].startswith("geometric_")]
        self.assertEqual(10, len(canaries))
        self.assertTrue(all(row["truth_comparable"] is False for row in canaries))
        self.assertFalse(self.protocol["rgb_second_layer"]["cached_runner_implemented"])
        self.assertTrue(self.protocol["rgb_second_layer"]["rgb_runner_implemented"])
        self.assertTrue(
            self.protocol["rgb_second_layer"]["required_before_answering_blur_or_low_light"]
        )


class OfflineStressTransformTest(unittest.TestCase):
    def setUp(self) -> None:
        self.depth = np.full((100, 120), 2.0, dtype=np.float64)
        self.intrinsics = np.asarray(
            [[80.0, 0.0, 60.0], [0.0, 80.0, 50.0], [0.0, 0.0, 1.0]]
        )

    def test_local_fields_are_positive_and_median_one(self) -> None:
        for field in ("vertical_linear", "horizontal_linear", "bandwise_left_to_right"):
            multiplier = local_multiplier(
                self.depth.shape,
                {"field": field, "amplitude": 0.2, "polarity": -1},
            )
            self.assertTrue(np.all(multiplier > 0.0))
            self.assertAlmostEqual(1.0, float(np.median(multiplier)), places=12)
            self.assertGreater(float(np.std(multiplier)), 0.0)

    def test_height_focal_and_global_scale_inject_at_distinct_points(self) -> None:
        depth, matrix, height = apply_scenario(
            self.depth,
            self.intrinsics,
            1.5,
            {"family": "camera_height_error", "delta_m": 0.1},
            0.55,
        )
        np.testing.assert_array_equal(depth, self.depth)
        np.testing.assert_array_equal(matrix, self.intrinsics)
        self.assertAlmostEqual(1.6, height)

        depth, matrix, height = apply_scenario(
            self.depth,
            self.intrinsics,
            1.5,
            {"family": "focal_length_error", "fraction": -0.05},
            0.55,
        )
        self.assertAlmostEqual(76.0, matrix[0, 0])
        self.assertAlmostEqual(76.0, matrix[1, 1])
        np.testing.assert_array_equal(depth, self.depth)
        self.assertAlmostEqual(1.5, height)

        depth, matrix, height = apply_scenario(
            self.depth,
            self.intrinsics,
            1.5,
            {"family": "global_da_scale_error", "fraction": 0.4},
            0.55,
        )
        np.testing.assert_allclose(depth, self.depth * 1.4)
        np.testing.assert_array_equal(matrix, self.intrinsics)
        self.assertAlmostEqual(1.5, height)

    def test_roi_masks_are_deterministic_and_confined_to_lower_roi(self) -> None:
        scenario = {
            "family": "ground_roi_occlusion",
            "pattern": "center_block",
            "fraction": 0.5,
        }
        first = mask_ground_roi(self.depth, scenario, 0.55)
        second = mask_ground_roi(self.depth, scenario, 0.55)
        np.testing.assert_array_equal(np.isnan(first), np.isnan(second))
        self.assertFalse(np.any(np.isnan(first[:55])))
        roi_fraction = float(np.mean(np.isnan(first[55:])))
        self.assertAlmostEqual(0.5, roi_fraction, delta=0.03)

    def test_geometric_canaries_preserve_shape_but_are_not_identity(self) -> None:
        gradient = np.arange(self.depth.size, dtype=np.float64).reshape(self.depth.shape)
        for scenario in (
            {"family": "geometric_pitch_canary", "degrees": 5.0},
            {"family": "geometric_roll_canary", "degrees": -5.0},
            {"family": "geometric_crop_canary", "retained_fraction": 0.8},
        ):
            output, matrix, height = apply_scenario(
                gradient, self.intrinsics, 1.5, scenario, 0.55
            )
            self.assertEqual(gradient.shape, output.shape)
            self.assertFalse(np.array_equal(gradient, output, equal_nan=True))
            np.testing.assert_array_equal(matrix, self.intrinsics)
            self.assertEqual(1.5, height)


class OfflineStressMetricTest(unittest.TestCase):
    @staticmethod
    def row(
        parent: str,
        anchor: int,
        candidate: dict[str, float] | None,
        truth: dict[str, float] | None,
        comparable: bool = True,
        reason: str | None = None,
    ) -> dict:
        return {
            "scenario_id": "s",
            "scenario_family": "test",
            "truth_comparable": comparable,
            "parent_id": parent,
            "anchor_frame_id": anchor,
            "candidate": candidate,
            "truth": truth,
            "accepted_bad": is_accepted_bad(candidate, truth),
            "unknown_reason": reason,
        }

    def test_accepted_bad_catches_metric_error_and_false_clear(self) -> None:
        truth = {"left": 0.5, "center": 1.2, "right": 2.5}
        good = {"left": 0.6, "center": 1.3, "right": 2.4}
        bad = {"left": 1.2, "center": 2.2, "right": 2.5}
        self.assertFalse(is_accepted_bad(good, truth))
        self.assertTrue(is_accepted_bad(bad, truth))
        self.assertIsNone(is_accepted_bad(None, truth))

    def test_summary_reports_unknown_worst_parent_and_clean_delta(self) -> None:
        truth = {band: 1.0 for band in ("left", "center", "right")}
        exact = truth.copy()
        rows = [
            self.row("a", 1, exact, truth),
            self.row("a", 2, None, truth, reason="NO_GROUND_CONSENSUS"),
            self.row("b", 1, {band: 2.0 for band in truth}, truth),
            self.row("b", 2, {band: 2.0 for band in truth}, truth),
        ]
        scenario = {"id": "s", "family": "test", "truth_comparable": True}
        summary = summarize_scenario(rows, scenario)
        self.assertEqual(1, summary["unknown_reason_counts"]["NO_GROUND_CONSENSUS"])
        self.assertEqual("a", summary["worst_parent"]["known_coverage"]["parent_id"])
        self.assertGreater(summary["parent_macro"]["accepted_bad_rate_all_truth_frames"], 0.0)
        add_clean_delta(summary, summary)
        self.assertTrue(all(value == 0.0 for value in summary["delta_vs_clean"].values()))

    def test_canary_summary_reports_admission_only(self) -> None:
        rows = [self.row("a", 1, {"left": 1.0, "center": 1.0, "right": 1.0}, None, False)]
        scenario = {"id": "c", "family": "geometric_pitch_canary", "truth_comparable": False}
        summary = summarize_scenario(rows, scenario)
        self.assertEqual(1.0, summary["parent_macro"]["known_coverage"])
        self.assertIsNone(summary["parent_macro"]["clearance_mae_m"])
        self.assertIsNone(summary["parent_macro"]["accepted_bad_rate_all_truth_frames"])


if __name__ == "__main__":
    unittest.main()
