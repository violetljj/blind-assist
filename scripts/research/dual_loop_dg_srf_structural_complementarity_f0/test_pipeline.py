"""Focused contract tests for DG-SRF image-space F0."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import numpy as np

from .common import SHAPE, validate_config
from .evaluate import (
    _choose_threshold,
    average_precision_tie_group,
    gate_report,
)
from .operators import (
    build_proxy_mask,
    depth_health_and_proximity,
    structural_scores,
    validate_depth_direction_canary,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "dg_srf_image_space_structural_complementarity_f0"
    / "default.json"
)


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


class FrozenContractTests(unittest.TestCase):
    def test_config_exact_locks(self) -> None:
        config = load_config()
        validate_config(config)
        self.assertEqual(
            config["model_contract"]["exact_parameter_count"], 24_785_089
        )
        self.assertEqual(config["model_contract"]["input_size"], 518)
        self.assertEqual(
            config["structural_signal"]["gradient_gaussian_sigmas_pixels"],
            [0.0, 1.5, 3.0],
        )
        self.assertEqual(
            config["structural_signal"]["surface_trend"],
            {
                "name": "lower_image_surface_trend",
                "lower_image_start_fraction": 0.45,
                "row_statistic": "median_full_width",
                "polynomial_degree": 2,
                "fit": "ordinary_least_squares_on_row_medians",
                "residual_dead_zone": 0.03,
                "truth_or_yolo_assisted_fit": False,
                "failure_on_nonfinite_or_rank_defect": True,
                "forbidden_names": ["ground", "floor", "height", "plane"],
            },
        )

    def test_mutated_weight_is_rejected(self) -> None:
        config = load_config()
        config["structural_signal"]["D4_weights"]["N"] = 0.30
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_direction_canary_cannot_flip(self) -> None:
        passed = validate_depth_direction_canary(
            [0.3, 0.2, 0.1, -0.01],
            frozen_direction="RAW_LARGER_IS_NEARER",
            minimum_consistent=3,
            minimum_median_margin=0.02,
        )
        self.assertTrue(passed["passed"])
        failed = validate_depth_direction_canary(
            [-0.3, -0.2, -0.1, 0.01],
            frozen_direction="RAW_LARGER_IS_NEARER",
            minimum_consistent=3,
            minimum_median_margin=0.02,
        )
        self.assertFalse(failed["passed"])

    def test_per_frame_positive_affine_invariance(self) -> None:
        config = load_config()
        y, x = np.mgrid[: SHAPE[0], : SHAPE[1]]
        raw = (1.0 + y / 100.0 + x / 500.0).astype(np.float32)
        health_a, p_a = depth_health_and_proximity(
            raw, direction="RAW_LARGER_IS_NEARER", config=config
        )
        health_b, p_b = depth_health_and_proximity(
            raw * 3.25 + 7.0,
            direction="RAW_LARGER_IS_NEARER",
            config=config,
        )
        self.assertEqual(health_a["q"], 1)
        self.assertEqual(health_b["q"], 1)
        np.testing.assert_allclose(p_a, p_b, atol=2e-6, rtol=0)

    def test_q_failures_remain_zero_score(self) -> None:
        config = load_config()
        health, proximity = depth_health_and_proximity(
            np.ones(SHAPE, dtype=np.float32),
            direction="RAW_LARGER_IS_NEARER",
            config=config,
        )
        self.assertEqual(health["q"], 0)
        scores = structural_scores(
            proximity,
            q=health["q"],
            yolo_mask=np.zeros(SHAPE, dtype=bool),
            config=config,
        )
        for arm in ("D1", "D2", "D3", "D4", "D5"):
            self.assertEqual(float(np.max(scores[arm])), 0.0)

    def test_signals_are_bounded_and_d4_is_exact_equal_weight(self) -> None:
        config = load_config()
        rng = np.random.default_rng(20260801)
        p = rng.random(SHAPE, dtype=np.float32)
        yolo = np.zeros(SHAPE, dtype=bool)
        yolo[30:50, 40:70] = True
        scores = structural_scores(p, q=1, yolo_mask=yolo, config=config)
        for name in ("N", "E", "R_plus", "R_minus"):
            self.assertGreaterEqual(float(np.min(scores[name])), 0.0)
            self.assertLessEqual(float(np.max(scores[name])), 1.0)
        expected = (
            scores["N"]
            + scores["E"]
            + scores["R_plus"]
            + scores["R_minus"]
        ) / 4.0
        expected[yolo] = 0.0
        np.testing.assert_allclose(scores["D4"], expected, atol=2e-7)
        self.assertTrue(np.all(scores["D4"][yolo] == 0))

    def test_d5_only_applies_frozen_proxy_prior(self) -> None:
        config = load_config()
        rng = np.random.default_rng(11)
        p = rng.random(SHAPE, dtype=np.float32)
        scores = structural_scores(
            p,
            q=1,
            yolo_mask=np.zeros(SHAPE, dtype=bool),
            config=config,
        )
        proxy = build_proxy_mask(SHAPE, config)
        np.testing.assert_allclose(
            scores["D5"][proxy], scores["D4"][proxy], atol=1e-7
        )
        np.testing.assert_allclose(
            scores["D5"][~proxy], scores["D4"][~proxy] * 0.25, atol=1e-7
        )

    def test_tie_group_average_precision(self) -> None:
        truth = np.array([1, 0, 1, 0], dtype=np.uint8)
        score = np.array([0.9, 0.9, 0.2, 0.1], dtype=np.float64)
        # First tied group contributes .5 recall at .5 precision; second
        # positive contributes .5 recall at 2/3 precision.
        expected = 0.5 * 0.5 + 0.5 * (2.0 / 3.0)
        self.assertAlmostEqual(
            average_precision_tie_group(truth, score), expected
        )

    def test_maximin_threshold_tie_breakers(self) -> None:
        base = {
            "minimum_normalized_gate_margin": -0.2,
            "utility_values": {
                "minimum_group_residual_recall_retention_vs_B": 0.8,
                "fp_pixel_reduction_vs_B": 0.4,
            },
        }
        candidates = [
            {**copy.deepcopy(base), "threshold": 0.4},
            {**copy.deepcopy(base), "threshold": 0.3},
        ]
        self.assertEqual(_choose_threshold(candidates)["threshold"], 0.3)

    def test_gate_report_uses_fixed_normalized_margins(self) -> None:
        config = load_config()
        values = {
            "fp_pixel_reduction_vs_B": 0.30,
            "overall_residual_recall_retention_vs_B": 0.90,
            "minimum_group_residual_recall_retention_vs_B": 0.80,
            "boundary_step_curb_recall_retention_vs_B": 0.80,
            "obstacle_recall_retention_vs_B": 0.80,
            "delta_recall_C_minus_A": 0.05,
            "delta_false_positive_area_fraction_C_minus_A": 0.05,
            "residual_truth_component_recall": 0.50,
            "false_activation_components_per_frame": 3.0,
        }
        report, minimum = gate_report(values, config)
        self.assertEqual(minimum, 0.0)
        self.assertTrue(all(row["passed"] for row in report.values()))


if __name__ == "__main__":
    unittest.main()
