from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    generator_geometry as p1,
)
from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    quality_calibration_r0 as producer,
)
from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    quality_interventions_r0 as quality,
)


class QualityInterventionR0Test(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = quality.add_calibration_plate(
            p1.build_scene("ADVIO_13", 0, "CAL")
        )
        self.clean = quality.render_calibration_frame(
            self.scene,
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )

    def test_frozen_candidate_grid_and_panel_identity(self) -> None:
        self.assertEqual(
            producer.BLUR_CANDIDATES,
            (0.75, 1.0, 1.25, 1.5, 2.0, 2.5),
        )
        self.assertEqual(
            producer.LOW_TEXTURE_CANDIDATES,
            (0.75, 0.60, 0.45, 0.30, 0.15),
        )
        self.assertEqual(len(producer.BLOCKS), 4)
        self.assertEqual(len(producer.CAL_ORDINALS), 4)
        self.assertEqual(len(producer.MOTIONS), 2)
        self.assertEqual(len(producer.FRAME_POSITIONS), 16)
        self.assertEqual(producer.EXPECTED_LEDGER_ROWS, 6144)

    def test_linear_srgb_round_trip_is_exact_for_uint8(self) -> None:
        values = np.arange(256, dtype=np.uint8).reshape(16, 16, 1)
        rgb = np.repeat(values, 3, axis=2)
        rebuilt = quality.linear_to_srgb_u8(
            quality.srgb_u8_to_linear(rgb)
        )
        np.testing.assert_array_equal(rebuilt, rgb)

    def test_blur_is_linear_rgb_psf_and_does_not_mutate_clean(self) -> None:
        before = self.clean["rgb"].copy()
        blurred = quality.apply_blur(self.clean["rgb"], 0.75)
        np.testing.assert_array_equal(self.clean["rgb"], before)
        self.assertEqual(blurred.dtype, np.uint8)
        self.assertEqual(blurred.shape, before.shape)
        self.assertFalse(np.array_equal(blurred, before))
        metrics = quality.blur_frame_metrics(
            quality.prepare_clean_frame_metrics(self.clean),
            blurred,
        )
        self.assertTrue(
            math.isfinite(metrics["laplacian_variance_ratio"])
        )
        self.assertTrue(
            math.isfinite(metrics["local_rms_contrast_ratio"])
        )

    def test_low_texture_preserves_geometry_and_has_no_psf(self) -> None:
        prepared = quality.prepare_clean_frame_metrics(self.clean)
        degraded = quality.apply_low_texture(self.clean, 0.30)
        metrics = quality.low_texture_frame_metrics(
            self.clean,
            prepared,
            degraded,
        )
        self.assertEqual(
            self.clean["geometry_identity"]["valid_mask_sha256"],
            quality.sha256_bytes(
                self.clean["valid_mask"].astype(np.uint8).tobytes()
            ),
        )
        self.assertGreaterEqual(metrics["edge_spread_ratio"], 0.90)
        self.assertLessEqual(metrics["edge_spread_ratio"], 1.10)
        self.assertLess(
            metrics["multiscale_gradient_density_ratio"],
            1.0,
        )

    def test_analytic_fixture_has_32_edges_and_preserves_spread(self) -> None:
        for alpha in producer.LOW_TEXTURE_CANDIDATES:
            clean, degraded = quality.analytic_edge_fixture(alpha)
            self.assertEqual(len(clean["edges"]), 32)
            clean_width, clean_count = quality.source_known_edge_spread(
                quality.linear_luminance(clean["rgb"]),
                clean["object_id"],
                clean["edges"],
            )
            degraded_width, degraded_count = (
                quality.source_known_edge_spread(
                    quality.linear_luminance(degraded["rgb"]),
                    degraded["object_id"],
                    degraded["edges"],
                )
            )
            self.assertEqual(clean_count, 32)
            self.assertEqual(degraded_count, 32)
            self.assertAlmostEqual(degraded_width / clean_width, 1.0, places=12)

    def test_zero_denominator_and_nonfinite_are_invalid(self) -> None:
        with self.assertRaises(quality.InvalidQualityMetric):
            quality.safe_ratio(1.0, 0.0, "TEST")
        with self.assertRaises(quality.InvalidQualityMetric):
            quality.average_rank_median([1.0, float("nan")])

    def test_sequence_then_subgroup_then_overall_hierarchy(self) -> None:
        values = {}
        for block_index, block in enumerate(producer.BLOCKS):
            for ordinal in producer.CAL_ORDINALS:
                for motion_index, motion in enumerate(producer.MOTIONS):
                    base = float(block_index + ordinal + motion_index)
                    values[(block, ordinal, motion)] = {
                        "metric": [base] * 16,
                    }
        summary = producer._summarize_candidate(values, ("metric",))
        self.assertEqual(len(summary["sequence_medians"]), 32)
        self.assertEqual(len(summary["subgroup_medians"]), 8)
        self.assertEqual(
            summary["overall_median"]["metric"],
            np.median(
                [
                    row["metrics"]["metric"]
                    for row in summary["sequence_medians"]
                ]
            ),
        )

    def test_import_firewall_rejects_algorithm_runner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.py"
            path.write_text(
                "from scripts.research.egomotion_compensated_looming."
                "ecological_response_discovery_r0 import runner\n",
                encoding="utf-8",
            )
            with self.assertRaises(producer.CalibrationIdentityError):
                producer._validate_import_firewall(path)

    def test_calibration_sources_do_not_import_forbidden_modules(self) -> None:
        for path in (producer.INTERVENTION_PATH, producer.IMPLEMENTATION_PATH):
            audit = producer._validate_import_firewall(path)
            self.assertEqual(audit["forbidden_imports"], [])

    def test_contract_and_budget_exact_freeze_validate(self) -> None:
        read_ledger: list[dict[str, str]] = []
        contract = producer._read_json(
            producer.CONTRACT_PATH,
            read_ledger,
        )
        budget = producer._read_json(
            producer.BUDGET_PATH,
            read_ledger,
        )
        lock = producer._read_json(
            producer.P1_LOCK_PATH,
            read_ledger,
        )
        receipt = producer._read_json(
            producer.P1_RECEIPT_PATH,
            read_ledger,
        )
        producer._verify_freeze(contract, budget, lock, receipt)
        self.assertEqual(len(read_ledger), 4)

    def test_lock_serializer_rejects_nan(self) -> None:
        with self.assertRaises(ValueError):
            quality.canonical_bytes({"metric": float("nan")})

    def test_source_ast_has_no_write_to_algorithm_or_p3_paths(self) -> None:
        tree = ast.parse(
            producer.IMPLEMENTATION_PATH.read_text(encoding="utf-8")
        )
        string_literals = {
            node.value.lower()
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        self.assertFalse(
            any("sequence16" in value and "path" in value for value in string_literals)
        )
        self.assertFalse(any("cotracker/" in value for value in string_literals))

    def test_clean_row_shape_can_be_canonicalized_without_private_arrays(self) -> None:
        prepared = quality.prepare_clean_frame_metrics(self.clean)
        public = {
            "metrics": {
                "laplacian_variance": prepared["laplacian_variance"],
                "local_rms_contrast": prepared["local_rms_contrast"],
                "multiscale_gradient_density": prepared[
                    "multiscale_gradient_density"
                ],
                "edge_spread_px": prepared["edge_spread_px"],
            }
        }
        parsed = json.loads(quality.canonical_bytes(public))
        self.assertEqual(set(parsed), {"metrics"})


if __name__ == "__main__":
    unittest.main()
