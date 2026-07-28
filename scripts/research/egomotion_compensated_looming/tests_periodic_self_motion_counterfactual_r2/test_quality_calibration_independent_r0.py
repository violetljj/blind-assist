from __future__ import annotations

import ast
import math
from pathlib import Path
import tempfile
import unittest

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_quality_calibration_independent_r0 as validator,
)


def _summary(laplacian: float, rms: float, gradient: float, edge: float):
    metrics = {
        "laplacian_variance_ratio": laplacian,
        "local_rms_contrast_ratio": rms,
        "multiscale_gradient_density_ratio": gradient,
        "edge_spread_ratio": edge,
    }
    return {
        "overall_median": dict(metrics),
        "subgroup_medians": [
            {"metrics": dict(metrics)}
            for _ in range(8)
        ],
    }


class IndependentQualityValidationR0Test(unittest.TestCase):
    def test_validator_does_not_import_producer_or_intervention(self) -> None:
        source = Path(validator.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertFalse(
            any("quality_calibration_r0" in name for name in imports)
        )
        self.assertFalse(
            any("quality_interventions_r0" in name for name in imports)
        )
        self.assertFalse(
            any("ecological_response_discovery" in name for name in imports)
        )

    def test_exact_frozen_grid_and_counts(self) -> None:
        self.assertEqual(
            validator.BLUR,
            (0.75, 1.0, 1.25, 1.5, 2.0, 2.5),
        )
        self.assertEqual(
            validator.LOW_TEXTURE,
            (0.75, 0.60, 0.45, 0.30, 0.15),
        )
        self.assertEqual(validator.EXPECTED_ROW_COUNT, 6144)
        self.assertEqual(len(validator.FRAMES), 16)

    def test_blur_gate_requires_overall_and_all_subgroups(self) -> None:
        passing = _summary(0.45, 0.75, 0.45, 1.0)
        self.assertTrue(validator._passes_blur(passing))
        passing["subgroup_medians"][7]["metrics"][
            "laplacian_variance_ratio"
        ] = 0.349999
        self.assertFalse(validator._passes_blur(passing))

    def test_low_texture_gate_requires_fixture_and_all_subgroups(self) -> None:
        passing = _summary(0.45, 0.75, 0.45, 1.0)
        self.assertTrue(validator._passes_low(passing, 1.0))
        self.assertFalse(validator._passes_low(passing, 1.100001))
        passing["subgroup_medians"][0]["metrics"][
            "multiscale_gradient_density_ratio"
        ] = 0.550001
        self.assertFalse(validator._passes_low(passing, 1.0))

    def test_nonfinite_is_invalid(self) -> None:
        self.assertFalse(validator._finite_tree({"metric": float("inf")}))
        self.assertFalse(validator._finite_tree({"metric": float("nan")}))
        with self.assertRaises(ValueError):
            validator._median([1.0, float("nan")])

    def test_forbidden_algorithm_key_is_rejected(self) -> None:
        errors: list[str] = []
        validator._validate_row_keys(
            {"metrics": {"trigger_density": 0.2}},
            errors,
        )
        self.assertEqual(errors, ["FORBIDDEN_LEDGER_KEY:trigger_density"])

    def test_raw_ratio_recalculation_detects_mutation(self) -> None:
        errors: list[str] = []
        validator._ratio(
            {
                "degraded": 1.0,
                "clean": 2.0,
                "ratio": 0.6,
            },
            "degraded",
            "clean",
            "ratio",
            errors,
            "fixture",
        )
        self.assertEqual(
            errors,
            ["ROW_RATIO_MISMATCH:fixture:ratio"],
        )

    def test_fixed_predecessor_hashes_are_current(self) -> None:
        errors: list[str] = []
        validator._validate_fixed_hashes(errors)
        self.assertEqual(errors, [])

    def test_receipt_write_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            validator._exclusive_write(path, b"{}\n")
            with self.assertRaises(FileExistsError):
                validator._exclusive_write(path, b"{}\n")

    def test_canonical_bytes_rejects_nan(self) -> None:
        with self.assertRaises(ValueError):
            validator.canonical_bytes({"value": math.nan})


if __name__ == "__main__":
    unittest.main()
