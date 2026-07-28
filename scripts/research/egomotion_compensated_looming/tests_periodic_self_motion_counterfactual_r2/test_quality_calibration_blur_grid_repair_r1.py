from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    quality_calibration_blur_grid_repair_r1 as producer,
)
from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_quality_calibration_blur_grid_repair_independent_r1 as validator,
)


class BlurGridRepairR1Test(unittest.TestCase):
    def test_one_shot_grid_and_counts_are_exact(self) -> None:
        expected = (
            0.35,
            0.40,
            0.425,
            0.45,
            0.475,
            0.50,
            0.55,
            0.60,
            0.65,
        )
        self.assertEqual(producer.SIGMA_CANDIDATES, expected)
        self.assertEqual(validator.SIGMA_CANDIDATES, expected)
        self.assertEqual(producer.EXPECTED_ROWS, 5120)
        self.assertEqual(validator.EXPECTED_ROWS, 5120)

    def test_contract_precedes_access_and_forbids_second_repair(self) -> None:
        contract = json.loads(
            producer.CONTRACT_PATH.read_text(encoding="utf-8")
        )
        self.assertEqual(
            contract["status"],
            "R1_GRID_FROZEN_BEFORE_NEW_CAL_METRIC_ACCESS",
        )
        self.assertFalse(
            contract["authorization"]["automatic_second_repair"]
        )
        self.assertFalse(contract["authorization"]["p3_authorized"])
        self.assertEqual(
            contract["frozen_repair"]["calibration_panel"][
                "candidate_image_evaluations"
            ],
            5120,
        )

    def test_predecessor_chain_and_contract_validate(self) -> None:
        reads: list[dict[str, str]] = []
        contract = producer._read_json(producer.CONTRACT_PATH, reads)
        r0_lock = producer._read_json(producer.R0_LOCK_PATH, reads)
        receipt = producer._read_json(producer.R0_RECEIPT_PATH, reads)
        producer._verify_contract(contract, r0_lock, receipt)
        self.assertEqual(
            r0_lock["selected_global_strengths"]["low_texture_alpha"],
            0.15,
        )
        self.assertEqual(
            receipt["evidence_sha256"]["response_blind_metric_ledger"],
            producer.R0_LEDGER_SHA256,
        )

    def test_fixed_hashes_are_current(self) -> None:
        errors: list[str] = []
        validator._validate_fixed_hashes(errors)
        self.assertEqual(errors, [])

    def test_validator_does_not_import_r1_producer_or_quality(self) -> None:
        tree = ast.parse(Path(validator.__file__).read_text(encoding="utf-8"))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        self.assertFalse(
            any("quality_calibration_blur_grid_repair_r1" in item for item in imports)
        )
        self.assertFalse(
            any("quality_interventions_r0" in item for item in imports)
        )

    def test_smallest_passing_sigma_selection_direction(self) -> None:
        passes = {
            0.35: False,
            0.40: False,
            0.425: True,
            0.45: True,
            0.475: True,
            0.50: False,
            0.55: False,
            0.60: False,
            0.65: False,
        }
        selected = next(
            sigma
            for sigma in producer.SIGMA_CANDIDATES
            if passes[sigma]
        )
        self.assertEqual(selected, 0.425)

    def test_overall_and_each_subgroup_must_pass(self) -> None:
        summary = {
            "overall_median": {
                "laplacian_variance_ratio": 0.45,
                "local_rms_contrast_ratio": 0.80,
            },
            "subgroup_medians": [
                {
                    "metrics": {
                        "laplacian_variance_ratio": 0.45,
                        "local_rms_contrast_ratio": 0.80,
                    }
                }
                for _ in range(8)
            ],
        }
        self.assertTrue(validator.base._passes_blur(summary))
        summary["subgroup_medians"][3]["metrics"][
            "local_rms_contrast_ratio"
        ] = 0.699
        self.assertFalse(validator.base._passes_blur(summary))

    def test_nonfinite_remains_invalid(self) -> None:
        self.assertFalse(validator.base._finite_tree({"ratio": math.inf}))
        self.assertFalse(validator.base._finite_tree({"ratio": math.nan}))

    def test_r1_producer_import_firewall(self) -> None:
        audit = producer.r0._validate_import_firewall(
            producer.IMPLEMENTATION_PATH
        )
        self.assertEqual(audit["forbidden_imports"], [])

    def test_receipt_write_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            validator.base._exclusive_write(path, b"{}\n")
            with self.assertRaises(FileExistsError):
                validator.base._exclusive_write(path, b"{}\n")


if __name__ == "__main__":
    unittest.main()
