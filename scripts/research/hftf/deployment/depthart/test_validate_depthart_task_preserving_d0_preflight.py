from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.research.hftf.deployment.depthart.validate_depthart_task_preserving_d0_preflight import (
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[5]
PROTOCOL_PATH = ROOT / "docs/research/hftf/DEPTHART_TASK_PRESERVING_D0_PRECISION_SCREEN_PROTOCOL_2026-08-09.json"
SOURCE_PATH = ROOT / "docs/research/hftf/DEPTHART_TASK_PRESERVING_D0_SOURCE_CONTROL_LOCK_2026-08-09.json"


class ValidateDepthArtTaskPreservingD0PreflightTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        self.source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))

    def test_frozen_contract_passes(self) -> None:
        receipt = validate_contract(self.protocol, self.source)
        self.assertTrue(receipt["contract_valid"])
        self.assertEqual(set(receipt["arms"]), {"D0_FP16_R0", "D0_W8A16_R0", "D0_INT8_R0"})

    def test_r2_cohort_selection_drift_fails(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["selection"]["r2_cohort_access_during_selection"] = True
        with self.assertRaisesRegex(ValueError, "R2 cohort"):
            validate_contract(protocol, self.source)

    def test_quantization_recipe_drift_fails(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        arm = next(item for item in protocol["arms"] if item["arm_id"] == "D0_W8A16_R0")
        index = arm["quantizer_args"].index("--act_bitwidth")
        arm["quantizer_args"][index + 1] = "8"
        with self.assertRaisesRegex(ValueError, "activation bitwidth drift"):
            validate_contract(protocol, self.source)


if __name__ == "__main__":
    unittest.main()
