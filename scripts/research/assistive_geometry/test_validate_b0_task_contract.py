from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from scripts.research.assistive_geometry.validate_b0_task_contract import validate_contract


ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TASK_CONTRACT_2026-08-09.json"


class ValidateB0TaskContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_frozen_contract_is_valid(self) -> None:
        validate_contract(copy.deepcopy(self.payload))

    def test_unknown_cannot_become_negative(self) -> None:
        mutated = copy.deepcopy(self.payload)
        mutated["unknown_contract"]["unknown_is_negative"] = True
        with self.assertRaisesRegex(ValueError, "UNKNOWN"):
            validate_contract(mutated)

    def test_student_training_cannot_be_authorized(self) -> None:
        mutated = copy.deepcopy(self.payload)
        mutated["authority"]["student_training"] = True
        with self.assertRaisesRegex(ValueError, "student_training"):
            validate_contract(mutated)

    def test_square_canary_cannot_replace_product_aspect_shape(self) -> None:
        mutated = copy.deepcopy(self.payload)
        mutated["input_geometry"]["resize"]["fixed_tensor_nchw"] = [1, 3, 448, 448]
        with self.assertRaisesRegex(ValueError, "tensor shape"):
            validate_contract(mutated)

    def test_consumed_or_r2_roster_cannot_be_reused(self) -> None:
        for key in ("consumed_120_frame_cohort_forbidden", "existing_arkitscenes_r2_roster_forbidden"):
            mutated = copy.deepcopy(self.payload)
            mutated["data_firewall"][key] = False
            with self.assertRaises(ValueError):
                validate_contract(mutated)


if __name__ == "__main__":
    unittest.main()
