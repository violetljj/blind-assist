from __future__ import annotations

import copy
from pathlib import Path
import unittest

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    p4_formal_independent_validator_r0 as validator,
)
from scripts.research.egomotion_compensated_looming.tests_periodic_self_motion_counterfactual_r2.test_p4_formal_analysis_r0 import (
    FormalFixture,
)


class FormalIndependentValidatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = FormalFixture()
        cls.result = cls.fixture.assemble()

    def validate(self, result=None):
        return validator.validate_loaded_result(
            self.result if result is None else result,
            "6" * 64,
            self.fixture.bundle,
            self.fixture.identity_lock,
            lambda path: self.fixture.receipts[path],
            lambda path: self.fixture.receipt_hashes[path],
            self.fixture.ledger,
            lambda path: self.fixture.ledger_hashes[path],
            bundle_sha256="5" * 64,
            identity_lock_sha256=self.fixture.identity_sha,
        )

    def test_independent_full_recomputation_validates(self) -> None:
        receipt = self.validate()
        self.assertTrue(receipt["validated"])
        self.assertEqual([], receipt["errors"])
        self.assertEqual(
            "FORMAL_RESULT_VALID / SCIENTIFIC_TERMINAL_SIGNED",
            receipt["terminal"],
        )
        self.assertFalse(
            receipt["independence"]["formal_assembler_imported"]
        )

    def test_forged_terminal_and_analysis_are_rejected(self) -> None:
        result = copy.deepcopy(self.result)
        result["scientific_terminal"] = "MOTION_SUPPORTED"
        result["analysis"]["estimands"]["MOTION_CLEAN"]["theta"] = 0.5
        receipt = self.validate(result)
        self.assertFalse(receipt["validated"])
        self.assertIn(
            "RESULT_MISMATCH:scientific_terminal", receipt["errors"]
        )
        self.assertIn("RESULT_MISMATCH:analysis", receipt["errors"])

    def test_forged_closure_is_rejected(self) -> None:
        result = copy.deepcopy(self.result)
        result["closure"]["atomic_receipt_count"] = 495
        receipt = self.validate(result)
        self.assertFalse(receipt["validated"])
        self.assertIn("RESULT_CLOSURE", receipt["errors"])

    def test_validator_has_no_producer_or_analysis_import(self) -> None:
        source = Path(validator.__file__).read_text(encoding="utf-8")
        forbidden = (
            "import p4_formal_analysis_r0",
            "import p3_analysis_r0",
            "import p4_formal_runner",
            "import p3_transport_r0",
            "import rgb_algorithm_development_canary",
        )
        for token in forbidden:
            with self.subTest(token=token):
                self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
