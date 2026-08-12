"""Schema-shaped fixtures and cross-field canaries for YOLO export receipt v1."""

import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "validate_yolo_export_receipt.py"
SPEC = importlib.util.spec_from_file_location("yolo_receipt", PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
FIXTURES = ROOT / "schemas" / "fixtures" / "yolo-export-receipt-v1"


class YoloExportReceiptTest(unittest.TestCase):
    def load(self, name):
        return json.loads((FIXTURES / name).read_text(encoding="utf-8"))

    def rejected(self, receipt):
        with self.assertRaises(module.ValidationError):
            module.validate(receipt)

    def test_positive_negative_and_unknown_fixtures(self):
        for name in ("positive.json", "negative.json", "unknown.json"):
            module.validate(self.load(name))

    def test_missing_checkpoint_identity_cannot_claim_pass(self):
        receipt = self.load("positive.json")
        receipt["source"]["checkpoint_sha256"] = None
        self.rejected(receipt)

    def test_missing_environment_cannot_claim_pass(self):
        receipt = self.load("positive.json")
        receipt["exporter"]["lock_sha256"] = None
        self.rejected(receipt)

    def test_tensor_mismatch_cannot_claim_pass(self):
        receipt = self.load("positive.json")
        receipt["inspection"]["observed"][1]["shape"] = [1, 85, 2100]
        self.rejected(receipt)

    def test_missing_fixture_denominator_cannot_claim_pass(self):
        receipt = self.load("positive.json")
        receipt["equivalence"]["denominator"] = None
        self.rejected(receipt)

    def test_non_finite_output_cannot_claim_pass(self):
        receipt = self.load("positive.json")
        receipt["equivalence"]["outputs"][0]["max_abs_error"] = float("nan")
        self.rejected(receipt)

    def test_numerical_pass_cannot_rescue_byte_mismatch(self):
        receipt = self.load("positive.json")
        receipt["artifacts"]["tflite"]["sha256"] = "a" * 64
        receipt["verdicts"]["byte_reproducibility"] = {
            "decision": "FAIL",
            "reason_codes": ["TFLITE_SHA256_MISMATCH"],
        }
        module.validate(receipt)
        self.assertEqual("PASS", receipt["verdicts"]["numerical_equivalence"]["decision"])

    def test_receipt_never_grants_downstream_authority(self):
        receipt = self.load("positive.json")
        receipt["authority"]["safety"] = True
        self.rejected(receipt)


if __name__ == "__main__":
    unittest.main()

