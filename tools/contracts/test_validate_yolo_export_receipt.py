"""Schema-shaped fixtures and cross-field canaries for YOLO export receipt v1."""

import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PATH = ROOT / "validate_yolo_export_receipt.py"
SPEC = importlib.util.spec_from_file_location("yolo_receipt", PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)
FIXTURES = ROOT / "fixtures" / "yolo-export-receipt-v1"


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

    def test_schema_rejects_missing_and_extra_top_level_fields(self):
        missing = self.load("positive.json")
        del missing["receipt_id"]
        self.rejected(missing)
        extra = self.load("positive.json")
        extra["unexpected"] = True
        self.rejected(extra)

    def test_schema_rejects_missing_and_extra_nested_fields(self):
        missing = self.load("positive.json")
        del missing["artifacts"]["labels"]["sha256"]
        self.rejected(missing)
        extra = self.load("positive.json")
        extra["equivalence"]["outputs"][0]["unexpected"] = 1
        self.rejected(extra)

    def test_checkpoint_identity_mismatch_is_fail_closed(self):
        receipt = self.load("positive.json")
        receipt["artifacts"]["checkpoint"]["sha256"] = "a" * 64
        self.assertEqual(
            {"decision": "FAIL", "reason_codes": ["CHECKPOINT_SHA256_MISMATCH"]},
            module.derive(receipt)["byte_reproducibility"],
        )
        self.rejected(receipt)

    def test_missing_labels_identity_cannot_claim_byte_pass(self):
        receipt = self.load("positive.json")
        receipt["artifacts"]["labels"]["sha256"] = None
        self.assertEqual("UNKNOWN", module.derive(receipt)["byte_reproducibility"]["decision"])
        self.rejected(receipt)

    def test_export_parameters_are_exact_and_non_empty(self):
        empty = self.load("positive.json")
        empty["parameters"] = {}
        self.rejected(empty)
        extra = self.load("positive.json")
        extra["parameters"]["unfrozen_option"] = True
        self.rejected(extra)

    def test_repository_identity_uses_url_and_git_object_id(self):
        missing_url = self.load("positive.json")
        missing_url["exporter"]["repository_url"] = None
        self.rejected(missing_url)
        invalid_url = self.load("positive.json")
        invalid_url["exporter"]["repository_url"] = "not-a-url"
        self.rejected(invalid_url)
        wrong_commit = self.load("positive.json")
        wrong_commit["exporter"]["repository_commit"] = "3" * 64 + "00"
        self.rejected(wrong_commit)

    def test_negative_absolute_error_is_rejected(self):
        receipt = self.load("positive.json")
        receipt["equivalence"]["outputs"][0]["max_abs_error"] = -0.001
        self.rejected(receipt)

    def test_numerical_outputs_bind_exactly_to_inspected_outputs(self):
        unrelated = self.load("positive.json")
        unrelated["equivalence"]["outputs"][0]["name"] = "not-a-model-output"
        self.assertEqual(
            "OUTPUT_IDENTITY_MISMATCH",
            module.derive(unrelated)["numerical_equivalence"]["reason_codes"][0],
        )
        self.rejected(unrelated)

        duplicate = self.load("positive.json")
        duplicate["equivalence"]["outputs"].append(copy.deepcopy(duplicate["equivalence"]["outputs"][0]))
        self.rejected(duplicate)

        missing = self.load("positive.json")
        missing["equivalence"]["outputs"] = []
        self.rejected(missing)

    def test_boolean_denominator_is_not_an_integer(self):
        receipt = self.load("positive.json")
        receipt["equivalence"]["denominator"] = True
        self.rejected(receipt)

    def test_receipt_never_grants_downstream_authority(self):
        receipt = self.load("positive.json")
        receipt["authority"]["safety"] = True
        self.rejected(receipt)

    def test_receipt_cannot_choose_its_own_asset_pin(self):
        receipt = self.load("positive.json")
        receipt["artifacts"]["tflite"]["sha256"] = "a" * 64
        receipt["artifacts"]["pinned_tflite_sha256"] = "a" * 64
        self.assertEqual("PINNED_ASSET_IDENTITY_MISMATCH", module.derive(receipt)["byte_reproducibility"]["reason_codes"][0])
        self.rejected(receipt)

    def test_complete_source_environment_artifact_evidence_is_required(self):
        mutations = [
            ("source", "checkpoint_url", None), ("source", "upstream_project", None),
            ("exporter", "packages", []), ("exporter", "lock_path", None),
        ]
        for section, key, value in mutations:
            with self.subTest(section=section, key=key):
                receipt = self.load("positive.json")
                receipt[section][key] = value
                self.assertEqual("UNKNOWN", module.derive(receipt)["byte_reproducibility"]["decision"])
                self.rejected(receipt)
        for artifact in ("checkpoint", "labels", "tflite"):
            for key, value in (("path", None), ("path", "  "), ("size_bytes", None), ("size_bytes", 0)):
                with self.subTest(artifact=artifact, key=key, value=value):
                    receipt = self.load("positive.json")
                    receipt["artifacts"][artifact][key] = value
                    self.assertEqual("UNKNOWN", module.derive(receipt)["byte_reproducibility"]["decision"])
                    self.rejected(receipt)

    def test_parameters_bind_batch_spatial_shape_and_dtype(self):
        for key, value in (("imgsz", 640), ("batch", 2)):
            with self.subTest(key=key):
                receipt = self.load("positive.json")
                receipt["parameters"][key] = value
                self.assertEqual("PARAMETER_TENSOR_MISMATCH", module.derive(receipt)["structural_equivalence"]["reason_codes"][0])
                self.rejected(receipt)
        receipt = self.load("positive.json")
        for side in ("expected", "observed"):
            receipt["inspection"][side][0]["dtype"] = "float16"
        self.rejected(receipt)

    def test_export_contract_hashes_and_flags_are_required(self):
        for key in ("preprocess_contract", "postprocess_contract", "metadata_sha256"):
            with self.subTest(key=key):
                receipt = self.load("positive.json")
                receipt["parameters"][key] = None
                self.assertEqual("UNKNOWN", module.derive(receipt)["structural_equivalence"]["decision"])
                self.rejected(receipt)
        for key, value in (("half", False), ("nms", True), ("int8", True), ("dynamic", True), ("input_layout", "NCHW")):
            with self.subTest(key=key):
                receipt = self.load("positive.json")
                receipt["parameters"][key] = value
                self.rejected(receipt)

    def test_equal_malformed_tensor_contracts_do_not_pass(self):
        for key, value in (("name", ""), ("name", "  "), ("dtype", ""), ("shape", []), ("shape", [0]), ("shape", [-1]), ("quantization", {})):
            with self.subTest(key=key, value=value):
                receipt = self.load("positive.json")
                for side in ("expected", "observed"):
                    receipt["inspection"][side][0][key] = value
                self.rejected(receipt)

    def test_tensor_inventory_requires_unique_single_input_and_output(self):
        for operation in ("duplicate", "no_input", "no_output"):
            with self.subTest(operation=operation):
                receipt = self.load("positive.json")
                for side in ("expected", "observed"):
                    tensors = receipt["inspection"][side]
                    if operation == "duplicate": tensors.append(copy.deepcopy(tensors[0]))
                    elif operation == "no_input": tensors.pop(0)
                    else: tensors.pop(1)
                self.assertEqual("FAIL", module.derive(receipt)["structural_equivalence"]["decision"])
                self.rejected(receipt)

    def test_known_byte_contradiction_is_not_hidden_by_missing_evidence(self):
        receipt = self.load("positive.json")
        receipt["exporter"]["packages"] = []
        receipt["artifacts"]["tflite"]["sha256"] = "a" * 64
        self.assertEqual("FAIL", module.derive(receipt)["byte_reproducibility"]["decision"])

    def test_authority_zero_is_not_boolean_false(self):
        receipt = self.load("positive.json")
        receipt["authority"]["safety"] = 0
        self.rejected(receipt)

    def test_observed_output_names_also_bind_numerical_evidence(self):
        receipt = self.load("positive.json")
        receipt["inspection"]["observed"][1]["name"] = "different"
        self.assertEqual("FAIL", module.derive(receipt)["numerical_equivalence"]["decision"])

    def test_missing_external_pin_is_invalid(self):
        from unittest.mock import patch
        with patch.object(module, "PUBLIC_ASSETS_PATH", Path("does-not-exist/assets.json")):
            self.rejected(self.load("positive.json"))

    def test_conflicting_package_versions_are_rejected(self):
        receipt = self.load("positive.json")
        receipt["exporter"]["packages"] = ["Torch==2.14", "torch==2.13"]
        self.rejected(receipt)

    def test_versioned_package_inventory_is_required(self):
        receipt = self.load("positive.json")
        receipt["exporter"]["packages"] = ["torch"]
        self.rejected(receipt)


if __name__ == "__main__":
    unittest.main()
