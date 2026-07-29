from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.research.egomotion_compensated_looming.periodic_self_motion_counterfactual_r2 import (
    validate_qms_r1_independent as validator,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE = (
    REPO_ROOT
    / "artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/"
    "qms_r1/new_cal"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ValidateQmsR1IndependentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evidence = self.root / "new_cal"
        shutil.copytree(SOURCE, self.evidence)

    def _json(self, name: str):
        return json.loads((self.evidence / name).read_text(encoding="utf-8"))

    def _write_json(self, name: str, value) -> None:
        (self.evidence / name).write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def _ledger(self):
        return [
            json.loads(line)
            for line in (
                self.evidence / "response_blind_ledger.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]

    def _write_ledger(self, rows, refresh_binding: bool = True) -> None:
        ledger = self.evidence / "response_blind_ledger.jsonl"
        ledger.write_text(
            "".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )
        if refresh_binding:
            receipt = self._json("receipt.json")
            receipt["bindings"]["response_blind_ledger_sha256"] = _sha(ledger)
            self._write_json("receipt.json", receipt)

    def _write_identities(self, rows, refresh_binding: bool = True) -> None:
        self._write_json("identity_manifest.json", rows)
        if refresh_binding:
            receipt = self._json("receipt.json")
            receipt["bindings"]["identity_manifest_sha256"] = _sha(
                self.evidence / "identity_manifest.json"
            )
            self._write_json("receipt.json", receipt)

    def _assert_invalid(self, code: str) -> None:
        with self.assertRaisesRegex(
            validator.InvalidQmsR1Independent, code
        ):
            validator.validate(REPO_ROOT, self.evidence)

    def test_valid_new_cal_and_exclusive_receipt(self):
        result = validator.validate(REPO_ROOT, self.evidence)
        self.assertEqual(result["validation"], "VALID")
        self.assertEqual(result["counts"]["frame_states"], 512)
        target = self.root / "independent_receipt.json"
        validator.write_exclusive(target, result)
        self.assertTrue(target.is_file())
        with self.assertRaisesRegex(
            validator.InvalidQmsR1Independent, "OUTPUT_EXISTS"
        ):
            validator.write_exclusive(target, result)

    def test_bound_ledger_hash_mutation(self):
        ledger = self.evidence / "response_blind_ledger.jsonl"
        ledger.write_bytes(ledger.read_bytes() + b" ")
        self._assert_invalid("BINDING_RESPONSE_BLIND_LEDGER_SHA256")

    def test_frame_hash_field_mutation(self):
        rows = self._ledger()
        rows[0]["frame_rows"][0]["clean_rgb_sha256"] = "not-a-digest"
        self._write_ledger(rows)
        self._assert_invalid("FRAME_HASH_FIELD")

    def test_residual_ratio_mutation(self):
        rows = self._ledger()
        for frame in rows[0]["frame_rows"]:
            frame["material_residual_ratio"] = 0.25
        rows[0]["sequence_medians"]["material_residual_ratio"] = 0.25
        self._write_ledger(rows)
        self._assert_invalid("MATERIAL_RESIDUAL_GATE")

    def test_seed_mutation(self):
        identities = self._json("identity_manifest.json")
        identities[0]["numeric_seed_uint64"] += 1
        self._write_identities(identities)
        self._assert_invalid("IDENTITY_TOKEN_OR_SEED")

    def test_frame_order_mutation(self):
        rows = self._ledger()
        rows[0]["frame_rows"][0], rows[0]["frame_rows"][1] = (
            rows[0]["frame_rows"][1],
            rows[0]["frame_rows"][0],
        )
        self._write_ledger(rows)
        self._assert_invalid("FRAME_ORDER")

    def test_subgroup_mutation(self):
        receipt = self._json("receipt.json")
        receipt["subgroups"][0]["pass_count"] = 3
        receipt["subgroups"][0]["subgroup_pass"] = False
        self._write_json("receipt.json", receipt)
        self._assert_invalid("SUBGROUP_RECEIPT")

    def test_prequantization_mutation(self):
        rows = self._ledger()
        rows[0]["frame_rows"][0][
            "prequantization_residual_max_abs_error"
        ] = 1e-12
        self._write_ledger(rows)
        self._assert_invalid("PREQUANTIZATION_IDENTITY")

    def test_source_hash_mutation(self):
        receipt = self._json("receipt.json")
        receipt["bindings"]["operator_sha256"] = "0" * 64
        self._write_json("receipt.json", receipt)
        self._assert_invalid("BINDING_OPERATOR_SHA256")

    def test_firewall_mutation(self):
        receipt = self._json("receipt.json")
        receipt["firewall"]["r3_outcome_read"] = True
        self._write_json("receipt.json", receipt)
        self._assert_invalid("RECEIPT_FIREWALL")

    def test_validator_has_no_forbidden_project_imports(self):
        source = Path(validator.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        forbidden = (
            "qms_r1_qualification",
            "material_residual_contraction_r1",
            "p4_",
            "p3_transport_r0",
        )
        self.assertFalse(
            any(token in module for module in imported for token in forbidden)
        )


if __name__ == "__main__":
    unittest.main()
