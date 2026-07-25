#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_r1_claim_scoped_source_program_r0 as subject  # noqa: E402


RECEIPT = (
    Path(__file__).resolve().parents[3]
    / "artifacts.local"
    / "evidence"
    / "ustrf"
    / "egomotion_compensated_looming_r1"
    / "r1_claim_scoped_source_program_r0.json"
)


class R1ClaimScopedSourceProgramTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))

    def test_current_receipt_is_valid(self) -> None:
        subject.validate(self.receipt)

    def test_same_bag_role_split_is_rejected(self) -> None:
        altered = copy.deepcopy(self.receipt)
        revel = next(
            item for item in altered["sources"] if item["source_family"] == "REVEL"
        )
        revel["same_bag_segmented_role_split_allowed"] = True
        with self.assertRaisesRegex(ValueError, "same-bag"):
            subject.validate(altered)

    def test_incomplete_bonn_denylist_is_rejected(self) -> None:
        altered = copy.deepcopy(self.receipt)
        bonn = next(
            item
            for item in altered["sources"]
            if item["source_family"] == "BONN_RGBD_DYNAMIC"
        )
        bonn["prior_inspected_units"].pop()
        with self.assertRaisesRegex(ValueError, "denylist"):
            subject.validate(altered)

    def test_bonn_receipt_hash_mismatch_is_rejected(self) -> None:
        altered = copy.deepcopy(self.receipt)
        bonn = next(
            item
            for item in altered["sources"]
            if item["source_family"] == "BONN_RGBD_DYNAMIC"
        )
        bonn["claim_scoped_role_freeze_receipt"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash"):
            subject.validate(altered)

    def test_authoritative_result_is_rejected(self) -> None:
        altered = copy.deepcopy(self.receipt)
        altered["hard_boundaries"][
            "authoritative_algorithm_result_available"
        ] = True
        with self.assertRaisesRegex(ValueError, "authority"):
            subject.validate(altered)

    def test_nonauthoritative_evaluation_hash_mismatch_is_rejected(self) -> None:
        altered = copy.deepcopy(self.receipt)
        bonn = next(
            item
            for item in altered["sources"]
            if item["source_family"] == "BONN_RGBD_DYNAMIC"
        )
        bonn["nonauthoritative_continuous_signal_evaluation_review"][
            "sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(ValueError, "evaluation review hash"):
            subject.validate(altered)


if __name__ == "__main__":
    unittest.main()
