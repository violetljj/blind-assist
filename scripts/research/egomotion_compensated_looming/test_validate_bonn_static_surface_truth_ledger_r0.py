#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_bonn_static_surface_truth_ledger_r0 as subject  # noqa: E402


class BonnStaticSurfaceTruthValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        root = Path(__file__).resolve().parents[3]
        self.receipt = json.loads(
            (
                root
                / "artifacts.local/evidence/ustrf/"
                "egomotion_compensated_looming_r1/"
                "bonn_static_surface_truth_ledger_recheck.json"
            ).read_text(encoding="utf-8")
        )
        self.points = (
            root
            / "artifacts.local/datasets/"
            "egomotion_compensated_looming_r1/bonn_discovery_r0/"
            "rgbd_bonn_groundtruth_hash64_r0.npz"
        )

    def test_current_receipt_validates(self) -> None:
        result = subject.validate(self.receipt, self.points)
        self.assertEqual("VALID", result["status"])

    def test_rgb_firewall_mutation_fails(self) -> None:
        mutated = copy.deepcopy(self.receipt)
        mutated["counts"]["rgb_member_read_or_decode_count"] = 1
        with self.assertRaisesRegex(ValueError, "RGB firewall"):
            subject.validate(mutated, self.points)

    def test_static_map_identity_mutation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.npz"
            changed.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "identity"):
                subject.validate(self.receipt, changed)


if __name__ == "__main__":
    unittest.main()
