from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_r3_1_inventory_candidates import _validate_source_pool


class R31InventoryCandidatePlanTest(unittest.TestCase):
    def test_frozen_split_and_unique_burns_are_required(self) -> None:
        split = "a\nb\n"
        import hashlib

        protocol = {
            "schema": "blindassist_hftf_stage_b_reference_only_opportunity_qualification_r3_1",
            "source_pool": {
                "split_object_generation": "7",
                "split_text_sha256": hashlib.sha256(
                    split.encode()
                ).hexdigest(),
            },
        }
        ledger = {
            "schema": "blindassist_hftf_r3_1_source_pool_burn_ledger",
            "burned_session_ids": ["a"],
            "burned_session_count": 1,
        }
        self.assertEqual(
            {"a"},
            _validate_source_pool(protocol, ledger, "7", split),
        )

    def test_split_generation_drift_fails_closed(self) -> None:
        protocol = {
            "schema": "blindassist_hftf_stage_b_reference_only_opportunity_qualification_r3_1",
            "source_pool": {
                "split_object_generation": "7",
                "split_text_sha256": "unused",
            },
        }
        ledger = {
            "schema": "blindassist_hftf_r3_1_source_pool_burn_ledger",
            "burned_session_ids": [],
            "burned_session_count": 0,
        }
        with self.assertRaisesRegex(ValueError, "generation drift"):
            _validate_source_pool(protocol, ledger, "8", "")


if __name__ == "__main__":
    unittest.main()
