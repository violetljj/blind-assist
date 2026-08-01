from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_stage_c_f0_1_sanpo_cross_split_inventory import (
    _validate_protocol_and_f0_plan,
    _validate_test_split,
)


class StageCF01CrossSplitInventoryPlanTest(unittest.TestCase):
    def test_test_split_generation_hash_count_and_uniqueness(self) -> None:
        text = "a\nb\nc\n"
        selection = {
            "split_object_generation": "7",
            "split_text_sha256": hashlib.sha256(
                text.encode()
            ).hexdigest(),
            "split_session_count": 3,
        }
        _validate_test_split(selection, "7", text)
        with self.assertRaisesRegex(ValueError, "generation drift"):
            _validate_test_split(selection, "8", text)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            duplicate = "a\na\nb\n"
            selection["split_text_sha256"] = hashlib.sha256(
                duplicate.encode()
            ).hexdigest()
            _validate_test_split(selection, "7", duplicate)

    def test_f0_metadata_firewall_and_first_nine_roles_are_required(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            docs = Path(temp)
            parent = docs / "f0.json"
            parent.write_text("f0", encoding="utf-8")
            parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
            parent_ledger = docs / "r4-ledger.json"
            parent_ledger.write_text(
                json.dumps(
                    {
                        "burned_session_ids": ["old"],
                    }
                ),
                encoding="utf-8",
            )
            ledger = docs / "ledger.json"
            ledger.write_text(
                json.dumps(
                    {
                        "schema": (
                            "blindassist_hftf_stage_c_sanpo_body_head_"
                            "source_pool_burn_ledger_f0"
                        ),
                        "status": "FROZEN_BEFORE_F0_SOURCE_OUTCOME",
                        "parent_r4_burn_ledger": {
                            "path": "r4-ledger.json"
                        },
                        "additional_r4_outcome_open_session_ids": [
                            "new"
                        ],
                    }
                ),
                encoding="utf-8",
            )
            ledger_sha = hashlib.sha256(ledger.read_bytes()).hexdigest()
            protocol = docs / "f0-1.json"
            protocol.write_text(
                json.dumps(
                    {
                        "schema": (
                            "blindassist_hftf_stage_c_sanpo_cross_split_"
                            "body_head_temporal_student_canary_f0_1"
                        ),
                        "status": (
                            "FROZEN_BEFORE_F0_1_SOURCE_OUTCOME"
                        ),
                        "parent_f0_protocol": {
                            "path": "f0.json",
                            "sha256": parent_sha,
                        },
                        "source_pool_burn_ledger": {
                            "path": "ledger.json",
                            "sha256": ledger_sha,
                            "effective_train_burned_session_count": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            candidates = [
                {
                    "inventory_eligible_rank": rank,
                    "role": (
                        "train"
                        if rank <= 6
                        else "dev"
                        if rank <= 9
                        else "heldout"
                    ),
                }
                for rank in range(1, 13)
            ]
            f0_plan = docs / "plan.json"
            f0_plan.write_text(
                json.dumps(
                    {
                        "schema": (
                            "blindassist_hftf_stage_c_f0_"
                            "sanpo_inventory_plan"
                        ),
                        "terminal": (
                            "F0_SANPO_FIXED_SOURCE_INVENTORY_READY"
                        ),
                        "protocol_sha256": parent_sha,
                        "burn_ledger_sha256": ledger_sha,
                        "geometry_outcome_read": False,
                        "teacher_outcome_read": False,
                        "student_outcome_read": False,
                        "inventory_candidates": candidates,
                    }
                ),
                encoding="utf-8",
            )
            _, _, burned = _validate_protocol_and_f0_plan(
                protocol, ledger, f0_plan
            )
            self.assertEqual({"old", "new"}, burned)
            payload = json.loads(f0_plan.read_text(encoding="utf-8"))
            payload["geometry_outcome_read"] = True
            f0_plan.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "firewall"):
                _validate_protocol_and_f0_plan(
                    protocol, ledger, f0_plan
                )


if __name__ == "__main__":
    unittest.main()
