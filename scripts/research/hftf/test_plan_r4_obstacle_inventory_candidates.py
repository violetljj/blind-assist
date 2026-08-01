from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_r4_obstacle_inventory_candidates import (
    _validate_frozen_inputs,
    _validate_split,
)


class R4ObstacleInventoryCandidatePlanTest(unittest.TestCase):
    def test_frozen_parent_bindings_and_unique_burns_are_required(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            docs = root / "docs/research/hftf"
            docs.mkdir(parents=True)
            artifacts = (
                root
                / "artifacts.local/evidence/hftf/r3-1/cohort_result.json"
            )
            artifacts.parent.mkdir(parents=True)
            parent_result = docs / "parent.md"
            old_ledger = docs / "old.json"
            parent_result.write_text("parent", encoding="utf-8")
            old_ledger.write_text("old", encoding="utf-8")
            artifacts.write_text("cohort", encoding="utf-8")
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            protocol_path = docs / "protocol.json"
            protocol = {
                "schema": (
                    "blindassist_hftf_stage_b_split_source_validation_r4"
                ),
                "status": "FROZEN_BEFORE_R4_OUTCOME",
                "parent_result_path": "parent.md",
                "parent_result_sha256": sha(parent_result),
            }
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            ledger_path = docs / "ledger.json"
            ledger = {
                "schema": "blindassist_hftf_r4_source_pool_burn_ledger",
                "status": "FROZEN_BEFORE_R4_OUTCOME",
                "parent_r3_1_burn_ledger": {
                    "path": "old.json",
                    "sha256": sha(old_ledger),
                },
                "parent_r3_1_cohort_report": {
                    "path": (
                        "artifacts.local/evidence/hftf/r3-1/"
                        "cohort_result.json"
                    ),
                    "sha256": sha(artifacts),
                },
                "burned_session_ids": ["a"],
                "burned_session_count": 1,
            }
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            self.assertEqual(
                {"a"},
                _validate_frozen_inputs(
                    protocol, protocol_path, ledger, ledger_path, root
                ),
            )

    def test_split_drift_fails_closed(self) -> None:
        split = "a\nb\n"
        source = {
            "split_object_generation": "7",
            "split_text_sha256": hashlib.sha256(
                split.encode()
            ).hexdigest(),
        }
        _validate_split(source, "7", split)
        with self.assertRaisesRegex(ValueError, "generation drift"):
            _validate_split(source, "8", split)


if __name__ == "__main__":
    unittest.main()
