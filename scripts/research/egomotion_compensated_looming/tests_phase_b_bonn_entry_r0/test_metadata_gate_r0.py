from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import random
import unittest

from scripts.research.egomotion_compensated_looming.rcle_phase_b_bonn_entry_r0 import (
    gate,
)


def inventory() -> list[dict[str, object]]:
    rows = []
    for index in range(26):
        sequence_id = f"rgbd_bonn_sequence_{index:02d}"
        rows.append(
            {
                "sequence_id": sequence_id,
                "display_size": "100 MB",
                "display_size_mb_normalized": 100.0,
                "url": (
                    "https://www.ipb.uni-bonn.de/html/projects/"
                    f"rgbd_dynamic2019/{sequence_id}.zip"
                ),
            }
        )
    return rows


class BonnMetadataGateTest(unittest.TestCase):
    def test_selection_hash_contract_includes_tab(self) -> None:
        sequence_id = "rgbd_bonn_example"
        expected = hashlib.sha256(
            f"{gate.SELECTION_SALT}\t{sequence_id}".encode()
        ).hexdigest()
        self.assertEqual(gate.selection_hash(sequence_id), expected)

    def test_deterministic_selection_is_input_order_independent(self) -> None:
        rows = inventory()
        exclusions = {
            row["sequence_id"]: "HISTORICAL"
            for row in rows[:9]
        }
        first = gate.build_metadata_decision(rows, exclusions)
        shuffled = copy.deepcopy(rows)
        random.Random(123).shuffle(shuffled)
        second = gate.build_metadata_decision(shuffled, exclusions)
        self.assertEqual(
            first["selected_sequence_ids"],
            second["selected_sequence_ids"],
        )
        self.assertEqual(
            first["cohort_identity_sha256"],
            second["cohort_identity_sha256"],
        )
        self.assertEqual(first["metadata_selection_denominator_count"], 26)
        self.assertEqual(first["selected_sequence_count"], 6)

    def test_historical_identity_can_never_be_selected(self) -> None:
        rows = inventory()
        forbidden = {row["sequence_id"]: "HISTORICAL" for row in rows[:9]}
        result = gate.build_metadata_decision(rows, forbidden)
        self.assertTrue(
            set(result["selected_sequence_ids"]).isdisjoint(forbidden)
        )
        historical_rows = [
            row
            for row in result["metadata_selection_denominator"]
            if row["sequence_id"] in forbidden
        ]
        self.assertEqual(len(historical_rows), 9)
        self.assertTrue(
            all(row["disposition"] == "EXCLUDED_HISTORICAL" for row in historical_rows)
        )

    def test_fewer_than_six_eligible_closes_candidate(self) -> None:
        rows = inventory()
        exclusions = {
            row["sequence_id"]: "HISTORICAL"
            for row in rows[:21]
        }
        result = gate.build_metadata_decision(rows, exclusions)
        self.assertFalse(result["gate_pass"])
        self.assertEqual(result["terminal_state"], gate.TERMINAL_CLOSE)
        self.assertEqual(result["selected_sequence_count"], 5)

    def test_missing_historical_identity_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "HISTORICAL_IDENTITY_MISSING"
        ):
            gate.build_metadata_decision(
                inventory(), {"rgbd_bonn_missing": "HISTORICAL"}
            )

    def test_receipt_firewall_rejects_any_read(self) -> None:
        receipt = {
            "schema_version": gate.RECEIPT_SCHEMA_VERSION,
            "candidate_id": gate.CANDIDATE_ID,
            "design_lock_sha256": gate.DESIGN_LOCK_SHA256,
            "historical_exclusion_manifest_sha256": (
                gate.HISTORICAL_MANIFEST_SHA256
            ),
            "implementation_lock_sha256": "0" * 64,
            "official_page": {},
            "selection_contract": {},
            "metadata_selection_denominator": [{}] * 26,
            "selected_sequence_ids": [str(index) for index in range(6)],
            "selected_sequence_count": 6,
            "cohort_identity_sha256": "0" * 64,
            "read_firewall": {
                "rgb_payload_members_read": 1,
                "candidate_signal_computed": False,
            },
            "gate_pass": True,
            "terminal_state": gate.TERMINAL_PASS,
            "authority": "METADATA_ONLY",
            "formal_phase_b_authorized": False,
            "payload_access_authorized": False,
        }
        with self.assertRaisesRegex(ValueError, "READ_FIREWALL"):
            gate.validate_receipt_shape(receipt)

    def test_receipt_shape_rejects_phase_b_authority(self) -> None:
        receipt = {
            "schema_version": gate.RECEIPT_SCHEMA_VERSION,
            "candidate_id": gate.CANDIDATE_ID,
            "design_lock_sha256": gate.DESIGN_LOCK_SHA256,
            "historical_exclusion_manifest_sha256": (
                gate.HISTORICAL_MANIFEST_SHA256
            ),
            "implementation_lock_sha256": "0" * 64,
            "official_page": {},
            "selection_contract": {},
            "metadata_selection_denominator": [{}] * 26,
            "selected_sequence_ids": [str(index) for index in range(6)],
            "selected_sequence_count": 6,
            "cohort_identity_sha256": "0" * 64,
            "read_firewall": {
                "rgb_payload_members_read": 0,
                "candidate_signal_computed": False,
            },
            "gate_pass": True,
            "terminal_state": gate.TERMINAL_PASS,
            "authority": "METADATA_ONLY",
            "formal_phase_b_authorized": True,
            "payload_access_authorized": False,
        }
        with self.assertRaisesRegex(ValueError, "PHASE_B_AUTHORITY"):
            gate.validate_receipt_shape(receipt)

    def test_runner_has_no_tzdata_dependency(self) -> None:
        runner = (
            Path(__file__).resolve().parents[1]
            / "run_phase_b_bonn_metadata_gate_r0.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("ZoneInfo", runner)
        self.assertIn("timezone(timedelta(hours=8))", runner)


if __name__ == "__main__":
    unittest.main()
