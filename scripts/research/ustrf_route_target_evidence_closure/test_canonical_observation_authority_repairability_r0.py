#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import canonical_observation_authority_inventory_r0 as inventory
import canonical_observation_denominator_availability_r0 as availability


class CanonicalObservationAuthorityR0Test(unittest.TestCase):
    def test_terminal_priority(self) -> None:
        self.assertEqual(
            availability._terminal(
                audit_complete=False,
                required_source_authority_absent=True,
                availability_complete=False,
            ),
            "FAIL_CLOSED_AUDIT_INCOMPLETE",
        )
        self.assertEqual(
            availability._terminal(
                audit_complete=True,
                required_source_authority_absent=True,
                availability_complete=False,
            ),
            "SOURCE_AUTHORITY_ABSENT",
        )
        self.assertEqual(
            availability._terminal(
                audit_complete=True,
                required_source_authority_absent=False,
                availability_complete=False,
            ),
            "AVAILABILITY_UPPER_BOUND_INSUFFICIENT",
        )

    def test_unknown_is_not_eligible(self) -> None:
        self.assertNotIn("unknown", availability.ELIGIBLE_STATES)
        self.assertNotIn("inferred", availability.ELIGIBLE_STATES)

    def test_authority_record_requires_legal_state(self) -> None:
        with self.assertRaisesRegex(
            inventory.AuthorityInventoryError, "illegal_authority_state"
        ):
            inventory._field(
                origin_authority="x",
                transform_status="x",
                value_state="x",
                scope="frame",
                state="heuristic",
                reason_code="x",
                parent_path="x",
                parent_sha256="0" * 64,
                join_key="x",
            )

    def test_a_config_forbidden_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            inventory.AuthorityInventoryError, "forbidden_key_in_a_config"
        ):
            inventory._forbidden_key_scan({"nested": {"truth_status": False}})

    def test_png_header_is_read_without_decoder_assumption(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.png"
            path.write_bytes(
                b"\x89PNG\r\n\x1a\n"
                + struct.pack(">I", 13)
                + b"IHDR"
                + struct.pack(">II", 640, 480)
            )
            self.assertEqual(inventory._png_size(path), (640, 480))

    def test_inventory_sha_is_verified_before_json_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "inventory.json"
            path.write_text("{invalid", encoding="utf-8")
            with self.assertRaisesRegex(
                availability.AvailabilityAuditError,
                "inventory_sha256_drift_before_denominator_open",
            ):
                availability.verify_inventory_first(root, path, "0" * 64)

    def test_inventory_first_accepts_exact_minimal_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "inventory.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": inventory.INVENTORY_SCHEMA,
                        "status": "AUTHORITY_INVENTORY_FROZEN",
                    }
                ),
                encoding="utf-8",
            )
            digest = availability.sha256_file(path)
            value, first = availability.verify_inventory_first(root, path, digest)
            self.assertEqual(value["status"], "AUTHORITY_INVENTORY_FROZEN")
            self.assertEqual(first["role"], "authority_inventory_first_read_and_verify")

    def test_contaminated_modules_are_not_imported(self) -> None:
        source = Path(inventory.__file__).read_text(encoding="utf-8")
        for fragment in (
            "metric_eligibility",
            "causal_per_track",
            "known_route",
            "route_conditioned_scale_growth_separability_r0",
        ):
            self.assertNotIn(f"import {fragment}", source)


if __name__ == "__main__":
    unittest.main()
