from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_stage_c_e0_1_fresh_transport as subject


class E01FreshTransportTest(unittest.TestCase):
    def test_unacquired_terminal_fails_closed(self) -> None:
        acquisition = {
            "schema": subject.ACQUISITION_SCHEMA,
            "terminal": "NO",
        }
        with self.assertRaisesRegex(ValueError, "not acquired"):
            subject._validate_acquisition(
                {}, Path(__file__), acquisition, Path.cwd()
            )

    def test_missing_burn_fails_closed(self) -> None:
        path = Path(__file__)
        acquisition = {
            "schema": subject.ACQUISITION_SCHEMA,
            "terminal": (
                "E0_1_FRESH_EVALUATION_MEDIA_BYTES_ACQUIRED_AND_HASH_BOUND"
            ),
            "protocol_sha256": subject._sha256(path),
            "output_root": str(Path.cwd()),
            "new_dev_and_heldout_burned": False,
            "fresh_evaluation_geometry_label_outcome_read": False,
            "fresh_transport_audit_authorized": True,
        }
        with self.assertRaisesRegex(ValueError, "burn"):
            subject._validate_acquisition(
                {}, path, acquisition, Path.cwd().resolve()
            )


if __name__ == "__main__":
    unittest.main()
