from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_stage_c_e0_fresh_media_transport as subject


class FreshMediaTransportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = {
            "frozen_sources": [
                {
                    "role": "train",
                    "trajectory": "one",
                    "files": {
                        "pose": {
                            "path": "data/one.parquet",
                            "size_bytes": 1,
                            "sha256": "a" * 64,
                        },
                        "rgb": {
                            "path": "video/rgb/one.mp4",
                            "size_bytes": 2,
                            "sha256": "b" * 64,
                        },
                        "depth": {
                            "path": "video/depth/one.mkv",
                            "size_bytes": 3,
                            "sha256": "c" * 64,
                        },
                    },
                }
            ]
        }

    def _acquisition(self, protocol_path: Path) -> dict:
        source = self.protocol["frozen_sources"][0]
        return {
            "schema": subject.ACQUISITION_SCHEMA,
            "terminal": "E0_FRESH_MEDIA_BYTES_ACQUIRED_AND_HASH_BOUND",
            "protocol_sha256": subject._sha256(protocol_path),
            "output_root": str(Path.cwd().resolve()),
            "selected_sources_burned": True,
            "fresh_geometry_label_outcome_read": False,
            "transport_decode_audit_authorized": True,
            "downloaded_files": [
                {
                    "kind": kind,
                    "role": source["role"],
                    "trajectory": source["trajectory"],
                    **source["files"][kind],
                }
                for kind in ("pose", "rgb", "depth")
            ],
        }

    def test_exact_acquisition_ledger_validates(self) -> None:
        path = Path(__file__)
        subject._validate_acquisition(
            self.protocol,
            path,
            self._acquisition(path),
            Path.cwd().resolve(),
        )

    def test_unrecorded_burn_fails_closed(self) -> None:
        path = Path(__file__)
        acquisition = self._acquisition(path)
        acquisition["selected_sources_burned"] = False
        with self.assertRaisesRegex(ValueError, "burn"):
            subject._validate_acquisition(
                self.protocol,
                path,
                acquisition,
                Path.cwd().resolve(),
            )

    def test_file_ledger_mutation_fails_closed(self) -> None:
        path = Path(__file__)
        acquisition = copy.deepcopy(self._acquisition(path))
        acquisition["downloaded_files"][0]["sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "ledger mismatch"):
            subject._validate_acquisition(
                self.protocol,
                path,
                acquisition,
                Path.cwd().resolve(),
            )


if __name__ == "__main__":
    unittest.main()
