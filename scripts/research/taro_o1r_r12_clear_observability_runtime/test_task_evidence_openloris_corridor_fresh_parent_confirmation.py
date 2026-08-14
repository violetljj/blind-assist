from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.research.taro_o1r_r12_clear_observability_runtime import (
    task_evidence_openloris_corridor_fresh_parent_confirmation as subject,
)


class OpenLorisCorridorFreshParentConfirmationTest(unittest.TestCase):
    def test_tar_header_authenticates_name_size_and_checksum(self) -> None:
        info = tarfile.TarInfo("corridor1-3.7z")
        info.size = subject.EXPECTED_TAR_ENTRIES["corridor1-3.7z"]["bytes"]
        payload = info.tobuf(format=tarfile.GNU_FORMAT)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "header.bin"
            path.write_bytes(payload)
            receipt = subject._load_tar_header(path)
            self.assertEqual("corridor1-3.7z", receipt["name"])
            self.assertEqual(info.size, receipt["bytes"])
            corrupted = bytearray(payload)
            corrupted[0] ^= 1
            path.write_bytes(corrupted)
            with self.assertRaises(subject.R36Error):
                subject._load_tar_header(path)

    def test_confirmation_lock_binds_runner_and_exact_tar_entries(self) -> None:
        lock = json.loads(subject.LOCK_PATH.read_text(encoding="utf-8"))
        self.assertEqual("LOCKED_BEFORE_FRESH_PAYLOAD_OPEN", lock["status"])
        frozen = lock["frozen_implementation"]["confirmation_runner"]
        runner = subject.REPO_ROOT / frozen["path"]
        self.assertEqual(runner.stat().st_size, frozen["bytes"])
        self.assertEqual(
            hashlib.sha256(runner.read_bytes()).hexdigest().upper(),
            frozen["sha256"],
        )
        locked_entries = {
            entry["name"]: {
                "header_offset": entry["header_offset"],
                "data_offset": entry["data_offset"],
                "bytes": entry["bytes"],
            }
            for entry in lock["publisher_inputs"]["required_tar_entries"]
        }
        self.assertEqual(subject.EXPECTED_TAR_ENTRIES, locked_entries)


if __name__ == "__main__":
    unittest.main()
