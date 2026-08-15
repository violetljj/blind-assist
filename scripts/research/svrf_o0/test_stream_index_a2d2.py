from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import sqlite3
import tarfile
import tempfile
import unittest

from scripts.research.svrf_o0.stream_index_a2d2 import (
    CountingReader,
    connect_index,
    find_archive,
    tar_header_checksum,
)


class SvrfO0A2D2StreamIndexTest(unittest.TestCase):
    def test_counting_reader_hashes_without_retaining_payload(self) -> None:
        reader = CountingReader(BytesIO(b"abcdef"))
        digest, retained = reader.hash_exact(6)
        self.assertEqual(reader.bytes_read, 6)
        self.assertEqual(retained, b"")
        self.assertEqual(digest, "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721")

    def test_tar_checksum_and_frozen_archive_lookup(self) -> None:
        payload = BytesIO()
        with tarfile.open(fileobj=payload, mode="w") as archive:
            info = tarfile.TarInfo("frame.png")
            info.size = 3
            archive.addfile(info, BytesIO(b"rgb"))
        block = payload.getvalue()[:512]
        self.assertGreater(tar_header_checksum(block), 0)
        lock = json.loads(
            Path("docs/research/svrf/SVRF_O0_A2D2_SPRING_SOURCE_LOCK_2026-08-15.json").read_text(
                encoding="utf-8"
            )
        )
        binding = find_archive(lock, "camera_lidar-20180810150607_bus_signals.tar")
        self.assertEqual(binding["parent_id"], "20180810150607")
        self.assertEqual(binding["modality"], "bus")

    def test_existing_database_identity_drift_is_rejected(self) -> None:
        binding = {
            "name": "archive.tar",
            "bytes": 1024,
            "md5": "0" * 32,
            "parent_id": "parent",
            "modality": "camera",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "index.sqlite3"
            connection = connect_index(path, binding, "1" * 64)
            connection.close()
            changed = dict(binding, bytes=2048)
            with self.assertRaisesRegex(ValueError, "archive_bytes"):
                connect_index(path, changed, "1" * 64)
            connection = sqlite3.connect(path)
            try:
                self.assertEqual(connection.execute("SELECT value FROM metadata WHERE key='status'").fetchone()[0], "IN_PROGRESS")
            finally:
                connection.close()


if __name__ == "__main__":
    unittest.main()
