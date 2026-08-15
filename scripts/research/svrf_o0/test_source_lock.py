from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import zipfile

from scripts.research.svrf_o0.validate_source_lock import parse_a2d2_checksums, select_spring, spring_sequence_rows


class SvrfO0SourceLockTest(unittest.TestCase):
    def test_a2d2_checksum_parser_rejects_non_md5_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checksums.txt"
            path.write_text("0123456789abcdef0123456789abcdef  parent.tar\nnot-a-checksum\n", encoding="utf-8")
            self.assertEqual(parse_a2d2_checksums(path), {"parent.tar": "0123456789abcdef0123456789abcdef"})

    def test_spring_selection_is_hash_ranked_after_support_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for sequence_id, rows in (("0001", 100), ("0002", 99), ("0003", 120), ("0004", 130)):
                    archive.writestr(f"spring/train/{sequence_id}/cam_data/extrinsics.txt", "x\n" * rows)
            counts = spring_sequence_rows(path)
            selected = select_spring(counts, "salt", 100, 2)
            self.assertEqual(len(selected), 2)
            self.assertNotIn("0002", {item["parent_id"] for item in selected})
            self.assertEqual(selected, sorted(selected, key=lambda item: item["selection_rank_sha256"]))


if __name__ == "__main__":
    unittest.main()
