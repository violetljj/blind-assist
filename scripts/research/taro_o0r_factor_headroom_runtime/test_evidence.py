from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.research.taro_o0r_factor_headroom_runtime.evidence import (
    FactorEvidenceError,
    FactorEvidenceWriter,
    deterministic_gzip,
)


class FactorEvidenceWriterTests(unittest.TestCase):
    def test_exclusive_atomic_budgeted_writer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "factor"
            writer = FactorEvidenceWriter(root, 256)
            writer.activate({"schema": "start"})
            receipt = writer.write_bytes("nested/value.bin", b"abc")
            self.assertEqual((root / "nested/value.bin").read_bytes(), b"abc")
            self.assertEqual(receipt["bytes"], 3)
            with self.assertRaises(FactorEvidenceError) as overwrite:
                writer.write_bytes("nested/value.bin", b"abc")
            self.assertEqual(overwrite.exception.code, "EVIDENCE_OVERWRITE_FORBIDDEN")
            with self.assertRaises(FactorEvidenceError) as escape:
                writer.write_bytes("../escape", b"x")
            self.assertEqual(escape.exception.code, "EVIDENCE_PATH_INVALID")
            with self.assertRaises(FactorEvidenceError) as collision:
                FactorEvidenceWriter(root, 256).activate({"schema": "again"})
            self.assertEqual(collision.exception.code, "FACTOR_ROOT_COLLISION")

    def test_deterministic_gzip(self) -> None:
        self.assertEqual(deterministic_gzip(b"payload"), deterministic_gzip(b"payload"))


if __name__ == "__main__":
    unittest.main()
