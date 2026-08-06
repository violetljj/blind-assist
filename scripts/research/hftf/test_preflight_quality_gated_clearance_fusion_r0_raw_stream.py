import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight_quality_gated_clearance_fusion_r0_raw_stream as subject


class RawStreamPreflightTest(unittest.TestCase):
    def test_bind_rejects_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "missing"):
                subject.bind(Path(folder), {"path": "x", "sha256": "0" * 64}, "asset")

    def test_bind_sha(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x"
            path.write_bytes(b"abc")
            sha = subject.sha256_file(path)
            self.assertEqual(path, subject.bind(Path(folder), {"path": "x", "sha256": sha}, "asset"))

    def test_overwrite_forbidden(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out"
            path.write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                subject.require(not path.exists(), "overwrite forbidden")


if __name__ == "__main__":
    unittest.main()
