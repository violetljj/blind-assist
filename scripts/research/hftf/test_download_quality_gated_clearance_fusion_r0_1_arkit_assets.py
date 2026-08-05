import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_quality_gated_clearance_fusion_r0_1_arkit_assets as subject


class DownloadTest(unittest.TestCase):
    def test_overwrite_forbidden(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "x.bin"
            target.write_bytes(b"x")
            with self.assertRaises(FileExistsError):
                subject.download("https://example.invalid", target, 1)

    def test_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "x.bin"
            path.write_bytes(b"abc")
            self.assertEqual("BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD", subject.sha256_file(path))


if __name__ == "__main__":
    unittest.main()
