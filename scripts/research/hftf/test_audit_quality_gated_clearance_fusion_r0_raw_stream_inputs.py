import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_quality_gated_clearance_fusion_r0_raw_stream_inputs as subject


class RawStreamInputAuditTest(unittest.TestCase):
    def test_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out"
            path.write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                subject.require(not path.exists(), "overwrite forbidden")


if __name__ == "__main__":
    unittest.main()
