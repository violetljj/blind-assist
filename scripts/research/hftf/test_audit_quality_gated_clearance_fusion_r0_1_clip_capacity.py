import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_quality_gated_clearance_fusion_r0_1_clip_capacity as subject


class ClipCapacityTest(unittest.TestCase):
    def test_nonoverlap_count(self):
        count, clips = subject.nonoverlap_clip_count([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], 4, 0.5)
        self.assertEqual(2, count)
        self.assertEqual(4, len(clips[0]))

    def test_gap_breaks_window(self):
        count, _ = subject.nonoverlap_clip_count([0.0, 0.1, 0.9, 1.0], 4, 0.5)
        self.assertEqual(0, count)

    def test_overwrite_forbidden(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "result.json"
            output.write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                subject.require(not output.exists(), "overwrite forbidden")


if __name__ == "__main__":
    unittest.main()
