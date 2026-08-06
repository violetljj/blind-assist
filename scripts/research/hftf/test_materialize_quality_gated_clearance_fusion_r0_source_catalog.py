import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import materialize_quality_gated_clearance_fusion_r0_source_catalog as subject


class SourceCatalogTest(unittest.TestCase):
    def test_clip_window(self):
        rows = [{"frame_id": str(i), "parent_id": "p", "video_id": "v", "timestamp_ns": i * 100_000_000} for i in range(8)]
        clips = subject.nonoverlap_clips(rows)
        self.assertEqual(2, len(clips))

    def test_gap_breaks_clip(self):
        rows = [{"frame_id": str(i), "parent_id": "p", "video_id": "v", "timestamp_ns": value} for i, value in enumerate((0, 100_000_000, 900_000_000, 1_000_000_000))]
        self.assertEqual([], subject.nonoverlap_clips(rows))

    def test_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.json"
            path.write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                subject.require(not path.exists(), "overwrite forbidden")


if __name__ == "__main__":
    unittest.main()
