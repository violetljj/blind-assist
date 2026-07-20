import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_public_video_dinov2_prospective_features as extractor


class DinoV2ProspectiveExtractorTest(unittest.TestCase):
    def test_half_open_schedule(self):
        self.assertEqual([0, 1000, 2000], extractor.schedule_timestamps(3000))
        self.assertEqual([0, 1000, 2000, 3000], extractor.schedule_timestamps(3001))

    def test_schedule_rejects_invalid_duration(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            extractor.schedule_timestamps(0)

    def test_scheduled_frame_index_clamps_to_last_frame(self):
        self.assertEqual(30, extractor.scheduled_frame_index(1000, 30.0, 100))
        self.assertEqual(99, extractor.scheduled_frame_index(5000, 30.0, 100))


if __name__ == "__main__":
    unittest.main()
