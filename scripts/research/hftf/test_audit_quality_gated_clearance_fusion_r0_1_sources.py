import csv
import tempfile
import unittest
from pathlib import Path

from audit_quality_gated_clearance_fusion_r0_1_sources import arkit_selection


class SourceAdmissionTest(unittest.TestCase):
    def test_arkit_selection_is_identity_only_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "split.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["video_id", "visit_id", "fold"])
                writer.writeheader()
                writer.writerows([
                    {"video_id": "v2", "visit_id": "200", "fold": "Validation"},
                    {"video_id": "v1", "visit_id": "100", "fold": "Validation"},
                    {"video_id": "v3", "visit_id": "100", "fold": "Validation"},
                ])
            visit, videos, count = arkit_selection(path, {"200"})
            self.assertEqual((visit, videos, count), ("100", ["v1", "v3"], 3))


if __name__ == "__main__":
    unittest.main()
