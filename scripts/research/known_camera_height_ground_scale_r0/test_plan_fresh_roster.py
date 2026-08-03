import csv
import json
import tempfile
import unittest
from pathlib import Path

from plan_fresh_roster import plan


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT / "docs" / "research" / "hftf" / "KNOWN_CAMERA_HEIGHT_GROUND_SCALE_R0_PROTOCOL_2026-08-04.json"


class PlanFreshRosterTest(unittest.TestCase):
    def test_is_deterministic_disjoint_and_excludes_cross_fold_visits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "metadata.csv"
            with metadata.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["video_id", "visit_id", "fold"])
                writer.writeheader()
                writer.writerow({"video_id": "1", "visit_id": "old", "fold": "Validation"})
                for index in range(8):
                    writer.writerow({"video_id": str(100 + index), "visit_id": f"new-{index}", "fold": "Validation"})
                writer.writerow({"video_id": "500", "visit_id": "cross", "fold": "Training"})
                writer.writerow({"video_id": "501", "visit_id": "cross", "fold": "Validation"})
            predecessor = root / "predecessor.json"
            predecessor.write_text(
                json.dumps(
                    {
                        "roles": {"train": [{"visit_id": "old"}], "validation": [], "sealed": []},
                        "source": {"repository_commit": "commit"},
                    }
                ),
                encoding="utf-8",
            )
            first = plan(metadata, predecessor, PROTOCOL)
            second = plan(metadata, predecessor, PROTOCOL)
            self.assertEqual(first, second)
            selected = {row["visit_id"] for row in first["fresh_evaluation"]}
            self.assertEqual(4, len(selected))
            self.assertNotIn("old", selected)
            self.assertNotIn("cross", selected)
            self.assertFalse(first["media_bytes_read"])
            self.assertFalse(first["metric_truth_opened"])


if __name__ == "__main__":
    unittest.main()
