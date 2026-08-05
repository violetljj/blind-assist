import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight_quality_gated_clearance_fusion_r0_1_arkit_assets as subject


class PreflightTest(unittest.TestCase):
    def test_request_roster_is_exact(self):
        roster = {"selected": [{"visit_id": "381644", "video_id": "1", "assets": list(subject.ASSETS)}]}
        rows = subject.requests_for(roster, "https://example.invalid")
        self.assertEqual(5, len(rows))
        self.assertTrue(all(row["visit_id"] == "381644" for row in rows))

    def test_asset_drift_fails_closed(self):
        roster = {"selected": [{"visit_id": "381644", "video_id": "1", "assets": ["lowres_wide.zip"]}]}
        with self.assertRaisesRegex(ValueError, "asset roster drift"):
            subject.requests_for(roster, "https://example.invalid")

    def test_disposition(self):
        good = [{"http_status": 200, "content_length_bytes": 1}]
        self.assertTrue(subject.disposition(good).endswith("AVAILABLE_MEDIA_UNOPENED"))
        self.assertIn("UNAVAILABLE", subject.disposition([{"http_status": 404, "content_length_bytes": 1}]))
        self.assertIn("INCOMPLETE", subject.disposition([{"http_status": None, "content_length_bytes": None}]))

    def test_overwrite_forbidden_before_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "result.json"
            output.write_text("occupied", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overwrite forbidden"):
                subject.produce(Path(folder), Path(folder) / "missing.json", output)


if __name__ == "__main__":
    unittest.main()
