#!/usr/bin/env python3

import json
import unittest
from pathlib import Path

import search_internet_archive_public_video_candidates as subject


FIXTURE = {
    "response": {
        "docs": [
            {
                "identifier": "walking-detour",
                "title": "POV walking sidewalk construction detour",
                "description": "A pedestrian walks around a barricade",
                "creator": "Example",
                "licenseurl": "https://creativecommons.org/licenses/by/4.0/",
                "downloads": 12,
            },
            {"identifier": "missing-license", "title": "walking", "downloads": 99},
        ]
    }
}


class InternetArchiveCandidateSearchTest(unittest.TestCase):
    def test_api_url_is_bounded_without_license_filter(self) -> None:
        url = subject.api_query_url("walking detour", 25)
        self.assertIn("advancedsearch.php", url)
        self.assertIn("rows=25", url)
        self.assertIn("page=1", url)
        self.assertNotIn("licenseurl%3A%2A", url)

    def test_parser_keeps_missing_license_as_nonblocking_metadata(self) -> None:
        rows = subject.parse_api_payload(json.dumps(FIXTURE).encode(), "walking detour")
        self.assertEqual(2, len(rows))
        self.assertEqual("walking-detour", rows[0]["identifier"])
        missing = next(row for row in rows if row["identifier"] == "missing-license")
        self.assertEqual("unknown_recorded_nonblocking", missing["item_license_status"])
        self.assertFalse(rows[0]["training_eligible"])
        self.assertGreater(rows[0]["title_priority_score"], 0)

    def test_report_deduplicates_identifiers(self) -> None:
        payload = json.dumps(FIXTURE).encode()
        report = subject.build_report({"contract_id": "test"}, [
            ("walking", "https://example/1", payload),
            ("detour", "https://example/2", payload),
        ])
        self.assertEqual(2, report["candidate_count"])
        self.assertEqual(2, report["request_count"])

    def test_write_report_hashes_and_refuses_overwrite(self) -> None:
        root = Path("artifacts.local/tests/internet_archive_candidate_search")
        root.mkdir(parents=True, exist_ok=True)
        output = root / "ledger.json"
        output.unlink(missing_ok=True)
        Path(str(output) + ".sha256").unlink(missing_ok=True)
        subject.write_report({"schema": subject.SCHEMA}, output)
        self.assertEqual(subject.sha256_bytes(output.read_bytes()), Path(str(output) + ".sha256").read_text().strip())
        with self.assertRaises(ValueError):
            subject.write_report({"schema": subject.SCHEMA}, output)


if __name__ == "__main__":
    unittest.main()
