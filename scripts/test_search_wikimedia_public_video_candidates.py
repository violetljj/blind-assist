#!/usr/bin/env python3

import json
import unittest
from pathlib import Path

import search_wikimedia_public_video_candidates as subject


FIXTURE = {
    "query": {
        "pages": [
            {
                "pageid": 11,
                "title": "File:POV walking construction detour.webm",
                "imageinfo": [{
                    "url": "https://upload.wikimedia.org/example.webm",
                    "descriptionurl": "https://commons.wikimedia.org/wiki/File:Example.webm",
                    "mime": "video/webm",
                    "width": 1280,
                    "height": 720,
                    "duration": 42.5,
                    "extmetadata": {
                        "Artist": {"value": "Example"},
                        "LicenseShortName": {"value": "CC BY-SA 4.0"},
                        "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0/"},
                        "UsageTerms": {"value": "Creative Commons Attribution-Share Alike 4.0"},
                        "ImageDescription": {"value": "Pedestrian sidewalk barricade"},
                    },
                }],
            },
            {
                "pageid": 12,
                "title": "File:Still.jpg",
                "imageinfo": [{"mime": "image/jpeg", "url": "x", "descriptionurl": "y"}],
            },
        ]
    }
}


class WikimediaCandidateSearchTest(unittest.TestCase):
    def test_api_url_is_video_file_namespace_and_bounded(self) -> None:
        url = subject.api_query_url("walking construction filetype:video", 25)
        self.assertIn("gsrnamespace=6", url)
        self.assertIn("gsrlimit=25", url)
        self.assertIn("iiprop=url%7Cmime%7Csize%7Cextmetadata", url)

    def test_parser_keeps_video_and_license_metadata(self) -> None:
        rows = subject.parse_api_payload(json.dumps(FIXTURE).encode(), "walking construction")
        self.assertEqual(1, len(rows))
        self.assertEqual("video/webm", rows[0]["mime"])
        self.assertTrue(rows[0]["item_license_metadata_present"])
        self.assertFalse(rows[0]["training_eligible"])

    def test_report_deduplicates_page_ids(self) -> None:
        payload = json.dumps(FIXTURE).encode()
        report = subject.build_report(
            {"contract_id": "test"},
            [("walking", "https://example/1", payload), ("detour", "https://example/2", payload)],
        )
        self.assertEqual(1, report["candidate_count"])
        self.assertEqual(2, report["request_count"])

    def test_write_report_adds_sidecar_and_refuses_overwrite(self) -> None:
        root = Path("artifacts.local/tests/wikimedia_candidate_search")
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
