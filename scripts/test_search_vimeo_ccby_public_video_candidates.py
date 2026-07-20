#!/usr/bin/env python3
"""Pure tests for the bounded Vimeo CC-BY candidate ledger."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import search_vimeo_ccby_public_video_candidates as subject


TEST_ARTIFACTS = (
    Path(__file__).resolve().parent.parent
    / "artifacts.local"
    / "tests"
    / "vimeo_ccby_candidate_ledger"
)
TEST_ARTIFACTS.mkdir(parents=True, exist_ok=True)


FIXTURE = """
<ul>
  <li class="first" id="clip_123" data-position="2">
    <a href="/123" title="POV Walking &amp; Roadworks">preview</a>
  </li>
  <li id="clip_456" data-position="3">
    <a href="https://vimeo.com/456" title="Aerial Construction Update">preview</a>
  </li>
  <li id="clip_123" data-position="9">
    <a href="/123" title="duplicate">preview</a>
  </li>
</ul>
"""


class VimeoCcByCandidateLedgerTest(unittest.TestCase):
    def test_search_url_uses_official_ccby_filter(self) -> None:
        self.assertEqual(
            "https://vimeo.com/creativecommons/by?search=walking%20construction",
            subject.build_search_url("walking construction"),
        )

    def test_parser_extracts_and_deduplicates_video_entries(self) -> None:
        candidates = subject.parse_search_html(FIXTURE)
        self.assertEqual(["123", "456"], [row["video_id"] for row in candidates])
        self.assertEqual("POV Walking & Roadworks", candidates[0]["source_title"])
        self.assertEqual("https://vimeo.com/123", candidates[0]["source_page_url"])

    def test_title_priority_penalizes_likely_montage(self) -> None:
        walking = subject.title_priority("POV walking roadworks", "walking construction")
        aerial = subject.title_priority(
            "Aerial construction update", "walking construction"
        )
        self.assertGreater(
            walking["title_priority_score"], aerial["title_priority_score"]
        )
        self.assertIn("construction update", aerial["negative_title_hits"])

    def test_offline_report_is_never_training_eligible(self) -> None:
        report = subject.build_report(
            query="walking construction",
            search_url=subject.build_search_url("walking construction"),
            html_payload=FIXTURE.encode("utf-8"),
            charset="utf-8",
            retrieval_mode="offline_saved_html",
            max_results=10,
        )
        self.assertEqual(subject.SCHEMA, report["schema"])
        self.assertEqual(0, report["request_page_count"])
        self.assertTrue(all(not row["training_eligible"] for row in report["candidates"]))
        self.assertEqual("123", report["candidates"][0]["video_id"])

    def test_external_single_page_mode_records_one_request(self) -> None:
        report = subject.build_report(
            query="walking construction",
            search_url=subject.build_search_url("walking construction"),
            html_payload=FIXTURE.encode("utf-8"),
            charset="utf-8",
            retrieval_mode="online_single_page_external_fetch",
            max_results=10,
        )
        self.assertEqual(1, report["request_page_count"])

    def test_write_report_adds_matching_sidecar(self) -> None:
        output = TEST_ARTIFACTS / "write-report.json"
        report = {"schema": subject.SCHEMA, "candidates": []}
        subject.write_report(report, output)
        encoded = output.read_bytes()
        self.assertEqual(
            subject.sha256_bytes(encoded),
            output.with_suffix(".json.sha256").read_text(encoding="ascii").strip(),
        )
        self.assertEqual(report, json.loads(encoded))


if __name__ == "__main__":
    unittest.main()
