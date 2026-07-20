#!/usr/bin/env python3
"""Pure tests for prompt-free exit persistence re-filtering."""

from __future__ import annotations

import unittest

import refilter_public_video_prompt_free_exit_candidates as subject
import scan_public_video_prompt_free_exit_candidates as discovery


class ExitPersistenceRefilterTest(unittest.TestCase):
    def test_refilter_keeps_only_stable_absence(self) -> None:
        present = [{"class_name": "barrier", "semantic_group": "barrier_structure", "confidence": 0.5}]
        samples = [
            discovery.summarize_sample(0, present),
            discovery.summarize_sample(1000, []),
            discovery.summarize_sample(2000, present),
            discovery.summarize_sample(3000, []),
            discovery.summarize_sample(4000, []),
        ]
        report = {
            "sampling": {"sample_interval_ms": 1000},
            "sources": [{"source_id": "s", "sample_count": 5, "samples": samples}],
        }
        filtered = subject.refilter(report, min_absent_run_samples=2)
        self.assertEqual(1, filtered["summary"]["exit_candidate_count"])
        self.assertEqual(2000, filtered["exit_candidates"][0]["present_timestamp_ms"])


if __name__ == "__main__":
    unittest.main()
