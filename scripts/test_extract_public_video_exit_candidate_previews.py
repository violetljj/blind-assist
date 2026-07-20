#!/usr/bin/env python3
"""Pure tests for public-video exit candidate preview selection."""

from __future__ import annotations

import unittest

import extract_public_video_exit_candidate_previews as subject


class PublicVideoExitCandidatePreviewTest(unittest.TestCase):
    def test_selects_highest_confidence_per_source_and_group(self) -> None:
        rows = [
            {"source_id": "a", "semantic_group": "surface_material", "present_group_max_confidence": 0.2, "present_timestamp_ms": 1000},
            {"source_id": "a", "semantic_group": "surface_material", "present_group_max_confidence": 0.8, "present_timestamp_ms": 2000},
            {"source_id": "a", "semantic_group": "barrier_structure", "present_group_max_confidence": 0.9, "present_timestamp_ms": 3000},
        ]
        selected = subject.select_candidates(
            rows, groups={"surface_material"}, top_per_source_group=1
        )
        self.assertEqual([2000], [row["present_timestamp_ms"] for row in selected])


if __name__ == "__main__":
    unittest.main()
