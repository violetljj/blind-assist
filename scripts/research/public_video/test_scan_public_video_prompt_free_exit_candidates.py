#!/usr/bin/env python3
"""Pure tests for public-video prompt-free exit discovery."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import scan_public_video_prompt_free_exit_candidates as subject


class PublicVideoPromptFreeExitDiscoveryTest(unittest.TestCase):
    def test_source_scan_range_accepts_bounded_continuous_window(self) -> None:
        self.assertEqual(
            (1000, 5000),
            subject.source_scan_range(
                {"scan_start_ms": 1000, "scan_end_ms": 5000}, 10000
            ),
        )

    def test_source_scan_range_rejects_out_of_video_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid source scan range"):
            subject.source_scan_range(
                {"scan_start_ms": 1000, "scan_end_ms": 11000}, 10000
            )

    def test_workzone_marker_additions_are_explicit_and_default_off(self) -> None:
        self.assertIsNone(subject.semantic_group_for_class("traffic cone"))
        self.assertEqual(
            "barrier_structure",
            subject.semantic_group_for_class(
                "traffic cone", include_workzone_markers=True
            ),
        )
        self.assertEqual(
            "barrier_structure",
            subject.semantic_group_for_class(
                "barricade", include_workzone_markers=True
            ),
        )
        self.assertIsNone(subject.semantic_group_for_class(
            "ice cream cone", include_workzone_markers=True
        ))

    def test_registry_v2_accepts_non_commons_licensed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "candidate.mp4"
            video.write_bytes(b"discovery-placeholder")
            registry_path = root / "registry.json"
            registry = {
                "schema": subject.REGISTRY_SCHEMA_V2,
                "sources": [{
                    "source_id": "vimeo-example",
                    "local_video_path": str(video),
                    "source_platform": "vimeo",
                    "source_title": "Example",
                    "source_page_url": "https://vimeo.com/1",
                    "author": "Example author",
                    "license": "CC BY 3.0",
                }],
            }
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            sources = subject.validate_registry(registry, registry_path)
            self.assertEqual("vimeo", sources[0]["source_platform"])
            self.assertNotIn("commons_title", sources[0])

    def test_registry_v1_normalizes_commons_source_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "candidate.webm"
            video.write_bytes(b"discovery-placeholder")
            registry_path = root / "registry.json"
            registry = {
                "schema": subject.REGISTRY_SCHEMA,
                "sources": [{
                    "source_id": "commons-example",
                    "local_video_path": str(video),
                    "commons_title": "File:Example.webm",
                    "commons_page_url": "https://commons.wikimedia.org/wiki/File:Example.webm",
                    "author": "Example author",
                    "license": "CC BY 4.0",
                }],
            }
            sources = subject.validate_registry(registry, registry_path)
            self.assertEqual("wikimedia_commons", sources[0]["source_platform"])
            self.assertEqual("File:Example.webm", sources[0]["source_title"])

    def test_nearfield_corridor_geometry_rejects_side_and_giant_boxes(self) -> None:
        accepted = subject.nearfield_corridor_geometry([40, 40, 60, 95], [100, 100])
        side = subject.nearfield_corridor_geometry([0, 40, 15, 95], [100, 100])
        giant = subject.nearfield_corridor_geometry([0, 0, 100, 100], [100, 100])
        self.assertTrue(accepted["nearfield_corridor_accepted"])
        self.assertFalse(side["nearfield_corridor_accepted"])
        self.assertFalse(giant["nearfield_corridor_accepted"])

    def test_summary_keeps_only_preregistered_semantics(self) -> None:
        sample = subject.summarize_sample(5000, [
            {"class_name": "sand", "semantic_group": "surface_material", "confidence": 0.4},
            {"class_name": "sand", "semantic_group": "surface_material", "confidence": 0.7},
            {"class_name": "barrier", "semantic_group": "barrier_structure", "confidence": 0.2},
        ])
        self.assertEqual(3, sample["semantic_detection_count"])
        self.assertEqual(2, sample["semantic_group_counts"]["surface_material"])
        self.assertEqual(0.7, sample["semantic_group_max_confidence"]["surface_material"])

    def test_adjacent_presence_to_absence_becomes_candidate(self) -> None:
        samples = [
            subject.summarize_sample(0, [
                {"class_name": "sand box", "semantic_group": "surface_material", "confidence": 0.6}
            ]),
            subject.summarize_sample(5000, []),
        ]
        candidates = subject.discover_adjacent_exits(
            "source-a", samples, sample_interval_ms=5000
        )
        self.assertEqual(1, len(candidates))
        self.assertEqual("surface_material", candidates[0]["semantic_group"])
        self.assertEqual(5000, candidates[0]["gap_ms"])
        self.assertEqual(1, candidates[0]["absent_run_sample_count"])

    def test_absence_persistence_rejects_detector_flicker(self) -> None:
        present = [
            {"class_name": "construction site", "semantic_group": "barrier_structure", "confidence": 0.4}
        ]
        samples = [
            subject.summarize_sample(0, present),
            subject.summarize_sample(1000, []),
            subject.summarize_sample(2000, []),
            subject.summarize_sample(3000, present),
            subject.summarize_sample(4000, []),
            subject.summarize_sample(5000, []),
            subject.summarize_sample(6000, []),
        ]
        candidates = subject.discover_adjacent_exits(
            "source-a",
            samples,
            sample_interval_ms=1000,
            min_absent_run_samples=3,
        )
        self.assertEqual(1, len(candidates))
        self.assertEqual(3000, candidates[0]["present_timestamp_ms"])
        self.assertEqual(3, candidates[0]["absent_run_sample_count"])

    def test_gap_larger_than_frozen_interval_is_rejected(self) -> None:
        samples = [
            subject.summarize_sample(0, [
                {"class_name": "barrier", "semantic_group": "barrier_structure", "confidence": 0.8}
            ]),
            subject.summarize_sample(5001, []),
        ]
        self.assertEqual([], subject.discover_adjacent_exits(
            "source-a", samples, sample_interval_ms=5000
        ))


if __name__ == "__main__":
    unittest.main()
