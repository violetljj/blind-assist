#!/usr/bin/env python3
"""Pure tests for chromatic cone evidence filtering."""

from __future__ import annotations

import unittest

import audit_public_video_chromatic_cone_lifecycle as subject


class ChromaticConeLifecycleAuditTest(unittest.TestCase):
    def test_color_balance_accepts_chromatic_detection(self) -> None:
        rows = [{
            "timestamp_ms": 0,
            "detections": [{
                "features": {
                    "high_saturation_fraction": 0.2,
                    "dark_fraction": 0.1,
                }
            }],
        }]
        filtered = subject.chromatic_samples(rows)[0]
        self.assertEqual({"barrier_structure": 1}, filtered["semantic_group_counts"])

    def test_color_balance_rejects_dark_detection(self) -> None:
        rows = [{
            "timestamp_ms": 0,
            "detections": [{
                "features": {
                    "high_saturation_fraction": 0.05,
                    "dark_fraction": 0.2,
                }
            }],
        }]
        filtered = subject.chromatic_samples(rows)[0]
        self.assertEqual({}, filtered["semantic_group_counts"])

    def test_equal_color_and_dark_is_rejected(self) -> None:
        rows = [{
            "timestamp_ms": 0,
            "detections": [{
                "features": {
                    "high_saturation_fraction": 0.1,
                    "dark_fraction": 0.1,
                }
            }],
        }]
        self.assertEqual({}, subject.chromatic_samples(rows)[0]["semantic_group_counts"])


if __name__ == "__main__":
    unittest.main()
