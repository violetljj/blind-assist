#!/usr/bin/env python3
"""Tests for D12 future-onset target materialization."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from materialize_stage_c_d12_thor_magni_future_onset import (
    derive_onset_row,
)


def record(future_proximity: bool, future_corridor: bool) -> dict:
    return {
        "sample_id": "sample",
        "source_session_id": "source",
        "ancestry_group": "ancestry",
        "fold": 0,
        "video_path": "video.mp4",
        "video_sha256": "0" * 64,
        "anchor_scene_frame": 30,
        "history_scene_frames": [6, 12, 18, 24, 30],
        "target": {
            "future_proximity_le_1_25m": future_proximity,
            "future_corridor_intrusion": future_corridor,
        },
    }


class D12FutureOnsetTests(unittest.TestCase):
    def test_current_positive_is_ineligible_not_onset(self) -> None:
        row = derive_onset_row(
            record(True, True),
            {
                "current_static": {
                    "proximity": -1.0,
                    "corridor": 0.1,
                }
            },
        )
        self.assertFalse(
            row["future_onset_target"]["proximity_eligible"]
        )
        self.assertFalse(
            row["future_onset_target"]["proximity_onset"]
        )
        self.assertFalse(
            row["future_onset_target"]["corridor_eligible"]
        )
        self.assertFalse(
            row["future_onset_target"]["corridor_onset"]
        )

    def test_current_safe_future_positive_is_onset(self) -> None:
        row = derive_onset_row(
            record(True, True),
            {
                "current_static": {
                    "proximity": -2.0,
                    "corridor": -0.2,
                }
            },
        )
        self.assertTrue(
            row["future_onset_target"]["proximity_onset"]
        )
        self.assertTrue(
            row["future_onset_target"]["corridor_onset"]
        )


if __name__ == "__main__":
    unittest.main()
