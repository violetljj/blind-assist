from __future__ import annotations

import unittest

from scripts.research.egomotion_compensated_looming.motion_diverse_rgbd_source_search_r1.run_geometry_batch import (
    summarize,
)


def row(index: int, band: str) -> dict[str, object]:
    return {
        "pair_index": index,
        "previous_timestamp_s": index * 0.1,
        "current_timestamp_s": (index + 1) * 0.1,
        "geometry_evaluable": True,
        "geometry_abstention_reason": None,
        "geometry_band": band,
        "geometry_signed_radial_expansion_per_s": (
            0.1 if band == "POSITIVE_APPROACH_GEOMETRY" else 0.0
        ),
    }


class GeometryBatchSummaryTest(unittest.TestCase):
    def window(self) -> dict[str, object]:
        return {
            "window_id": "fixture@0",
            "sequence_id": "fixture",
            "proxy_queue": "positive",
            "proxy_queue_index": 0,
            "start_timestamp_s": "0",
            "end_timestamp_s": "10",
            "pair_count": 100,
        }

    def test_positive_requires_fixed_denominator_and_five_seconds(self) -> None:
        rows = [row(index, "POSITIVE_APPROACH_GEOMETRY") for index in range(80)]
        rows.extend(row(index, "BELOW_TRIGGER_REFERENCE") for index in range(80, 100))
        summary = summarize(self.window(), rows)
        self.assertEqual(summary["role"], "POSITIVE_APPROACH_WINDOW")
        self.assertEqual(summary["positive_fraction_fixed_denominator"], 0.8)
        self.assertGreaterEqual(summary["longest_positive_run_duration_s"], 5.0)

    def test_fragmented_positive_is_ambiguous(self) -> None:
        rows = [
            row(
                index,
                (
                    "POSITIVE_APPROACH_GEOMETRY"
                    if index % 5 != 4
                    else "BELOW_TRIGGER_REFERENCE"
                ),
            )
            for index in range(100)
        ]
        summary = summarize(self.window(), rows)
        self.assertEqual(summary["positive_fraction_fixed_denominator"], 0.8)
        self.assertEqual(summary["role"], "AMBIGUOUS_OR_INELIGIBLE")


if __name__ == "__main__":
    unittest.main()
