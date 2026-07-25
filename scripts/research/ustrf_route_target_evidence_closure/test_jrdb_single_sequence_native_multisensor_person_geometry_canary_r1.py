#!/usr/bin/env python3
"""Focused tests for elastic denominators and localized abstention."""
from __future__ import annotations

import unittest

from run_jrdb_single_sequence_native_multisensor_person_geometry_canary_r1 import (
    compact_ranges,
    coverage_band,
    denominator,
    index_unique,
)


class ElasticGeometryR1Tests(unittest.TestCase):
    def test_denominator_conservation(self) -> None:
        row = denominator(100, 97, 3, 0)
        self.assertEqual(row["coverage"], 0.97)
        self.assertEqual(row["coverage_band"], "HIGH_COVERAGE")

    def test_denominator_drift_rejected(self) -> None:
        with self.assertRaises(RuntimeError):
            denominator(100, 97, 2, 0)

    def test_coverage_band_is_descriptive(self) -> None:
        self.assertEqual(coverage_band(1.0), "HIGH_COVERAGE")
        self.assertEqual(coverage_band(0.9), "MODERATE_COVERAGE")
        self.assertEqual(coverage_band(0.5), "LOW_COVERAGE")

    def test_duplicate_id_is_localized(self) -> None:
        values, ambiguous = index_unique(
            [{"label_id": "pedestrian:1"}, {"label_id": "pedestrian:1"}, {"label_id": "pedestrian:2"}]
        )
        self.assertEqual(set(values), {"pedestrian:2"})
        self.assertEqual(ambiguous, {"pedestrian:1"})

    def test_compact_ranges_preserve_clusters(self) -> None:
        self.assertEqual(compact_ranges([1, 2, 3, 7, 9, 10]), [[1, 3], [7, 7], [9, 10]])


if __name__ == "__main__":
    unittest.main()
