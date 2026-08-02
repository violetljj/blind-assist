#!/usr/bin/env python3
"""Tests for D23 proximity multi-seed aggregation."""

from __future__ import annotations

import unittest

from aggregate_stage_c_d23_thor_magni_proximity_multiseed import (
    build_gate,
    summarize,
)


class D23ProximityMultiSeedTests(unittest.TestCase):
    def test_summary_and_supported_gate(self) -> None:
        units = []
        for fold in range(5):
            for seed in (17, 23, 41):
                units.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "history_minus_current": {
                            "metric": 0.02,
                        },
                    }
                )
        summary = summarize(units, "metric")
        self.assertEqual(summary["positive_units"], 15)
        self.assertEqual(summary["positive_seeds"], 3)
        self.assertEqual(summary["positive_folds"], 5)
        aggregate = {
            "by_target.proximity.source_macro.auroc": summary,
            "by_target.proximity.source_macro.average_precision": summary,
            "by_target.proximity.pooled.auroc": summary,
            "by_target.proximity.pooled.average_precision": summary,
        }
        self.assertTrue(build_gate(aggregate)["supported"])


if __name__ == "__main__":
    unittest.main()
