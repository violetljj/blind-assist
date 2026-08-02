#!/usr/bin/env python3
"""Tests for D15 JRDB onset replication gate."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d15_jrdb_future_onset_replication import (
    PRIMARY_PATHS,
    replication_supported,
)


class D15JrdbReplicationTests(unittest.TestCase):
    def test_gate_requires_both_folds_and_four_units(self) -> None:
        aggregate = {
            path: {"mean": 0.1, "positive_count": 2}
            for path in PRIMARY_PATHS
        }
        units = [
            {
                "history_minus_current": {
                    path: (0.1 if index < 4 else -0.1)
                    for path in PRIMARY_PATHS
                }
            }
            for index in range(6)
        ]
        self.assertTrue(replication_supported(aggregate, units))
        aggregate["corridor.average_precision"]["positive_count"] = 1
        self.assertFalse(replication_supported(aggregate, units))


if __name__ == "__main__":
    unittest.main()
