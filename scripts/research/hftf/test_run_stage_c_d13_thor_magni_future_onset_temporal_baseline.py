#!/usr/bin/env python3
"""Tests for the D13 true-future onset baseline."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d13_thor_magni_future_onset_temporal_baseline import (
    masked_source_weights,
)


class D13FutureOnsetBaselineTests(unittest.TestCase):
    def test_source_weight_is_balanced_within_each_target_mask(self) -> None:
        sources = np.asarray(["a", "a", "b", "b", "b"])
        eligibility = np.asarray(
            [
                [True, True],
                [True, False],
                [True, True],
                [False, True],
                [False, True],
            ]
        )
        weights = masked_source_weights(sources, eligibility)
        for target in range(2):
            source_sums = [
                np.sum(
                    weights[
                        (sources == source)
                        & eligibility[:, target],
                        target,
                    ]
                )
                for source in ("a", "b")
            ]
            np.testing.assert_allclose(source_sums[0], source_sums[1])


if __name__ == "__main__":
    unittest.main()
