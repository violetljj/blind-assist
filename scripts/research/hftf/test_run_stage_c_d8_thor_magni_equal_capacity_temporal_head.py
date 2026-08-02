#!/usr/bin/env python3
"""Tests for the equal-capacity THOR temporal actionability head."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d8_thor_magni_equal_capacity_temporal_head import (
    TemporalActionabilityHead,
    source_balanced_weights,
)


class EqualCapacityTemporalHeadTests(unittest.TestCase):
    def test_repeated_current_is_invariant_to_temporal_weights(self) -> None:
        model = TemporalActionabilityHead(3)
        current = torch.asarray([[1.0, 2.0, 3.0]])
        repeated = current[:, None].repeat(1, 5, 1)
        first = model(repeated)
        with torch.no_grad():
            model.temporal_residual_weight.fill_(2.0)
        second = model(repeated)
        torch.testing.assert_close(first, second)

    def test_real_history_can_change_fused_output(self) -> None:
        model = TemporalActionabilityHead(2, output_count=1)
        with torch.no_grad():
            model.temporal_residual_weight.fill_(1.0)
            model.head.weight.copy_(torch.asarray([[1.0, -1.0]]))
            model.head.bias.zero_()
        history = torch.asarray(
            [[[0.0, 3.0], [0.0, 3.0], [0.0, 3.0], [0.0, 3.0], [2.0, 1.0]]]
        )
        repeated = history[:, -1:].repeat(1, 5, 1)
        self.assertNotEqual(
            float(model(history).item()),
            float(model(repeated).item()),
        )

    def test_each_source_has_equal_total_training_weight(self) -> None:
        sources = np.asarray(["a", "a", "a", "b", "c", "c"])
        weights = source_balanced_weights(sources)
        totals = [
            float(np.sum(weights[sources == source]))
            for source in ("a", "b", "c")
        ]
        np.testing.assert_allclose(totals, np.repeat(totals[0], 3))


if __name__ == "__main__":
    unittest.main()
