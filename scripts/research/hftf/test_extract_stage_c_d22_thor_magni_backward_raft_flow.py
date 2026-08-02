#!/usr/bin/env python3
"""Unit tests for D22 THOR current-to-history flow extraction."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from extract_stage_c_d22_thor_magni_backward_raft_flow import (
    PAIRS_PER_SAMPLE,
    ThorCurrentToHistoryPairDataset,
)


class D22ThorBackwardFlowTests(unittest.TestCase):
    def test_pair_dataset_uses_current_as_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.npy"
            cache = np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.uint8,
                shape=(1, 5, 128, 224, 3),
            )
            cache[0, 0].fill(0)
            cache[0, 4].fill(255)
            cache.flush()
            del cache
            dataset = ThorCurrentToHistoryPairDataset(
                path,
                expected_samples=1,
            )
            current, earlier, sample_index, history_index = dataset[0]
            self.assertEqual(sample_index, 0)
            self.assertEqual(history_index, 0)
            self.assertTrue(torch.all(current == 1.0))
            self.assertTrue(torch.all(earlier == -1.0))
            self.assertEqual(
                len(dataset),
                PAIRS_PER_SAMPLE,
            )
            dataset.cache._mmap.close()
            del dataset


if __name__ == "__main__":
    unittest.main()
