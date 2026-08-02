#!/usr/bin/env python3
"""Tests for the THOR-MAGNI spatial feature extractor."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_stage_c_d8_thor_magni_spatial_features import flush_batch


class FakeSpatialModel:
    def features(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            frames.shape[0],
            576,
            4,
            7,
            device=frames.device,
        )


class ThorMagniSpatialFeatureTests(unittest.TestCase):
    def test_flush_batch_preserves_keys_and_float16_shape(self) -> None:
        frames = [torch.zeros(3, 128, 224), torch.ones(3, 128, 224)]
        keys = [("video", 1), ("video", 2)]
        output: dict[tuple[str, int], np.ndarray] = {}
        flush_batch(
            FakeSpatialModel(),
            torch.device("cpu"),
            frames,
            keys,
            output,
        )
        self.assertEqual(frames, [])
        self.assertEqual(keys, [])
        self.assertEqual(output[("video", 1)].shape, (576, 4, 7))
        self.assertEqual(output[("video", 1)].dtype, np.float16)


if __name__ == "__main__":
    unittest.main()
