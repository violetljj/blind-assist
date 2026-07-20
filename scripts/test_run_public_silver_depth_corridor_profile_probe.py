#!/usr/bin/env python3
"""Pure tests for deterministic relative-depth corridor profiles."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import run_public_silver_depth_corridor_profile_probe as probe


class PublicSilverDepthCorridorProfileProbeTest(unittest.TestCase):
    def test_central_protrusion_increases_core_and_blockage_features(self) -> None:
        height, width = 48, 64
        clear = np.tile(np.linspace(0.1, 1.0, height)[:, None], (1, width))
        blocked = clear.copy()
        blocked[22:42, 25:39] += 0.8
        clear_vector = probe.depth_frame_vector(clear)
        blocked_vector = probe.depth_frame_vector(blocked)
        self.assertEqual(30, len(clear_vector))
        self.assertGreater(blocked_vector[12], clear_vector[12])
        self.assertGreater(blocked_vector[-2], clear_vector[-2])

    def test_episode_vector_is_fixed_and_temporally_sensitive(self) -> None:
        base = np.tile(np.linspace(0.1, 1.0, 40)[:, None], (1, 52))
        approaching = []
        for scale in (0.0, 0.2, 0.5):
            frame = base.copy()
            frame[18:36, 21:31] += scale
            approaching.append(frame)
        forward = probe.episode_vector(approaching)
        reverse = probe.episode_vector(list(reversed(approaching)))
        self.assertEqual(120, len(forward))
        self.assertFalse(np.array_equal(forward, reverse))

    def test_constant_depth_is_finite(self) -> None:
        vector = probe.depth_frame_vector(np.ones((32, 32), dtype=np.float32))
        self.assertTrue(np.isfinite(vector).all())

    def test_independent_direction_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "secondary-corridor-causal" / "probe.json"
            with self.assertRaisesRegex(ValueError, "independent model direction"):
                probe.reject_independent_direction(path)


if __name__ == "__main__":
    unittest.main()
