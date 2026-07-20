#!/usr/bin/env python3
"""Pure tests for the isolated public-silver free-space topology descriptor."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import run_public_silver_free_space_topology_probe as probe


def logits_from_classes(classes: np.ndarray) -> np.ndarray:
    height, width = classes.shape
    logits = np.full((height, width, 4), -4.0, dtype=np.float64)
    for class_id in range(4):
        logits[..., class_id][classes == class_id] = 4.0
    return logits


class PublicSilverFreeSpaceTopologyProbeTest(unittest.TestCase):
    def test_full_width_obstruction_reduces_walkable_width(self) -> None:
        clear_classes = np.zeros((48, 64), dtype=np.int64)
        blocked_classes = clear_classes.copy()
        blocked_classes[24:39, :] = 2
        clear, _ = probe.free_space_topology_frame(logits_from_classes(clear_classes))
        blocked, _ = probe.free_space_topology_frame(logits_from_classes(blocked_classes))
        minimum_width_index = probe.FRAME_FEATURE_NAMES.index("path_width_t05_minimum")
        obstacle_index = probe.FRAME_FEATURE_NAMES.index("path_obstacle_maximum")
        self.assertGreater(clear[minimum_width_index], blocked[minimum_width_index])
        self.assertGreater(blocked[obstacle_index], clear[obstacle_index])

    def test_lateral_obstruction_changes_adaptive_path(self) -> None:
        classes = np.full((48, 64), 2, dtype=np.int64)
        classes[:, 16:48] = 0
        classes[12:30, 20:36] = 2
        _vector, summary = probe.free_space_topology_frame(logits_from_classes(classes))
        self.assertGreater(summary["path_center_range"], 0.05)

    def test_episode_vector_is_temporally_sensitive(self) -> None:
        first = np.zeros(len(probe.FRAME_FEATURE_NAMES), dtype=np.float64)
        second = np.ones(len(probe.FRAME_FEATURE_NAMES), dtype=np.float64)
        forward = probe.topology_episode_vector(np.stack([first, second]))
        reverse = probe.topology_episode_vector(np.stack([second, first]))
        self.assertEqual(len(probe.FRAME_FEATURE_NAMES) * 4, len(forward))
        self.assertFalse(np.array_equal(forward, reverse))

    def test_independent_direction_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "secondary-corridor-causal" / "probe.json"
            with self.assertRaisesRegex(ValueError, "independent model direction"):
                probe.reject_independent_direction(path)


if __name__ == "__main__":
    unittest.main()
