from __future__ import annotations

import unittest

import numpy as np

from .core import corridor_zone_masks, pool_probabilities, stable_softmax_from_int8


CONTRACT = {
    "corridor_geometry": {
        "top_ratio": 0.42,
        "top_half_width_ratio": 0.16,
        "bottom_half_width_ratio": 0.42,
    }
}
CONFIGS = [
    {
        "config_id": "test",
        "boundary_weight": 1.0,
        "top_fraction": 0.01,
        "lateral_weights": [0.5, 1.0, 0.5],
    }
]


class CoreTest(unittest.TestCase):
    def test_softmax_is_finite_and_normalized(self) -> None:
        raw = np.asarray([[[[-128, -10, 20, 127]]]], dtype=np.int8)
        result = stable_softmax_from_int8(raw, 0.1)
        self.assertTrue(np.isfinite(result).all())
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, rtol=1e-6)

    def test_corridor_zones_are_disjoint_and_cover_corridor(self) -> None:
        corridor, zones = corridor_zone_masks(
            288, 512, CONTRACT["corridor_geometry"]
        )
        combined = zones[0].astype(int) + zones[1].astype(int) + zones[2].astype(int)
        self.assertTrue(np.array_equal(combined > 0, corridor))
        self.assertEqual(int(combined.max()), 1)

    def test_center_obstacle_scores_above_walkable(self) -> None:
        walkable = np.zeros((32, 32, 4), dtype=np.float32)
        walkable[..., 0] = 1.0
        danger = walkable.copy()
        _, zones = corridor_zone_masks(32, 32, CONTRACT["corridor_geometry"])
        danger[zones[1], 0] = 0.0
        danger[zones[1], 1] = 1.0
        safe_scores, _ = pool_probabilities(walkable, CONTRACT, CONFIGS)
        danger_scores, _ = pool_probabilities(danger, CONTRACT, CONFIGS)
        self.assertEqual(safe_scores["test"], 0.0)
        self.assertGreater(danger_scores["test"], 0.9)


if __name__ == "__main__":
    unittest.main()

