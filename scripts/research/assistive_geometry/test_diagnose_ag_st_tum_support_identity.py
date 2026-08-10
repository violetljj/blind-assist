#!/usr/bin/env python3

import unittest

import numpy as np

from diagnose_ag_st_tum_support_identity import (
    SupportIdentityPolicy,
    _mode_candidates,
    classify_dominant_plane,
)


class TumSupportIdentityTest(unittest.TestCase):
    def test_lowest_mode_must_persist(self) -> None:
        policy = SupportIdentityPolicy(
            minimum_frame_points=20,
            minimum_total_points=60,
            sample_stride=1,
        )
        rng = np.random.default_rng(17)
        frames = []
        for _ in range(3):
            floor = rng.normal(0.0, 0.01, 400)
            table = rng.normal(0.78, 0.01, 900)
            frames.append(np.concatenate((floor, table)))
        frames[0] = np.concatenate((frames[0], rng.normal(-0.5, 0.01, 200)))
        modes = _mode_candidates(frames, policy)
        self.assertGreaterEqual(len(modes), 2)
        self.assertAlmostEqual(0.0, modes[0]["world_height_m"], delta=0.04)
        self.assertGreaterEqual(modes[0]["persistent_frame_count"], 2)

    def test_elevated_and_supported_classification(self) -> None:
        policy = SupportIdentityPolicy()
        self.assertEqual(
            "LOWEST_PERSISTENT_SURFACE_SUPPORTED",
            classify_dominant_plane(0.08, 0.02, policy),
        )
        self.assertEqual(
            "ELEVATED_DOMINANT_SURFACE_REJECTED",
            classify_dominant_plane(0.78, 0.02, policy),
        )
        self.assertEqual(
            "UNKNOWN_HEIGHT_IDENTITY_AMBIGUOUS",
            classify_dominant_plane(0.20, 0.02, policy),
        )
        self.assertEqual(
            "UNKNOWN_NO_PERSISTENT_HORIZONTAL_MODE",
            classify_dominant_plane(0.78, None, policy),
        )


if __name__ == "__main__":
    unittest.main()
