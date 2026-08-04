#!/usr/bin/env python3

import math
import unittest

import numpy as np

from evaluate_fresh_tf_r0 import UNKNOWN, arm_state, scene_change


POLICY = {
    "hard_ttl_s": 0.75,
    "quality_threshold": 0.5,
    "uniform_tau_s": 0.5,
    "selective_tau_s": {"CLEAR_OBSERVED": 0.35, "OCCUPIED_OBSERVED": 0.75},
    "rgb_change": {"scale": 0.08},
}


class FreshTfR0Test(unittest.TestCase):
    def test_zero_order_hold_never_abstains(self) -> None:
        state, quality = arm_state(
            "fixed_2hz_zero_order_hold",
            "CLEAR_OBSERVED",
            age_s=99.0,
            rgb_change=1.0,
            policy=POLICY,
        )
        self.assertEqual("CLEAR_OBSERVED", state)
        self.assertEqual(1.0, quality)

    def test_hard_ttl_fails_closed(self) -> None:
        for arm in (
            "fixed_2hz_ttl_750ms",
            "uniform_age_freshness",
            "selective_rgb_change_freshness",
        ):
            state, _ = arm_state(
                arm,
                "CLEAR_OBSERVED",
                age_s=0.751,
                rgb_change=0.0,
                policy=POLICY,
            )
            self.assertEqual(UNKNOWN, state)

    def test_unknown_anchor_never_becomes_clear(self) -> None:
        for arm in (
            "fixed_2hz_zero_order_hold",
            "fixed_2hz_ttl_750ms",
            "uniform_age_freshness",
            "selective_rgb_change_freshness",
        ):
            state, quality = arm_state(
                arm,
                "UNKNOWN_SUPPORT",
                age_s=0.0,
                rgb_change=0.0,
                policy=POLICY,
            )
            self.assertEqual(UNKNOWN, state)
            self.assertEqual(0.0, quality)

    def test_selective_policy_expires_clear_before_blocked(self) -> None:
        clear, clear_quality = arm_state(
            "selective_rgb_change_freshness",
            "CLEAR_OBSERVED",
            age_s=0.4,
            rgb_change=0.0,
            policy=POLICY,
        )
        blocked, blocked_quality = arm_state(
            "selective_rgb_change_freshness",
            "OCCUPIED_OBSERVED",
            age_s=0.4,
            rgb_change=0.0,
            policy=POLICY,
        )
        self.assertEqual(UNKNOWN, clear)
        self.assertEqual("OCCUPIED_OBSERVED", blocked)
        self.assertLess(clear_quality, blocked_quality)

    def test_rgb_change_only_reduces_quality(self) -> None:
        stable = arm_state(
            "selective_rgb_change_freshness",
            "OCCUPIED_OBSERVED",
            age_s=0.1,
            rgb_change=0.0,
            policy=POLICY,
        )[1]
        changed = arm_state(
            "selective_rgb_change_freshness",
            "OCCUPIED_OBSERVED",
            age_s=0.1,
            rgb_change=0.08,
            policy=POLICY,
        )[1]
        self.assertAlmostEqual(stable / math.e, changed)

    def test_scene_change_has_fixed_normalization(self) -> None:
        first = np.zeros((48, 64), dtype=np.float32)
        second = np.full((48, 64), 0.25, dtype=np.float32)
        self.assertAlmostEqual(0.25, scene_change(first, second))


if __name__ == "__main__":
    unittest.main()
