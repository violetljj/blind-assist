from __future__ import annotations

import math
import unittest

from scripts.research.dual_loop_causal_track_tristate_r0.common import (
    frame_detection_id,
    immutable_roi_id,
    ols_slope,
    source_decision,
)


class CausalTrackTristateTest(unittest.TestCase):
    def test_ols_recovers_known_slope(self) -> None:
        times = [index / 15 for index in range(7)]
        values = [0.3 * value + 2.0 for value in times]
        self.assertAlmostEqual(ols_slope(times, values), 0.3)

    def test_unanimous_growth_confirms(self) -> None:
        times = [index / 15 for index in range(7)]
        values = [math.log(100 * math.exp(0.25 * value)) for value in times]
        decision, slope, reason = source_decision(times, values)
        self.assertEqual(decision, "CONFIRM_APPROACH")
        self.assertGreaterEqual(slope or 0.0, 0.2)
        self.assertEqual(reason, "SEVEN_FRAME_UNANIMOUS_GROWTH")

    def test_one_opposite_step_abstains(self) -> None:
        times = [index / 15 for index in range(7)]
        values = [0.0, 0.02, 0.04, 0.03, 0.08, 0.10, 0.12]
        self.assertEqual(source_decision(times, values)[0], "ABSTAIN")

    def test_identity_binds_frame_track_and_roi(self) -> None:
        first = frame_detection_id("s", "000001", "pedestrian:1")
        second = frame_detection_id("s", "000001", "pedestrian:2")
        self.assertNotEqual(first, second)
        self.assertNotEqual(
            immutable_roi_id(first, [1.0, 2.0, 3.0, 4.0]),
            immutable_roi_id(first, [1.0, 2.0, 3.0, 5.0]),
        )


if __name__ == "__main__":
    unittest.main()
