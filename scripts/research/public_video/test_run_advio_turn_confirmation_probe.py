import unittest

import numpy as np

import run_advio_turn_confirmation_probe as subject


class AdvioTurnConfirmationProbeTest(unittest.TestCase):
    def test_target_is_past_and_current_only(self) -> None:
        pose_time = np.linspace(0.0, 3.0, 301)
        x = np.minimum(pose_time, 1.5)
        z = np.maximum(pose_time - 1.5, 0.0)
        pose = np.column_stack([pose_time, x, np.zeros_like(x), z,
                                np.ones_like(x), np.zeros((len(x), 3))])
        first = subject.base.interpolate_xz(pose, np.asarray([1.5])) - subject.base.interpolate_xz(pose, np.asarray([1.0]))
        second = subject.base.interpolate_xz(pose, np.asarray([2.0])) - subject.base.interpolate_xz(pose, np.asarray([1.5]))
        self.assertAlmostEqual(90.0, abs(subject.base.signed_angle_degrees(first, second)[0]))


if __name__ == "__main__":
    unittest.main()
