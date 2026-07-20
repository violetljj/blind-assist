import unittest

import numpy as np

import run_advio_turn_intent_invariant_probe as subject


class AdvioTurnIntentInvariantProbeTest(unittest.TestCase):
    def test_invariant_statistics_ignore_axis_rotation(self) -> None:
        time = np.linspace(0.0, 1.0, 101)
        values = np.column_stack([time, time, 2.0 * time, -time])
        rotated = values.copy()
        rotated[:, 1:4] = values[:, 1:4] @ np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(subject.invariant_window_statistics(values, 1.0, 0.5),
                                   subject.invariant_window_statistics(rotated, 1.0, 0.5), atol=1e-10)

    def test_feature_width_is_eleven_per_sensor(self) -> None:
        time = np.linspace(0.0, 1.0, 101)
        values = np.column_stack([time, time, 2.0 * time, -time])
        self.assertEqual(11, len(subject.invariant_window_statistics(values, 1.0, 0.5)))


if __name__ == "__main__":
    unittest.main()
