import unittest

import numpy as np

import run_advio_turn_intent_probe as subject


class AdvioTurnIntentProbeTest(unittest.TestCase):
    def test_signed_angle_has_direction(self) -> None:
        first = np.asarray([[1.0, 0.0], [1.0, 0.0]])
        second = np.asarray([[0.0, 1.0], [0.0, -1.0]])
        np.testing.assert_allclose([90.0, -90.0], subject.signed_angle_degrees(first, second))

    def test_window_statistics_is_causal_and_fixed_width(self) -> None:
        sensor = np.asarray([[t, t, 2 * t, -t] for t in np.linspace(0.0, 1.0, 101)])
        stats = subject.window_statistics(sensor, 1.0, 0.5)
        self.assertEqual(18, len(stats))
        np.testing.assert_allclose([1.0, 2.0, -1.0], stats[6:9])
        np.testing.assert_allclose([1.0, 2.0, -1.0], stats[-3:], atol=1e-10)

    def test_auc_handles_ties(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.5, 0.5, 0.9])
        self.assertAlmostEqual(0.875, subject.roc_auc(labels, scores))


if __name__ == "__main__":
    unittest.main()
