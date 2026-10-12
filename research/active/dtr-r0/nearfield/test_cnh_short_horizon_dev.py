"""Focused synthetic checks for the short-horizon trajectory diagnostic."""
import unittest

import numpy as np

import cnh_short_horizon_dev as S


def straight_clip(speed=.8, frames=600):
    x = np.zeros((frames, 24, 3), dtype=float)
    x[:, :, 0] = (np.arange(frames)*speed/S.HZ)[:, None]
    x[:, 7, 1], x[:, 11, 1] = -.2, .2
    x[:, 6, 2] = 1.6
    return x


class ShortHorizonTests(unittest.TestCase):
    def test_constant_velocity_all_targets_and_methods(self):
        d, receipt = S.analyse_clip(straight_clip(), 2, 0)
        self.assertFalse(receipt['invalid'])
        self.assertTrue(d['valid'].all(1).any())
        np.testing.assert_allclose(d['speed_causal'], .8, atol=1e-12)
        for j, duration in enumerate((.5, 1., 2., 1.875)):
            valid = d['valid'][:, j]
            np.testing.assert_allclose(d['elapsed'][valid, j], duration, atol=1e-12)
            np.testing.assert_allclose(d['arc'][valid, j], .8*duration, atol=1e-12)
            np.testing.assert_allclose(d['values'][valid, j, :, :7], 0., atol=1e-12)
            np.testing.assert_allclose(d['values'][valid, j, :, 7:], 1., atol=1e-12)

    def test_distance_censoring_does_not_remove_short_time_targets(self):
        d, _ = S.analyse_clip(straight_clip(.4), 2, 0)
        # At 7 seconds the clip retains every fixed-time target but not 1.5 m.
        row = np.flatnonzero(d['frame'] == 426)[0]
        self.assertTrue(d['base'][row])
        np.testing.assert_array_equal(d['valid'][row], [True, True, True, False])
        self.assertTrue(np.isfinite(d['values'][row, :3]).all())
        self.assertTrue(np.isnan(d['values'][row, 3]).all())
        self.assertGreater(d['valid'][:, 0].sum(), d['valid'][:, 3].sum())

    def test_prediction_inputs_do_not_read_future_at_each_anchor(self):
        original = straight_clip(.7)
        # Give the original a changing trajectory so EMA causality is exercised.
        original[:, :, 1] += (.05*np.sin(np.arange(len(original))*.04))[:, None]
        for anchor in (90, 210, 450):
            expected = S.predict_inputs(original, np.array([anchor]))
            changed = original.copy()
            rng = np.random.default_rng(anchor)
            changed[anchor+1:] = rng.normal(size=changed[anchor+1:].shape)*10
            actual = S.predict_inputs(changed, np.array([anchor]))
            np.testing.assert_array_equal(actual[0], expected[0])
            np.testing.assert_array_equal(actual[1], expected[1])

    def test_bidirectional_coverage_uses_finite_segments(self):
        for true_length, pred_length, expected_coverage, expected_support in (
            (2., 1., 2/3, 1.), (1., 2., 1., 2/3),
        ):
            path = np.array([[[0., 0.], [true_length/2, 0.], [true_length, 0.]]])
            fde, _, cross, miss, covered, supported = S.geometry_metrics(
                path, path[:, -1], np.array([[0.]]), np.array([pred_length]))
            np.testing.assert_allclose(fde, 1.)
            np.testing.assert_allclose(cross, 0.)
            np.testing.assert_allclose(miss, 1.)
            np.testing.assert_allclose(covered, expected_coverage)
            np.testing.assert_allclose(supported, expected_support)

    def test_bent_path_vertex_is_not_replaced_by_endpoint_chord(self):
        path = np.array([[[0., 0.], [1., 0.], [1., 1.]]])
        fde, _, _, miss, covered, supported = S.geometry_metrics(
            path, path[:, -1], np.array([[45.]]), np.array([np.sqrt(2.)]))
        np.testing.assert_allclose(fde, 0., atol=1e-12)
        np.testing.assert_allclose(miss, np.sqrt(.5))
        np.testing.assert_allclose(covered, 2/3)
        np.testing.assert_allclose(supported, 2/3)


if __name__ == '__main__':
    unittest.main()
