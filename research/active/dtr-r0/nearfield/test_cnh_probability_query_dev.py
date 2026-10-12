"""CPU-only checks for probability-query algebra and calibration separation."""

import unittest

import numpy as np

from cnh_probability_query_dev import (
    combine,
    split_units,
    symmetric_weights,
    threshold_at_cap,
)


class ProbabilityQueryTests(unittest.TestCase):
    def test_probability_mass_is_normalized_and_mirror_symmetric(self):
        errors = np.array([-8., -2., -1., 0., 1., 2., 8.])
        weights = symmetric_weights(errors, 4.)
        np.testing.assert_allclose(weights, [5 / 7, 1 / 7, 1 / 7])
        np.testing.assert_allclose(symmetric_weights(-errors, 4.), weights)
        skewed_errors = np.array([0., 1., 8., 9., 10.])
        np.testing.assert_allclose(symmetric_weights(skewed_errors, 4.), [.4, .3, .3])
        np.testing.assert_allclose(symmetric_weights(-skewed_errors, 4.), [.4, .3, .3])
        self.assertAlmostEqual(float(np.sum(weights)), 1.)
        self.assertTrue(np.all(np.asarray(weights) >= 0.))

        # Tail direction must not introduce a handedness preference.
        scores = np.array([[.1, .9, .3], [.7, .2, .6]])
        np.testing.assert_allclose(combine(scores, weights),
                                   combine(scores[:, [0, 2, 1]], weights))
        np.testing.assert_allclose(symmetric_weights([0., 1., -1.], 4.), [1., 0., 0.])
        np.testing.assert_allclose(symmetric_weights([-8., 8.], 4.), [0., .5, .5])

    def test_combine_broadcasts_per_frame_weights_without_changing_equal_scores(self):
        scores = np.array([[[1., 3., 5.], [2., 4., 6.]],
                           [[5., 3., 1.], [6., 4., 2.]]])
        weights = np.array([[.5, .25, .25], [1., 0., 0.]])
        np.testing.assert_allclose(combine(scores, weights), [[2.5, 2.], [3.5, 6.]])
        equal_scores = np.repeat(np.array([[.1, .8], [.3, .5]])[..., None], 3, axis=-1)
        np.testing.assert_allclose(combine(equal_scores, weights), equal_scores[..., 0])

    def test_threshold_is_lowest_float_that_obeys_cap_with_ties(self):
        cases = (
            (np.array([0., 1., 2., 3.]), .25),
            (np.array([0.] * 50 + [1.] * 46 + [2.] * 4), .05),
            (np.array([0.] * 50 + [1.] * 46 + [2.] * 4), .025),
            (np.full(40, .7), .025),
            (np.array([-1., 0., 1.]), 0.),
        )
        for controls, cap in cases:
            with self.subTest(cap=cap, values=np.unique(controls).tolist()):
                threshold = float(threshold_at_cap(controls, cap))
                self.assertLessEqual(float(np.mean(controls >= threshold)), cap)
                predecessor = np.nextafter(threshold, -np.inf)
                self.assertGreater(float(np.mean(controls >= predecessor)), cap)

    def test_positive_affine_rescaling_preserves_calibrated_alarms(self):
        rng = np.random.default_rng(2026100711)
        weights = symmetric_weights(rng.normal(0., 8., 1000), 12.)
        controls = rng.uniform(0., 1., (800, 3))
        candidates = rng.uniform(0., 1., (400, 3))
        control_scores = combine(controls, weights)
        candidate_scores = combine(candidates, weights)
        threshold = threshold_at_cap(control_scores, .025)
        for scale, offset in ((4., 3.), (.25, -2.)):
            with self.subTest(scale=scale, offset=offset):
                transformed_control = combine(scale * controls + offset, weights)
                transformed_candidates = combine(scale * candidates + offset, weights)
                transformed_threshold = threshold_at_cap(transformed_control, .025)
                np.testing.assert_array_equal(candidate_scores >= threshold,
                                              transformed_candidates >= transformed_threshold)
                np.testing.assert_array_equal(np.argsort(candidate_scores),
                                              np.argsort(transformed_candidates))
                self.assertLessEqual(float(np.mean(transformed_control >= transformed_threshold)), .025)

    def test_split_uses_whole_units_and_retains_both_sides_of_each_stratum(self):
        rows = []
        strata = {}
        unit = 98000
        for batch in (98000, 99000):
            for mode in (0, 1, 2):
                for turn in (False, True):
                    key = (batch, mode, turn)
                    strata[key] = set()
                    for _ in range(9):
                        strata[key].add(unit)
                        # Each unit contributes several observations, never split independently.
                        rows.extend(dict(unit=unit, batch=batch, mode=mode, turn=turn)
                                    for _ in range(3))
                        unit += 1
        calibration = set(split_units(rows))
        self.assertEqual(calibration, set(split_units(rows)))
        all_units = {row["unit"] for row in rows}
        self.assertTrue(calibration <= all_units)
        evaluation = all_units - calibration
        self.assertFalse(calibration & evaluation)
        for key, units in strata.items():
            with self.subTest(stratum=key):
                self.assertGreaterEqual(len(units & calibration), 2)
                self.assertLessEqual(len(units & calibration), 4)
                self.assertTrue(units & evaluation)
        # Duplicate rows are extra observations of the same unit, not more split units.
        self.assertEqual(calibration, set(split_units(rows + rows)))


if __name__ == "__main__":
    unittest.main()
