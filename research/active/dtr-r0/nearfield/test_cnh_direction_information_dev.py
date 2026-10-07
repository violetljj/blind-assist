"""Focused CPU checks for causal readout features and unit-held-out probes."""

from collections import Counter
import unittest

import numpy as np

from cnh_direction_information_dev import causal_features, make_folds, scramble_sides, training_rows


class DirectionInformationTests(unittest.TestCase):
    def test_side_scramble_preserves_role_centers_widths_and_width_matched_history(self):
        rows = [dict(batch=98000, mode=0, turn=False, config=0) for _ in range(7)]
        x = np.zeros((7, 13, 36))
        for i in range(7):
            for t in range(13):
                x[i, t] = i * 10000 + t * 100 + np.arange(36)
        # Three width-matched pairs and one held-out role. Width differences in
        # either current or historical coordinates must prohibit cross-pair donors.
        for i, widths in enumerate(((1., 2., 3.), (1., 2., 3.),
                                   (9., 2., 3.), (9., 2., 3.),
                                   (1., 2., 8.), (1., 2., 8.), (1., 2., 3.))):
            for t in range(13):
                x[i, t, [10, 22, 34]] = widths
        mask = np.array([True] * 6 + [False])
        out, stats = scramble_sides(x, rows, mask, 2026100711)
        side_columns = np.array([off + j for off in (0, 12, 24) for j in range(2, 10)])
        fixed_columns = np.array([j for j in range(36) if j not in side_columns])
        np.testing.assert_array_equal(out[:, :, fixed_columns], x[:, :, fixed_columns])
        np.testing.assert_array_equal(out[~mask], x[~mask])
        for receiver, donor in ((0, 1), (1, 0), (2, 3), (3, 2), (4, 5), (5, 4)):
            np.testing.assert_array_equal(out[receiver][:, side_columns], x[donor][:, side_columns])
        self.assertEqual(6 * 13, stats["matched_changed_frames"])
        self.assertEqual(6 * 13, stats["total_frames"])

    def test_lag_features_have_explicit_order_and_clip_only_at_start(self):
        data = np.arange(2 * 6 * 2, dtype=float).reshape(2, 6, 2)
        actual = causal_features(data, [0, 1, 3])
        expected = np.concatenate([
            data[:, np.maximum(np.arange(6) - lag, 0), :]
            for lag in (0, 1, 3)
        ], axis=-1)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual((2, 6, 6), actual.shape)

    def test_future_mutation_cannot_change_current_or_past_features(self):
        data = np.arange(3 * 13 * 4, dtype=float).reshape(3, 13, 4)
        lags = [0, 1, 2, 4]
        before = causal_features(data, lags)
        changed = data.copy()
        changed[:, 6:, :] = -100000.
        after = causal_features(changed, lags)
        np.testing.assert_array_equal(before[:, :6, :], after[:, :6, :])
        self.assertFalse(np.array_equal(before[:, 6:, :], after[:, 6:, :]))

    def test_folds_keep_all_configs_of_a_unit_together_and_evaluate_once(self):
        rows = []
        for batch in (98000, 99000):
            offset = 0
            for mode in (0, 1, 2):
                for turn in (False, True):
                    for _ in range(9):
                        unit = batch + offset
                        offset += 1
                        rows.extend(dict(unit=unit, batch=batch, mode=mode,
                                         turn=turn, config=config)
                                    for config in ("single", "dual"))
        all_units = {row["unit"] for row in rows}
        folds = make_folds(rows)
        self.assertEqual(3, len(folds))
        evaluations = Counter()
        for number, fold in enumerate(folds):
            with self.subTest(fold=number):
                train, calibration, evaluation = (
                    set(fold[key]) for key in ("train", "calibration", "evaluation")
                )
                self.assertTrue(train and calibration and evaluation)
                self.assertFalse(train & calibration)
                self.assertFalse(train & evaluation)
                self.assertFalse(calibration & evaluation)
                self.assertEqual(all_units, train | calibration | evaluation)
                for key in ("train", "calibration", "evaluation"):
                    self.assertEqual(len(fold[key]), len(set(fold[key])))
                evaluations.update(evaluation)
                for unit in all_units:
                    config_memberships = {
                        (row["unit"] in train, row["unit"] in calibration,
                         row["unit"] in evaluation)
                        for row in rows if row["unit"] == unit
                    }
                    self.assertEqual(1, len(config_memberships))
        self.assertEqual({unit: 1 for unit in all_units}, dict(evaluations))
        # Config count cannot alter which independent units are held out.
        deduplicated = [row for row in rows if row["config"] == "single"]
        canonical = lambda fs: [{key: set(fold[key]) for key in
                                ("train", "calibration", "evaluation")} for fold in fs]
        self.assertEqual(canonical(folds), canonical(make_folds(deduplicated)))

    def test_training_rows_respect_masks_deadlines_and_time_balanced_weights(self):
        contact = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
        control = np.array([0, 0, 0, 0, 1, 1, 1, 0, 0], dtype=bool)
        deadline = np.array([2, 2, 8, 10, -1, -1, -1, -1, -1])
        trainmask = np.array([1, 1, 1, 0, 1, 1, 0, 1, 0], dtype=bool)
        ri, ti, y, w = map(np.asarray, training_rows(contact, control, deadline, trainmask))
        self.assertEqual(ri.shape, ti.shape)
        self.assertEqual(ri.shape, y.shape)
        self.assertEqual(ri.shape, w.shape)
        self.assertTrue(np.all(trainmask[ri]))
        self.assertTrue(np.all(np.isfinite(w) & (w >= 0)))
        self.assertTrue(np.all((ti >= 0) & (ti < 13)))
        self.assertEqual({0, 1}, set(y.tolist()))
        positive = y == 1
        negative = y == 0
        self.assertEqual({(0, 2), (1, 2), (2, 8)},
                         set(zip(ri[positive].tolist(), ti[positive].tolist())))
        self.assertEqual(3, int(np.sum(positive)))
        np.testing.assert_allclose(w[positive], 1.)
        self.assertTrue(np.all(control[ri[negative]]))
        self.assertFalse(np.any(contact[ri[negative]]))
        self.assertFalse(np.any(np.isin(ri, [3, 6, 7, 8])))
        self.assertAlmostEqual(float(w[positive].sum()), float(w[negative].sum()))
        self.assertEqual({(4, 2), (5, 2), (4, 8), (5, 8)},
                         set(zip(ri[negative & (w > 0)].tolist(),
                                 ti[negative & (w > 0)].tolist())))
        for frame in range(13):
            with self.subTest(frame=frame):
                pos_mass = w[positive & (ti == frame)].sum()
                neg_mass = w[negative & (ti == frame)].sum()
                self.assertAlmostEqual(float(pos_mass), float(neg_mass))

    def test_unknown_is_not_a_control_and_conflicting_labels_are_rejected(self):
        with self.assertRaises(ValueError):
            training_rows(np.array([True, False]), np.array([True, True]),
                          np.array([2, -1]), np.array([True, True]))
        contact = np.array([True, False, False])
        control = np.array([False, True, False])
        mask = np.ones(3, dtype=bool)
        baseline = training_rows(contact, control, np.array([2, -1, -1]), mask)
        unknown_changed = training_rows(contact, control, np.array([2, -1, 12]), mask)
        for before, after in zip(baseline, unknown_changed):
            np.testing.assert_array_equal(before, after)
        for invalid_deadline in (-1, 13):
            with self.subTest(invalid_deadline=invalid_deadline), self.assertRaises(ValueError):
                training_rows(contact, control, np.array([invalid_deadline, -1, -1]), mask)


if __name__ == "__main__":
    unittest.main()
