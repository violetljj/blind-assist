"""Focused no-fit checks of querywise supervision and monotone union scoring."""
import unittest

import numpy as np

from cnh_querywise_calibration_dev import (
    event_states,
    metrics,
    predict_query_models,
    supervision_masks,
    training_rows_for_arm,
)


class MonotoneModel:
    def __init__(self, slope, offset):
        self.slope, self.offset = slope, offset

    def predict(self, x):
        return np.clip(self.slope * np.asarray(x) + self.offset, 0., 1.)


class QuerywiseCalibrationTests(unittest.TestCase):
    def test_hb_clear_with_missing_deadline_encoded_minus_one_counts_fa_not_contact(self):
        alarm = np.zeros((2,13),dtype=bool)
        alarm[0,3] = True
        alarm[1,:] = True
        contact = np.array([True,False])
        control = np.array([False,True])
        deadline = np.array([4,-1])
        timely,post = event_states(alarm,contact,deadline)
        np.testing.assert_array_equal(timely,[True,False])
        np.testing.assert_array_equal(post,[True,False])
        result = metrics(alarm,contact,control,deadline,np.ones(2,dtype=bool))
        self.assertEqual(1,result['events'])
        self.assertEqual(1,result['timely_excluding_warmup'])
        self.assertEqual(10,result['false_alarm_intervals'])
        self.assertEqual(10,result['control_intervals'])
        self.assertEqual(2,result['warmup_alarm_frames'])

    def setUp(self):
        self.query = np.array([[1,0], [0,1], [1,1], [0,0], [0,0], [0,0], [1,0]], dtype=bool)
        self.contact = self.query.any(axis=1)
        self.control = np.array([0,0,0,1,1,0,0], dtype=bool)
        self.deadline = np.array([10,11,9,-1,-1,-1,12])
        self.train = np.array([1,1,0,1,1,1,0], dtype=bool)

    def test_query_supervision_does_not_relabel_other_height_contact_as_clear(self):
        for q, expected in ((0,{0,2,6}), (1,{1,2})):
            with self.subTest(query=q):
                positive, negative, ignored = map(np.asarray, supervision_masks(
                    'query_isotonic', q, self.query, self.contact, self.control))
                self.assertEqual(expected, set(np.flatnonzero(positive)))
                np.testing.assert_array_equal(negative, self.control)
                np.testing.assert_array_equal(ignored, ~(positive | negative))
                self.assertFalse(np.any(positive & negative))
                self.assertTrue(ignored[5])  # Neither contact nor strict clear.
                self.assertTrue(ignored[1 if q == 0 else 0])

    def test_event_supervision_is_same_label_for_both_queries_with_strict_controls(self):
        for q in (0,1):
            positive, negative, ignored = supervision_masks(
                'event_isotonic', q, self.query, self.contact, self.control)
            np.testing.assert_array_equal(positive, self.contact)
            np.testing.assert_array_equal(negative, self.control)
            np.testing.assert_array_equal(ignored, ~(self.contact | self.control))

    def test_training_uses_only_own_positive_deadlines_and_train_strict_controls(self):
        for arm in ('query_isotonic', 'event_isotonic'):
            for q in (0,1):
                with self.subTest(arm=arm, query=q):
                    ri, ti, y, w = map(np.asarray, training_rows_for_arm(
                        arm, q, self.query, self.contact, self.control, self.deadline, self.train))
                    expected = {(0,10), (1,11)} if arm == 'event_isotonic' else {(q,10+q)}
                    positive, negative = y == 1, y == 0
                    self.assertEqual(expected, set(zip(ri[positive],ti[positive])))
                    self.assertEqual(len(expected), int(positive.sum()))
                    self.assertTrue(np.all(self.train[ri]))
                    self.assertTrue(np.all(self.control[ri[negative]]))
                    self.assertFalse(np.any(np.isin(ri, [2,5,6])))
                    np.testing.assert_allclose(w[positive], 1.)
                    self.assertAlmostEqual(float(w[positive].sum()), float(w[negative].sum()))
                    for frame in range(13):
                        self.assertAlmostEqual(float(w[positive & (ti == frame)].sum()),
                                               float(w[negative & (ti == frame)].sum()))
                    # Changing held-out labels/deadlines cannot change fitted rows or weights.
                    changed_q = self.query.copy(); changed_q[[2,6]] = False
                    changed_d = self.deadline.copy(); changed_d[[2,6]] = -1
                    changed = training_rows_for_arm(arm, q, changed_q, changed_q.any(axis=1),
                                                    self.control, changed_d, self.train)
                    for before, after in zip((ri,ti,y,w), changed):
                        np.testing.assert_array_equal(before, after)

    def test_query_models_keep_head_body_order_and_max_only_after_independent_predictions(self):
        x = np.array([[[-3.,8.], [6.,-2.], [2.,3.]],
                      [[4.,1.], [-1.,-3.], [0.,7.]]])
        models = [MonotoneModel(.1,.2), MonotoneModel(.05,.3)]
        per_query, score = predict_query_models(models, x)
        expected = np.stack([models[q].predict(x[...,q]) for q in (0,1)], axis=-1)
        self.assertEqual(x.shape, np.asarray(per_query).shape)
        self.assertEqual(x.shape[:-1], np.asarray(score).shape)
        np.testing.assert_array_equal(per_query, expected)
        np.testing.assert_array_equal(score, expected.max(axis=-1))
        for q in (0,1):
            raised = x.copy(); raised[...,q] += 4.
            raised_per, raised_score = predict_query_models(models, raised)
            self.assertTrue(np.all(raised_score >= score))
            np.testing.assert_array_equal(raised_per[...,1-q], per_query[...,1-q])
            self.assertTrue(np.all(raised_per[...,q] >= per_query[...,q]))


if __name__ == '__main__':
    unittest.main()
