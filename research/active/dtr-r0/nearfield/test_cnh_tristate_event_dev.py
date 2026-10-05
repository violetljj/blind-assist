"""Focused causal-deadline and observed-exposure regression checks."""
import unittest

import numpy as np

import cnh_tristate_event_dev as E


class EventContractTests(unittest.TestCase):
    def outputs(self, n=1):
        return np.full((n, 13), -1.), np.ones((n, 13), bool)

    def test_exact_saved_deadline_includes_current_frame(self):
        score, gate = self.outputs()
        score[0, 10] = E.R.THRESHOLD  # Frame 13.
        state, _, _, alarm = E.partition(score, gate, 0., [13.], [True])
        self.assertEqual(int(E.causal_index(13.)), 10)
        self.assertTrue(alarm[0, 10])
        self.assertEqual(state.tolist(), [0])

    def test_interpolated_deadline_does_not_use_right_frame(self):
        score, gate = self.outputs()
        score[0, 11] = E.R.THRESHOLD  # Frame 14 is after deadline.
        state, clear, _, _ = E.partition(score, gate, 0., [13.25], [True])
        self.assertEqual(int(E.causal_index(13.25)), 10)
        self.assertTrue(clear[0, 10])
        self.assertEqual(state.tolist(), [2])

    def test_before_integer_deadline_uses_previous_frame(self):
        score, gate = self.outputs()
        score[0, 10] = E.R.THRESHOLD
        state, _, _, _ = E.partition(score, gate, 0., [13.-1e-8], [True])
        self.assertEqual(int(E.causal_index(13.-1e-8)), 9)
        self.assertEqual(state.tolist(), [2])

    def test_saved_last_frame_is_usable_for_event(self):
        score, gate = self.outputs()
        score[0, 12] = E.R.THRESHOLD
        state, _, _, _ = E.partition(score, gate, 0., [15.], [True])
        self.assertEqual(state.tolist(), [0])

    def test_prior_alarm_remains_timely_when_current_state_is_clear(self):
        score, gate = self.outputs()
        score[0, 0] = E.R.THRESHOLD
        state, clear, _, _ = E.partition(score, gate, 0., [13.25], [True])
        self.assertTrue(clear[0, 10])
        self.assertEqual(state.tolist(), [0])

    def test_noncontact_is_not_evaluated_and_partition_is_exhaustive(self):
        score, gate = self.outputs(4)
        score[0, 10] = E.R.THRESHOLD
        gate[1, 10] = False
        state, clear, unknown, alarm = E.partition(
            score, gate, 0., [13.25, 13.25, 13.25, np.nan],
            [True, True, True, False])
        self.assertEqual(state.tolist(), [0, 1, 2, -1])
        np.testing.assert_array_equal(
            clear.astype(int)+unknown.astype(int)+alarm.astype(int),
            np.ones_like(score, int))

    def test_excluded_frames_do_not_add_exposure_or_entries(self):
        unknown = np.zeros((1, 13), bool)
        unknown[0, [0, 1, 12]] = True  # Warmup and frame 15 only.
        b = E.burden(unknown)
        np.testing.assert_array_equal(b['seconds'], [0.])
        np.testing.assert_array_equal(b['total_seconds'], [2.])
        np.testing.assert_array_equal(b['starts'], [0])
        np.testing.assert_array_equal(b['initial_unknown'], [0])

    def test_initial_unknown_is_left_censored_not_a_new_entry(self):
        unknown = np.zeros((2, 13), bool)
        unknown[0, 2:5] = True  # Frame 5 begins an already-active segment.
        unknown[0, 6:8] = True  # One new segment at frame 9.
        unknown[1, 2:12] = True  # Entire observed two-second interval.
        b = E.burden(unknown)
        np.testing.assert_allclose(b['seconds'], [1., 2.])
        np.testing.assert_array_equal(b['total_seconds'], [2., 2.])
        np.testing.assert_array_equal(b['starts'], [1, 0])
        np.testing.assert_array_equal(b['initial_unknown'], [1, 1])


if __name__ == '__main__':
    unittest.main()
