"""CPU contract checks for the frozen EMA and its disclosed origin adapter."""
import unittest
import numpy as np
import cnh_torso_ema_compare_dev as E
import cnh_real_head_confirm as RC
import cnh_cvr_pilot as CP


def fixture():
    sensor = np.broadcast_to(np.eye(4), (2, 16, 4, 4)).copy()
    noisy = sensor.copy()
    origin = np.zeros((2, 16, 3))
    for c in range(2):
        for f in range(16):
            sensor[c, f, :3, :3] = CP.rotation(17., 'y') @ CP.rotation(-10., 'x')
            sensor[c, f, :3, 3] = [.3 + .03*f, .04, .16*f]
            noisy[c, f, :3, :3] = CP.rotation(29. + c, 'y') @ CP.rotation(-10., 'x')
            noisy[c, f, :3, 3] = [.04*f + .02*np.sin(f), 0., .14*f]
            origin[c, f] = [.02*f, 0., .15*f]
    return sensor, noisy, origin


class FrozenEMAContracts(unittest.TestCase):
    def test_original_frozen_function_and_first_difference_seed(self):
        sensor, noisy, origin = fixture()
        q, rel, heading = E.ema_queries(sensor, noisy, origin)
        for c in range(2):
            np.testing.assert_array_equal(rel[c], RC.ema_rel(noisy[c]))
            for f in range(16):
                np.testing.assert_array_equal(q[c, f, :3, :3], RC.rel_query(rel[c, f])[:3, :3])
        np.testing.assert_allclose(heading[:, 0], 0., atol=1e-12)
        d = noisy[:, 1, [0, 2], 3] - noisy[:, 0, [0, 2], 3]
        np.testing.assert_allclose(heading[:, 1], np.degrees(np.arctan2(d[:, 0], d[:, 1])), atol=1e-12)

    def test_current_noisy_yaw_is_preserved(self):
        sensor, noisy, origin = fixture()
        q, rel, heading = E.ema_queries(sensor, noisy, origin)
        changed = noisy.copy()
        changed[..., :3, :3] = CP.rotation(11., 'y') @ changed[..., :3, :3]
        qq, rr, hh = E.ema_queries(sensor, changed, origin)
        np.testing.assert_allclose(RC.wrap(rr-rel), 11., atol=1e-12)
        np.testing.assert_allclose(RC.wrap(hh-heading), 0., atol=1e-12)
        np.testing.assert_allclose(qq[..., :3, :3], CP.rotation(11., 'y') @ q[..., :3, :3], atol=1e-12)
        np.testing.assert_allclose(qq[..., :3, 3], q[..., :3, 3], atol=1e-12)

    def test_full_se3_recovers_common_ideal_pelvis_origin(self):
        sensor, noisy, origin = fixture()
        q, rel, heading = E.ema_queries(sensor, noisy, origin)
        for c in range(2):
            for f in range(16):
                rotation = CP.rotation(float(heading[c, f]), 'y')
                recovered = sensor[c, f, :3, 3] - rotation @ q[c, f, :3, 3]
                np.testing.assert_allclose(recovered, origin[c, f], atol=1e-12)
        self.assertGreater(float(np.abs(q[..., :3, 3]).max()), .1)

    def test_future_suffix_cannot_change_past_queries(self):
        sensor, noisy, origin = fixture()
        reference = E.ema_queries(sensor, noisy, origin)
        sensor[... ,10:, :3, 3] += 50.
        noisy[..., 10:, :3, 3] -= 12.
        noisy[..., 10:, :3, :3] = CP.rotation(-140., 'y')
        origin[..., 10:, :] += 4.
        changed = E.ema_queries(sensor, noisy, origin)
        for before, after in zip(reference, changed):
            np.testing.assert_array_equal(before[:, :10], after[:, :10])

    def test_configuration_decisions_are_separate_and_include_zero(self):
        self.assertEqual(E.decision([0, 2, 0, 1]), 'SUPPORT_COLD_START_SCREEN')
        self.assertEqual(E.decision([0, -1, -4, 0]), 'LOWER_GAIT_PRIORITY')
        self.assertEqual(E.decision([3, -1, 0, 0]), 'MIXED_TRADEOFF_NO_AUTOMATIC_FOLLOWUP')
        self.assertEqual(E.decision([0, 0, 0, 0]), 'ALL_ZERO_RETAIN_NO_AUTOMATIC_FOLLOWUP')
        with self.assertRaises(ValueError):
            E.decision([1, 1, 1])


if __name__ == '__main__':
    unittest.main()
