"""Independent physical head-yaw contracts; synthetic CPU fixtures, no GPU."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

import cnh_torso_head_yaw_dev as H
import test_cnh_torso_native_dev as F


def fixture(start=0):
    bank = F.fixture_bank(F.skeleton(offset=(.27, -.19), yaw_offset=24.))
    positive = H.M.poses(bank, F.row(start), 1.)
    negative = H.M.poses(bank, F.row(start), -1.)
    f = dict(unit=99000, sign=np.array([1., -1.]),
             native_frame_index=np.stack([positive['native_frame_index'], negative['native_frame_index']]),
             sensor=np.stack([positive['sensor'], negative['sensor']]))
    for old in H.OLD_ARMS:
        f[old+'_query'] = np.stack([positive['queries'][old], negative['queries'][old]])
    return bank, f


def yaw_rotation(degrees):
    angle = np.radians(degrees)
    out = np.eye(3)
    out[0, 0] = out[2, 2] = np.cos(angle)
    out[0, 2] = np.sin(angle)
    out[2, 0] = -np.sin(angle)
    return out


def fake_noise(sensor, unit, config):
    # CPU deterministic input-sensitive stand-in; calls and physical inputs audited.
    out = sensor.copy()
    out[:, 0, 3] += .001 * (config+1)
    return out


class PhysicalHeadYawChecks(unittest.TestCase):
    def test_profiles_native_clock_signs_and_exact_zero_prefixes(self):
        np.testing.assert_array_equal(H.perturbation('zero'), np.zeros(600))
        np.testing.assert_array_equal(H.perturbation('const_pos'), np.full(600, 15.))
        np.testing.assert_array_equal(H.perturbation('const_neg'), np.full(600, -15.))
        pulse = H.perturbation('pulse_pos')
        np.testing.assert_array_equal(H.perturbation('pulse_neg'), -pulse)
        np.testing.assert_array_equal(pulse[:121], 0.)
        np.testing.assert_array_equal(pulse[240:], 0.)
        self.assertEqual(float(pulse[180]), 20.)
        self.assertGreater(float(pulse[150]), 0.)
        for length in (1, 120, 121, 180, 240, 300):
            np.testing.assert_array_equal(H.perturbation('pulse_pos', length), pulse[:length])
        with self.assertRaises(ValueError):
            H.perturbation('unfrozen_variant')

    def test_world_yaw_left_multiply_retains_translation_pitch_and_so3(self):
        _, f = fixture()
        sensor = f['sensor']
        delta = np.linspace(-20., 20., 16)
        rotated = H.rotate_sensor(sensor, delta)
        expected = np.stack([yaw_rotation(d) for d in delta])[None] @ sensor[..., :3, :3]
        np.testing.assert_allclose(rotated[..., :3, :3], expected, atol=1e-14, rtol=0)
        np.testing.assert_array_equal(rotated[..., :3, 3], sensor[..., :3, 3])
        np.testing.assert_array_equal(rotated[..., 3, :], sensor[..., 3, :])
        np.testing.assert_array_equal(rotated[..., 1, :3], sensor[..., 1, :3])
        np.testing.assert_allclose(np.linalg.det(rotated[..., :3, :3]), 1., atol=1e-14)
        self.assertFalse(np.array_equal(rotated[..., :3, :3], sensor[..., :3, :3]))
        np.testing.assert_array_equal(sensor, f['sensor'])

    def test_complete_se3_queries_preserve_every_estimated_travel_frame(self):
        _, f = fixture()
        original = copy.deepcopy(f)
        with patch.object(H.N.RC, 'noisy_for', side_effect=fake_noise) as noise:
            sensor, noisy, queries, delta = H.unit_inputs(f, 'const_pos')
        self.assertEqual(noise.call_count, 2)
        for config, call in enumerate(noise.call_args_list):
            np.testing.assert_array_equal(call.args[0], sensor[config])
            self.assertEqual(call.args[1:], (99000, config))
            np.testing.assert_array_equal(noisy[config], fake_noise(sensor[config], 99000, config))
        np.testing.assert_array_equal(delta[0], 15.)
        np.testing.assert_array_equal(delta[1], -15.)
        for arm, old in zip(H.ARMS, H.OLD_ARMS):
            expected = f[old+'_query'] @ np.linalg.inv(f['sensor']) @ sensor
            np.testing.assert_allclose(queries[arm], expected, atol=1e-14, rtol=0)
            old_travel = f['sensor'] @ np.linalg.inv(f[old+'_query'])
            new_travel = sensor @ np.linalg.inv(queries[arm])
            np.testing.assert_allclose(new_travel, old_travel, atol=2e-14, rtol=0)
            np.testing.assert_allclose(queries[arm][..., :3, 3], f[old+'_query'][..., :3, 3], atol=1e-14, rtol=0)
        for key in f:
            np.testing.assert_array_equal(f[key], original[key])

    def test_zero_all_inputs_and_queries_are_bit_identical(self):
        _, f = fixture()
        with patch.object(H.N.RC, 'noisy_for', side_effect=fake_noise):
            sensor, noisy, queries, delta = H.unit_inputs(f, 'zero')
        np.testing.assert_array_equal(sensor, f['sensor'])
        np.testing.assert_array_equal(delta, 0.)
        for arm, old in zip(H.ARMS, H.OLD_ARMS):
            np.testing.assert_array_equal(queries[arm], f[old+'_query'])
        for config in range(2):
            np.testing.assert_array_equal(noisy[config], fake_noise(f['sensor'][config], 99000, config))

    def test_native_age_pulse_zero_prefix_and_recovered_pose_are_exact(self):
        for start in (0, 180):
            _, f = fixture(start)
            with patch.object(H.N.RC, 'noisy_for', side_effect=fake_noise):
                sensor, _, queries, delta = H.unit_inputs(f, 'pulse_pos')
            age = f['native_frame_index']
            np.testing.assert_array_equal(delta, f['sign'][:, None]*H.perturbation('pulse_pos')[age])
            zeros = delta == 0
            self.assertTrue(zeros.any())
            np.testing.assert_array_equal(sensor[zeros], f['sensor'][zeros])
            for arm, old in zip(H.ARMS, H.OLD_ARMS):
                np.testing.assert_array_equal(queries[arm][zeros], f[old+'_query'][zeros])
            np.testing.assert_array_equal(sensor[..., :3, 3], f['sensor'][..., :3, 3])
        # Pose recovery does not assert recovery photons; actual observations
        # must be regenerated by the renderer, whose RNG consumption can differ.

    def test_mirrored_profile_and_queries_conjugate_consistently(self):
        _, f = fixture()
        reflection = np.diag([-1., 1., 1., 1.])
        with patch.object(H.N.RC, 'noisy_for', side_effect=fake_noise):
            sensor, _, queries, _ = H.unit_inputs(f, 'pulse_pos')
        np.testing.assert_allclose(sensor[1], reflection@sensor[0]@reflection, atol=1e-14, rtol=0)
        for query in queries.values():
            np.testing.assert_allclose(query[1], reflection@query[0]@reflection, atol=1e-14, rtol=0)

    def test_native_e1_position_chord_and_online_states_unchanged_by_yaw(self):
        bank, f = fixture()
        before = bank['e1'].copy()
        head = bank['head']
        idx = np.maximum(np.arange(head.shape[1])-60, 0)
        rebuilt = H.M.T.B.yaw(head-head[:, idx])
        np.testing.assert_allclose(H.M.T.B.wrap(rebuilt-before), 0., atol=1e-12, rtol=0)
        with patch.object(H.N.RC, 'noisy_for', side_effect=fake_noise):
            sensor, _, queries, _ = H.unit_inputs(f, 'const_neg')
        np.testing.assert_array_equal(bank['e1'], before)
        old_e1_travel = f['sensor']@np.linalg.inv(f['e1_query'])
        new_e1_travel = sensor@np.linalg.inv(queries['e1'])
        np.testing.assert_allclose(new_e1_travel, old_e1_travel, atol=2e-14, rtol=0)
        # Head yaw changes the physical sensor/query, not position-derived E1.
        self.assertGreater(float(np.abs(queries['e1']-f['e1_query']).max()), .1)

    def test_future_delta_suffix_does_not_change_any_physical_query_prefix(self):
        _, f = fixture()
        delta = H.perturbation('pulse_pos')
        changed = delta.copy(); changed[150:] = -123.
        with patch.object(H.N.RC, 'noisy_for', side_effect=fake_noise), patch.object(H, 'perturbation', return_value=delta):
            a = H.unit_inputs(f, 'pulse_pos')
        with patch.object(H.N.RC, 'noisy_for', side_effect=fake_noise), patch.object(H, 'perturbation', return_value=changed):
            b = H.unit_inputs(f, 'pulse_pos')
        prefix = f['native_frame_index'] < 150
        for index in (0, 1, 3):
            np.testing.assert_array_equal(a[index][prefix], b[index][prefix])
        for arm in H.ARMS:
            np.testing.assert_array_equal(a[2][arm][prefix], b[2][arm][prefix])


if __name__ == '__main__':
    unittest.main()
