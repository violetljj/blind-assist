"""Same-native-trajectory pose/query integrity and frozen source-bank checks."""
import json
import unittest

import numpy as np

import cnh_torso_native_motion_dev as N


def skeleton(n=600, speed=1., offset=(.2, -.1), yaw_offset=20.):
    t = np.arange(n)/60.
    pelvis = np.c_[speed*t, np.zeros(n)]
    x = np.zeros((n, 24, 3))
    x[:, :, :2] = pelvis[:, None, :]
    x[:, 6, :2] += offset
    x[:, 6, 2] = 1.7
    yaw = np.full(n, yaw_offset)
    sh = .4*np.c_[-np.sin(np.radians(yaw)), np.cos(np.radians(yaw))]
    x[:, 11, :2] += sh/2
    x[:, 7, :2] -= sh/2
    return x


def fixture_bank(x):
    initial = dict(tau_seconds=2., straight_rate_deg_s=3., torso_rate_deg_s=12.,
                   min_speed=.3, max_bias_rate_deg_s=3.)
    d = N.estimates(x, initial, N.G.CONFIGS[0])
    bank = {key: np.asarray(value)[None] for key, value in d.items() if key != 'onsets'}
    bank.update(pid=np.array(['P06']), clip=np.array(['P06_synthetic.npy']))
    return bank


def row(start=0, group=0):
    return dict(clip_index=0, start=start, pid='P06', role='evaluation', requested_group=group)


class NativePoseChecks(unittest.TestCase):
    def test_full_se3_exact_query_recovery_and_history_composition(self):
        bank = fixture_bank(skeleton())
        poses = N.poses(bank, row(), 1.)
        sensor, travel = poses['sensor'], poses['travel']
        query = poses['queries']['exact']
        np.testing.assert_allclose(travel@query, sensor, atol=1e-12)
        for frame in (4, 10, 15):
            history = sensor[:frame+1]
            projected = query[frame]@np.linalg.inv(sensor[frame])@history
            expected = np.linalg.inv(travel[frame])@history
            np.testing.assert_allclose(projected, expected, atol=1e-12)
        for arm in N.ARMS:
            # Recover every estimated travel frame from the complete query.
            estimated = sensor@np.linalg.inv(poses['queries'][arm])
            np.testing.assert_allclose(estimated[:, :3, 3], travel[:, :3, 3], atol=1e-12)
            np.testing.assert_allclose(np.linalg.det(estimated[:, :3, :3]), 1., atol=1e-12)

    def test_nonzero_horizontal_head_pelvis_offset_is_retained(self):
        offset = np.array([.2, -.1])
        bank = fixture_bank(skeleton(offset=offset))
        poses = N.poses(bank, row(), 1.)
        sensor_offset = poses['sensor'][:, :3, 3]-poses['travel'][:, :3, 3]
        np.testing.assert_allclose(np.linalg.norm(sensor_offset, axis=1), np.linalg.norm(offset), atol=1e-12)
        np.testing.assert_allclose(sensor_offset[:, [2, 0]], np.tile(offset, (16, 1)), atol=1e-12)
        for arm in N.ARMS:
            translated = poses['queries'][arm][:, :3, 3]
            np.testing.assert_allclose(np.linalg.norm(translated, axis=1), np.linalg.norm(offset), atol=1e-12)
        # The documented sensor proxy uses constant vertical zero; this test
        # deliberately checks native horizontal offset, not native head height.
        np.testing.assert_array_equal(poses['sensor'][:, 1, 3], 0.)

    def test_mirror_conjugation_and_wrapped_yaw_equivalence(self):
        bank = fixture_bank(skeleton())
        f = 12*np.arange(16)
        # Reference around +179 while torso crosses -180 tests the short delta.
        bank['exact'][0, f] = 179.
        bank['torso'][0, f] = -179.
        plus = N.poses(bank, row(), 1.)
        minus = N.poses(bank, row(), -1.)
        reflection = np.diag([-1., 1., 1., 1.])
        for key in ('sensor', 'travel'):
            np.testing.assert_allclose(minus[key], reflection@plus[key]@reflection, atol=1e-12)
        for arm in N.ARMS:
            np.testing.assert_allclose(minus['queries'][arm], reflection@plus['queries'][arm]@reflection, atol=1e-12)
        torso_arm = N.ARMS.index('torso')
        np.testing.assert_array_equal(plus['heading_error'][torso_arm], 2.)
        np.testing.assert_array_equal(minus['heading_error'][torso_arm], -2.)
        bank['torso'][0, f] += 360.
        shifted = N.poses(bank, row(), 1.)
        np.testing.assert_allclose(shifted['sensor'], plus['sensor'], atol=1e-12)
        np.testing.assert_allclose(shifted['queries']['torso'], plus['queries']['torso'], atol=1e-12)

    def test_future_perturbation_preserves_causal_physical_pose_prefix(self):
        x = skeleton()
        changed = x.copy()
        cutoff = 150
        # A future turn changes the future1.5m evaluator direction at frame 100.
        p = x[:, 0, :2]
        q = p.copy()
        q[cutoff:, 0] = p[cutoff, 0]
        q[cutoff:, 1] = (np.arange(len(x)-cutoff))/60.
        changed[cutoff:, :, :2] += (q-p)[cutoff:, None, :]
        original = fixture_bank(x)
        altered = fixture_bank(changed)
        self.assertGreater(abs(float(N.T.B.wrap(original['exact'][0, 100]-altered['exact'][0, 100]))), 1.)
        for arm in N.ARMS[1:]:
            np.testing.assert_array_equal(original[arm][0, :cutoff], altered[arm][0, :cutoff])
        for key in ('bias_initial', 'bias_gait', 'updated_initial', 'updated_gait'):
            np.testing.assert_array_equal(original[key][0, :cutoff], altered[key][0, :cutoff])
        original['support'][:] = True
        altered['support'][:] = True
        before, after = N.poses(original, row(), 1.), N.poses(altered, row(), 1.)
        # Final evaluator phi and pelvis anchor both differ. E and S share
        # that world rigid transform, so inv(E)@S cancels it without using truth.
        prefix = before['native_frame_index'] < cutoff
        for arm in N.ARMS[1:]:
            np.testing.assert_allclose(before['queries'][arm][prefix], after['queries'][arm][prefix], atol=1e-12)

    def test_cold_queries_equal_raw_torso_before_first_bias_update(self):
        bank = fixture_bank(skeleton())
        poses = N.poses(bank, row(), 1.)
        f = poses['native_frame_index']
        for arm, first, bias_key in [('corrected', 60, 'bias_initial'), ('corrected_gait', 90, 'bias_gait')]:
            cold = f < first
            np.testing.assert_array_equal(bank[bias_key][0, f[cold]], 0.)
            np.testing.assert_allclose(poses['queries'][arm][cold], poses['queries']['torso'][cold], atol=1e-12)

    @unittest.skipUnless((N.OUT/'bank.npz').exists(), 'native source bank unavailable')
    def test_all_source_windows_role_strata_and_saved_heading_identity(self):
        bank, windows = N.load_bank()
        plan = json.loads((N.OUT/'PLAN.json').read_text(encoding='utf8'))
        files = plan['files']
        self.assertEqual(len(bank['clip']), len(files))
        onsets = {}
        for ci, record in enumerate(files):
            self.assertEqual(str(bank['clip'][ci]), record['name'])
            self.assertEqual(str(bank['pid'][ci]), record['pid'])
            x = np.load(N.T.B.SRC/record['name'])
            lab = N.T.labels(x)
            np.testing.assert_array_equal(bank['pelvis'][ci], x[:, 0, :2])
            np.testing.assert_array_equal(bank['head'][ci], x[:, 6, :2])
            np.testing.assert_array_equal(bank['exact'][ci], lab['truth'])
            np.testing.assert_array_equal(bank['turn_group'][ci], lab['turn_group'])
            onsets[ci] = {max(0, onset-72) for onset in lab['onsets']}
        counts = {(role, group): 0 for role in ('calibration', 'evaluation') for group in range(4)}
        for source in windows:
            ci, start, pid = source['clip_index'], source['start'], source['pid']
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(start+192, len(bank['exact'][ci]))
            self.assertEqual(pid, str(bank['pid'][ci]))
            self.assertTrue(bank['support'][ci, start:start+192].all())
            role = 'calibration' if pid in N.T.DEV else 'evaluation'
            self.assertEqual(source['role'], role)
            self.assertIn(pid, N.T.DEV if role == 'calibration' else N.T.EVAL)
            group = source['requested_group']
            self.assertIn(group, range(4))
            counts[(role, group)] += 1
            if group == 0:
                self.assertEqual(start, 0)
            elif group == 1:
                self.assertIn(start, onsets[ci])
            else:
                self.assertEqual(int(bank['turn_group'][ci, start+120]), 1 if group == 2 else 0)
        self.assertTrue(all(value > 0 for value in counts.values()))
        with np.load(N.T.OUT/'series.npz') as original, np.load(N.G.OUT/'series.npz') as gait:
            for arm in N.ARMS[1:]:
                stored = gait['corrected_gait'] if arm == 'corrected_gait' else original[arm]
                rebuilt = N.T.B.wrap(bank[arm]-bank['exact']).ravel()
                np.testing.assert_allclose(N.T.B.wrap(rebuilt-stored), 0., atol=1e-9)


if __name__ == '__main__':
    unittest.main()
