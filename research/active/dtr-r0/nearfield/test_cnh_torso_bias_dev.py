"""Synthetic causality checks plus generated clip-window integrity; no data rerun."""
import json
import unittest
from unittest.mock import patch

import numpy as np

import cnh_torso_bias_dev as T
import cnh_torso_gait_bias_dev as G


def path_from_yaw(yaw, speed=1.):
    direction = np.c_[np.cos(np.radians(yaw)), np.sin(np.radians(yaw))]
    return np.cumsum(direction * speed / T.HZ, axis=0)


def skeleton(pelvis, torso):
    x = np.zeros((len(pelvis), 24, 3))
    x[:, :, :2] = pelvis[:, None, :]
    # L-R shoulder line rotates clockwise into the requested forward normal.
    sh = .4 * np.c_[-np.sin(np.radians(torso)), np.cos(np.radians(torso))]
    x[:, T.B.J['lsh'], :2] += sh / 2
    x[:, T.B.J['rsh'], :2] -= sh / 2
    return x


class TorsoBiasChecks(unittest.TestCase):
    def setUp(self):
        self.config = dict(tau_seconds=2., straight_rate_deg_s=3.,
                           torso_rate_deg_s=12., min_speed=.3,
                           max_bias_rate_deg_s=3.)

    def test_future_perturbation_preserves_every_prefix_output(self):
        n = 600
        theta = 2. * np.sin(np.arange(n) / 120.)
        p = path_from_yaw(theta)
        torso = T.B.wrap(theta + 17.)
        reference = T.causal_correct(p, torso, self.config)
        rng = np.random.default_rng(73)
        for end in (20, 60, 125, 301, 540):
            q, changed_torso = p.copy(), torso.copy()
            q[end:] += rng.normal(0, 20, (n-end, 2))
            changed_torso[end:] = rng.uniform(-180, 180, n-end)
            altered = T.causal_correct(q, changed_torso, self.config)
            short = T.causal_correct(p[:end], torso[:end], self.config)
            for expected, actual, prefix in zip(reference, altered, short):
                np.testing.assert_array_equal(expected[:end], actual[:end])
                np.testing.assert_array_equal(expected[:end], prefix)

    def test_straight_known_offset_converges_with_capped_update(self):
        p = path_from_yaw(np.zeros(1800))
        corrected, bias, updated = T.causal_correct(p, np.full(1800, 20.), self.config)
        self.assertTrue(updated[60:].all())
        self.assertFalse(updated[:60].any())
        np.testing.assert_array_equal(bias[:60], 0.)
        self.assertGreater(bias[-1], 19.99)
        self.assertLess(abs(corrected[-1]), .01)
        self.assertLessEqual(np.abs(np.diff(bias)).max(), 3./60. + 1e-12)

    def test_observed_turn_freezes_bias_without_future_label(self):
        theta = np.zeros(600)
        theta[240:] = 90.
        p = path_from_yaw(theta)
        torso = theta + 20.
        corrected, bias, updated = T.causal_correct(p, torso, self.config)
        self.assertTrue(updated[239])
        self.assertFalse(updated[240:300].any())
        np.testing.assert_array_equal(bias[240:300], np.full(60, bias[239]))
        # The current torso turn is retained immediately while the bias freezes.
        self.assertAlmostEqual(float(T.B.wrap(corrected[240]-corrected[239])), 90.)

    def test_wrap_boundary_keeps_short_offset_and_turn_rate(self):
        theta = np.full(1800, 179.)
        corrected, bias, updated = T.causal_correct(
            path_from_yaw(theta), T.B.wrap(theta+20.), self.config)
        self.assertTrue(updated[60:].all())
        self.assertLess(abs(bias[-1]-20.), .01)
        self.assertLess(abs(float(T.B.wrap(corrected[-1]-179.))), .01)

    def test_clip_reset_and_zero_speed_keep_zero_bias(self):
        moving = path_from_yaw(np.zeros(300))
        self.assertGreater(T.causal_correct(moving, np.full(300, 20.), self.config)[1][-1], 0.)
        corrected, bias, updated = T.causal_correct(moving[:60], np.full(60, -30.), self.config)
        np.testing.assert_array_equal(corrected, -30.)
        np.testing.assert_array_equal(bias, 0.)
        self.assertFalse(updated.any())
        torso = np.linspace(-20, 20, 600)
        corrected, bias, updated = T.causal_correct(np.zeros((600, 2)), torso, self.config)
        np.testing.assert_array_equal(bias, 0.)
        np.testing.assert_allclose(corrected, torso)
        self.assertFalse(updated.any())

    def test_future_truth_can_change_without_changing_online_heading(self):
        theta = np.zeros(600)
        p = path_from_yaw(theta)
        x = skeleton(p, np.full(600, 20.))
        changed = x.copy()
        turn = np.zeros(600)
        turn[151:] = 90.
        q = path_from_yaw(turn)
        changed[151:, :, :2] += (q-p)[151:, None, :]
        before, after = T.errors(x, self.config), T.errors(changed, self.config)
        self.assertGreater(abs(float(T.B.wrap(before['truth'][100]-after['truth'][100]))), 1.)
        for key in ('bias_deg', 'updated'):
            np.testing.assert_array_equal(before[key][:151], after[key][:151])
        estimate = lambda d: T.B.wrap(d['corrected']+d['truth'])
        np.testing.assert_allclose(estimate(before)[:151], estimate(after)[:151], atol=1e-12)

    def test_evaluator_labels_are_never_read_by_causal_correct(self):
        p = path_from_yaw(np.zeros(180))
        with patch.object(T, 'labels', side_effect=AssertionError('evaluator accessed')):
            corrected, bias, updated = T.causal_correct(p, np.full(180, 20.), self.config)
        self.assertTrue(updated[-1])
        self.assertGreater(bias[-1], 0.)
        self.assertLess(corrected[-1], 20.)

    def test_label_support_excludes_cold_start_and_incomplete_future(self):
        x = skeleton(path_from_yaw(np.zeros(600)), np.zeros(600))
        d = T.labels(x)
        self.assertFalse(d['valid'][:60].any())
        self.assertFalse(d['valid'][-90:].any())
        self.assertTrue(d['valid'][100])
        np.testing.assert_allclose(d['truth'][d['valid']], 0.)

    def test_invalid_clip_edges_separate_persistent_runs(self):
        mask = np.zeros(240, bool)
        mask[10:70] = True
        mask[140:200] = True
        result = T.persistent(np.full(240, 20.), mask)
        self.assertEqual(result, dict(runs_ge1s=2, frames_in_runs_ge1s=120))
        # A sign change likewise starts a new persistent interval.
        signed = np.r_[np.full(60, 20.), np.full(60, -20.)]
        self.assertEqual(T.persistent(signed, np.ones(120, bool))['runs_ge1s'], 2)

    @unittest.skipUnless((T.OUT/'windows.json').exists() and (T.OUT/'series.npz').exists(),
                         'generated estimator artifacts unavailable')
    def test_saved_replay_windows_stay_inside_source_clip(self):
        rows = json.loads((T.OUT/'windows.json').read_text(encoding='utf8'))
        self.assertGreater(len(rows), 0)
        with np.load(T.OUT/'series.npz') as series:
            valid = series['valid']
            files = series['file_index']
            frames = series['frame_index']
            for base in range(0, len(rows), 512):
                batch = rows[base:base+512]
                starts = np.array([row['start'] for row in batch])
                positions = starts[:, None] + np.arange(192)
                self.assertLess(int(positions.max()), len(valid))
                self.assertTrue(valid[positions].all())
                expected_files = np.array([row['file_index'] for row in batch])[:, None]
                self.assertTrue((files[positions] == expected_files).all())
                first = frames[starts]
                np.testing.assert_array_equal(frames[positions], first[:, None]+np.arange(192))
                self.assertTrue((first >= 60).all())
                self.assertTrue((first+191 < 540).all())
                self.assertTrue(all(row['pid'] in T.EVAL for row in batch))


class GaitBiasChecks(unittest.TestCase):
    def setUp(self):
        self.config = G.CONFIGS[1].copy()

    def test_boxcar_prefix_seed_torso_geometry_and_midpoint_phase(self):
        p = np.c_[np.arange(180), np.arange(180)**2].astype(float)
        q = G.smooth_past(p)
        np.testing.assert_array_equal(q[0], p[0])
        for i in (1, 22, 23, 90, 179):
            np.testing.assert_allclose(q[i], p[max(0, i-23):i+1].mean(axis=0))
        changed = p.copy()
        changed[91:] += 10000.
        np.testing.assert_array_equal(G.smooth_past(changed)[:91], q[:91])
        np.testing.assert_array_equal(G.smooth_past(p[:91]), q[:91])
        np.testing.assert_allclose(G.smooth_past(p+10000.), q+10000.)
        # A quadratic position has a linear velocity: exact filtered chord
        # midpoint is t-30-11.5, hence integer torso[t-42] differs by half frame.
        n = 600
        time = np.arange(n)/60.
        acceleration = .1
        p = np.c_[time, .5*acceleration*time**2]
        theta = np.degrees(np.arctan2(acceleration*time, 1.))
        torso = T.B.wrap(theta+20.)
        x = skeleton(p, torso)
        np.testing.assert_allclose(T.torso_yaw(x), torso, atol=1e-12)
        _, _, _, gate = G.causal_correct(p, torso, self.config)
        ids = gate['ids']
        full = ids >= 113  # all samples of both 24-frame endpoint means present
        exact_chord = np.degrees(np.arctan2(acceleration*(time[ids]-41.5/60.), 1.))
        expected = T.B.wrap(torso[ids-42]-exact_chord)
        np.testing.assert_allclose(gate['residual'][full], expected[full], atol=1e-10)
        self.assertLess(np.abs(gate['residual'][full]-20.).max(), .05)

    def test_causal_output_and_gate_prefix_perturbation_and_truncation(self):
        n = 600
        theta = 2.*np.sin(np.arange(n)/150.)
        p = path_from_yaw(theta)
        torso = theta+17.
        original = G.causal_correct(p, torso, self.config)
        rng = np.random.default_rng(144)
        for end in (20, 90, 91, 160, 300, 540):
            changed_p, changed_torso = p.copy(), torso.copy()
            changed_p[end:] += rng.normal(0, 20, (n-end, 2))
            changed_torso[end:] = rng.uniform(-180, 180, n-end)
            future = G.causal_correct(changed_p, changed_torso, self.config)
            short = G.causal_correct(p[:end], torso[:end], self.config)
            for expected, altered, truncated in zip(original[:3], future[:3], short[:3]):
                np.testing.assert_array_equal(expected[:end], altered[:end])
                np.testing.assert_array_equal(expected[:end], truncated)
            mask = original[3]['ids'] < end
            self.assertEqual(len(short[3]['ids']), max(end-90, 0))
            for key in original[3]:
                np.testing.assert_array_equal(original[3][key][mask], future[3][key][mask])
                np.testing.assert_array_equal(original[3][key][mask], short[3][key])

    def test_cold_zero_speed_and_known_offset_convergence(self):
        n = 1800
        p = path_from_yaw(np.zeros(n))
        corrected, bias, updated, gate = G.causal_correct(p, np.full(n, 20.), self.config)
        self.assertFalse(updated[:90].any())
        self.assertTrue(updated[90:].all())
        np.testing.assert_array_equal(bias[:90], 0.)
        self.assertLess(abs(corrected[-1]), .01)
        self.assertGreater(bias[-1], 19.99)
        self.assertLessEqual(np.abs(np.diff(bias)).max(), .05+1e-12)
        self.assertTrue(all(len(v) == n-90 for v in gate.values()))
        raw = np.full(300, -30.)
        corrected, bias, updated, gate = G.causal_correct(np.ones((300, 2))*5., raw, self.config)
        np.testing.assert_array_equal(corrected, raw)
        np.testing.assert_array_equal(bias, 0.)
        self.assertFalse(updated.any())
        self.assertFalse(gate['speed_ok'].any())

    def test_slow_turn_gates_use_observed_signals_and_retain_body_rotation(self):
        time = np.arange(600)/60.
        theta = 10.*time
        p = path_from_yaw(theta)
        torso = theta+20.
        with patch.object(T, 'labels', side_effect=AssertionError('future label access')):
            corrected, bias, updated, gate = G.causal_correct(p, torso, self.config)
        full = gate['ids'] >= 114
        np.testing.assert_allclose(gate['rate'][full], 10., atol=1e-9)
        self.assertTrue(gate['body_ok'][full].all())
        self.assertFalse(gate['direction_ok'][full].any())
        self.assertFalse(updated[114:].any())
        np.testing.assert_allclose(T.B.wrap(np.diff(corrected[114:])), 10./60., atol=1e-12)

    @unittest.skipUnless((G.OUT/'series.npz').exists(), 'gait pilot artifacts unavailable')
    def test_shared_baseline_support_and_windows_have_exact_identity(self):
        # Derived geometry, evaluator truth and baseline arms also agree for a
        # constructed trajectory, independently of the saved production arrays.
        theta = np.arange(600)/60.
        x = skeleton(path_from_yaw(theta), theta+20.)
        initial = T.errors(x, self.config)
        new, _ = G.errors(x, self.config)
        for key in ('e1', 'torso', 'oracle', 'valid', 'truth', 'turn_group', 'speed', 'onsets'):
            np.testing.assert_array_equal(initial[key], new[key])
        with np.load(T.OUT/'series.npz') as original, np.load(G.OUT/'series.npz') as pilot:
            for key in ('e1', 'torso', 'oracle', 'valid', 'turn_group', 'file_index', 'frame_index'):
                np.testing.assert_array_equal(original[key], pilot[key])
        self.assertEqual((T.OUT/'windows.json').read_bytes(), (G.OUT/'windows.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
