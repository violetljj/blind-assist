"""Independent R3 geometry and truth contract checks; no consumed payload reads."""
from __future__ import annotations

import unittest
import numpy as np
import cnh_tristate_dev as R1
import cnh_tristate_dev_r2 as R2
import cnh_tristate_dev_r3_geometry as G
import cnh_tristate_dev_r3_truth as T


def straight():
    p = np.repeat(np.eye(4)[None], 16, axis=0)
    p[:, :3, :3] = R1.rotation(-10, 'x')
    p[:, 2, 3] = np.arange(16) * .16 - 2.4
    return p


def direct_mask(points, poses, frame, margin):
    result = np.zeros(len(points), bool)
    edge = np.tan(np.deg2rad(22.5 - margin))
    for h in range(max(0, frame - 3), frame + 1):
        pose = R2.nominal(poses, h)
        if pose is None:
            continue
        local = (points - pose[:3, 3]) @ pose[:3, :3]
        z = local[:, 2]
        result |= ((z > 0) & (np.abs(local[:, 0]) <= edge*z + R1.EPS)
                   & (np.abs(local[:, 1]) <= edge*z + R1.EPS)
                   & (np.linalg.norm(local, axis=1) <= G.RANGE + R1.EPS))
    return result


class GeometryContract(unittest.TestCase):
    def test_warmup_and_invalid_pose_close_gate(self):
        poses = straight()
        for f in (3, 4):
            ev = G.gates(G.prepared(poses, f), [0, 3, 4])
            self.assertEqual(ev['passed'].shape, (3, 2))
            self.assertFalse(ev['passed'].any())
            self.assertTrue((ev['mask_count'] == 0).all())
        poses[8, 0, 3] = np.nan
        self.assertFalse(G.gates(G.prepared(poses, 10), [3])['passed'].any())

    def test_aligned_ideal_core_fresh_all_nonwarm_frames(self):
        poses = straight()
        for f in range(5, 16):
            ev = G.gates(G.prepared(poses, f), [0, 3, 7])
            self.assertTrue(ev['passed'][:, 0].all())
            self.assertTrue(ev['passed'][1:, 1].all())

    def test_contraction_matches_independent_fov_history_union(self):
        rng = np.random.default_rng(941)
        poses = G.noise_pose(straight(), 2304)
        for f in (5, 8, 15):
            points = rng.uniform([-.6, -.35, -1], [.6, 1.1, 2.5], (1500, 3))
            for margin in (0, .5, 3, 7, 15):
                np.testing.assert_array_equal(G.core_mask(points, poses, f, margin),
                                              direct_mask(points, poses, f, margin))

    def test_dual_uses_union_never_intersection(self):
        pose = np.eye(4)
        angles = [-15, 15]
        points = np.array([[.65, 0, 1], [-.65, 0, 1], [0, 0, 1]])
        local = points[None]
        union = np.any([R1.fov(points, pose @ R1.extrinsic(a), G.RANGE)
                       for a in angles], axis=0)
        np.testing.assert_array_equal(G._fresh(local, angles), union)
        self.assertTrue(union.all())
        both = np.all([R1.fov(points, pose @ R1.extrinsic(a), G.RANGE)
                       for a in angles], axis=0)
        self.assertFalse(both.all())

    def test_freshness_history_excludes_older_than_point6_seconds(self):
        poses = np.repeat(np.eye(4)[None], 16, axis=0)
        # Current path gives a valid forward direction; old sensor sees point,
        # whereas frames 7..10 point sideways.
        poses[:, 2, 3] = np.arange(16)*.16
        for f in range(7, 11):
            poses[f, :3, :3] = R1.rotation(90)
        point = np.array([[0., 0., 2.]])
        self.assertTrue(R1.fov(point, poses[6], G.RANGE)[0])
        self.assertFalse(G.fresh(point, poses, 10, [0])[0])

    def test_voxel_weights_cover_actual_check_volume_and_monotone_core(self):
        poses = straight()
        points, weight, group = G.quadrature(poses, 15)
        self.assertAlmostEqual(weight.sum(), .58*1.2*1.1, places=12)
        expected = np.array([.58*.3*.62, .58*.3*.48]*4)
        np.testing.assert_allclose(np.bincount(group, weights=weight), expected, atol=1e-12)
        core, total = G._volume_arrays(poses, 15, [0, 3, 4, 8])
        self.assertTrue((np.diff(core, axis=0) <= 1e-12).all())
        self.assertTrue((core <= total + 1e-12).all())

    def test_synthetic_motion_and_frozen_noise_same_schedule(self):
        true, noisy = G.synthetic(142)
        np.testing.assert_allclose(true, straight(), atol=0)
        np.testing.assert_allclose(np.diff(true[:, 2, 3]), .8*.2, atol=1e-14)
        np.testing.assert_array_equal(noisy, G.frozen_noisy()(true, 142, dt=.2))

    def test_truth_gap_distinct_from_online_gate(self):
        estimated = straight()
        prep = G.prepared(estimated, 15)
        ev = G.gates(prep, [3])
        self.assertTrue(ev['passed'][0, 0])
        true_sensor = estimated.copy()
        true_sensor[:, :3, :3] = R1.rotation(30) @ R1.rotation(-10, 'x')
        true_fresh = T.fresh_true(prep['points'][ev['mask'][0]], true_sensor, 15, [0])
        self.assertFalse(true_fresh.all())


class TruthContract(unittest.TestCase):
    def test_near_and_shrunk_edge_contact_remain_label_errors(self):
        travel = np.eye(4)
        for box in [dict(lo=[-.1, .1, .2], hi=[.1, .2, .3]),
                    dict(lo=[-.1, .80, 1.], hi=[.1, .85, 1.1])]:
            contact = R1.contact_boxes([box], travel, zmax=2.1)
            self.assertTrue(contact)
            witnesses = np.asarray(contact[0]['witness_world'])
            poses = straight(); poses[:, 2, 3] -= poses[15, 2, 3]
            attr = T.unchecked_witnesses(witnesses, poses, 15, 5)
            self.assertTrue(attr['unchecked'])

    def test_current2point1_meter_window_and_closed_contact(self):
        travel = np.eye(4)
        near = dict(lo=[-.1, .1, .1], hi=[.1, .2, .2])
        boundary = dict(lo=[-.1, .1, 2.1], hi=[.1, .2, 2.2])
        beyond = dict(lo=[-.1, .1, 2.10001], hi=[.1, .2, 2.2])
        self.assertEqual(len(R1.contact_boxes([near, boundary, beyond], travel, zmax=2.1)), 2)

    def test_hold_has_alarm_precedence_and_known_history_only(self):
        scores = np.zeros((13, 2)); scores[4, 0] = R1.THRESHOLD
        alarm = scores >= R1.THRESHOLD
        hold = np.maximum.accumulate(alarm, axis=0)
        clear_a = ~alarm
        clear_b = clear_a & ~hold
        self.assertTrue(clear_b[:4].all())
        self.assertFalse(clear_b[4:, 0].any())
        unknown_b = ~alarm & ~clear_b
        self.assertFalse(unknown_b[4, 0])
        self.assertTrue(unknown_b[5:, 0].all())
        self.assertAlmostEqual((R1.FRAMES[-1]-R1.FRAMES[0])*.2, 2.4)


def audit_completed_artifacts(out):
    """Call explicitly only after the parent reports online/truth/result sealed."""
    from pathlib import Path
    out = Path(out)
    r = R1.read(out/'result.json')
    online_receipt = R1.read(out/'online_receipt.json')
    truth_receipt = R1.read(out/'truth_receipt.json')
    assert (out/'PLAN.sha256').read_text().strip() == R1.sha(out/'PLAN.json')
    for name in ('online', 'rows', 'calibration'):
        suffix = '.npz' if name == 'online' else '.json'
        assert online_receipt[name+'_sha256'] == R1.sha(out/(name+suffix))
    assert truth_receipt['truth_sha256'] == R1.sha(out/'truth.npz')
    assert truth_receipt['online_sha256'] == online_receipt['online_sha256']
    assert truth_receipt['calibration_sha256'] == online_receipt['calibration_sha256']
    for name, sha in r['provenance'].items():
        assert sha == R1.sha(out/name)
    calibration = R1.read(out/'calibration.json')
    with np.load(out/'calibration.npz') as z:
        for mi, model in enumerate(('original', 'gravity')):
            data = calibration['models'][model]
            eligible = [p for p in data['curve'] if p['pass_fraction'] >= .99
                        and p['minimum_mask_points'] > 0]
            assert eligible[0]['margin_deg'] == data['margin_deg']
            for j, p in enumerate(data['curve']):
                assert p['passed'] == int(z['passed'][mi, :, :, j].sum())
                assert p['pass_fraction'] == float(z['passed'][mi, :, :, j].mean())
                assert p['minimum_mask_points'] == int(z['mask_count'][mi, :, :, j].min())
                assert p['core_fraction'] == float(z['core_volume_sums'][mi, j].sum()/z['total_volume_sums'][mi].sum())
    expected_model = 'gravity' if calibration['models']['original']['core_fraction'] < .5 else 'original'
    assert calibration['selected_model'] == expected_model
    with np.load(out/'online.npz') as z:
        d = {k: z[k] for k in z.files}
    with np.load(out/'truth.npz') as z:
        t = {k: z[k] for k in z.files}
    rows = R1.read(out/'rows.json')
    assert rows == R1.read(R1.OUT/'rows.json')
    with np.load(R1.OUT/'online.npz') as z:
        for name in ('score', 'aug_score', 'thresholds'):
            assert np.array_equal(d[name], z[name], equal_nan=True)
    sampled_poses = 0
    if calibration['selected_model'] == 'original':
        for batch in sorted({r['batch'] for r in rows}):
            ids = [i for i, row in enumerate(rows) if row['batch'] == batch]
            for i in (ids[0], ids[len(ids)//2], ids[-1]):
                expected_pose = R2.metadata(rows[i])[2]
                np.testing.assert_array_equal(d['poses'][i], expected_pose)
                sampled_poses += 1
    alarm = d['score'] >= R1.THRESHOLD
    hold = np.maximum.accumulate(alarm, axis=1)
    safe = ~t['contact'] & ~t['graze']
    masks = {'all': np.ones(len(rows), bool)}
    masks |= {f'mode{k}': np.array([x['mode'] == k for x in rows]) for k in range(3)}
    masks |= {f'turn_{k}': np.array([x['turn'] == k for x in rows]) for k in ('left', 'right', 'none')}
    checks = 0
    for group_name, mask in masks.items():
        if not mask.any():
            continue
        for a, arm in enumerate(('single', 'dual')):
            previous = {'A': np.zeros_like(safe), 'B': np.zeros_like(safe)}
            for j, tau in enumerate(d['thresholds']):
                clear_a = ~alarm[..., a] & d['gate'][:, :, 1, a] & (d['score'][..., a] <= tau)
                for variant in ('A', 'B'):
                    clear = clear_a if variant == 'A' else clear_a & ~hold[..., a]
                    assert not (previous[variant] & ~clear).any()
                    previous[variant] = clear
                    error = clear & t['contact']
                    unknown = ~alarm[..., a] & ~clear
                    p = r['groups'][group_name]['curves'][arm+'/'+variant][j]
                    expected = {
                        'risk': (error, clear),
                        'unknown': (unknown, np.ones_like(safe)),
                        'clear': (clear, np.ones_like(safe)),
                        'obstacle': (alarm[..., a], np.ones_like(safe)),
                        'availability': (clear & safe, safe),
                        'graze_clear': (clear & t['graze'], clear),
                    }
                    for metric, (num, den) in expected.items():
                        nn, dd = int(num[mask].sum()), int(den[mask].sum())
                        assert p[metric]['numerator'] == nn
                        assert p[metric]['denominator'] == dd
                        assert p[metric]['value'] == (nn/dd if dd else None)
                    assert not (clear & ~t['core_gap_evaluated'][..., a]).any()
                    for source, count in p['sources'].items():
                        val = t[source][..., a] if t[source].ndim == 3 else t[source]
                        assert count == int((error & val)[mask].sum())
                    checks += 1
    assert d['poses'].shape == (len(rows), 16, 4, 4)
    from scipy.stats import binom
    import math
    assert math.ceil(math.log(.025)/math.log(1-.05)) == 72
    for key, planning in r['planning'].items():
        valid = [p for p in r['groups']['all']['curves'][key]
                 if p['risk']['value'] is not None and p['risk']['value'] <= .05]
        if not valid:
            assert planning['status'] == 'NO_DEVELOPMENT_POINT_AT5PERCENT'
            continue
        selected = valid[-1]
        assert planning['tau'] == selected['tau']
        assert planning['clear_yield'] == selected['clear']['value']
        yield_rate = planning['clear_yield']
        assert planning['expected_zero_error_units'] == math.ceil(72/yield_rate)
        n95 = planning['units95pct_atleast72clear']
        assert binom.sf(71, n95, yield_rate) >= .95
        assert binom.sf(71, n95-1, yield_rate) < .95
    return dict(status='INDEPENDENT_ARTIFACT_AUDIT_PASS', groups=len(masks),
                curve_points_recomputed=checks, rows=len(rows),
                hashes='PLAN/calibration/online/truth/provenance exact',
                inherited_score_aug_threshold_rows='EXACT_IDENTICAL_TO_R1',
                original_pose_identity_samples=sampled_poses,
                selected_model=calibration['selected_model'], selected_m=calibration['selected_m'])


def audit_certification_planning(out):
    """Independent PPF rejection boundary, crosschecked with beta CP limits."""
    from pathlib import Path
    from scipy.stats import beta, binom
    out = Path(out)
    planning = R1.read(out/'certification_planning.json')
    assert planning['source_result_sha256'] == R1.sha(out/'result.json')
    ns = np.arange(10001)
    critical = binom.ppf(.025, ns, .05).astype(int)
    critical -= binom.cdf(critical, ns, .05) > .025
    valid = (ns > 0) & (critical >= 0) & (critical < ns)
    assert (beta.ppf(.975, critical[valid]+1, ns[valid]-critical[valid]) <= .05).all()
    assert (beta.ppf(.975, critical[valid]+2, ns[valid]-critical[valid]-1) > .05).all()
    def power(n, p, r):
        counts = ns[:n+1]
        return float(binom.pmf(counts, n, p) @ binom.cdf(critical[:n+1], counts, r))
    checks = 0
    for arm, data in planning['arms'].items():
        for p in data['all_grid_at400']:
            np.testing.assert_allclose(power(400, p['p_clear'], p['risk']),
                                       p['cp_certificate_probability_at400'], rtol=1e-12)
            checks += 1
        for name in ('best_development_grid_at400', 'maximum_tau'):
            selected = data[name]
            if name == 'best_development_grid_at400':
                assert selected['index'] == max(data['all_grid_at400'],
                                                key=lambda x: x['cp_certificate_probability_at400'])['index']
            for percent in (80, 90):
                count = selected[f'units_for_{percent}pct']
                tested = [n for n in range(100, count+1, 100)
                          if power(n, selected['p_clear'], selected['risk']) >= percent/100]
                assert tested[0] == count
                np.testing.assert_allclose(power(count, selected['p_clear'], selected['risk']),
                    selected[f'achieved_at_{percent}pct'], rtol=1e-12)
                checks += 1
    return dict(status='INDEPENDENT_CP_PLANNING_AUDIT_PASS',
                all400_points_recomputed=76, step100_targets_recomputed=16,
                checks=checks, beta_cp_equivalence_checked_clear_counts=10000,
                scope='Fixed-tau iid plug-in budget model only; not full-sequence success or a risk certificate')


if __name__ == '__main__':
    unittest.main()
