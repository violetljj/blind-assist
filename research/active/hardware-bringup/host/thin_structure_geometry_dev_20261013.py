"""Consumed Development: candidate-local geometry under two existing scale fits.

No annotations are read. Forty line samples replace the old image-wide pixel
count only for this descriptive mechanism. Missing geometry stays UNKNOWN.
"""
from __future__ import annotations

import json
import shutil
import time
from collections import Counter

import numpy as np

from atom_camera import AtomCamera
import atom_tof_extrinsic_dev_20261012 as X
import real_rgb_query_feasibility_dev_20261012 as F
from fusion_refinement_readout_dev_20261013 import project

CACHE = F.ROOT / 'artifacts.local/work/fusion-refinement-dev-20261013'
SOURCE = F.ROOT / 'artifacts.local/work/thin-structure-dev-20261013/detection_ridge_frames.jsonl'
OUT = F.ROOT / 'artifacts.local/work/thin-local-geometry-dev-20261013'
ARMS = ('baseline', 'robust_replace')
SAMPLES = 40


def inside_boxes(points):
    return np.stack([(np.abs(points[:, 0]) <= .30) &
                     (points[:, 1] >= lo) & (points[:, 1] <= hi) &
                     (points[:, 2] >= .30) & (points[:, 2] <= 2.)
                     for lo, hi in F.BOXES.values()], axis=1)


def sample_geometry(candidate, disp, valid, fit, K, rotation, pose_ok, anchor):
    result = dict(status='UNKNOWN', reason=None, samples=SAMPLES, valid_points=0,
                  camera_z_median_m=None, corridor_median_m=None,
                  inside_count=None, inside_fraction_total=None,
                  inside_fraction_valid=None, query=[None, None], anchor=anchor,
                  outside_reason_count=None, domain_status='UNKNOWN')
    if not pose_ok:
        result['reason'] = 'pose'
        return result
    if disp is None or valid is None:
        result['reason'] = 'depth'
        return result
    if fit is None or not np.isfinite(fit).all() or abs(fit[0]) < 1e-12:
        result['reason'] = 'fit'
        return result
    p = np.array([[candidate['x1'], candidate['y1']],
                  [candidate['x2'], candidate['y2']]], float)
    xy = np.rint(p[0] + np.linspace(0, 1, SAMPLES)[:, None] * (p[1] - p[0])).astype(int)
    bounded = (xy[:, 0] >= 0) & (xy[:, 0] < disp.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < disp.shape[0])
    xy = xy[bounded]
    inv = (disp[xy[:, 1], xy[:, 0]] - fit[1]) / fit[0]
    good = valid[xy[:, 1], xy[:, 0]] & np.isfinite(inv) & (inv > 1e-3)
    xy = xy[good]
    inv = inv[good]
    result['valid_points'] = len(xy)
    if not len(xy):
        result['reason'] = 'valid_depth'
        return result
    dirs = np.stack(((xy[:, 0] - K[0, 2]) / K[0, 0],
                     (xy[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(xy))), axis=1)
    points = (dirs @ rotation.T) / inv[:, None]
    counts = inside_boxes(points).sum(axis=0)
    outside = dict(below_near_model_range=int(np.sum(points[:, 2] < .3)),
                   beyond_lookahead=int(np.sum(points[:, 2] > 2.)),
                   lateral_outside=int(np.sum(np.abs(points[:, 0]) > .3)),
                   vertical_outside=int(np.sum((points[:, 1] < -.20) | (points[:, 1] > .90))))
    result.update(camera_z_median_m=float(np.median(1 / inv)),
                  corridor_median_m=np.median(points, axis=0).tolist(),
                  inside_count=counts.tolist(), inside_fraction_total=(counts / SAMPLES).tolist(),
                  inside_fraction_valid=(counts / len(xy)).tolist(), outside_reason_count=outside)
    if len(xy) < SAMPLES // 2:
        result['reason'] = 'insufficient_valid_depth'
    else:
        domain = ('BELOW_NEAR_MODEL_RANGE' if outside['below_near_model_range'] > len(xy) / 2
                  else 'BEYOND_LOOKAHEAD' if outside['beyond_lookahead'] > len(xy) / 2
                  else 'IN_OR_MIXED_RANGE')
        result.update(status='AVAILABLE', query=(counts >= SAMPLES // 2).tolist(), domain_status=domain)
    return result


def checks():
    # Same boundary inequalities as old image-wide project, including rotation.
    rng = np.random.default_rng(20261013)
    pts = rng.uniform([-1, -.8, .1], [1, 1.2, 3.], (500, 3))
    pts = np.concatenate([pts, [[.3, .42, .3], [-.3, .9, 2.], [0, -.2, 1.]]])
    n = len(pts)
    old = project(np.ones((1, n)), np.ones((1, n), bool), [1., 0.], pts)
    assert old['px'] == inside_boxes(pts).sum(axis=0).tolist()
    K = np.array([[100., 0., 50.], [0., 100., 50.], [0., 0., 1.]])
    line = dict(x1=50., y1=100., x2=50., y2=120.)
    disp = np.ones((160, 100)); valid = np.ones_like(disp, bool)
    a = sample_geometry(line, disp, valid, [1., 0.], K, np.eye(3), True, 'synthetic')
    assert a['query'] == [False, True] and a['inside_count'] == [0, 40]
    assert np.allclose(a['corridor_median_m'], [0, .6, 1])
    assert a['domain_status'] == 'IN_OR_MIXED_RANGE'
    near = sample_geometry(line, disp, valid, [.02, 0.], K, np.eye(3), True, 'synthetic')
    far = sample_geometry(line, disp, valid, [3., 0.], K, np.eye(3), True, 'synthetic')
    assert near['domain_status'] == 'BELOW_NEAR_MODEL_RANGE' and near['outside_reason_count']['below_near_model_range'] == 40
    assert far['domain_status'] == 'BEYOND_LOOKAHEAD' and far['outside_reason_count']['beyond_lookahead'] == 40
    assert near['query'] == far['query'] == [False, False]
    assert sample_geometry(line, disp, valid, None, K, np.eye(3), True, None)['query'] == [None, None]
    assert sample_geometry(line, disp, valid, [1., 0.], K, np.eye(3), False, None)['query'] == [None, None]
    # Actual extrinsics and nonzero pose agree pointwise with full-image formula.
    Rct = np.array(json.loads((X.OUT / 'camera_tof_extrinsic.json').read_text())['rotation'])
    rotation = F.rot(-23., 7.) @ Rct.T
    a = sample_geometry(line, disp, valid, [1., 0.], K, rotation, True, 'synthetic')
    ys = np.rint(np.linspace(100, 120, SAMPLES))
    rays = np.stack([np.zeros(SAMPLES), (ys - 50) / 100, np.ones(SAMPLES)], axis=1)
    old = project(np.ones((1, SAMPLES)), np.ones((1, SAMPLES), bool), [1., 0.], rays @ rotation.T)
    assert a['inside_count'] == old['px']
    return dict(synthetic_projection=True, old_box_count_parity_points=n,
                actual_rotation_old_project_parity=True, unknown_semantics=True,
                near_far_diagnostic_preserves_query=True)


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    # Preserve the first delivered version before adding range diagnostics.
    for name in ('frames.jsonl', 'summary.json'):
        old = OUT / ('geometry_' + name)
        archived = OUT / ('geometry_initial_' + name)
        if old.exists() and not archived.exists():
            shutil.copy2(old, archived)
    verified = checks()
    manifest = json.loads((CACHE / 'cache_manifest.json').read_text())
    readout = json.loads((CACHE / 'robust_readout.json').read_text())
    frames = {(k, r['i']): r for k, s in manifest['sessions'].items() for r in s['frames']}
    fits = {(k, r['i'], arm): r for k, s in readout['sessions'].items()
            for arm in ARMS for r in s['arms'][arm]}
    cam = AtomCamera()
    Rct = np.array(json.loads((X.OUT / 'camera_tof_extrinsic.json').read_text())['rotation'])
    sessions = {}
    with SOURCE.open(encoding='utf8') as source, (OUT / 'geometry_frames.jsonl').open('w', encoding='utf8') as out:
        for line in source:
            if time.perf_counter() - start > 220:
                raise TimeoutError('geometry CPU budget reserve')
            row = json.loads(line); key = row['key']; i = row['i']; meta = frames[key, i]
            counts = sessions.setdefault(key, Counter())
            counts['frames'] += 1
            disp = valid = None
            if meta.get('cache'):
                with np.load(CACHE / meta['cache']) as payload:
                    disp, valid = payload['disp'], payload['valid']
            pose_ok = bool(meta['pose_ok']) and np.isfinite([meta['pitch'], meta['roll']]).all()
            rotation = F.rot(meta['pitch'], meta['roll']) @ Rct.T if pose_ok else None
            for candidate in row['candidates']:
                candidate['geometry'] = {}
                counts['candidates'] += 1
                for arm in ARMS:
                    fitrow = fits[key, i, arm]
                    g = sample_geometry(candidate, disp, valid, fitrow['fit'], cam.K_upright,
                                        rotation, pose_ok, fitrow['anchor'])
                    candidate['geometry'][arm] = g
                    counts[f'{arm}_{g["status"]}'] += 1
                    counts[f'{arm}_domain_{g["domain_status"]}'] += 1
                    for j, name in enumerate(F.BOXES):
                        counts[f'{arm}_{name}_candidates'] += g['query'][j] is True
                candidate['geometry_disagreement'] = candidate['geometry'][ARMS[0]]['query'] != candidate['geometry'][ARMS[1]]['query']
                counts['disagreement_candidates'] += candidate['geometry_disagreement']
            for arm in ARMS:
                for j, name in enumerate(F.BOXES):
                    counts[f'{arm}_{name}_frames'] += any(c['geometry'][arm]['query'][j] is True for c in row['candidates'])
            out.write(json.dumps(row, allow_nan=False) + '\n')
    totals = Counter()
    for c in sessions.values(): totals.update(c)
    result = dict(source=str(SOURCE), rule='40 equally spaced rounded line samples; >=20 valid for AVAILABLE; >=20 inside each original box for query True',
                  evidence='consumed Development, inherited unverified scale and pose; not true object distance or alarm',
                  range_diagnostic='Outside reason counts may overlap. Domain uses a strict majority of valid projected points; insufficient geometry is UNKNOWN. BELOW_NEAR_MODEL_RANGE flags the estimated projection outside model range, not evidence that a 0.02m estimate is physically true. query False is not safety clearance.',
                  checks=verified, sessions=sessions, totals=totals, wall_s=time.perf_counter() - start)
    (OUT / 'geometry_summary.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(dict(totals=totals, checks=verified, wall_s=result['wall_s'])))


if __name__ == '__main__':
    main()
