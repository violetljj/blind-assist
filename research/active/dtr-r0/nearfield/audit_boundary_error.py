"""Independent governed audit of boundary-error rows; no diagnostic producer imports."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np

from audit_contact_boundary import (world_boundaries, world_labels, expected_queries,
                                    boundary_report, read, digest, compare, BANDS)
from tof_fov45_core import boxes45, simulate
from query_occupancy_data import observation_tokens

HERE = Path(__file__).resolve().parent
ART = HERE.parents[3] / 'artifacts.local/evidence'
OLD = ART / 'ba-contact-boundary-20260923'
SOURCE = ART / 'ba-query-occupancy-20260922'
PREPARED = ART / 'ba-query-occupancy-20260922-prepared'
CAPTURE = ART / 'ba-query-occupancy-20260922-capture'
CURRENT = ART / 'ba-boundary-error-20260923-v3'
KINDS = ('width', 'horizon')


def number(value):
    return float('inf') if value is None else float(value)


def equal(actual, expected, label):
    if isinstance(expected, dict):
        for key, value in expected.items():
            equal(actual[key], value, label + '/' + key)
    elif isinstance(expected, (list, tuple, np.ndarray)):
        assert actual is not None and len(actual) == len(expected), label
        for a, b in zip(actual, expected):
            equal(a, b, label)
    elif isinstance(expected, (float, np.floating)):
        assert np.isclose(number(actual), expected, rtol=1e-9, atol=1e-9), label
    else:
        assert actual == expected, (label, actual, expected)


def extent(ax, ay, depth, kind, layer):
    """Enumerate endpoints of feasible Z intervals instead of producer clipping."""
    low, high = max(.3, depth[0]), min(3., depth[1]) if kind == 'width' else depth[1]
    yl, yh = BANDS[layer]
    candidates = [low, high]
    faces = [(yl, ay[1]), (yh, ay[0])]
    if kind == 'horizon':
        faces += [(-.3, ax[1]), (.3, ax[0])]
    candidates += [face / slope for face, slope in faces if abs(slope) >= 1e-15]
    feasible = [z for z in candidates if low - 1e-10 <= z <= high + 1e-10
                and ay[0] * z <= yh + 1e-10 and ay[1] * z >= yl - 1e-10
                and (kind == 'width' or (ax[0] * z <= .3 + 1e-10 and ax[1] * z >= -.3 - 1e-10))]
    if low > high or not feasible:
        return None
    a, b = min(feasible), max(feasible)
    if kind == 'horizon':
        return a, b
    lateral = [0.] if ax[0] <= 0 <= ax[1] else []
    lateral += [abs(ax[0]), abs(ax[1])]
    return 2 * min(lateral) * a, 2 * max(lateral) * b


def select(points, allowed, kind, layer, target):
    x, y, z = points.T
    yl, yh = BANDS[layer]
    eligible = allowed & (y >= yl) & (y <= yh) & (z >= .3)
    eligible &= z <= 3. if kind == 'width' else np.abs(x) <= .3
    ids = np.flatnonzero(eligible)
    values = 2 * np.abs(x) if kind == 'width' else z
    if not len(ids):
        return np.inf, np.inf, None
    first = float(values[ids].min())
    costs = abs(values[ids] - target) if np.isfinite(target) else values[ids]
    chosen = int(ids[np.argmin(costs)])
    return first, float(values[chosen]), chosen


def native_frame(i, geometry, case, tokens, rows, truths):
    geo = geometry[i]
    path = CAPTURE / 'evaluator' / geo['native_path']
    assert digest(path) == geo['native_sha256'], str(path)
    depth = np.load(path, allow_pickle=False)
    assert depth.shape == (360, 640)
    sy = np.floor((np.arange(192) + .5) * 360 / 192).astype(int)
    sx = np.floor((np.arange(256) + .5) * 640 / 256).astype(int)
    ids = (sy[:, None] * 640 + sx[None, :]).ravel()
    boxes = boxes45()
    values, traces = simulate(depth[np.ix_(sy, sx)].copy(),
                              'query-occupancy/' + case['sensor_noise_key'], boxes)
    assert np.array_equal(observation_tokens(values, boxes), tokens)
    valid = np.isfinite(depth.ravel()) & (depth.ravel() > 0)
    z = np.where(valid, depth.ravel(), 0).astype(float)
    py, px = np.divmod(np.arange(360 * 640), 640)
    focal = 320 / np.tan(np.deg2rad(50.))
    points = np.column_stack((((px + .5 - 320) / focal) * z, ((py + .5 - 180) / focal) * z, z))
    sampled = points[ids]
    zones = np.full((192, 256), -1, int)
    observed = np.zeros(192 * 256, bool)
    for zone, (y0, x0, y1, x1) in enumerate(boxes):
        zones[y0:y1, x0:x1] = zone
        if traces[zone]['observed']:
            observed[traces[zone]['pixel_indices']] = True
    for kind in KINDS:
        for layer in range(2):
            record = rows[i, 'direct', kind, layer]['native_info']
            target = truths[i][kind][layer]
            for name, cloud, mask in [('native', points, valid),
                    ('lattice', sampled, valid[ids] & (zones.ravel() >= 0)),
                    ('returned', sampled, valid[ids] & observed)]:
                first, nearest, index = select(cloud, mask, kind, layer, target)
                equal(record, {name: first, 'near_' + name: nearest}, f'native/{i}/{kind}/{layer}')
            equal(record['sampled_point'], index, 'selected sample')
            if index is None:
                assert record['zone'] is None and record['native_point'] is None
                continue
            zone = int(zones.ravel()[index])
            assert index in traces[zone]['pixel_indices'] and traces[zone]['observed']
            equal(record, {'zone': zone, 'native_point': int(ids[index])}, 'lineage')
            value = float(values[zone]); radius = .13 + .06 * value
            interval = (max(.1, value - radius), value + radius)
            point = sampled[index]; rayx, rayy = point[:2] / point[2]
            y0, x0, y1, x1 = boxes[zone]
            ax = ((x0 * 2.5 - 320) / focal, (x1 * 2.5 - 320) / focal)
            ay = ((y0 * 1.875 - 180) / focal, (y1 * 1.875 - 180) / focal)
            equal(record['public_interval'], interval, 'public interval')
            equal(record['actual_point_in_public_interval'], bool(interval[0] <= point[2] <= interval[1]), 'inclusion')
            for name, angles in [('zone', (ax, ay)), ('ray', ((rayx, rayx), (rayy, rayy)))]:
                span = extent(*angles, interval, kind, layer)
                equal(record[name + '_extent'], span, name + ' extent')
                equal(record[name + '_span'], None if span is None else span[1] - span[0], name + ' span')
    assert digest(path) == geo['native_sha256']


def audit():
    journal = os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    assert journal and read(journal)['state'] == 'running', 'Use governed research-ue execution'
    start = time.perf_counter()
    result = read(CURRENT / 'result.json'); rows = read(CURRENT / 'rows.json')
    assert result['status'] == 'PASS' and len(rows) == 13824
    assert digest(CURRENT / 'rows.json') == result['rows_sha256']
    assert rows == read(ART / 'ba-boundary-error-20260923-v2/rows.json'), 'v2/v3 row drift'
    recorded_seal = read(CURRENT / 'input-seal.json')
    seal = dict(recorded_seal['files'])
    seal.update({str(HERE / n): h for n, h in recorded_seal['code'].items()})
    seal.update(read(OLD / 'source-seal.json')['input_hashes'])
    extra = [CURRENT / x for x in ('result.json', 'rows.json', 'input-seal.json')]
    extra += [ART / 'ba-boundary-error-20260923-v2/rows.json', Path(__file__),
              HERE / 'audit_contact_boundary.py', HERE / 'query_occupancy_data.py']
    seal = dict(seal, **{str(p): digest(p) for p in extra})
    oldseal = read(OLD / 'prediction-seal.json')
    seal.update({str(OLD / n): h for n, h in oldseal['files'].items()})
    seal.update({str(OLD / (a + '-selection.json')): h for a, h in oldseal['selection_hashes'].items()})
    for path, expected in seal.items():
        assert digest(path) == expected, path
    geometry = read(CAPTURE / 'evaluator/geometry.json')
    meta = read(PREPARED / 'observations/identities.json')
    cases = read(SOURCE / 'plan/spec.json')['cases']; cohort = read(OLD / 'cohort.json')['indices']
    truths = [world_boundaries(g) for g in geometry]
    query = expected_queries()['seen']; labels = [world_labels(g, query) for g in geometry]
    lookup = {(r['index'], r['arm'], r['kind'], r['layer']): r for r in rows}
    assert len(lookup) == len(rows)
    grouped = defaultdict(list); checks = Counter()
    old = read(OLD / 'result.json')
    for split, indices in cohort.items():
        for arm in ('direct', 'geometry'):
            threshold = read(OLD / (arm + '-selection.json'))['threshold']
            with np.load(OLD / f'{arm}-{split}-predictions.npz', allow_pickle=False) as saved:
                for kind in KINDS:
                    axis = np.linspace(.2, 1.2, 51) if kind == 'width' else np.linspace(.3, 3., 55)
                    curves = saved[kind + '_curve'].reshape(-1, 2, len(axis))
                    assert not (np.diff(curves, axis=-1) < -1e-6).any()
                    truth_array = np.asarray([truths[i][kind] for i in indices])
                    if split != 'dev':
                        reference = old['arms'][arm] if split == 'evaluation' else old['arms'][arm]['training_layouts']
                        compare(boundary_report(saved[kind + '_curve'], threshold, truth_array, kind),
                                reference['boundaries'][kind], 'old metrics'); checks['original_metrics'] += 1
                    for local, i in enumerate(indices):
                        for layer in range(2):
                            r = lookup[i, arm, kind, layer]; t = truths[i][kind][layer]
                            assert r['split'] == split and r['id'] == meta[i]['id'] == geometry[i]['id']
                            col, fixed = (0, 1) if kind == 'width' else (1, 0)
                            mask = (query[:, 2] == layer) & np.isclose(query[:, fixed], 3. if kind == 'width' else .6)
                            coordinates = np.round(query[mask, col].astype(float), 6); y = labels[i][mask]
                            assert not np.any(y[:-1] & ~y[1:])
                            lo = max(coordinates[~y], default=0. if kind == 'width' else .3)
                            hi = min(coordinates[y], default=np.inf)
                            hit = np.flatnonzero(curves[local, layer].astype(float) >= threshold)
                            j = int(hit[0]) if len(hit) else len(axis)
                            pc = float(axis[j]) if j < len(axis) else np.inf
                            cl = -np.inf if j == 0 else round(float(axis[j - 1]), 6)
                            cu = round(pc, 6)
                            incompatible = max(lo, cl) >= min(hi, cu) if np.isfinite(max(lo, cl)) else False
                            tc = 'left' if t <= axis[0] + 1e-6 else 'right' if t > axis[-1] + 1e-6 else 'interior'
                            good = tc == 'interior' and np.isfinite(pc) and abs(pc - t) <= .050001
                            equal(r, dict(truth=t, predicted=pc, label_lower=lo, label_upper=hi,
                                  true_class=tc, predicted_class='left' if j == 0 else 'right' if j == len(axis) else 'interior',
                                  within_5cm=bool(good), bracket_incompatible=bool(incompatible)), 'row')
                            if split == 'evaluation':
                                s = r['native_info']; state = 'none_near'
                                if np.isfinite(t):
                                    gaps = [abs(number(s['near_' + k]) - t) for k in ('native', 'lattice', 'returned')]
                                    assert gaps[0] <= gaps[1] + 1e-9 and gaps[1] <= gaps[2] + 1e-9
                                    for gap, name in zip(gaps, ('native_only_near', 'lattice_only_near', 'returned_near')):
                                        if gap <= .050001: state = name
                                assert r['support_class'] == state
                                assert s == lookup[i, 'direct', kind, layer]['native_info']
                            grouped[f'{split}/{arm}/{kind}'].append(r); checks['rows'] += 1
    for key, group in grouped.items():
        interior = [r for r in group if r['true_class'] == 'interior']
        errors = [r for r in interior if not r['within_5cm']]
        fields = dict(rows=len(group), interior_true=len(interior), errors=len(errors),
                      within_5cm=sum(r['within_5cm'] for r in interior),
                      errors_bracket_incompatible=sum(r['bracket_incompatible'] for r in errors),
                      errors_bracket_compatible=sum(not r['bracket_incompatible'] for r in errors),
                      true_censoring=dict(Counter(r['true_class'] for r in group)),
                      predicted_censoring=dict(Counter(r['predicted_class'] for r in group)),
                      support_classes=dict(Counter(r.get('support_class', 'not_audited') for r in interior)))
        equal(result['summaries'][key], fields, key); checks['summaries'] += 1
    frames = [i for i in cohort['evaluation'] if meta[i]['frame_in_clip'] == 5]
    assert len(frames) == 48 and len({meta[i]['base_group_id'] for i in frames}) == 16
    assert all(len({meta[i]['layout_relation'] for i in frames if meta[i]['base_group_id'] == g}) == 3
               for g in {meta[i]['base_group_id'] for i in frames})
    tokens = np.load(PREPARED / 'observations/tof.npy', mmap_mode='r')
    for i in frames:
        native_frame(i, geometry, cases[i], tokens[i], lookup, truths); checks['native_frames'] += 1
    for path, expected in seal.items():
        assert digest(path) == expected, path
    return dict(status='PASS', checks=dict(checks), elapsed_s=time.perf_counter() - start,
                rows_sha256=digest(CURRENT / 'rows.json'), audit_sha256=digest(__file__),
                v2_v3_rows_equal=True, source_and_prediction_hashes_unchanged=True,
                backend='TASK_NOT_GPU_SUITABLE', limits='Consumed evaluator diagnostic; no model or information-ceiling claim')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--result', required=True)
    destination = Path(parser.parse_args().result)
    journal = os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    assert journal and read(journal)['state'] == 'running', 'Use governed research-ue execution'
    assert not destination.exists(), 'Preserve existing audit receipts'
    try:
        receipt = audit()
    except Exception as exc:
        destination.write_text(json.dumps(dict(status='FAIL', error=repr(exc)), indent=2), encoding='utf-8')
        raise
    destination.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(receipt))
