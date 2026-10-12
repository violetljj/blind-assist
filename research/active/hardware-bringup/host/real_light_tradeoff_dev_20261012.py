"""Consumed real Development: quiet-only light cuts and causal persistence tradeoffs.

Strong grades are copied unchanged. UNKNOWN scenes never select a cut. A held-out
segment fold excludes every pose variant of that recording; this is descriptive
Development reuse, not new validation. No model or live default is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def grades(data, tau, consecutive):
    ready = data['ready'].astype(bool)
    reset = data['reset'].astype(bool)
    original = data['S_grade']
    # Compare in float64: Python scalar coercion to float32 would round a
    # nextafter cut back onto a score tie after selection/JSON serialization.
    score = data['s_score'].astype(np.float64)
    hit = np.isfinite(score) & (score >= float(tau))
    count = np.zeros(2, int)
    light = np.zeros_like(hit)
    for i in range(len(hit)):
        if reset[i]:
            count[:] = 0
        count = np.where(hit[i] & ready[i], count + 1, 0)
        light[i] = count >= consecutive
    result = np.where(original == 2, 2, np.where(light & ready[:, None], 1, 0)).astype(np.int8)
    if not np.array_equal(result == 2, original == 2):
        raise AssertionError('Strong grades changed')
    return result


def describe(data, grade):
    ready = data['ready'].astype(bool)
    old = data['S_grade'][ready]
    new = grade[ready]
    out = {'ready_frames': int(ready.sum())}
    for q, name in enumerate(('HEAD', 'BODY')):
        out[name] = {
            'strong': int((new[:, q] == 2).sum()),
            'light': int((new[:, q] == 1).sum()),
            'old_light': int((old[:, q] == 1).sum()),
            'lost_any_frames': int(((old[:, q] > 0) & (new[:, q] == 0)).sum()),
            'rescued_any_frames': int(((old[:, q] == 0) & (new[:, q] > 0)).sum()),
        }
    out['light_union_frames'] = int((new == 1).any(1).sum())
    out['strong_union_frames'] = int((new == 2).any(1).sum())
    out['any_union_frames'] = int((new > 0).any(1).sum())
    return out


def quiet_rate(records, arrays, tau, consecutive):
    # Repeated assumed poses do not become independent negative recordings.
    groups = {}
    for rec in records:
        if rec['expect'] != 'quiet':
            continue
        data = arrays[rec['id']]
        d = describe(data, grades(data, tau, consecutive))
        if d['ready_frames']:
            key = rec['source'] + ':' + rec['key']
            groups.setdefault(key, []).append(d['light_union_frames'] / d['ready_frames'])
    return float(np.mean([max(v) for v in groups.values()])) if groups else None


def select(records, arrays, baseline_tau, consecutive, cap):
    values = [np.array([baseline_tau])]
    for rec in records:
        if rec['expect'] == 'quiet':
            d = arrays[rec['id']]
            s = d['s_score'][d['ready'].astype(bool)]
            values.append(s[np.isfinite(s) & (s >= baseline_tau)])
    # nextafter removes complete ties and the final value permits no light.
    # A float32-nextafter cut also survives the existing replay engine's
    # float32 scalar comparison; preserve whole ties in the exported manifest.
    above = np.nextafter(np.concatenate(values).astype(np.float32), np.float32(np.inf)).astype(float)
    cuts = np.unique(np.r_[baseline_tau, above])
    lo, hi = 0, len(cuts) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if quiet_rate(records, arrays, cuts[mid], consecutive) <= cap:
            hi = mid
        else:
            lo = mid + 1
    tau = float(cuts[lo])
    actual = quiet_rate(records, arrays, tau, consecutive)
    if actual is None or actual > cap:
        raise AssertionError('Selected whole-tie cut does not satisfy quiet cap')
    return tau, actual


def run(index, output, baseline_tau):
    payload = json.loads(index.read_text(encoding='utf8'))
    records = payload['records']
    arrays = {}
    for rec in records:
        path = Path(rec['npz'])
        if not path.is_absolute():
            path = index.parent / path
        with np.load(path) as data:
            arrays[rec['id']] = {k: data[k] for k in ('ready', 'reset', 's_score', 'S_grade')}
    output.mkdir(parents=True, exist_ok=True)
    result = {'scope': 'consumed real Development; scene expectations are weak labels',
              'selection': 'quiet only, equal recording weight, worst assumed pose per recording',
              'baseline_tau': baseline_tau, 'rules': {}, 'leave_one_recording_out': {}}
    specs = [('baseline', baseline_tau, 1, None)]
    for consecutive in (1, 2, 3):
        specs.append((f'persistence_{consecutive}', baseline_tau, consecutive, None))
        for cap in (.20, .10, .05):
            tau, rate = select(records, arrays, baseline_tau, consecutive, cap)
            specs.append((f'quiet{int(cap*100)}_persist{consecutive}', tau, consecutive, cap))
    for name, tau, consecutive, cap in specs:
        result['rules'][name] = {'tau': tau, 'consecutive': consecutive, 'quiet_cap': cap,
            'quiet_macro_worst_pose_light_rate': quiet_rate(records, arrays, tau, consecutive),
            'records': {r['id']: dict(source=r['source'], key=r['key'], arm=r['arm'], expect=r['expect'],
                                     **describe(arrays[r['id']], grades(arrays[r['id']], tau, consecutive))) for r in records}}
    keys = sorted({r['source'] + ':' + r['key'] for r in records if r['expect'] == 'quiet'})
    for consecutive in (1, 2, 3):
        folds = []
        for key in keys:
            train = [r for r in records if r['source'] + ':' + r['key'] != key]
            held = [r for r in records if r['source'] + ':' + r['key'] == key]
            tau, rate = select(train, arrays, baseline_tau, consecutive, .10)
            folds.append({'held': key, 'tau': tau, 'train_rate': rate,
                          'held_rate': quiet_rate(held, arrays, tau, consecutive)})
        result['leave_one_recording_out'][str(consecutive)] = folds
    (output / 'light_tradeoff.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({k: {f: v[f] for f in ('tau', 'consecutive', 'quiet_macro_worst_pose_light_rate')}
                      for k, v in result['rules'].items()}, indent=2))


def simulate(output):
    """Apply already selected rules to the consumed synthetic joint-stack scores."""
    root = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(root / 'research/active/dtr-r0/nearfield'))
    import cnh_pose_floor_dr_dev_20261012 as P
    EV = P._ev()
    folder = root / 'artifacts.local/work/cnh-pose-floor-noise-dr-dev-20261012'
    cuts = json.loads((folder / 'cuts.json').read_text())['cuts']
    rules = json.loads((output / 'light_tradeoff.json').read_text())['rules']
    names = ('baseline', 'persistence_2', 'quiet10_persist1', 'quiet10_persist2', 'quiet5_persist2')
    rows = EV.C.read(EV.OUT / 'scene_rows.json')['hold']
    with np.load(EV.OUT / 'data/hold/geometry.npz') as g:
        category = g['category']
    report = {}
    for cond in P.HOLD_CONDITIONS:
        sc = dict(np.load(folder / f'scores/hold_{cond}_new.npz'))
        strong = P._strong(sc, cuts)
        score = EV.T.predict_s(folder / 'S', EV.s_features(sc, cuts)).mean(0).astype(float)
        original = EV.TM.apply_gate(strong, score, cuts['s_tau'])
        report[cond] = {}
        for name in names:
            rule = rules[name]
            hits = score >= rule['tau']
            count = np.zeros(hits.shape[:2] + (2,), int)
            light = np.zeros_like(hits)
            for f in range(hits.shape[2]):
                count = np.where(hits[:, :, f], count + 1, 0)
                light[:, :, f] = count >= rule['consecutive']
            grade = np.where(strong == 2, 2, light.astype(np.int8))
            assert np.array_equal(grade == 2, original == 2)
            report[cond][name] = EV.describe(grade, original, category, rows)[0]
        print('SIM', cond, json.dumps(report[cond]), flush=True)
    (output / 'sim_light_tradeoff.json').write_text(json.dumps(report, indent=2), encoding='utf8')


def export_manifest(output, source, rule_name):
    rule = json.loads((output / 'light_tradeoff.json').read_text())['rules'][rule_name]
    if rule['consecutive'] != 1:
        raise ValueError('Existing replay manifest supports a scalar cut only')
    manifest = json.loads(source.read_text(encoding='utf8'))
    manifest['candidate'] = 'joint_noise_quieter_light_development'
    manifest['cuts']['s_tau'] = rule['tau']
    manifest['development_only'] = True
    manifest['selection'] = dict(rule=rule_name, source_manifest=str(source.resolve()),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        data='six consumed quiet recordings; worst pose per recording, equal recording weight',
        scope='optional replay candidate; not independently validated or a global default',
        strong_unchanged=True, consecutive=1)
    path = output / 'real_manifest_quieter_light.json'
    path.write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print(path)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--baseline-tau', type=float, default=.9520257115364076)
    p.add_argument('--simulate', action='store_true')
    p.add_argument('--export-from', type=Path)
    p.add_argument('--rule', default='quiet10_persist1')
    a = p.parse_args()
    if a.export_from:
        export_manifest(a.output, a.export_from, a.rule)
    elif a.simulate:
        simulate(a.output)
    elif a.index:
        run(a.index, a.output, a.baseline_tau)
    else:
        p.error('--index required for real-data analysis')
