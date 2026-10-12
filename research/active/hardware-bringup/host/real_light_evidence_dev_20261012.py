"""Quiet-selected native observation gates; consumed Development, not confirmation.

Only existing S light grades may be suppressed. Strong is invariant. Native
bin0 remains excluded by the inherited contract; no other bins/rows are removed.
Four fixed, causal observation features test persistence and spatial coherence.
All pose variants of a recording stay together in leave-one-recording-out.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from real_light_tradeoff_dev_20261012 import describe


def gate(data, evidence, cut):
    original = data['S_grade']
    result = np.where((original == 1) & (evidence < cut), 0, original)
    assert np.array_equal(result == 2, original == 2)
    return result


def rate(records, arrays, features, feature, cut):
    groups = {}
    for r in records:
        if r['expect'] != 'quiet':
            continue
        d = describe(arrays[r['id']], gate(arrays[r['id']], features[r['id']][feature], cut))
        groups.setdefault(r['source'] + ':' + r['key'], []).append(d['light_union_frames'] / d['ready_frames'])
    return float(np.mean([max(x) for x in groups.values()]))


def select(records, arrays, features, feature, cap):
    vals = [np.array([0.])]
    for r in records:
        if r['expect'] == 'quiet':
            d = arrays[r['id']]
            vals.append(features[r['id']][feature][(d['S_grade'] == 1) & d['ready'][:, None]])
    cuts = np.unique(np.r_[0., np.nextafter(np.concatenate(vals).astype(float), np.inf)])
    lo, hi = 0, len(cuts) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if rate(records, arrays, features, feature, cuts[mid]) <= cap:
            hi = mid
        else:
            lo = mid + 1
    cut = float(cuts[lo])
    actual = rate(records, arrays, features, feature, cut)
    assert actual <= cap
    return cut, actual


def features_for(diagnosis, r, data):
    raw = np.load(diagnosis / 'inputs' / (r['source'] + '_' + r['key'] + '.npz'))
    z = raw['z'].astype(float)
    starts = np.maximum.accumulate(np.where(data['reset'], np.arange(len(z)), 0))
    avg = np.stack([z[max(starts[i], i-7):i+1].mean(0) for i in range(len(z))])
    # Remove only the per-bin spatial common mode, not selected native cells.
    contrast = avg - np.median(avg, axis=(1, 2), keepdims=True)
    weight = np.load(diagnosis / (r['id'] + '_geometry.npz'))['source_weight'].astype(float)
    weight[..., 0] = 0
    support = weight > 0
    cached = np.load(diagnosis / (r['id'] + '_evidence.npz'))
    native_max = np.stack([avg[:, support[q]].max(1) if support[q].any() else np.zeros(len(z)) for q in range(2)], 1)
    assert np.allclose(native_max, cached['history8_max_bin1plus'], atol=1e-5)
    out = {'native_max8': native_max}
    out['contrast_max8'] = np.stack([contrast[:, support[q]].max(1) if support[q].any() else np.zeros(len(z)) for q in range(2)], 1)
    out['contrast_top4_mean8'] = np.stack([np.sort(contrast[:, support[q]], axis=1)[:, -4:].mean(1) if support[q].any() else np.zeros(len(z)) for q in range(2)], 1)
    # Coherent signed sum; normalized by spatial white-noise norm. Correlation
    # and nuisance are left for quiet-only calibration, not assumed away.
    out['weighted_signed8'] = np.stack([(avg * weight[q]).sum((1, 2, 3)) / max(1e-30, np.sqrt((weight[q]**2).sum())) for q in range(2)], 1)
    return out


def run(index, output):
    records = json.loads(index.read_text())['records']
    diagnosis = index.parent.parent
    arrays, features = {}, {}
    output.mkdir(parents=True, exist_ok=True)
    for r in records:
        arrays[r['id']] = dict(np.load(r['npz']))
        features[r['id']] = features_for(diagnosis, r, arrays[r['id']])
        np.savez_compressed(output / (r['id'] + '_features.npz'), **features[r['id']])
    result = {'scope': 'posthoc consumed real Development; weak scene labels; no live default change',
              'selection': 'quiet only, recording-macro worst pose; UNKNOWN and near-pass excluded',
              'features': {}, 'strong_invariant_all_records': True}
    keys = sorted({r['source'] + ':' + r['key'] for r in records if r['expect'] == 'quiet'})
    for feature in next(iter(features.values())):
        desc = {}
        for cap in (.20, .10, .05):
            cut, actual = select(records, arrays, features, feature, cap)
            desc[str(cap)] = {'cut': cut, 'quiet_macro_worst_pose_light_rate': actual,
                'records': {r['id']: dict(source=r['source'], key=r['key'], arm=r['arm'], expect=r['expect'],
                    **describe(arrays[r['id']], gate(arrays[r['id']], features[r['id']][feature], cut))) for r in records}}
        folds = []
        for key in keys:
            train = [r for r in records if r['source'] + ':' + r['key'] != key]
            held = [r for r in records if r['source'] + ':' + r['key'] == key]
            cut, actual = select(train, arrays, features, feature, .10)
            folds.append({'held': key, 'cut': cut, 'train_rate': actual,
                          'held_rate': rate(held, arrays, features, feature, cut)})
        result['features'][feature] = {'cuts': desc, 'leave_one_recording_out': folds}
    (output / 'light_evidence.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps({f: {'cap10_cut': v['cuts']['0.1']['cut'], 'cap10_quiet': v['cuts']['0.1']['quiet_macro_worst_pose_light_rate'],
                         'held_macro': np.mean([x['held_rate'] for x in v['leave_one_recording_out']])} for f, v in result['features'].items()}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.index, a.output)
