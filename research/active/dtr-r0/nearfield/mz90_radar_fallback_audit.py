"""Posthoc provenance audit of sealed MZ90 predictions; never fits a policy."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def instrument(text):
    """Carry labels through the original RNG stream and stable numeric sort."""
    replacements = [
        ('    return raw, evaluator', '    return raw, evaluator, origins'),
        ('    for e, scene in enumerate(source):', '    origins = np.full((n, 4), "NONE", dtype="U24")\n    for e, scene in enumerate(source):'),
        ('for r, a, rv, _ in returns:', 'for origin_index, (r, a, rv, _) in enumerate(returns):'),
        ('radar.append((r, a, rv))', 'radar.append((r, a, rv, f"real:{origin_index}"))'),
        ('radar.append((r, a, -WEARER_SPEED*z/max(r, 1e-12)))', 'radar.append((r, a, -WEARER_SPEED*z/max(r, 1e-12), "persistent"))'),
        ('for r, a, rv in radar:', 'for r, a, rv, origin in radar:'),
        ('measured.append((r, a, rv))', 'measured.append((r, a, rv, origin))'),
        ('float(np.round(rng.uniform(-1, 1)/0.1)*0.1)))', 'float(np.round(rng.uniform(-1, 1)/0.1)*0.1), "transient"))'),
        ('for k, (r, a, rv) in enumerate(sorted(measured)[:4]):', 'for k, (r, a, rv, origin) in enumerate(sorted(measured, key=lambda row: row[:3])[:4]):\n                origins[j, k] = origin'),
    ]
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError(f'Sealed instrumentation anchor changed: {old}')
        text = text.replace(old, new)
    return text


def parity(actual, expected):
    assert set(actual) == set(expected)
    for key, value in actual.items():
        np.testing.assert_array_equal(value, expected[key], err_msg=key)


def descriptors(obs, q, i, s):
    """Causal observable features only; no truth or provenance argument."""
    slots = np.flatnonzero(q[s])
    start = i
    while start > max(0, i-2) and obs['episode_id'][start-1] == obs['episode_id'][i]:
        start -= 1
    return dict(support_age_frames=i-s,
                qualifying_frames_last3=int(q[start:i+1].any(axis=1).sum()),
                min_range_m=float(obs['radar_range_m'][s, slots].min()),
                min_abs_bearing_deg=float(np.abs(obs['radar_angle'][s, slots]).min()),
                most_negative_velocity=float(obs['radar_velocity'][s, slots].min()))


def run(root, out):
    receipt = json.loads((root/'receipt.json').read_text())
    for name, digest in receipt['hashes'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    namespace = {'__name__': 'mz90_provenance_replay'}
    exec(compile(instrument((root/'mz90_observation_source.py').read_text()),
                 '<sealed-source-provenance>', 'exec'), namespace)
    source = json.loads((root/'source.json').read_text())
    raw, evaluator, origins = namespace['materialize'](source, 'sensor_proxy')
    parity(raw, dict(np.load(root/'sensor_proxy/raw-observations.npz')))
    parity(evaluator, dict(np.load(root/'sensor_proxy/evaluator.npz')))
    obs = dict(np.load(root/'sensor_proxy/adapted-observations.npz'))
    prediction = np.load(root/'sensor_proxy/predictions.npz')['joint_spatial']
    q = (obs['radar_valid'] & (obs['radar_range_m'] < 3.18)
         & (obs['radar_velocity'] <= -.35) & (np.abs(obs['radar_angle']) <= 20))
    rows = []
    for i in np.flatnonzero(prediction & ~obs['tof_known']):
        s = int(i if q[i].any() else i-1)
        assert s >= 0 and obs['episode_id'][s] == obs['episode_id'][i] and q[s].any()
        labels = origins[s, q[s]].tolist()
        kinds = sorted(set('real' if x.startswith('real:') else x for x in labels))
        scene = source[int(evaluator['scene'][i])]
        t = float(raw['time_s'][i])
        hazard_support = False
        for label in labels:
            if label.startswith('real:'):
                obj = scene['objects'][int(label.split(':')[1])]
                hazard_support |= namespace['geometric_hazard'](
                    obj['x']+obj['vx']*t, obj['z']+(obj['vz']-.7)*t, obj['vx'], obj['vz']-.7)
        rows.append(dict(frame=int(i), outcome='TP' if evaluator['truth'][i] else 'FP',
                         origins=labels, category=kinds[0]+'_only' if len(kinds)==1 else 'mixed',
                         real_hazard_support=bool(hazard_support),
                         tof_packet_received=bool(raw['tof_packet_received'][i]),
                         **descriptors(obs, q, int(i), s)))
    counts = Counter(r['outcome'] for r in rows)
    assert counts == {'TP': 202, 'FP': 126}, counts
    # Same trajectory equations, not an assertion of equal observation laws.
    alias_checks = 0
    for scene in source:
        if not scene['persistent_clutter']:
            continue
        for t in np.arange(namespace['FRAMES'])*namespace['DT']:
            x, z = scene['clutter_x'], scene['clutter_z']-.7*t
            r = np.hypot(x, z)
            np.testing.assert_allclose((x*0+z*(-.7))/max(r, 1e-12), -.7*z/max(r, 1e-12), atol=1e-15)
            alias_checks += 1
    summary = dict(status='PASS', analysis='POSTHOC_CONSUMED_MZ90',
                   sealed_hashes_verified=len(receipt['hashes']), exact_raw_evaluator_parity=True,
                   predictions_unchanged=True, new_predictions=0, threshold_searches=0,
                   static_phantom_kinematic_checks=alias_checks, outcomes={})
    for outcome in ('TP', 'FP'):
        group = [r for r in rows if r['outcome']==outcome]
        summary['outcomes'][outcome] = dict(count=len(group),
            categories=dict(Counter(r['category'] for r in group)),
            real_hazard_support=sum(r['real_hazard_support'] for r in group),
            packet_gap=sum(not r['tof_packet_received'] for r in group),
            descriptors={k:np.quantile([r[k] for r in group], [0,.25,.5,.75,1]).tolist()
                for k in descriptors(obs,q,rows[0]['frame'],rows[0]['frame']-rows[0]['support_age_frames'])})
    out.mkdir(parents=True, exist_ok=False)
    (out/'rows.json').write_text(json.dumps(rows, indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    np.savez_compressed(out/'evaluator-only-radar-origins.npz', origins=origins)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source_run, args.output)
