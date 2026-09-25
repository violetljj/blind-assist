"""Fast-lane diagnosis: why multi-frame accumulation does not help HEAD (v1.3 pilot).

Condition fixed to the pilot's primary: mount -10, SNR6, 5 Hz, GT ego-motion.
Variants of the S2 transport weight g = (r_past/r_now)^(2*power):
power 1 = current r^2 compensation, 0 = none, -1 = SNR-matched down-weighting;
history K in {2, 4}. Part A uses the oracle noise-free expected response (audit
only) to measure the ideal scan Z with/without accumulation; Part B re-scores
the noisy stream (tau chosen on calib) and reports audit macro AP.
"""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse
from cnh_track_a_readout import cell_points, _transform, _indices, query_weights
from cnh_track_a_v12_readout import s1, scan_transported
from cnh_track_a_v13_evaluate import load_records, macro_ap, GROUPS, TAUS

ROOT = Path(r'E:\linnan\linnan\artifacts.local\evidence\cnh-track-a-v13-pilot12-20260926-v1')
VARIANTS = [(p, k) for k in (4, 2) for p in (1, 0, -1)]


def transport(current_from_past, power):
    original = cell_points()
    points = _transform(original, current_from_past)
    dest, valid = _indices(points)
    radius = np.linalg.norm(points, axis=-1)
    valid &= radius > 0
    gain = np.ones(radius.shape)
    np.divide(np.linalg.norm(original, axis=-1), radius, out=gain, where=radius > 0)
    gain = gain**(2*power)
    source = np.broadcast_to(np.arange(1024)[:, None], valid.shape)
    a = sparse.coo_matrix((gain[valid]/16, (dest[valid], source[valid])), shape=(1024, 1024)).tocsr()
    a.sum_duplicates()
    return a


def s2_sequence(r, v, poses, weights, tau, power, k):
    out = np.empty((len(r), 6))
    identity = sparse.eye(1024, format='csr')
    for i, w in enumerate(weights):
        if i == 0 or k == 1:
            out[i] = s1(r[i:i+1], v[i:i+1], w, tau)[0]
            continue
        sources = [(i, identity)] + [(j, transport(np.linalg.inv(poses[i]) @ poses[j], power))
                                     for j in range(max(0, i-k+1), i)]
        out[i] = scan_transported(r, v, sources, w, tau)
    return out


def variant(args):
    power, k = args
    records = load_records(ROOT/'geometry', ROOT/'sensor', -10, 1, 5)
    bias = np.load(ROOT/'readouts'/'mount-10-snr6-5hz-scores.npz')['bias']
    oracle = {}
    scores = {tau: [] for tau in TAUS}
    ideal_s1, ideal_s2 = [], []
    for rec in records:
        r = rec['hist']-bias
        v = 16*rec['ambient'][..., None]+np.maximum(bias, 0)
        weights = [query_weights(t) for t in rec['tq']]
        for tau in TAUS:
            scores[tau].append(s2_sequence(r, v, rec['poses'], weights, tau, power, k))
        if rec['split'] == 'audit':
            if rec['unit'] not in oracle:
                with np.load(ROOT/'sensor'/f"unit{rec['unit']:02d}-mount-10-oracle.npz") as f:
                    e = f['expected_H3'][1]
                with np.load(ROOT/'sensor'/f"unit{rec['unit']:02d}-mount-10-observations.npz") as f:
                    oracle[rec['unit']] = (e, f['config'])
            e, cfg = oracle[rec['unit']]
            re = e[np.flatnonzero(cfg == rec['config'])[::2]]-bias
            ideal_s1.append(np.array([s1(re[i:i+1], v[i:i+1], w, .75)[0] for i, w in enumerate(weights)]))
            ideal_s2.append(s2_sequence(re, v, rec['poses'], weights, .75, power, k))
    y = np.concatenate([rec['labels'] for rec in records])
    units = np.concatenate([np.full(len(rec['labels']), rec['unit']) for rec in records])
    split = np.concatenate([np.full(len(rec['labels']), rec['split']) for rec in records])
    main = np.concatenate([rec['main'] for rec in records])
    calib, audit = (split == 'calib') & main, (split == 'audit') & main
    scores = {tau: np.concatenate(s) for tau, s in scores.items()}
    cand = {tau: np.mean([macro_ap(y[:, ix], scores[tau][:, ix], units, calib) for _, ix in GROUPS]) for tau in TAUS}
    tau = max(TAUS, key=lambda t: (cand[t], -t))
    result = dict(power=power, K=k, tau=tau, calib=cand,
                  audit_macro_AP={g: macro_ap(y[:, ix], scores[tau][:, ix], units, audit) for g, ix in GROUPS})
    i1, i2 = np.concatenate(ideal_s1), np.concatenate(ideal_s2)
    ya, ma = y[split == 'audit'], main[split == 'audit']
    ideal = {}
    for g, ix in GROUPS:
        pos = (ma[:, None] & (ya[:, ix] == 1))
        a, b = i1[:, ix][pos], i2[:, ix][pos]
        ok = a > 0
        ideal[g] = dict(positives=int(pos.sum()), median_ideal_Z_S1=float(np.median(a)), median_ideal_Z_S2=float(np.median(b)),
                        median_ratio=float(np.median(b[ok]/a[ok])), frac_S2_gt_S1=float(np.mean(b > a)))
    result['ideal'] = ideal
    return result


def target_alignment(step=2):
    """Offline: share of past-frame target-only energy (crosstalk removed) that lands on
    current target cells (>=20% of max) after correct / identity / reverse transport."""
    from dataclasses import replace
    from cnh_route_sensor import SensorParameters, angular_rays, synthesize_response
    p6 = replace(SensorParameters(), signal_counts=json.loads(
        (ROOT/'sensor'/'unit08-mount-10.json').read_text(encoding='utf-8'))['parameters']['6']['signal_counts'], noise_scale=0.)
    _, w = angular_rays(16)

    def h3(d, rho, cos):
        return synthesize_response(d, rho, cos, w, params=p6, seed=0)['histogram'].reshape(8, 8, 16, 8).sum(-1).reshape(1024)
    empty = h3(np.full((8, 8, 256), np.inf), 0., 0.)
    records = [r for r in load_records(ROOT/'geometry', ROOT/'sensor', -10, 1, 5) if r['split'] == 'audit'][::step]
    shares = {k: {'HEAD': [], 'BODY': []} for k in ('correct', 'identity', 'reverse')}
    cache = {}
    for rec in records:
        u = rec['unit']
        if u not in cache:
            with np.load(ROOT/'sensor'/f'unit{u:02d}-mount-10-oracle.npz') as f:
                cache[u] = {k: f[k] for k in ('raydistance', 'object_id', 'raycos')}
            with np.load(ROOT/'sensor'/f'unit{u:02d}-mount-10-observations.npz') as f:
                cache[u]['config'] = f['config']
            cache[u]['geo'] = {c['config']: c for c in json.loads(
                (ROOT/'geometry'/f'unit{u:02d}'/f'unit{u:02d}.json').read_text(encoding='utf-8-sig'))['configs']}
        o = cache[u]
        rows = np.flatnonzero(o['config'] == rec['config'])[::2]
        objects = [ob for ob in o['geo'][rec['config']]['objects'] if ob.get('category') in ('HEAD', 'BODY')]
        for cat in ('HEAD', 'BODY'):
            sub = {ob['id']: ob['rho'] for ob in objects if ob['category'] == cat}
            if not sub:
                continue
            frames = []
            for row in rows:
                oid = o['object_id'][row]
                rho = np.zeros(oid.shape)
                for i, r in sub.items():
                    rho[oid == i] = r
                mask = np.isin(oid, list(sub))
                frames.append(h3(np.where(mask, o['raydistance'][row], np.inf), rho, o['raycos'][row])-empty)
            poses = rec['poses']
            for i in range(1, len(frames)):
                cur, past = frames[i], frames[i-1]
                if cur.sum() < 5 or past.sum() < 5:
                    continue
                top = cur >= .2*cur.max()
                for name, t in (('correct', np.linalg.inv(poses[i]) @ poses[i-1]),
                                ('reverse', np.linalg.inv(poses[i-1]) @ poses[i]), ('identity', None)):
                    moved = past if t is None else transport(t, 0) @ past
                    shares[name][cat].append(float(moved[top].sum()/past.sum()))
    return {k: {c: dict(pairs=len(v), median_share=float(np.median(v))) for c, v in d.items()} for k, d in shares.items()}


def ideal_ceiling():
    """Offline: median sqrt(sum_{j=t-3..t} Z_j^2)/Z_t of noise-free S1 (perfect alignment, optimal weights)."""
    bias = np.load(ROOT/'readouts'/'mount-10-snr6-5hz-scores.npz')['bias']
    records = [r for r in load_records(ROOT/'geometry', ROOT/'sensor', -10, 1, 5) if r['split'] == 'audit']
    cache, z, y, m = {}, [], [], []
    for rec in records:
        u = rec['unit']
        if u not in cache:
            with np.load(ROOT/'sensor'/f'unit{u:02d}-mount-10-oracle.npz') as f:
                e = f['expected_H3'][1]
            with np.load(ROOT/'sensor'/f'unit{u:02d}-mount-10-observations.npz') as f:
                cache[u] = (e, f['config'])
        e, cfg = cache[u]
        re = e[np.flatnonzero(cfg == rec['config'])[::2]]-bias
        v = 16*rec['ambient'][..., None]+np.maximum(bias, 0)
        z.append(np.array([s1(re[i:i+1], v[i:i+1], query_weights(t), .75)[0] for i, t in enumerate(rec['tq'])]))
        y.append(rec['labels'])
        m.append(rec['main'])
    z, y, m = np.stack(z), np.stack(y), np.stack(m)
    out = {}
    for g, ix in GROUPS:
        ratios = []
        for t in range(3, 12):
            now, past = z[:, t][:, ix], np.clip(z[:, t-3:t][:, :, ix], 0, None)
            pos = (y[:, t][:, ix] == 1) & m[:, t][:, None] & (now > .5)
            ratios.append((np.sqrt(now**2+(past**2).sum(1))/now)[pos])
        r = np.concatenate(ratios)
        out[g] = dict(positives=len(r), median=float(np.median(r)), p25=float(np.percentile(r, 25)), p75=float(np.percentile(r, 75)))
    return out


if __name__ == '__main__':
    if sys.argv[1:] == ['offline']:
        result = dict(alignment=target_alignment(), ceiling=ideal_ceiling())
        (ROOT/'head-diagnosis').mkdir(exist_ok=True)
        (ROOT/'head-diagnosis'/'offline.json').write_text(json.dumps(result, indent=1)+'\n', encoding='utf-8')
        print(json.dumps(result))
        sys.exit()
    # Optional argv: "power,K" pairs; default is the six-variant grid.
    chosen = [tuple(int(x) for x in a.split(',')) for a in sys.argv[1:]] or VARIANTS
    with ProcessPoolExecutor(len(chosen)) as pool:
        results = list(pool.map(variant, chosen))
    out = ROOT/'head-diagnosis'
    out.mkdir(exist_ok=True)
    name = 'variants.json' if chosen == VARIANTS else 'variants-' + '_'.join(f'{p}k{k}' for p, k in chosen) + '.json'
    (out/name).write_text(json.dumps(results, indent=1)+'\n', encoding='utf-8')
    for r in results:
        print(json.dumps(dict(power=r['power'], K=r['K'], tau=r['tau'],
                              AP={g: round(x, 4) for g, x in r['audit_macro_AP'].items()},
                              ideal={g: (round(v['median_ideal_Z_S1'], 2), round(v['median_ideal_Z_S2'], 2),
                                         round(v['median_ratio'], 3), round(v['frac_S2_gt_S1'], 3)) for g, v in r['ideal'].items()})))
    sys.stdout.flush()
