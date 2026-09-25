"""Track A v1.3 pilot readouts (fast lane, descriptive; 2 calib / 4 audit units).

Readouts see only H3, public ambient, public T_Q_tof and (GT or assumed-noisy)
ego-motion. calib fits bias, tau and alarm thresholds; audit is never used to
choose anything. Oracle rays are opened only for offline observability strata.
Equal history duration: 5 Hz K=4 and 10 Hz K=7 (0.6 s); S3 (k=8) at 5 Hz only.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from cnh_route_sensor import angular_rays
from cnh_track_a_generate import BOXES
from cnh_track_a_readout import query_weights, noisy_poses
from cnh_track_a_v12_readout import accumulate, s1, scan_transported, memory_scan
from cnh_track_a_v13_sensor import seed_for

TAUS = (.25, .5, .75)
GROUPS = (('HEAD', [0, 2, 4]), ('BODY', [1, 3, 5]))
CALIB, AUDIT = (6, 7), (8, 9, 10, 11)


def metric(y, s, threshold):
    y, s = np.asarray(y).ravel().astype(bool), np.asarray(s).ravel()
    pos, neg = int(y.sum()), int((~y).sum())
    pred = s >= threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    return dict(positive=pos, negative=neg, AP=float(average_precision_score(y, s)) if pos and neg else None,
                AUROC=float(roc_auc_score(y, s)) if pos and neg else None, TP=tp, FP=fp,
                recall=tp/pos if pos else None, FPR=fp/neg if neg else None)


def calib_threshold(y, s):
    """Highest recall with calib FPR <= 5%; ties -> higher threshold."""
    y, s = np.asarray(y).ravel().astype(bool), np.asarray(s).ravel()
    order = np.argsort(-s, kind='stable')
    values, truth = s[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(values)), len(values)-1]
    tp, fp = np.cumsum(truth)[ends], np.cumsum(~truth)[ends]
    ok = fp/max(1, (~y).sum()) <= .05
    if not ok.any():
        return float(np.nextafter(s.max(), np.inf))
    best = tp[ok].max()
    return float(values[ends[np.flatnonzero(ok & (tp == best))[0]]])


def macro_ap(y, s, units, mask):
    values = []
    for u in sorted(set(units[mask].tolist())):
        m = mask & (units == u)
        if y[m].any() and (~y[m].astype(bool)).any():
            values.append(float(average_precision_score(y[m].ravel(), s[m].ravel())))
    return float(np.mean(values)) if values else None


def load_records(geometry, sensor, mount, snr_index, rate):
    records = []
    for unit in CALIB + AUDIT:
        data = json.loads((geometry/f'unit{unit:02d}'/f'unit{unit:02d}.json').read_text(encoding='utf-8-sig'))
        with np.load(sensor/f'unit{unit:02d}-mount{mount}-observations.npz') as f:
            obs = {k: f[k] for k in ('hist', 'ambient', 'world_from_tof', 'T_Q_tof', 'config', 'frame')}
        step = 2 if rate == 5 else 1
        for c in data['configs']:
            sel = np.flatnonzero(obs['config'] == c['config'])[::step]
            suffix = '' if rate == 5 else '_10hz'
            records.append(dict(unit=unit, config=c['config'], split=data['split'],
                                hist=obs['hist'][snr_index, sel].astype(float), ambient=obs['ambient'][snr_index, sel].astype(float),
                                poses=obs['world_from_tof'][sel], tq=obs['T_Q_tof'][sel], frames=obs['frame'][sel],
                                labels=np.asarray(c['labels'+suffix]), main=np.asarray(c['main'+suffix], bool),
                                boundary=np.asarray(c['boundary'+suffix], bool),
                                contributors=c['contributors'+suffix] if suffix else c['contributors'],
                                size_classes=c['size_classes'],
                                world_from_Q=np.asarray(c['world_from_Q_10hz'])[obs['frame'][sel]],
                                ego_seed=seed_for(unit, c['config'], 'ego', rate=rate)))
    return records


def sequence_scores(rec, bias, k, with_memory):
    r = rec['hist']-bias
    v = 16*rec['ambient'][..., None]+np.maximum(bias, 0)
    weights = [query_weights(t) for t in rec['tq']]
    out = {'B0': np.array([np.einsum('zb,qzb->q', y.reshape(64, 16), w) for y, w in zip(rec['hist'], weights)])}
    for tau in TAUS:
        out[f'S1/{tau}'] = np.array([s1(r[i:i+1], v[i:i+1], w, tau)[0] for i, w in enumerate(weights)])
    dt = .2 if k == 4 else .1
    for motion, p in (('GT', rec['poses']), ('noisy', noisy_poses(rec['poses'], rec['ego_seed'], dt=dt))):
        acc = accumulate(r, v, p, k)
        out[f'B1-R/{motion}'] = np.array([np.einsum('zb,qzb->q', m.reshape(64, 16), w) for m, w in zip(acc['mean'], weights)])
        for tau in TAUS:
            out[f'S2/{motion}/{tau}'] = np.array([s1(r[i:i+1], v[i:i+1], w, tau)[0] if i == 0 else
                                                  scan_transported(r, v, acc['sources'][i], w, tau) for i, w in enumerate(weights)])
        if with_memory:
            out[f'memory/{motion}'] = np.array([memory_scan(r, v, p, i, rec['tq'][i]) for i in range(len(r))])
    return out


def positive_strata(records):
    """Per (frame, query): size stratum of label-causing objects (offline only)."""
    strata = []
    for rec in records:
        for frame in rec['contributors']:
            row = []
            for ids in frame:
                classes = {rec['size_classes'].get(str(i), 'background') for i in ids}
                row.append('none' if not ids else 'wide' if 'wide' in classes else
                           'tiny' if classes == {'tiny'} else 'background' if classes == {'background'} else 'realistic')
            strata.append(row)
    return np.asarray(strata)


def observability(oracle_dir, records, mount, rate):
    """Offline ray-sampled visibility: solid-angle fraction of ToF rays whose first hit
    is a label-causing object inside the query box; plus expected target counts."""
    dirs, w = angular_rays(16)
    w = w/w.sum()
    out = []
    cache = {}
    for rec in records:
        if rec['split'] != 'audit':
            continue
        if rec['unit'] not in cache:
            with np.load(oracle_dir/f"unit{rec['unit']:02d}-mount{mount}-oracle.npz") as f:
                cache[rec['unit']] = {k: f[k] for k in ('raydistance', 'object_id')}
            with np.load(oracle_dir/f"unit{rec['unit']:02d}-mount{mount}-observations.npz") as f:
                cache[rec['unit']]['config'] = f['config']
        o = cache[rec['unit']]
        rows = np.flatnonzero(o['config'] == rec['config'])[::(2 if rate == 5 else 1)]
        vis = np.zeros((len(rows), 6))
        for i, (row, pose, wq) in enumerate(zip(rows, rec['poses'], rec['world_from_Q'])):
            dist, oid = o['raydistance'][row], o['object_id'][row]
            points = pose[:3, 3]+(dirs@pose[:3, :3].T)*dist[..., None]
            q = (points-wq[:3, 3])@wq[:3, :3]
            for qi, (lo, hi) in enumerate(BOXES):
                ids = rec['contributors'][i][qi]
                if ids:
                    hit = np.isin(oid, ids) & (q >= lo).all(-1) & (q <= hi).all(-1)
                    vis[i, qi] = float(w[hit].sum())
        out.append(vis)
    return np.concatenate(out)


def evaluate(geometry, sensor, mount, snr_index, rate, output):
    t0 = time.monotonic()
    k = 4 if rate == 5 else 7
    records = load_records(geometry, sensor, mount, snr_index, rate)
    bias = np.median(np.concatenate([r['hist'] for r in records if r['split'] == 'calib']), axis=0)
    raw = {}
    for rec in records:
        for key, value in sequence_scores(rec, bias, k, rate == 5).items():
            raw.setdefault(key, []).append(value)
    raw = {key: np.concatenate(v) for key, v in raw.items()}
    y = np.concatenate([r['labels'] for r in records])
    units = np.concatenate([np.full(len(r['labels']), r['unit']) for r in records])
    split = np.concatenate([np.full(len(r['labels']), r['split']) for r in records])
    main = np.concatenate([r['main'] for r in records])
    boundary = np.concatenate([r['boundary'] for r in records])
    calib, audit = (split == 'calib') & main, (split == 'audit') & main
    arms, selection = {'B0': raw['B0']}, {}
    for name, keyfmt in [('S1', 'S1/{}')] + [(f'S2/{m}', f'S2/{m}/{{}}') for m in ('GT', 'noisy')]:
        cand = {tau: np.mean([macro_ap(y[:, ix], raw[keyfmt.format(tau)][:, ix], units, calib) for _, ix in GROUPS]) for tau in TAUS}
        tau = max(TAUS, key=lambda t: (cand[t], -t))
        arms[name] = raw[keyfmt.format(tau)]
        selection[name] = dict(tau=tau, calib_macro_AP=cand)
    for m in ('GT', 'noisy'):
        arms[f'B1-R/{m}'] = raw[f'B1-R/{m}']
        if rate == 5:
            arms[f'S3/{m}'] = np.maximum(arms[f'S2/{m}'], raw[f'memory/{m}'])
    strata = positive_strata(records)
    audit_strata = strata[split == 'audit']
    vis = observability(sensor, records, mount, rate)
    report = dict(mount=mount, snr=(3, 6, 12)[snr_index], rate=rate, K=k, selection=selection, arms={},
                  denominators=dict(audit_main_frames=int(audit.sum()), calib_main_frames=int(calib.sum())))
    for arm, s in arms.items():
        row = {}
        for group, ix in GROUPS:
            thr = calib_threshold(y[calib][:, ix], s[calib][:, ix])
            res = metric(y[audit][:, ix], s[audit][:, ix], thr)
            res['macro_AP'] = macro_ap(y[:, ix], s[:, ix], units, audit)
            res['per_query_AP'] = [metric(y[audit][:, q], s[audit][:, q], thr)['AP'] for q in ix]
            am, st, ya, sa = audit[split == 'audit'], audit_strata[:, ix], y[split == 'audit'][:, ix], s[split == 'audit'][:, ix]
            va = vis[:, ix]
            res['strata_AP'] = {}
            for label, mask in [('tiny', st == 'tiny'), ('realistic', st == 'realistic'), ('wide', st == 'wide'),
                                ('invisible', (ya == 1) & (va == 0)), ('visible', (ya == 1) & (va > 0))]:
                sel = am[:, None] & (mask | (ya == 0))
                pos = int((sel & (ya == 1)).sum())
                res['strata_AP'][label] = dict(positive=pos, AP=float(average_precision_score(ya[sel], sa[sel])) if pos else None)
            row[group] = res
        report['arms'][arm] = row
    am = audit[split == 'audit']
    ya, va = y[split == 'audit'], vis
    report['observability'] = {}
    for g, ix in GROUPS:
        pos = (am[:, None] & (ya == 1))[:, ix]
        report['observability'][g] = dict(positives=int(pos.sum()), invisible=int((pos & (va[:, ix] == 0)).sum()),
                                          visible_solid_angle_median=float(np.median(va[:, ix][pos])) if pos.any() else None)
    report['wall_s'] = time.monotonic()-t0
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = f'mount{mount}-snr{(3, 6, 12)[snr_index]}-{rate}hz'
    np.savez_compressed(output/f'{stem}-scores.npz', labels=y, units=units, split=split, main=main, boundary=boundary,
                        strata=strata, visibility_audit=vis, bias=bias, **{k.replace('/', '_'): v for k, v in arms.items()})
    (output/f'{stem}.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--sensor', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mount', type=int, required=True)
    p.add_argument('--snr', type=int, choices=(3, 6, 12), required=True)
    p.add_argument('--rate', type=int, choices=(5, 10), required=True)
    a = p.parse_args()
    r = evaluate(a.geometry, a.sensor, a.mount, (3, 6, 12).index(a.snr), a.rate, a.output)
    print(json.dumps({arm: {g: round(v[g]['macro_AP'] or -1, 4) for g in ('HEAD', 'BODY')} for arm, v in r['arms'].items()}))
