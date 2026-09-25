"""Track A scale-up readouts (formal lane). Frozen by CNH_TRACK_A_SCALE_PROTOCOL_20260926.md.

Stages: bias (calib only) -> per-unit scores (parallel) -> analysis (calib selects
tau and thresholds; audit reports; paired unit bootstrap + Holm on the six primary
tests; alert-level and stratum metrics descriptive). Readouts see only H3, public
ambient, public T_Q_tof and GT / assumed-noisy ego-motion; labels, witnesses and
oracle visibility are opened only for evaluation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from sklearn.metrics import average_precision_score
from cnh_track_a_readout import query_weights, noisy_poses
from cnh_track_a_v12_readout import accumulate, s1, scan_transported, memory_scan
from cnh_track_a_v13_evaluate import metric, calib_threshold, observability, GROUPS, TAUS
from cnh_track_a_v13_head_diag import s2_sequence
from cnh_track_a_v13_sensor import seed_for
import cnh_track_a_v13_sensor as sensor_module

PRIMARY_TESTS = (('S2/noisy', 'B1-R/noisy'), ('S1', 'B0'), ('S3/noisy', 'S2/noisy'))
B, BOOT_SEED, ALPHA = 10000, 20260926, .05


def unit_records(geometry, sensor, unit, mount, snr_index):
    data = json.loads((geometry/f'unit{unit:02d}'/f'unit{unit:02d}.json').read_text(encoding='utf-8-sig'))
    with np.load(sensor/f'unit{unit:02d}-mount{mount}-observations.npz') as f:
        obs = {k: f[k] for k in ('hist', 'ambient', 'world_from_tof', 'T_Q_tof', 'config', 'frame', 'rate')}
    step = 2 if int(obs['rate']) == 10 else 1
    records = []
    for c in data['configs']:
        sel = np.flatnonzero(obs['config'] == c['config'])[::step]
        records.append(dict(unit=unit, config=c['config'], split=data['split'],
                            hist=obs['hist'][snr_index, sel].astype(float), ambient=obs['ambient'][snr_index, sel].astype(float),
                            poses=obs['world_from_tof'][sel], tq=obs['T_Q_tof'][sel],
                            labels=np.asarray(c['labels']), main=np.asarray(c['main'], bool),
                            boundary=np.asarray(c['boundary'], bool), contributors=c['contributors'],
                            witness=np.array([[np.nan if z is None else z for z in row] for row in c['witness_z']]),
                            size_classes=c['size_classes'],
                            world_from_Q=np.asarray(c['world_from_Q_10hz'])[::2] if 'world_from_Q_10hz' in c else np.asarray(c['world_from_Q']),
                            ego_seed=seed_for(unit, c['config'], 'ego', rate=5)))
    return data['split'], records, step


def s1_cell(r, v, w, tau):
    """Per-cell z test inside the query support (single-window ablation of S1)."""
    z = (r/np.sqrt(np.maximum(v, 1e-9))).reshape(64, 16)
    out = np.full(6, -50.)
    for q in range(6):
        support = np.asarray(w).reshape(6, 64, 16)[q] >= tau
        if support.any():
            out[q] = max(-50., float(z[support].max()))
    return out


def strata_of(records):
    out = []
    for rec in records:
        for frame in rec['contributors']:
            row = []
            for ids in frame:
                classes = {rec['size_classes'].get(str(i), 'background') for i in ids}
                row.append('none' if not ids else 'wide' if 'wide' in classes else
                           'tiny' if classes == {'tiny'} else 'background' if classes == {'background'} else 'realistic')
            out.append(row)
    return np.asarray(out)


def score_unit(job):
    geometry, sensor, output, unit, mount, snr_index, family, with_r4 = job
    sensor_module.FAMILY = family
    target = Path(output)/f'unit{unit:02d}.npz'
    if target.exists():
        return unit
    bias = np.load(Path(output)/'bias.npy')
    split, records, step = unit_records(Path(geometry), Path(sensor), unit, mount, snr_index)
    scores = {}
    for rec in records:
        r = rec['hist']-bias
        v = 16*rec['ambient'][..., None]+np.maximum(bias, 0)
        weights = [query_weights(t) for t in rec['tq']]
        out = {'B0': np.array([np.einsum('zb,qzb->q', y.reshape(64, 16), w) for y, w in zip(rec['hist'], weights)])}
        for tau in TAUS:
            out[f'S1|{tau}'] = np.array([s1(r[i:i+1], v[i:i+1], w, tau)[0] for i, w in enumerate(weights)])
            out[f'S1cell|{tau}'] = np.array([s1_cell(r[i], v[i], w, tau) for i, w in enumerate(weights)])
        for motion, p in (('GT', rec['poses']), ('noisy', noisy_poses(rec['poses'], rec['ego_seed'], dt=.2))):
            acc = accumulate(r, v, p, 4)
            out[f'B1-R/{motion}'] = np.array([np.einsum('zb,qzb->q', m.reshape(64, 16), w) for m, w in zip(acc['mean'], weights)])
            for tau in TAUS:
                out[f'S2/{motion}|{tau}'] = np.array([s1(r[i:i+1], v[i:i+1], w, tau)[0] if i == 0 else
                                                      scan_transported(r, v, acc['sources'][i], w, tau) for i, w in enumerate(weights)])
            out[f'memory/{motion}'] = np.array([memory_scan(r, v, p, i, rec['tq'][i]) for i in range(len(r))])
            if with_r4 and motion == 'noisy':
                for tau in TAUS:
                    out[f'S2r4/noisy|{tau}'] = s2_sequence(r, v, p, weights, tau, -2, 4)
        for key, value in out.items():
            scores.setdefault(key, []).append(value)
    arrays = {k.replace('/', '__').replace('|', '@'): np.concatenate(v) for k, v in scores.items()}
    extra = dict(labels=np.concatenate([r['labels'] for r in records]), main=np.concatenate([r['main'] for r in records]),
                 boundary=np.concatenate([r['boundary'] for r in records]),
                 witness=np.concatenate([r['witness'] for r in records]), strata=strata_of(records),
                 config=np.repeat([r['config'] for r in records], 12), frame=np.tile(np.arange(12), len(records)),
                 split=np.array(split))
    if split == 'audit':
        extra['visibility'] = observability(Path(sensor), records, mount, 5, step=step)
    np.savez_compressed(target, **arrays, **extra)
    return unit


def per_unit_ap(y, s, units, mask):
    out = {}
    for u in sorted(set(units[mask].tolist())):
        m = mask & (units == u)
        yy, ss = y[m].ravel(), s[m].ravel()
        if yy.any() and (~yy.astype(bool)).any():
            out[u] = float(average_precision_score(yy, ss))
    return out


def events(y, s, witness, main_any, seq, threshold, ix):
    """Alert-level metrics over (sequence, query) pairs using frames t>=3 (history >= 4)."""
    rows = dict(events=0, alerted=0, lead=[], false_pairs=0, nonevent_pairs=0)
    for sid in np.unique(seq[main_any]):
        m = (seq == sid) & main_any
        yy, ss, ww = y[m][:, ix], s[m][:, ix], witness[m][:, ix]
        for q in range(len(ix)):
            pos = np.flatnonzero(yy[:, q] == 1)
            if len(pos):
                rows['events'] += 1
                hits = pos[ss[pos, q] >= threshold]
                if len(hits):
                    rows['alerted'] += 1
                    lead = ww[hits[0], q]
                    if np.isfinite(lead):
                        rows['lead'].append(float(lead))
            else:
                rows['nonevent_pairs'] += 1
                rows['false_pairs'] += int((ss[:, q] >= threshold).any())
    lead = rows.pop('lead')
    return dict(rows, event_recall=rows['alerted']/max(1, rows['events']),
                missed=rows['events']-rows['alerted'],
                median_first_alert_distance_m=float(np.median(lead)) if lead else None,
                false_alert_pair_rate=rows['false_pairs']/max(1, rows['nonevent_pairs']))


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted, running = np.empty(len(pvalues)), 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(pvalues)-rank)*pvalues[i]))
        adjusted[i] = running
    return adjusted


def analyze(output, n_units, splits, primary):
    output = Path(output)
    data = {}
    for u in range(n_units):
        path = output/f'unit{u:02d}.npz'
        if path.exists():
            with np.load(path) as f:
                data[u] = {k: f[k] for k in f.files}
    units_all = sorted(data)
    keys = [k for k in data[units_all[0]] if '__' in k or k in ('B0',) or k.startswith('S1@') or k.startswith('S1cell@')]
    cat = lambda k: np.concatenate([data[u][k] for u in units_all])
    y, main, witness, strata = cat('labels'), cat('main'), cat('witness'), cat('strata')
    unit = np.concatenate([np.full(len(data[u]['labels']), u) for u in units_all])
    split = np.concatenate([np.full(len(data[u]['labels']), str(data[u]['split'])) for u in units_all])
    seq = unit*100+cat('config') % 32
    frame = cat('frame')
    raw = {k.replace('__', '/').replace('@', '|'): cat(k) for k in keys}
    calib, audit = (split == 'calib') & main, (split == 'audit') & main
    arms, selection = {'B0': raw['B0']}, {}
    families = [('S1', 'S1|{}'), ('S2/GT', 'S2/GT|{}'), ('S2/noisy', 'S2/noisy|{}')]
    if 'S1cell|0.25' in raw:
        families.append(('S1cell', 'S1cell|{}'))
    if 'S2r4/noisy|0.25' in raw:
        families.append(('S2r4/noisy', 'S2r4/noisy|{}'))
    for name, fmt in families:
        cand = {}
        for tau in TAUS:
            vals = [np.mean(list(per_unit_ap(y[:, ix], raw[fmt.format(tau)][:, ix], unit, calib).values())) for _, ix in GROUPS]
            cand[tau] = float(np.mean(vals))
        tau = max(TAUS, key=lambda t: (cand[t], -t))
        arms[name] = raw[fmt.format(tau)]
        selection[name] = dict(tau=tau, calib_macro_AP=cand)
    for m in ('GT', 'noisy'):
        arms[f'B1-R/{m}'] = raw[f'B1-R/{m}']
        arms[f'S3/{m}'] = np.maximum(arms[f'S2/{m}'], raw[f'memory/{m}'])
    report = dict(n_units=len(units_all), splits=splits, selection=selection, arms={}, denominators={})
    audit_units = sorted(set(unit[split == 'audit'].tolist()))
    report['denominators'] = dict(audit_units=len(audit_units), calib_units=len(set(unit[split == 'calib'].tolist())),
                                  audit_main_frames=int(audit.sum()),
                                  audit_positive_queries={g: int((y[audit][:, ix] == 1).sum()) for g, ix in GROUPS},
                                  audit_negative_queries={g: int((y[audit][:, ix] == 0).sum()) for g, ix in GROUPS})
    unit_ap = {}
    for arm, s in arms.items():
        row = {}
        for g, ix in GROUPS:
            thr = calib_threshold(y[calib][:, ix], s[calib][:, ix])
            res = metric(y[audit][:, ix], s[audit][:, ix], thr)
            ap_units = per_unit_ap(y[:, ix], s[:, ix], unit, audit)
            unit_ap[(arm, g)] = ap_units
            res['macro_AP'] = float(np.mean(list(ap_units.values())))
            res['units_evaluable'] = len(ap_units)
            res['threshold'] = thr
            sa, ya, st = s[audit][:, ix], y[audit][:, ix], strata[audit][:, ix]
            res['strata_AP'] = {}
            for label in ('tiny', 'realistic', 'wide'):
                sel = (st == label) | (ya == 0)
                pos = int((sel & (ya == 1)).sum())
                res['strata_AP'][label] = dict(positive=pos, AP=float(average_precision_score(ya[sel], sa[sel])) if pos else None)
            is_audit = split == 'audit'
            ev_mask = is_audit & (frame >= 3)
            res['events'] = events(y, s, witness, ev_mask, seq, thr, ix)
            row[g] = res
        report['arms'][arm] = row
    if 'visibility' in data[audit_units[0]]:
        vis = np.concatenate([data[u]['visibility'] for u in audit_units])
        am, ya = main[split == 'audit'], y[split == 'audit']
        report['visibility_strata'] = {}
        for arm in ('B0', 'S1', 'S2/noisy', 'S3/noisy'):
            sa = arms[arm][split == 'audit']
            report['visibility_strata'][arm] = {}
            for g, ix in GROUPS:
                va, yy, ss = vis[:, ix], ya[:, ix], sa[:, ix]
                out = {}
                for label, mask in (('invisible', (yy == 1) & (va == 0)), ('visible', (yy == 1) & (va > 0))):
                    sel = am[:, None] & (mask | (yy == 0))
                    pos = int((sel & (yy == 1)).sum())
                    out[label] = dict(positive=pos, AP=float(average_precision_score(yy[sel], ss[sel])) if pos else None)
                report['visibility_strata'][arm][g] = out
    if primary:
        tests, pvalues = [], []
        for a, b in PRIMARY_TESTS:
            for g, _ in GROUPS:
                ua, ub = unit_ap[(a, g)], unit_ap[(b, g)]
                common = [u for u in audit_units if u in ua and u in ub]
                diff = np.array([ua[u]-ub[u] for u in common])
                # Same seed per test: identical resamples whenever the evaluable unit sets match.
                draws = np.random.default_rng(BOOT_SEED).integers(0, len(common), size=(B, len(common)))
                boot = diff[draws].mean(1)
                p = (1+np.sum(boot <= 0))/(B+1)
                tests.append(dict(comparison=f'{a} - {b}', group=g, units=len(common), delta=float(diff.mean()),
                                  ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                                  p_one_sided=float(p), units_positive=int((diff > 0).sum())))
                pvalues.append(p)
        adj = holm(np.array(pvalues))
        for t, pa in zip(tests, adj):
            t['p_holm'] = float(pa)
            t['holm_reject_at_0.05'] = bool(pa < ALPHA)
        report['primary_tests'] = tests
        frozen = [t for t in tests if t['comparison'] == 'S2/noisy - B1-R/noisy']
        report['v12_frozen_criterion'] = dict(pass_gate=all(t['ci95'][0] > 0 for t in frozen),
                                              rule='S2-B1-R HEAD and BODY paired unit bootstrap 95% lower bound > 0')
    (output/'result.json').write_text(json.dumps(report, indent=1, allow_nan=False)+'\n', encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--sensor', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mount', type=int, required=True)
    p.add_argument('--snr', type=int, choices=(3, 6, 12), required=True)
    p.add_argument('--family', required=True)
    p.add_argument('--split-counts', type=int, nargs=3, required=True)
    p.add_argument('--primary', action='store_true')
    p.add_argument('--workers', type=int, default=16)
    a = p.parse_args()
    t0 = time.monotonic()
    ntr, nca, nau = a.split_counts
    n = ntr+nca+nau
    snr_index = (3, 6, 12).index(a.snr)
    a.output.mkdir(parents=True, exist_ok=True)
    if not (a.output/'bias.npy').exists():
        calib = []
        for u in range(ntr, ntr+nca):
            if not (a.geometry/f'unit{u:02d}'/f'unit{u:02d}.json').exists():
                continue
            _, recs, _ = unit_records(a.geometry, a.sensor, u, a.mount, snr_index)
            calib.extend(r['hist'] for r in recs)
        np.save(a.output/'bias.npy', np.median(np.concatenate(calib), axis=0))
    jobs = [(str(a.geometry), str(a.sensor), str(a.output), u, a.mount, snr_index, a.family, a.primary)
            for u in range(ntr, n) if (a.geometry/f'unit{u:02d}'/f'unit{u:02d}.json').exists()]
    with ProcessPoolExecutor(a.workers) as pool:
        for u in pool.map(score_unit, jobs):
            pass
    report = analyze(a.output, n, dict(train=ntr, calib=nca, audit=nau), a.primary)
    report['wall_s'] = time.monotonic()-t0
    (a.output/'result.json').write_text(json.dumps(report, indent=1, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({arm: {g: round(v[g]['macro_AP'], 4) for g in ('HEAD', 'BODY')} for arm, v in report['arms'].items()}))
    if a.primary:
        for t in report['primary_tests']:
            print(json.dumps(t))


if __name__ == '__main__':
    main()
