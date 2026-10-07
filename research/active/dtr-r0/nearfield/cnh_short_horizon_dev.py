"""CPU-only, consumed BlindWays Development: short-time vs 1.5 m path prediction.

All predictions use a pelvis origin and causal past-1s pelvis speed. Future poses
only define targets, strata, censoring, and scoring; no future heading flip.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import cnh_blindways_heading as B

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts.local/work/cnh-short-horizon-dev-20261007'
HZ = 60
METHODS = ('head_E1', 'pelvis_E1', 'torso', 'head_EMA', 'pelvis_EMA')
TARGETS = ('0.5s', '1s', '2s', '1.5m')
METRICS = ('angle_deg', 'fde_m', 'ade_m', 'fde_per_m', 'cross_m',
           'path_miss_m', 'path_miss_bodywidth', 'true_path_covered', 'pred_path_supported')
RADIUS = .30


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())


def unit_heading(deg):
    rad = np.radians(deg)
    return np.stack((np.cos(rad), np.sin(rad)), -1)


def predict_inputs(x, anchors):
    """Return angles and common speed; invariant to x after max(anchors)."""
    h, p = x[:, 6, :2], x[:, 0, :2]
    shoulder = x[:, 11, :2] - x[:, 7, :2]
    angles = np.stack((B.yaw(h[anchors]-h[anchors-HZ]),
                       B.yaw(p[anchors]-p[anchors-HZ]),
                       B.yaw(np.stack((shoulder[anchors, 1], -shoulder[anchors, 0]), -1)),
                       B.ema(h)[anchors], B.ema(p)[anchors]), -1)
    speed = np.linalg.norm(p[anchors]-p[anchors-HZ], axis=-1)
    return angles, speed


def interpolate(p, indices):
    lo = np.floor(indices).astype(int)
    hi = np.minimum(lo+1, len(p)-1)
    a = indices-lo
    return p[lo]*(1-a)[..., None]+p[hi]*a[..., None]


def geometry_metrics(paths, ends, angles, lengths):
    """21 points on the true path vs predicted straight segment.

    Coverage is sampled CENTERLINE containment in a 0.30 m tube, in both
    directions; neither polygon area IoU nor obstacle recall/safety.
    """
    dirs = unit_heading(angles)
    frac = np.linspace(0, 1, paths.shape[1])
    pred_end = dirs*lengths[:, None, None]
    pred = pred_end[:, :, None, :]*frac[None, None, :, None]
    diff = pred-paths[:, None, :, :]
    fde = np.linalg.norm(pred_end-ends[:, None, :], axis=-1)
    ade = np.linalg.norm(diff, axis=-1).mean(-1)
    # Actual path to finite predicted segment.
    proj = np.einsum('nqd,nmd->nmq', paths, dirs)
    nearest = dirs[:, :, None, :]*np.clip(proj, 0, lengths[:, None, None])[..., None]
    d_true = np.linalg.norm(paths[:, None, :, :]-nearest, axis=-1)
    # Predicted sample points to the actual piecewise-linear centreline.
    a, v = paths[:, :-1], np.diff(paths, axis=1)
    delta = pred[:, :, :, None, :]-a[:, None, None, :, :]
    denom = (v*v).sum(-1)
    weights = np.clip((delta*v[:, None, None]).sum(-1)/np.maximum(denom[:, None, None], 1e-15), 0, 1)
    d_pred = np.linalg.norm(delta-weights[..., None]*v[:, None, None], axis=-1).min(-1)
    miss = np.maximum(d_true.max(-1), d_pred.max(-1))
    cross = np.abs(dirs[..., 0]*ends[:, None, 1]-dirs[..., 1]*ends[:, None, 0])
    return fde, ade, cross, miss, (d_true <= RADIUS).mean(-1), (d_pred <= RADIUS).mean(-1)


def analyse_clip(x, pid, clip_id):
    if x.ndim != 3 or x.shape[1:] != (24, 3) or not np.isfinite(x).all():
        return None, {'file_id': clip_id, 'invalid': True}
    p = x[:, 0, :2]
    t = np.arange(90, len(x)-90, 12)  # 5 Hz, 1.5s history/future for evaluation turn split
    angles, speed_causal = predict_inputs(x, t)
    speed_eval = np.linalg.norm(p[t+30]-p[t-30], axis=1)
    rate = B.wrap(B.yaw(p[t+90]-p[t+30])-B.yaw(p[t-30]-p[t-90]))/2
    shoulder_norm = np.linalg.norm(x[t, 11, :2]-x[t, 7, :2], axis=1)
    # Whole file break mask; each target must have no >1m native jump in its own support.
    jumps = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1) > 1)]
    base = ((speed_eval >= .3) & (speed_causal > 1e-6) & (shoulder_norm > 1e-6)
            & np.isfinite(angles).all(1) & (jumps[t] == 0))
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    future_index = np.searchsorted(arc, arc[t]+1.5)
    ends_idx, valid = [], []
    for h in (30, 60, 120):
        end = t+h
        ok = (end < len(x))
        end = np.minimum(end, len(x)-1).astype(float)
        ok &= jumps[end.astype(int)] == jumps[t]
        ends_idx.append(end)
        valid.append(base & ok)
    ok = future_index < len(x)
    hi = np.minimum(future_index, len(x)-1)
    lo = np.maximum(hi-1, 0)
    fraction = np.clip((arc[t]+1.5-arc[lo])/np.maximum(arc[hi]-arc[lo], 1e-15), 0, 1)
    distance_end = lo+fraction
    ok &= jumps[hi] == jumps[t]
    ends_idx.append(distance_end)
    valid.append(base & ok)
    valid = np.stack(valid, -1)
    values = np.full((len(t), len(TARGETS), len(METHODS), len(METRICS)), np.nan)
    elapsed = np.full((len(t), len(TARGETS)), np.nan)
    arc_m = elapsed.copy()
    net_m = elapsed.copy()
    for j, end in enumerate(ends_idx):
        ids = np.flatnonzero(valid[:, j])
        if not len(ids):
            continue
        duration = (end[ids]-t[ids])/HZ
        at = t[ids, None]+(end[ids]-t[ids])[:, None]*np.linspace(0, 1, 21)
        paths = interpolate(p, at)-p[t[ids], None]
        endpoint = paths[:, -1]
        arclen = np.interp(end[ids], np.arange(len(x)), arc)-arc[t[ids]]
        net = np.linalg.norm(endpoint, axis=1)
        length = np.full(len(ids), 1.5) if j == 3 else speed_causal[ids]*duration
        fde, ade, cross, miss, coverage, support = geometry_metrics(paths, endpoint, angles[ids], length)
        err = B.wrap(angles[ids]-B.yaw(endpoint)[:, None])
        err[net < .05] = np.nan  # keep those anchors for position/region metrics
        normalized = fde/np.maximum(arclen[:, None], .05)
        normalized[arclen < .05] = np.nan
        values[ids, j] = np.stack((err, fde, ade, normalized, cross, miss,
                                   miss/(2*RADIUS), coverage, support), -1)
        elapsed[ids, j], arc_m[ids, j], net_m[ids, j] = duration, arclen, net
    result = dict(pid=np.full(len(t), pid, np.int16), clip=np.full(len(t), clip_id, np.int16),
                  frame=t, base=base, valid=valid, values=values, elapsed=elapsed,
                  arc=arc_m, net=net_m, speed=speed_eval, speed_causal=speed_causal,
                  turn=np.abs(rate) >= 10)
    return result, dict(file_id=clip_id, pid=pid, invalid=False, frames=len(x),
                       jumps=int(jumps[-1]), anchors=len(t), base=int(base.sum()))


def rms(v):
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    return None if not len(v) else float(np.sqrt(np.mean(v*v)))


def average(v):
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    return None if not len(v) else float(np.mean(v))


def pct(v, q):
    v = np.asarray(v)
    v = v[np.isfinite(v)]
    return None if not len(v) else float(np.percentile(v, q))


def stats(values):
    return dict(n=len(values), angle_n=int(np.isfinite(values[:, 0]).sum()),
                angle_rms=rms(values[:, 0]), angle_p50=pct(np.abs(values[:, 0]), 50),
                angle_p90=pct(np.abs(values[:, 0]), 90),
                **{name: average(values[:, k]) for k, name in enumerate(METRICS) if k > 0})


def summarise(d):
    groups = {'all': np.ones(len(d['pid']), bool), 'slow': d['speed'] < .6,
              'mid': (d['speed'] >= .6) & (d['speed'] < .9), 'fast': d['speed'] >= .9,
              'straight': ~d['turn'], 'turning': d['turn'],
              'slow_straight': (d['speed'] < .6) & ~d['turn'],
              'slow_turning': (d['speed'] < .6) & d['turn']}
    cohorts = {'odd_development': d['pid'] % 2 == 1, 'even_validation': d['pid'] % 2 == 0}
    for pid in np.unique(d['pid']):
        cohorts[f'P{pid:02d}'] = d['pid'] == pid
    counts, tables = {}, {}
    common = d['valid'].all(1)
    angle_common = common & np.isfinite(d['values'][..., 0]).all((1, 2))
    for cname, cohort in cohorts.items():
        counts[cname], tables[cname] = {}, {}
        for group, gmask in groups.items():
            base = cohort & gmask & d['base']
            counts[cname][group] = dict(eligible=int(base.sum()), common=int((base & common).sum()),
                angle_common=int((base & angle_common).sum()),
                targets={h: dict(valid=int((base & d['valid'][:, j]).sum()),
                    censored=int((base & ~d['valid'][:, j]).sum()),
                    angle_invalid=int((base & d['valid'][:, j] & ~np.isfinite(d['values'][:, j, 0, 0])).sum()),
                    elapsed_p50=pct(d['elapsed'][base & d['valid'][:, j], j], 50),
                    elapsed_p90=pct(d['elapsed'][base & d['valid'][:, j], j], 90),
                    arc_net_p50=pct(d['arc'][base & d['valid'][:, j], j] /
                                    np.maximum(d['net'][base & d['valid'][:, j], j], .05), 50))
                         for j, h in enumerate(TARGETS)})
            tables[cname][group] = {}
            for pop in ('own', 'common'):
                table = {}
                for j, h in enumerate(TARGETS):
                    mask = base & (d['valid'][:, j] if pop == 'own' else angle_common)
                    table[h] = {method: stats(d['values'][mask, j, k]) for k, method in enumerate(METHODS)}
                    reference = table[h]['pelvis_E1']['fde_m']
                    for val in table[h].values():
                        val['fde_skill_vs_pelvis_E1'] = None if reference in (None, 0.) else 1-val['fde_m']/reference
                tables[cname][group][pop] = table
    # Participant-balanced aggregate is primary; pooled anchors remain separately visible.
    macro = {}
    for name, parity in (('odd_development', 1), ('even_validation', 0)):
        pids = [f'P{p:02d}' for p in np.unique(d['pid']) if p % 2 == parity]
        macro[name] = {group: {pop: {h: {m: {metric: average([
            tables[p][group][pop][h][m][metric] for p in pids
            if tables[p][group][pop][h][m][metric] is not None])
            for metric in tables[pids[0]][group][pop][h][m] if metric not in ('n','angle_n')}
            for m in METHODS} for h in TARGETS} for pop in ('own','common')} for group in groups}
    selected = {}
    for h in TARGETS:
        dev = macro['odd_development']['all']['common'][h]
        best = min(METHODS, key=lambda m: dev[m]['angle_rms'])
        selected[h] = dict(method=best, criterion='odd participant mean angular RMS, common angular-valid anchors',
                           odd=dev[best], even=macro['even_validation']['all']['common'][h][best])
    return dict(counts=counts, pooled_and_participant=tables, participant_macro=macro, selected=selected)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'PLAN.json').exists():
        raise FileExistsError('plan already frozen')
    files = sorted(B.SRC.glob('*.npy'))
    manifest = [{'name': p.name, 'sha256': sha(p)} for p in files]
    save(OUT/'inputs.json', manifest)
    save(OUT/'PLAN.json', dict(task='CNH_SHORT_HORIZON_DEV_20261007',
        authorization='User 试试吧 after CPU short-time vs fixed-distance proposal',
        lane='EXPLORE consumed real motion, no M3/GPU/obstacle relabeling',
        question='Are 0.5/1/2s pelvis paths predictably easier than 1.5m future direction after controlling cohort and scale?',
        budget='CPU analysis <=600s wall; no GPU, training, download or hardware',
        targets=list(TARGETS), methods=list(METHODS), inputs_sha256=sha(OUT/'inputs.json'),
        source_sha256=sha(__file__), helper_sha256=sha(B.__file__),
        sample_hz=5, history_s=1, evaluation_turn_support_s=1.5,
        prediction='current pelvis origin; same past-1s pelvis speed; fixed direction per forecast. Time endpoints speed*h. Distance endpoint 1.5*direction, no true arrival time as prediction input.',
        strata='evaluation-only centred1s pelvis speed: walking>=0.3, slow<0.6, mid<0.9, fast>=0.9; 2s centred direction-change rate>=10deg/s turning',
        censoring='each horizon own endpoints; common mask plus all-horizon net displacement>=0.05m for matched angular primary; excluded angle cases retained in position metrics',
        geometry='21 equally spaced time samples; symmetric sampled centreline Hausdorff and bidirectional coverage in r=0.30m tubes; not body area IoU or collision test',
        selection='odd participant mean angular RMS selects method per target on common cohort; fixed EMA, no fit. Even participants consumed-development validation, no new unseen claim.',
        evidence='report per participant/macro/pooled and every source, horizon, stratum; no independent-frame CI. Easier only if angle/normalized error improve, not merely fewer metres.',
        delivery='result.json, per-anchor ledger, summary/report, current+run log, focused causal/geometry checks',
        start_unix=time.time()))
    print(f'FROZEN {len(files)} input files, CPU-only 600s analysis budget', flush=True)


def run():
    started = time.perf_counter()
    plan = json.loads((OUT/'PLAN.json').read_text())
    if (OUT/'result.json').exists() or (OUT/'ledger.npz').exists():
        raise FileExistsError('completed payload exists; no overwrite')
    assert sha(__file__) == plan['source_sha256']
    assert sha(B.__file__) == plan['helper_sha256']
    assert sha(OUT/'inputs.json') == plan['inputs_sha256']
    manifest = json.loads((OUT/'inputs.json').read_text())
    chunks, receipts = [], []
    for i, entry in enumerate(manifest):
        if time.perf_counter()-started > 600:
            raise TimeoutError('600s CPU analysis wall budget reached')
        f = B.SRC/entry['name']
        assert sha(f) == entry['sha256'], f.name
        data, receipt = analyse_clip(np.load(f), int(f.stem[1:3]), i)
        receipts.append(receipt)
        if data is not None:
            chunks.append(data)
        if (i+1) % 100 == 0:
            print(f'{i+1}/{len(manifest)} clips {time.perf_counter()-started:.1f}s', flush=True)
    d = {key: np.concatenate([c[key] for c in chunks]) for key in chunks[0]}
    np.savez_compressed(OUT/'ledger.npz', **d)
    result = summarise(d)
    result.update(methods=METHODS, targets=TARGETS, metrics=METRICS, files=len(manifest),
                  file_receipts=receipts, wall_s=time.perf_counter()-started, plan_sha256=sha(OUT/'PLAN.json'))
    save(OUT/'result.json', result)
    print_summary(result)
    print(f'COMPLETE {result["wall_s"]:.1f}s', flush=True)


def print_summary(result):
    lines = ['EVEN validation: participant-macro on common angular-valid anchors',
             'target method angle_RMS angle_p90 FDE_m FDE/arc path_miss_m coverage support']
    for h in TARGETS:
        for m in METHODS:
            r = result['participant_macro']['even_validation']['all']['common'][h][m]
            lines.append(f'{h:5} {m:11} '+ ' '.join(f'{r[k]:.3f}' for k in
                         ('angle_rms','angle_p90','fde_m','fde_per_m','path_miss_m','true_path_covered','pred_path_supported')))
    lines.append('SELECTED ON ODD: '+str({h:x['method'] for h,x in result['selected'].items()}))
    text='\n'.join(lines)+'\n'
    (OUT/'summary.txt').write_bytes(text.encode())
    print(text, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze','run'))
    args = parser.parse_args()
    (freeze if args.action == 'freeze' else run)()
