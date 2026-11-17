"""Direction A experiment 2 (DIRECTION_A2_OUTDOOR_SCALE_FALLBACK_20261117.md): outdoor scale fallback from the model's own
ground, evaluated with protocol v1.1 paired comparisons.
Usage: run_a2.py <out dir> [--smoke N]
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import adapters
import core
from methods import A2_METHODS, Methods
from run_a1 import FOV_HALF, REGIONS

SCALED = ('tof_unidepth_out', 'floor_unidepth', 'hybrid_out', 'hybrid_out_hm10', 'hybrid_out_hm5', 'hybrid_out_hp5', 'hybrid_out_hp10')
DATASETS = ('sanpo_head_seg', 'aria_gen2_walks')


def known_eye_heights(gen):
    """Evaluator side: per-sequence median camera height above the reference ground (separate RNG, so the evaluation
    truth below uses the same random stream as earlier runs)."""
    rng = np.random.default_rng(1); hs = {}
    for f in gen():
        z = f.ref_depth.reshape(-1); ok = np.isfinite(z) & (z > .3) & (z < 10)
        pts = core.rays(f.K, *f.ref_depth.shape)[ok] * z[ok, None]
        gp = (core.ground_plane(pts, f.down, rng) if f.down is not None
              else core.ground_plane(pts, np.array([0, 1., 0]), rng, max_deg=35.))
        if gp is not None:
            hs.setdefault(f.seq, []).append(gp[1])
    return {k: float(np.median(v)) for k, v in hs.items()}


def evaluate(name, M, rng, out, smoke=0, methods=A2_METHODS, scaled=SCALED):
    gen0 = adapters.ADAPTERS[name]
    eye = known_eye_heights(gen0)

    def gen():
        for i, f in enumerate(gen0()):
            if smoke and i >= smoke:
                return
            f.eye_height = eye.get(f.seq)
            yield f
    n_all = sum(1 for _ in gen()); k = max(1, n_all // 20)
    sc = core.self_check([f for i, f in enumerate(gen()) if i % k == 0][:20], rng)
    entry = dict(self_check=sc, eye_heights=eye)
    print(name, 'self-check pass', sc['pass'], 'eye heights median', float(np.median(list(eye.values()))), flush=True)
    if not sc['pass']:
        entry['status'] = 'STOPPED_AT_SELF_CHECK'; return entry, []
    rows, t0 = [], time.time()
    for f in gen():
        reg = core.regions(f, rng)
        d = core.rays(f.K, *f.ref_depth.shape)
        fov = (np.abs(np.arctan(d[:, 0])) <= FOV_HALF) & (np.abs(np.arctan(d[:, 1])) <= FOV_HALF)
        if 'near_nonground' in reg:
            reg['nng_in_fov'] = reg['near_nonground'] & fov; reg['nng_out_fov'] = reg['near_nonground'] & ~fov
        row = dict(seq=f.seq, t=f.t, block=f.block, evaluable='axes' in reg, truth={b: reg.get(f'truth_{b}') for b in core.BANDS},
                   truth_opened={b: reg.get(f'truth_opened_{b}') for b in core.BANDS},
                   truth_dist={b: reg.get(f'truth_dist_{b}') for b in core.BANDS}, depth={}, alert={}, source={})
        for m in methods:
            pred = getattr(M, m)(f)
            row['depth'][m] = {r: core.depth_metrics(pred, f.ref_depth, reg[r]) for r in REGIONS + ['nng_in_fov', 'nng_out_fov'] if r in reg}
            row['alert'][m] = core.band_alert(pred, f, reg)
            if m in scaled:
                row['source'][m] = M.source[m]
        rows.append(row)
    entry.update(frames=len(rows), seconds=round(time.time() - t0),
                 sources={m: dict(Counter(r['source'][m] for r in rows)) for m in scaled if m in methods})
    (out / f'rows_{name}.json').write_text(json.dumps(rows, default=float), encoding='utf-8')
    print(name, 'done', len(rows), round(time.time() - t0), 's', flush=True)
    return entry, rows


def judge(c):
    """Rule of DIRECTION_A2: tolerances in frames on the paired comparison c = core.compare(rows, a, b)."""
    a, b = c['methods']
    if not all(k in c['depth'] for k in ('corridor_LOW', 'corridor_BODY')):
        return dict(verdict='NOT_EVALUABLE', reason='corridor depth missing')
    prim = {m: (c['depth']['corridor_LOW']['absrel'][m] + c['depth']['corridor_BODY']['absrel'][m]) / 2 for m in (a, b)}
    rel = prim[b] / prim[a] - 1
    checks = {}
    for band in ('LOW', 'BODY'):
        d = c['detect'][band]
        rec_tol = max(3, .02 * d['positives']); fa_tol = .03 * d['negatives']
        checks[band] = dict(recall_change_frames=d['tp'][b] - d['tp'][a], recall_tol_frames=rec_tol,
                            fa_change_frames=d['fp'][b] - d['fp'][a], fa_tol_frames=fa_tol,
                            ok=(d['tp'][b] - d['tp'][a] >= -rec_tol) and (d['fp'][b] - d['fp'][a] <= fa_tol))
    within = all(v['ok'] for v in checks.values())
    verdict = 'ADOPT_B' if rel <= -.15 and within else 'KEEP_A' if (rel >= .15 or not within) else 'UNCERTAIN'
    return dict(a=a, b=b, primary=prim, relative_change=rel, checks=checks, within_tolerance=within, verdict=verdict)


def main(out_dir, smoke=0):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261117); M = Methods()
    report = dict(protocol=core.PROTOCOL_VERSION, methods=A2_METHODS, smoke=smoke)
    rows_by = {}
    for name in DATASETS:
        report[name], rows_by[name] = evaluate(name, M, rng, out, smoke)
        print(name, 'sources', json.dumps(report[name].get('sources')), flush=True)
    if smoke:
        (out / 'report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8'); return
    comps = {}
    for name in DATASETS:
        rows = rows_by[name]
        comps[name] = {f'{a}__{b}': core.compare(rows, a, b) for a, b in
                       [('tof_unidepth_out', 'hybrid_out'), ('unidepth_s', 'floor_unidepth'), ('tof_dav2s_out', 'hybrid_out')]
                       + [('tof_unidepth_out', h) for h in SCALED[3:]]}
        sub = [r for r in rows if r['source']['tof_unidepth_out'] != 'tof']        # frames without a current ToF scale
        comps[name]['tof_insufficient_subset'] = core.compare(sub, 'tof_unidepth_out', 'hybrid_out') if sub else None
    report['comparisons'] = comps
    d1 = judge(comps['sanpo_head_seg']['tof_unidepth_out__hybrid_out'])
    d2 = judge(comps['aria_gen2_walks']['tof_unidepth_out__hybrid_out'])
    overall = ('ADOPT_HYBRID' if d1['verdict'] == 'ADOPT_B' and d2['verdict'] != 'KEEP_A'
               else 'SCENE_DEPENDENT' if d1['verdict'] == 'ADOPT_B' else 'NO_BENEFIT' if d1['verdict'] == 'KEEP_A' else 'UNCERTAIN')
    report['decision'] = dict(D1_sanpo=d1, D2_aria=d2, overall=overall)
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps(report['decision'], indent=1, default=str))


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0], int(args[args.index('--smoke') + 1]) if '--smoke' in args else 0)
