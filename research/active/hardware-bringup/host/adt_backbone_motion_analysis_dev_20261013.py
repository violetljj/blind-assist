"""Offline analysis of the ADT blob dumps (plan: adt-eval-20261013/PLAN_BACKBONE_MOTION.md).

Q1: v6 readout per backbone and arm (chain / raw / oracle). Q2: world-consistency rule WC-a/WC-b on the chain arm
(plus degraded-pose WC-a: displacement along the current head heading with GT speed, or a fixed 1 m/s; added after
the plan as robustness checks) (and
raw arm of metric models), GT poses. Metrics: pooled query-frames, 2-of-3 confirmation; FA on GT-negative, recall on
GT-positive with z50 <= 1.5 m; splits by GT speed (still < 0.3 m/s) and GT distance (<= 0.7 m).
Usage: adt_backbone_motion_analysis_dev_20261013.py <bb dir> <prep dir> <out json> [backbone ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v2 as V2

K_LOOK, WALK, MIN_PTS = 5, .3, 3
GXC, GZC = np.linspace(-1.45, 1.45, 30), np.linspace(.35, 5.25, 50)


def corridor_axes(fwd):
    y = np.array([0., -1., 0.]); return np.stack([np.cross(y, fwd), y, fwd])


def load_seq(bb_dir, prep, seq):
    J = json.loads((bb_dir / f'{seq}.json').read_text(encoding='utf8'))
    G = np.load(bb_dir / f'{seq}.npz')
    P = np.load(prep / seq / 'poses.npz')
    with np.load(prep / seq / 'frames.npz') as Z:
        R_cc, t = Z['R_cc'], Z['t_ns'] / 1e9
    meta = json.loads((prep / seq / 'meta.json').read_text())
    return J['frames'], {k: G[k] for k in G.files}, P['pos'], P['fwd'], P['speed'], R_cc, t, meta


def wc_keep(blob, q, i, j, pos, fwd, R_cc, grid, K, wh, disp=None):
    """True if the blob of frame i is supported by frame j's predicted points at the same world spot (or unverifiable).
    disp: optional estimated horizontal displacement pos[i] - pos[j] replacing the GT one (rotation stays GT ~ gyro)."""
    c = np.array(blob[2:5])
    s = pos[i] + corridor_axes(fwd[i]).T @ c                         # scene point
    pj = pos[j] if disp is None else pos[i] - disp
    cj = corridor_axes(fwd[j]) @ (s - pj)                            # corridor of frame j
    pc = R_cc[j].T @ cj                                              # camera of frame j
    fx, fy, cx, cy = K; w, h = wh
    if pc[2] <= .3:
        return True
    u, v = fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy
    if not (0 <= u < w and 0 <= v < h) or cj[2] > 5.3 or abs(cj[0]) > 1.5:
        return True
    r = .2 + .1 * abs(cj[2])
    near = ((GXC[:, None] - cj[0]) ** 2 + (GZC[None, :] - cj[2]) ** 2) <= r * r
    return int(grid[j, q][near].sum()) >= MIN_PTS


def alerts(frames, arm, rule, ctx):
    """[N, 2] bool raw alerts for an arm under rule 'v6' | 'wca' | 'wcb'."""
    pos, fwd, speed, R_cc, t, meta, G = ctx
    out = np.zeros((len(frames), 2), bool)
    for i, f in enumerate(frames):
        a = f['arms'].get(arm)
        if a is None:
            continue
        j = i - K_LOOK
        walking = np.isfinite(speed[i]) and speed[i] >= WALK
        jok = j >= 0 and .6 <= t[i] - t[j] <= 1.5
        for q in (0, 1):
            cnt = 0
            for b in a[q][1]:
                if b[1] > 1.8:
                    continue
                if rule != 'v6':
                    disp = None
                    if walking and jok and rule in ('wca_head', 'wca_step'):
                        v = speed[i] if rule == 'wca_head' else 1.0      # GT speed, or a fixed 1 m/s pace
                        disp = v * (t[i] - t[j]) * fwd[i]              # horizontal only
                    if walking and jok and not wc_keep(b, q, i, j, pos, fwd, R_cc, G[arm], meta['projection_params'], meta['image_size'], disp):
                        continue
                    if rule == 'wcb' and not walking and b[1] > 1.0:
                        continue
                cnt += b[0]
            out[i, q] = cnt >= 150
    return out


def score(rows):
    """rows: list of (alerts [N,2] confirmed, frames, speed) -> dict of counts."""
    c = {k: [0, 0] for k in ('fa', 'fa_still', 'fa_walk', 'rec', 'rec_near', 'rec_mid', 'rec_walk_near')}
    for a, frames, speed in rows:
        for i, f in enumerate(frames):
            still = not (np.isfinite(speed[i]) and speed[i] >= WALK)
            for q in (0, 1):
                tr = f['truth'][q]
                if tr['n'] == 0:
                    for k in ('fa', 'fa_still' if still else 'fa_walk'):
                        c[k][0] += int(a[i, q]); c[k][1] += 1
                elif tr['n'] >= 150 and tr['z50'] <= 1.5:
                    ks = ['rec', 'rec_near' if tr['z50'] <= .7 else 'rec_mid'] + (['rec_walk_near'] if (not still and tr['z50'] <= .7) else [])
                    for k in ks:
                        c[k][0] += int(a[i, q]); c[k][1] += 1
    return {k: dict(n=v, rate=round(v[0] / max(v[1], 1), 4)) for k, v in c.items()}


PX_GRID, Z_GRID = (50, 75, 100, 150, 200, 300, 450, 600, 900), (1.2, 1.4, 1.6, 1.8, 2.0)


def curve(seq_frames, arm):
    """Operating-point sweep of the v6 readout (min px x blob z cut), 2-of-3 confirmed: list of (px, z, FA, recall)."""
    out = []
    for px in PX_GRID:
        for zc in Z_GRID:
            fa = [0, 0]; rc = [0, 0]
            for frames in seq_frames:
                a = np.zeros((len(frames), 2), bool)
                for i, f in enumerate(frames):
                    x = f['arms'].get(arm)
                    if x is not None:
                        a[i] = [sum(b[0] for b in x[q][1] if b[1] <= zc) >= px for q in (0, 1)]
                a = np.stack([V2.confirm(a[:, 0]), V2.confirm(a[:, 1])], 1)
                for i, f in enumerate(frames):
                    for q in (0, 1):
                        t = f['truth'][q]
                        if t['n'] == 0:
                            fa[0] += int(a[i, q]); fa[1] += 1
                        elif t['n'] >= 150 and t['z50'] <= 1.5:
                            rc[0] += int(a[i, q]); rc[1] += 1
            out.append((px, zc, fa[0] / fa[1], rc[0] / rc[1]))
    return out


def fa_at(cv, target):
    """Lowest FA among sweep points with recall >= target (None if unreachable)."""
    ok = [c for c in cv if c[3] >= target]
    return min(ok, key=lambda c: c[2]) if ok else None


def main(bb, prep, out, *names):
    bb, prep = Path(bb), Path(prep); res = {}
    names = names or [p.name for p in sorted(bb.iterdir()) if p.is_dir()]
    for name in names:
        seqs = sorted(p.stem for p in (bb / name).glob('*.json'))
        if len(seqs) < 13:
            print('incomplete', name, len(seqs)); continue
        per = {}; seq_frames = []
        for seq in seqs:
            frames, G, pos, fwd, speed, R_cc, t, meta = load_seq(bb / name, prep, seq); seq_frames.append(frames)
            ctx = (pos, fwd, speed, R_cc, t, meta, G)
            arms = [a for a in ('chain', 'raw', 'oracle') if a in frames[0]['arms']]
            for arm in arms:
                for rule in (('v6', 'wca', 'wcb', 'wca_head', 'wca_step') if arm != 'oracle' else ('v6',)):
                    a = alerts(frames, arm, rule, ctx)
                    a = np.stack([V2.confirm(a[:, 0]), V2.confirm(a[:, 1])], 1)
                    per.setdefault(f'{arm}/{rule}', []).append((a, frames, speed))
        res[name] = {k: score(v) for k, v in per.items()}
        res[name]['curves'] = {arm: curve(seq_frames, arm) for arm in ('chain', 'raw', 'oracle') if arm in seq_frames[0][0]['arms']}
        for arm, cv in res[name]['curves'].items():
            print(f"{name:13s} {arm:6s} FA at recall>=0.80/0.85/0.90:", [None if (c := fa_at(cv, r)) is None else (c[0], c[1], round(c[2], 4), round(c[3], 3)) for r in (.80, .85, .90)], flush=True)
        for k, v in res[name].items():
            if k == 'curves':
                continue
            print(f"{name:13s} {k:12s} FA {v['fa']['rate']:.4f} ({v['fa']['n'][0]}/{v['fa']['n'][1]}) still {v['fa_still']['rate']:.4f} walk {v['fa_walk']['rate']:.4f} | "
                  f"rec {v['rec']['rate']:.4f} ({v['rec']['n'][0]}/{v['rec']['n'][1]}) near {v['rec_near']['rate']:.3f} walk-near {v['rec_walk_near']['rate']:.3f} mid {v['rec_mid']['rate']:.3f}", flush=True)
    Path(out).write_text(json.dumps(res, indent=1), encoding='utf8')


if __name__ == '__main__':
    main(*sys.argv[1:])
