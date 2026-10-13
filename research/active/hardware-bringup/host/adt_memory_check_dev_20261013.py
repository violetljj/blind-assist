"""GT-only check on ADT: how much of the HEAD/BODY weakness is outside the camera view but was seen in the last 2 s
(plan: adt-eval-20261013/PLAN_MEMORY.md). No model is run; reads the main eval output (adt_eval_dev_20261013), the
prepared GT depth / gravity frames and the GT poses (adt_poses_dev_20261013).
Per frame and query: in-view GT in-box count (re-derived, must equal the main eval truth), border truncation of the
in-box mask, and "hidden" structure = GT points seen in the previous <= 2 s, moved into the current corridor frame with
GT poses, inside the box (z <= 1.5 m) but outside the current image (unique 5 cm voxels).
Usage: adt_memory_check_dev_20261013.py <prep dir> <adt_eval.json> <out json>
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v2 as V2

BOXES = [(-.20, .42), (.42, .90)]          # HEAD, BODY corridor-y bands (y down), = real_rgb_query_feasibility BOXES
Z_MAX, Z_HID, LOOK_S, LOOK_N, SUB, VOX, WALK, EDGE = 2.0, 1.5, 2.0, 10, 4, .05, .3, 2


def corridor_axes(fwd):
    y = np.array([0., -1., 0.]); return np.stack([np.cross(y, fwd), y, fwd])


def run_seq(args):
    prep, seq = args
    meta = json.loads((prep / seq / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']; w, h = meta['image_size']
    with np.load(prep / seq / 'frames.npz') as Z:
        dep, R_cc, t = Z['depth'], Z['R_cc'], Z['t_ns'] / 1e9
    P = np.load(prep / seq / 'poses.npz'); pos, fwd, speed = P['pos'], P['fwd'], P['speed']
    N = len(t); A = [corridor_axes(f) for f in fwd]
    vv, uu = np.mgrid[0:h, 0:w]
    dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], -1).reshape(-1, 3)
    sub = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu, float)], -1)[::SUB, ::SUB].reshape(-1, 3)
    clouds = []
    for j in range(N):                                                # scene-frame GT clouds, every SUB-th pixel
        z = dep[j, ::SUB, ::SUB].reshape(-1) / 1000.; ok = z > .1
        g = (R_cc[j] @ (sub[ok] * z[ok, None]).T).T
        clouds.append((pos[j] + g @ A[j]).astype(np.float32))
    out = []
    for i in range(N):
        zgt = dep[i].reshape(-1) / 1000.; gok = zgt > .1
        g = np.full((h * w, 3), np.nan); g[gok] = (R_cc[i] @ (dirs[gok] * zgt[gok, None]).T).T
        js = [j for j in range(i - 1, max(i - LOOK_N - 1, -1), -1) if t[i] - t[j] <= LOOK_S]
        if js:
            gi = (np.concatenate([clouds[j] for j in js]).astype(float) - pos[i]) @ A[i].T   # corridor frame of i
            pc = gi @ R_cc[i]                                                                # camera frame of i
            with np.errstate(divide='ignore', invalid='ignore'):
                u, v = fx * pc[:, 0] / pc[:, 2] + cx, fy * pc[:, 1] / pc[:, 2] + cy
            outv = (pc[:, 2] <= .1) | ~((u >= 0) & (u < w) & (v >= 0) & (v < h))
        qs = []
        for lo, hi in BOXES:
            m = gok & (np.abs(g[:, 0]) <= .30) & (g[:, 1] >= lo) & (g[:, 1] <= hi) & (g[:, 2] >= .30) & (g[:, 2] <= Z_MAX)
            M = m.reshape(h, w)
            edges = [bool(M[:EDGE].any()), bool(M[-EDGE:].any()), bool(M[:, :EDGE].any()), bool(M[:, -EDGE:].any())]
            hid, hz = 0, None
            if js:
                b = outv & (np.abs(gi[:, 0]) <= .30) & (gi[:, 1] >= lo) & (gi[:, 1] <= hi) & (gi[:, 2] >= .30) & (gi[:, 2] <= Z_HID)
                if b.any():
                    hid = len(np.unique(np.floor(gi[b] / VOX).astype(np.int32), axis=0)); hz = float(gi[b, 2].min())
            qs.append(dict(n=int(m.sum()), edges=edges, hid=hid, hz=hz))
        out.append(dict(walk=bool(np.isfinite(speed[i]) and speed[i] >= WALK), look=len(js), q=qs))
    return seq, out


def frac(a, b):
    return dict(n=[int(a), int(b)], rate=round(a / b, 4) if b else None)


def main(prep, eval_json, out_json):
    prep = Path(prep); ev = json.loads(Path(eval_json).read_text(encoding='utf8'))
    seqs = [s for s in ev if (prep / s / 'poses.npz').exists()]
    with ProcessPoolExecutor(4) as ex:
        per = dict(ex.map(run_seq, [(prep, s) for s in seqs]))
    recs, mismatch = [], 0
    for s in seqs:
        fr, mm = ev[s]['frames'], per[s]
        assert len(fr) == len(mm), s
        conf = {arm: [V2.confirm([f['arms'][arm][q][1] for f in fr]) for q in (0, 1)] for arm in ('chain', 'oracle')}
        for i, (f, m) in enumerate(zip(fr, mm)):
            for q in (0, 1):
                tr, x = f['truth'][q], m['q'][q]
                mismatch += int(tr['n'] != x['n'])
                recs.append(dict(seq=s, q=q, walk=m['walk'], look=m['look'], n=tr['n'], z50=tr['z50'], edges=x['edges'],
                                 hid=x['hid'], hz=x['hz'], v6=bool(conf['chain'][q][i]), v6o=bool(conf['oracle'][q][i])))
    Path(out_json).with_name(Path(out_json).stem + '_records.json').write_text(json.dumps(recs), encoding='utf8')
    res = dict(plan='PLAN_MEMORY.md', sequences=len(seqs), truth_mismatch=mismatch, query={})
    for q, name in enumerate(('HEAD', 'BODY')):
        R = {}
        for sub in ('all', 'walk'):
            rr = [r for r in recs if r['q'] == q and (sub == 'all' or r['walk'])]
            pos = [r for r in rr if r['n'] >= 150 and r['z50'] <= 1.5]
            miss = [r for r in pos if not r['v6']]
            tr = [r for r in pos if any(r['edges'])]; nt = [r for r in pos if not any(r['edges'])]
            d = dict(query_frames=len(rr), pos=len(pos), recall=frac(len(pos) - len(miss), len(pos)),
                     miss_rate_truncated=frac(sum(not r['v6'] for r in tr), len(tr)),
                     miss_rate_not_truncated=frac(sum(not r['v6'] for r in nt), len(nt)),
                     miss_edges={e: frac(sum(r['edges'][k] for r in miss), len(miss)) for k, e in enumerate(('top', 'bottom', 'left', 'right'))},
                     miss_also_oracle_miss=frac(sum(not r['v6o'] for r in miss), len(miss)),
                     miss_near_z07=frac(sum(r['z50'] <= .7 for r in miss), len(miss)))
            for k in (1, 4, 20):
                d[f'miss_hidden_ge{k}'] = frac(sum(r['hid'] >= k for r in miss), len(miss))
                hh = [r for r in rr if not (r['n'] >= 150) and r['hid'] >= k]
                d[f'hidden_hazard_ge{k}'] = dict(frames=len(hh), vs_pos=round(len(hh) / max(len(pos), 1), 4),
                                                 truth_negative=sum(r['n'] == 0 for r in hh),
                                                 hz_le07=sum(r['hz'] <= .7 for r in hh))
            d['pos_hidden_ge4'] = frac(sum(r['hid'] >= 4 for r in pos), len(pos))
            R[sub] = d
        R['decision'] = dict(a_miss_hidden_share=R['walk']['miss_hidden_ge4']['rate'],
                             b_hidden_hazard_vs_pos=R['walk']['hidden_hazard_ge4']['vs_pos'])
        res['query'][name] = R
    h = res['query']['HEAD']['decision']
    res['go_memory'] = bool((h['a_miss_hidden_share'] or 0) >= .30 or (h['b_hidden_hazard_vs_pos'] or 0) >= .25)
    Path(out_json).write_text(json.dumps(res, indent=1), encoding='utf8')
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
