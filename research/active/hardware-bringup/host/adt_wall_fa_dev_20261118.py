"""Do real plain walls make D8 false-alarm? ADT walking frames with an empty body corridor, split by a side surface and
its image texture (EXPLORE, 2026-11-18). Plan: artifacts.local/work/adt-wall-fa-20261118/PLAN.md

  python adt_wall_fa_dev_20261118.py strata  <adt data> <adt work> <out>
  python adt_wall_fa_dev_20261118.py score   <adt work> <out> <ase work>
  python adt_wall_fa_dev_20261118.py summary <out>
  python adt_wall_fa_dev_20261118.py sheet   <adt work> <out>

ADT license: non-commercial research; frames and derived data stay in ignored artifacts and are never redistributed.
"""
from __future__ import annotations

import json
import sys
from collections import deque
from itertools import groupby
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

STEP = 2
RX = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])                     # ADT scene y-up -> z-up (rotation about x)
UP = np.array([0., 0., 1.])
SIDE_LAT, SIDE_FWD, SIDE_H, SIDE_BINS, NONE_PTS = (.35, 1.20), (.30, 2.0), (.60, 1.90), 10, 50


def zup(T):
    T = T.copy(); T[..., :3, :3] = RX @ T[..., :3, :3]; T[..., :3, 3] = T[..., :3, 3] @ RX.T
    return T


def heading(T):
    f = T[:3, 2] - UP * (T[:3, 2] @ UP); n = np.linalg.norm(f)
    return f / n if n > .2 else None


def load_seq(work, s):
    meta = json.loads((work / 'prep_n' / s / 'meta.json').read_text()); fx, fy, cx, cy = meta['projection_params']
    T = zup(np.load(work / 'prep_n' / s / 'poses.npz')['T_sc'])
    return meta, (fx, fy, cx, cy), T


# ---------------------------------------------------------------- strata (ground truth only)
def strata(adt, work, out):
    import cv2
    import adt_cane_reach_summary_dev_20261015 as S
    adt, work, out = Path(adt), Path(work), Path(out); out.mkdir(parents=True, exist_ok=True)
    R = json.loads((work / 'adt_eval.json').read_text(encoding='utf8'))
    old = json.loads((work / 'cane_reach_v3.json').read_text(encoding='utf8'))['labels']
    L2 = json.loads((work / 'review_L2.json').read_text(encoding='utf8'))
    res = {}
    for s in R:
        floor = R[s]['floor']; meta, (fx, fy, cx, cy), T = load_seq(work, s)
        nd = S.nearest_dynamic(work / 'prep_b' / s, adt / s)
        with np.load(work / 'prep_n' / s / 'frames.npz') as Z:
            depth = Z['depth']
        with np.load(work / 'prep_b' / s / 'frames.npz') as Z:
            rgb = Z['rgb']
        h_, w_ = depth.shape[1:]
        vv, uu = np.mgrid[0:h_:STEP, 0:w_:STEP]
        dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(uu.shape)], -1)
        rows = []
        for k in range(len(depth)):
            lab = L2[s]['rows'][k]
            base = old[s]['speed'][k] >= .30 and nd[k] >= S.MANIP_M and bool(lab.get('neg'))
            f = heading(T[k])
            if not base or f is None:
                rows.append(None); continue
            z = depth[k][::STEP, ::STEP] / 1000.; ok = z > .1
            P = (dirs[ok] * z[ok, None]) @ T[k][:3, :3].T + T[k][:3, 3]
            d = P - T[k][:3, 3]; xr = np.cross(f, UP); lat, fwd, h = d @ xr, d @ f, P[:, 2] - floor
            zone = (fwd >= SIDE_FWD[0]) & (fwd <= SIDE_FWD[1]) & (h >= SIDE_H[0]) & (h <= SIDE_H[1])
            sides = {}
            for sg in (-1, 1):
                m = zone & (sg * lat >= SIDE_LAT[0]) & (sg * lat <= SIDE_LAT[1])
                nb = len(np.unique(((h[m] - SIDE_H[0]) // .1).astype(int))) if m.any() else 0
                sides[sg] = (m, nb)
            present = [sg for sg in (-1, 1) if sides[sg][1] >= SIDE_BINS]
            tot = int(sum(sides[sg][0].sum() for sg in (-1, 1)))
            r = dict(k=k)
            if present:
                m = np.zeros(len(P), bool)
                for sg in present:
                    m |= sides[sg][0]
                gray = cv2.GaussianBlur(cv2.cvtColor(rgb[k], cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.)
                gm = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
                iv, iu = vv[ok][m], uu[ok][m]
                marks = (iv % 16 == 0) & (iu % 16 == 0)
                r.update(stratum='side', lat=float(np.min(np.abs(lat[m]))),
                         tex=float(np.median(gm[iv, iu])), npts=int(m.sum()),
                         sample_pixels=np.column_stack((iu[marks], iv[marks])).tolist())
            elif tot < NONE_PTS:
                r.update(stratum='none')
            else:
                r.update(stratum='other')
            rows.append(r)
        res[s] = rows
        c = [r['stratum'] for r in rows if r]
        print(s, {x: c.count(x) for x in ('side', 'none', 'other')}, flush=True)
    (out / 'strata.json').write_text(json.dumps(res), encoding='utf8')
    rows = [(seq, r) for seq, frames in res.items() for r in frames if r]
    side = [(s, r) for s, r in rows if r['stratum'] == 'side']
    low_cut, high_cut = np.percentile([r['tex'] for _, r in side], [100 / 3, 200 / 3]) if side else (None, None)
    groups = {
        'eligible': rows,
        'none': [(s, r) for s, r in rows if r['stratum'] == 'none'],
        'other': [(s, r) for s, r in rows if r['stratum'] == 'other'],
        'side_low': [(s, r) for s, r in side if r['tex'] <= low_cut],
        'side_mid': [(s, r) for s, r in side if low_cut < r['tex'] <= high_cut],
        'side_high': [(s, r) for s, r in side if r['tex'] > high_cut],
    }
    counts = {name: dict(frames=len(group), sequences=len({seq for seq, _ in group}))
              for name, group in groups.items()}
    enough = all(counts[name]['frames'] >= 100 and counts[name]['sequences'] >= 4
                 for name in ('side_low', 'none'))
    coverage = dict(source_frames=sum(map(len, res.values())), sequences=len(res), groups=counts,
                    texture_cutoffs=[low_cut, high_cut], coverage_pass=enough)
    (out / 'coverage.json').write_text(json.dumps(coverage, indent=2), encoding='utf8')
    print(json.dumps(coverage), flush=True)


# ---------------------------------------------------------------- D8 on every target frame
def score(work, out, ase_work):
    import torch
    import torch.nn.functional as Fn
    import spires_insert_dev_20261117 as SI
    from spires_insert_dev_20261117 import A, V, I
    work, out, ase_work = Path(work), Path(out), Path(ase_work)
    R = json.loads((work / 'adt_eval.json').read_text(encoding='utf8'))
    ST = json.loads((out / 'strata.json').read_text(encoding='utf8'))
    model = SI._dav2s()
    net = I.D.make_head(8).cuda(); net.load_state_dict(torch.load(ase_work / 'decision_head8_all.pt')); net.eval()
    res = {}
    for s in R:
        tgt = {r['k'] for r in ST[s] if r and r['stratum'] in ('side', 'none')}
        if not tgt:
            continue
        floor = R[s]['floor']; meta, (fx, fy, cx, cy), T = load_seq(work, s)
        with np.load(work / 'prep_n' / s / 'frames.npz') as Z:
            depth, t_ns = Z['depth'], Z['t_ns']
        with np.load(work / 'prep_b' / s / 'frames.npz') as Z:
            rgb = Z['rgb']
        tt = (t_ns - t_ns[0]) / 1e9; h_, w_ = depth.shape[1:]
        vv, uu = np.mgrid[0:h_:STEP, 0:w_:STEP]
        dp = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(uu.shape)], -1)
        hc, fc = (np.arange(V.NH) + .5) * V.CELL, (np.arange(V.NF) + .5) * V.CELL
        L_, F_, H_ = np.meshgrid(np.array([-.2, 0., .2]), fc, hc, indexing='ij')
        gf, gh = np.meshgrid(fc / (V.NF * V.CELL), hc / (V.NH * V.CELL))
        last = max(tgt); buf = deque(); X, K = [], []
        for b0 in range(0, last + 1, 8):
            ims = [np.ascontiguousarray(rgb[k][..., ::-1]) for k in range(b0, min(b0 + 8, last + 1))]   # RGB -> BGR
            t_ = torch.cat([model.image2tensor(im, 518)[0] for im in ims])
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                dd = model(t_)
            zs = Fn.interpolate(dd[:, None].float(), (h_, w_), mode='bilinear', align_corners=True)[:, 0].cpu().numpy()
            for q, z in enumerate(zs):
                k = b0 + q
                gt2, z2 = depth[k][::STEP, ::STEP].astype(np.float32), z[::STEP, ::STEP] * 1000
                m = (gt2 > 0) & (gt2 < A.MAX_MM) & (z2 > 0)
                zz = np.where(gt2 > 0, z2 * (np.median(gt2[m] / z2[m]) if m.sum() > 100 else 1.), 0.) / 1000.
                okz = (zz > 0) & (zz < A.MAX_MM / 1000.)
                buf.append((k, tt[k], T[k], ((dp[okz] * zz[okz, None]) @ T[k][:3, :3].T + T[k][:3, 3]).astype(np.float32),
                            zz.astype(np.float32)))
                while buf and buf[0][1] < tt[k] - 3.0 - 1e-6:
                    buf.popleft()
                if k not in tgt:
                    continue
                Tk = T[k]; o = Tk[:3, 3]; f = heading(Tk); xr = np.cross(f, UP)

                def cells(P):
                    d = P - o; lat, fwd, h = d @ xr, d @ f, P[:, 2] - floor
                    mm = (np.abs(lat) <= V.X_HALF) & (fwd >= 0) & (fwd < V.NF * V.CELL) & (h >= 0) & (h < V.NH * V.CELL)
                    return np.bincount((h[mm] // V.CELL).astype(int) * V.NF + (fwd[mm] // V.CELL).astype(int),
                                       minlength=V.NH * V.NF).reshape(V.NH, V.NF)
                win = list(buf); cur = win[-1]
                hp = sum((cells(e[3]) for e in win[:-1]), np.zeros((V.NH, V.NF)))
                Wp = (o * (1 - UP) + L_[..., None] * xr + F_[..., None] * f + (floor + H_)[..., None] * UP).reshape(-1, 3)
                seen = np.zeros((2, V.NH, V.NF)); free = np.zeros((2, V.NH, V.NF))
                for e in win:
                    qq = (Wp - e[2][:3, 3]) @ e[2][:3, :3]; zq = qq[:, 2]; zs_ = np.where(zq > .05, zq, 1.)
                    u = (fx * qq[:, 0] / zs_ + cx) / STEP; v = (fy * qq[:, 1] / zs_ + cy) / STEP
                    ins = (zq > .05) & (u >= 0) & (u < e[4].shape[1]) & (v >= 0) & (v < e[4].shape[0])
                    zm = np.zeros(len(zq)); zm[ins] = e[4][v[ins].astype(int), u[ins].astype(int)]
                    vis = ins & (zm > 0); frr = vis & (zq < zm - .10); si = 0 if e is cur else 1
                    seen[si] += vis.reshape(3, V.NF, V.NH).sum(0).T; free[si] += frr.reshape(3, V.NF, V.NH).sum(0).T
                X.append(np.stack([np.log1p(cells(cur[3])), np.log1p(hp), free[0] / 3, np.log1p(free[1]), seen[0] / 3,
                                   np.log1p(seen[1]), gf, gh]).astype(np.float16)); K.append(k)
        with torch.no_grad():
            sc = torch.sigmoid(net(V.pad(torch.from_numpy(np.stack(X)).cuda().float()))).cpu().numpy()
        res[s] = {int(k): float(v) for k, v in zip(K, sc)}
        print(s, len(K), 'alert', int((sc > I.THETA).sum()), flush=True)
    (out / 'scores.json').write_text(json.dumps(res), encoding='utf8')


# ---------------------------------------------------------------- summary
def summary(out):
    out = Path(out); THETA = .5367
    ST = json.loads((out / 'strata.json').read_text(encoding='utf8')); SC = json.loads((out / 'scores.json').read_text(encoding='utf8'))
    rows = [dict(r, seq=s, s=SC[s][str(r['k'])]) for s in ST for r in ST[s] if r and r['stratum'] in ('side', 'none')]
    if any(not np.isfinite(r['s']) for r in rows):
        raise ValueError('Missing or non-finite D8 score')
    side = [r for r in rows if r['stratum'] == 'side']
    q1, q2 = np.percentile([r['tex'] for r in side], [100 / 3, 200 / 3])
    for r in side:
        r['tex3'] = 'low' if r['tex'] <= q1 else ('mid' if r['tex'] <= q2 else 'high')

    def rate(rr):
        a = sum(r['s'] > THETA for r in rr)
        per = {}
        for r in rr:
            p = per.setdefault(r['seq'][18:], [0, 0]); p[0] += 1; p[1] += r['s'] > THETA
        return dict(frames=len(rr), alerts=int(a), rate=a / len(rr) if rr else None, seqs=len(per),
                    per_seq={k: f'{v[1]}/{v[0]}' for k, v in sorted(per.items())})
    rep = dict(tex_tercile_edges=[round(float(q1), 2), round(float(q2), 2)],
               none=rate([r for r in rows if r['stratum'] == 'none']), side_all=rate(side),
               **{f'side_{t}': rate([r for r in side if r['tex3'] == t]) for t in ('low', 'mid', 'high')},
               **{f'side_lat_{a}-{b}': rate([r for r in side if a <= r['lat'] < b]) for a, b in ((.35, .6), (.6, .9), (.9, 1.21))})
    lo, no = rep['side_low'], rep['none']
    ok = lo['frames'] >= 100 and no['frames'] >= 100 and lo['seqs'] >= 4 and no['seqs'] >= 4
    d = lo['rate'] - no['rate'] if ok else None
    rep['verdict'] = ('NOT_EVALUABLE' if not ok else
                      'plain walls raise D8 false alarms' if (lo['rate'] >= 2 * no['rate'] and d >= .05) else
                      'no plain-wall false alarms seen' if d <= .02 else 'undetermined')
    if lo['frames'] and rep['side_high']['frames']:
        rep['low_minus_high'] = lo['rate'] - rep['side_high']['rate']
    (out / 'summary.json').write_text(json.dumps(rep, indent=1), encoding='utf8')
    print(json.dumps(rep, indent=1))


def sheet(work, out):
    import cv2
    work, out = Path(work), Path(out)
    ST = json.loads((out / 'strata.json').read_text(encoding='utf8'))
    side = [(s, r) for s in ST for r in ST[s] if r and r['stratum'] == 'side']
    if not side:
        print('No side-surface frames to inspect')
        return
    q1 = np.percentile([r['tex'] for _, r in side], 100 / 3)
    low = [(s, r) for s, r in side if r['tex'] <= q1]
    rng = np.random.default_rng(0); pick = [low[i] for i in sorted(rng.choice(len(low), min(16, len(low)), replace=False))]
    tiles = []
    for s, frames in groupby(pick, key=lambda item: item[0]):
        with np.load(work / 'prep_b' / s / 'frames.npz') as archive:
            rgb = archive['rgb']
        for _, r in frames:
            im = cv2.cvtColor(rgb[r['k']], cv2.COLOR_RGB2BGR)
            for x, y in r['sample_pixels']:
                cv2.circle(im, (x, y), 1, (255, 0, 255), -1)
            im = cv2.resize(im, (300, 400), interpolation=cv2.INTER_AREA)
            label = f"{len(tiles) + 1} {s[18:].split('_M')[0]} k{r['k']}"
            cv2.rectangle(im, (0, 0), (300, 18), (0, 0, 0), -1)
            cv2.putText(im, label, (2, 13), cv2.FONT_HERSHEY_SIMPLEX, .44, (255, 255, 255), 1)
            tiles.append(im)
        del rgb
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out / 'low_texture_sample.jpg'), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


if __name__ == '__main__':
    dict(strata=strata, score=score, summary=summary, sheet=sheet)[sys.argv[1]](*sys.argv[2:])
