"""Head-on approaches: insert overhangs (table edge, hanging sign, bar, shelf) and cane-reachable controls (low / tall cabinet)
on straight walking segments of untouched ASE scenes, then score rule H, true grid G and decision head D8 per approach.
(EXPLORE, 2026-10-15; rules frozen in <work>/PLAN_INSERT.md before any result.)
Usage:
  ase_cane_insert_dev_20261015.py runs   <scenes> <work>     select segments + place objects (insert/runs.json)
  ase_cane_insert_dev_20261015.py render <scenes> <work>     composite pinhole RGB + GT depth, DAV2-S, grids, labels
  ase_cane_insert_dev_20261015.py sheet  <work>              contact sheet for a visual check
  ase_cane_insert_dev_20261015.py head   <work>              train D8 once on all 175 training scenes
  ase_cane_insert_dev_20261015.py score  <work>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ase_cane_reach_dev_20261015 as A  # noqa: E402
import ase_cane_sideview_dev_20261015 as V  # noqa: E402
import ase_cane_decision_dev_20261015 as D  # noqa: E402

SCENES = list(range(0, 20)) + list(range(55, 60))
TYPES = ['table', 'sign', 'bar', 'shelf', 'lowcab', 'tallcab']
MISS_TYPES = {'table', 'sign', 'bar', 'shelf'}
THETA = .5367                                                          # median of the five calibration-fold thresholds
COLORS = dict(table=(60, 110, 160), sign=(40, 160, 220), bar=(90, 90, 90), shelf=(170, 140, 110), lowcab=(120, 150, 90),
              tallcab=(150, 120, 170))                                 # BGR


def ctx(scenes, work, k):
    work = Path(work); R = np.load(work / 'rays.npz'); T_dc = R['T_dc']
    meta = json.loads((work / 'pinhole_meta.json').read_text()); T_dc_up = np.array(meta['T_device_camera'])
    frames = A.load_scene(Path(scenes) / str(k), T_dc)
    T_up = [f[2] @ np.linalg.inv(T_dc) @ T_dc_up for f in frames]
    Z = np.load(work / 'grids' / f'{k}.npz')
    rows = {r['frame']: r for r in json.loads((work / 'labels_v2' / f'{k}.json').read_text())['rows']}
    return frames, T_up, Z, rows, meta


def heading(T):
    f = T[:3, 2] - A.UP * (T[:3, 2] @ A.UP); n = np.linalg.norm(f)
    return f / n if n > .2 else None


# ---------------------------------------------------------------- segment selection and placement
def boxes_for(kind, rng):
    """Boxes in the object frame: (a along walking dir from near face, b lateral, z above floor) centre and half sizes."""
    if kind == 'table':
        top = .74 + rng.uniform(-.04, .04)
        out = [((.3, 0, top - .02), (.3, .5, .02))]
        out += [((a, b, (top - .04) / 2), (.02, .02, (top - .04) / 2)) for a in (.04, .56) for b in (-.48, .48)]
        return out
    if kind == 'sign':
        bot = 1.45 + rng.uniform(-.1, .1); return [((.01, 0, bot + .2), (.01, .3, .2))]
    if kind == 'bar':
        zc = 1.2 + rng.uniform(-.15, .15); return [((.025, 0, zc), (.025, .8, .025))]
    if kind == 'shelf':
        zc = 1.05 + rng.uniform(-.1, .1); return [((.175, 0, zc), (.175, .3, .015))]
    h = .8 if kind == 'lowcab' else 1.8
    return [((.25, 0, h / 2), (.25, .3, h / 2))]


def runs(scenes, work):
    work = Path(work); rng = np.random.default_rng(20261015); out = []; n = 0
    for k in SCENES:
        frames, T_up, Z, rows, _ = ctx(scenes, work, k)
        fid = {int(f): j for j, f in enumerate(Z['frames'])}
        ok = []
        for j, (i, *_ ) in enumerate(frames):
            r = rows.get(i); h = heading(T_up[j])
            ok.append(bool(Z['valid'][fid[i]] and r is not None and 'skip' not in r and D.strata(r) and h is not None))
        j = 0
        while j < len(frames):
            if not ok[j]:
                j += 1; continue
            run = [j]; hs = [heading(T_up[j])]
            while run[-1] + 1 < len(frames) and ok[run[-1] + 1] and frames[run[-1] + 1][0] - frames[run[-1]][0] == 2:
                hn = heading(T_up[run[-1] + 1]); m = np.mean(hs, 0); m /= np.linalg.norm(m)
                if np.degrees(np.arccos(np.clip(hn @ m, -1, 1))) > 15:
                    break
                run.append(run[-1] + 1); hs.append(hn)
            j0, j1 = run[0], run[-1]; p0, p1 = T_up[j0][:3, 3], T_up[j1][:3, 3]
            disp = float(np.linalg.norm((p1 - p0)[:2]))
            if disp >= 2.0:
                kind = TYPES[n % len(TYPES)]
                u = np.mean(hs, 0); u /= np.linalg.norm(u); r = np.cross(u, A.UP)
                far = disp + .35 + .6
                Y0 = Z['Y'][fid[frames[j0][0]]]
                if not Y0[1:, :min(V.NF, int(np.ceil(far / V.CELL)))].any():
                    lat = rng.uniform(-.1, .1)
                    near = p1[:2] + .35 * u[:2] + lat * r[:2]
                    out.append(dict(id=n, scene=k, kind=kind, j0=j0, j1=j1, frames=[int(frames[q][0]) for q in range(max(0, j0 - 15), j1 + 1)],
                                    run_frames=[int(frames[q][0]) for q in range(j0, j1 + 1)], near=near.tolist(), u=u.tolist(),
                                    r=r.tolist(), floor=float(Z['floor']), boxes=boxes_for(kind, rng), disp=round(disp, 2)))
                    n += 1
                    j = j1 + 15; continue
            j = j1 + 1
    (work / 'insert').mkdir(exist_ok=True)
    (work / 'insert' / 'runs.json').write_text(json.dumps(out, indent=1), encoding='utf8')
    from collections import Counter
    print(len(out), 'runs', Counter(o['kind'] for o in out), 'scenes', len({o['scene'] for o in out}))


def world_boxes(run):
    u, r = np.array(run['u']), np.array(run['r']); base = np.r_[run['near'], run['floor']]
    Ax = np.stack([u, r, A.UP], 1)
    return [(base + Ax @ np.array(c), Ax, np.array(h)) for c, h in run['boxes']]


def surface_points(run, step=.02):
    pts = []
    for c, Ax, h in world_boxes(run):
        g = [np.arange(-hh, hh + 1e-9, step) if hh > step else np.array([0.]) for hh in h]
        a, b, z = np.meshgrid(*g, indexing='ij')
        pts.append(c + np.stack([a.ravel(), b.ravel(), z.ravel()], 1) @ Ax.T)
    return np.concatenate(pts)


# ---------------------------------------------------------------- rendering + inputs + labels
def raycast(T, K, H, W, boxes):
    fx, fy, cx, cy = K
    vv, uu = np.mgrid[0:H, 0:W]
    dc = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(uu.shape)], -1)
    dw = dc @ T[:3, :3].T; o = T[:3, 3]
    best = np.full((H, W), np.inf); nrm = np.zeros((H, W, 3))
    for c, Ax, h in boxes:
        p = (o - c) @ Ax; d = dw @ Ax; d = np.where(np.abs(d) < 1e-9, 1e-9, d)
        t1, t2 = (-h - p) / d, (h - p) / d
        tlo, thi = np.minimum(t1, t2), np.maximum(t1, t2)
        tmin, tmax = tlo.max(-1), thi.min(-1)
        hit = (tmax >= np.maximum(tmin, 0)) & (tmin > .05) & (tmin < best)
        ax = tlo.argmax(-1)
        n = -np.sign(np.take_along_axis(d, ax[..., None], -1)) * Ax.T[ax]
        best = np.where(hit, tmin, best); nrm[hit] = n[hit]
    return best, nrm                                                   # t is z-depth (camera ray has z = 1)


def render(scenes, work):
    import cv2
    import torch
    import torch.nn.functional as Fn
    work = Path(work); R = json.loads((work / 'insert' / 'runs.json').read_text())
    meta = json.loads((work / 'pinhole_meta.json').read_text()); K = meta['projection']; W, H = meta['size']
    L = np.array([.4, .3, .86]); L /= np.linalg.norm(L)
    model = V._dav2s(); rng = np.random.default_rng(7)
    cache = {}
    for run in R:
        k = run['scene']
        if k not in cache:
            cache = {k: ctx(scenes, work, k)}
        frames, T_up, Z, rows, _ = cache[k]
        jx = {int(f[0]): j for j, f in enumerate(frames)}
        od = work / 'insert' / str(run['id']); od.mkdir(exist_ok=True)
        boxes = world_boxes(run); col = np.array(COLORS[run['kind']], float)
        ims, gts = [], []
        for fr in run['frames']:
            T = T_up[jx[fr]]
            rgb = cv2.imread(str(work / 'pinhole' / str(k) / f'{fr:07d}.jpg'))
            gt = cv2.imread(str(work / 'pinhole_nn' / str(k) / f'{fr:07d}.png'), cv2.IMREAD_UNCHANGED).astype(np.float32)
            t, n = raycast(T, K, H, W, boxes)
            m = np.isfinite(t) & (gt > 0) & (t * 1000 < gt)
            shade = (.35 + .65 * np.clip(n @ L, 0, 1))[..., None] * col + rng.normal(0, 6, (H, W, 1))
            rgb[m] = np.clip(shade[m], 0, 255).astype(np.uint8); gt[m] = t[m] * 1000
            cv2.imwrite(str(od / f'{fr:07d}.jpg'), rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(od / f'{fr:07d}.png'), np.clip(gt, 0, 65535).astype(np.uint16))
            ims.append(rgb); gts.append(gt)
        zs = []                                                        # DAV2-S on the composites
        for b in range(0, len(ims), 8):
            t_ = torch.cat([model.image2tensor(im, 518)[0] for im in ims[b:b + 8]])
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                dd = model(t_)
            zs += list(Fn.interpolate(dd[:, None].float(), (H, W), mode='bilinear', align_corners=True)[:, 0].cpu().numpy())
        X, Yg, lab, dd_ = inputs_and_labels(run, frames, T_up, Z, jx, meta, gts, zs)
        np.savez_compressed(od / 'grid.npz', X=X, Y=Yg, d=np.array([np.nan if v is None else v for v in dd_]),
                            lab=np.array(lab), frames=np.array(run['run_frames']))
        print('run', run['id'], run['kind'], 'scene', k, 'frames', len(run['frames']), 'labels', dict(zip(*np.unique(lab, return_counts=True))), flush=True)


def inputs_and_labels(run, frames, T_up, Z, jx, meta, gts, zs):
    fx, fy, cx, cy = meta['projection']; floor = run['floor']; STEP = V.STEP
    vv, uu = np.mgrid[0:meta['size'][1]:STEP, 0:meta['size'][0]:STEP]
    dp = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(uu.shape)], -1)
    win_fr = run['frames']; Pm, Zm, Tw, tt = [], [], [], []
    for fr, gt, z in zip(win_fr, gts, zs):
        gt2, z2 = gt[::STEP, ::STEP], z[::STEP, ::STEP] * 1000
        m = (gt2 > 0) & (gt2 < A.MAX_MM) & (z2 > 0)
        zz = np.where(gt2 > 0, z2 * (np.median(gt2[m] / z2[m]) if m.sum() > 100 else 1.), 0.) / 1000.
        ok = (zz > 0) & (zz < A.MAX_MM / 1000.)
        T = T_up[jx[fr]]; Tw.append(T); tt.append(frames[jx[fr]][1])
        Pm.append(((dp[ok] * zz[ok, None]) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)); Zm.append(zz.astype(np.float32))
    tt = np.array(tt); obj = surface_points(run)
    hc, fc = (np.arange(V.NH) + .5) * V.CELL, (np.arange(V.NF) + .5) * V.CELL
    L_, F_, H_ = np.meshgrid(np.array([-.2, 0., .2]), fc, hc, indexing='ij')
    gf, gh = np.meshgrid(fc / (V.NF * V.CELL), hc / (V.NH * V.CELL))
    zid = {int(f): j for j, f in enumerate(Z['frames'])}
    X, Yg, lab, dd = [], [], [], []
    for fr in run['run_frames']:
        j = win_fr.index(fr); T = Tw[j]; o = T[:3, 3]; f = heading(T); xr = np.cross(f, A.UP)

        def cells(P, half=V.X_HALF):
            d = P - o; lat, fwd, h = d @ xr, d @ f, P[:, 2] - floor
            m = (np.abs(lat) <= half) & (fwd >= 0) & (fwd < V.NF * V.CELL) & (h >= 0) & (h < V.NH * V.CELL)
            return np.bincount((h[m] // V.CELL).astype(int) * V.NF + (fwd[m] // V.CELL).astype(int), minlength=V.NH * V.NF).reshape(V.NH, V.NF)
        win = [q for q in range(len(win_fr)) if tt[j] - WIN <= tt[q] <= tt[j]]
        hp = sum((cells(Pm[q]) for q in win if q != j), np.zeros((V.NH, V.NF)))
        Wp = (o * (1 - A.UP) + L_[..., None] * xr + F_[..., None] * f + (floor + H_)[..., None] * A.UP).reshape(-1, 3)
        seen = np.zeros((2, V.NH, V.NF)); free = np.zeros((2, V.NH, V.NF))
        for q in win:
            qq = (Wp - Tw[q][:3, 3]) @ Tw[q][:3, :3]; z = qq[:, 2]; zs_ = np.where(z > .05, z, 1.)
            u = (fx * qq[:, 0] / zs_ + cx) / STEP; v = (fy * qq[:, 1] / zs_ + cy) / STEP
            ins = (z > .05) & (u >= 0) & (u < Zm[q].shape[1]) & (v >= 0) & (v < Zm[q].shape[0])
            zm = np.zeros(len(z)); zm[ins] = Zm[q][v[ins].astype(int), u[ins].astype(int)]
            vis = ins & (zm > 0); frr = vis & (z < zm - .10); s = 0 if q == j else 1
            seen[s] += vis.reshape(3, V.NF, V.NH).sum(0).T; free[s] += frr.reshape(3, V.NF, V.NH).sum(0).T
        X.append(np.stack([np.log1p(cells(Pm[j])), np.log1p(hp), free[0] / 3, np.log1p(free[1]), seen[0] / 3, np.log1p(seen[1]), gf, gh]))
        Yg.append((Z['Y'][zid[fr]] > 0) | (cells(obj) > 0))
        d = obj - o; lat, fwd, h = d @ xr, d @ f, obj[:, 2] - floor
        body = (np.abs(lat) <= D.BODY_HALF) & (fwd >= A.Z_RANGE[0]) & (fwd <= A.Z_RANGE[1]) & (h >= A.BODY[0]) & (h <= A.BODY[1])
        if body.sum() < 3:
            lab.append('NONE'); dd.append(None); continue
        dz = float(np.percentile(fwd[body], 5))
        sup = ((np.abs(lat) <= D.CANE_HALF) & (h >= A.BAND_LO) & (h <= A.TOPS['R2']) & (fwd >= 0) & (fwd <= dz + A.MARGIN)).sum()
        lab.append(('CANE' if sup >= 3 else 'MISS') + ('_near' if dz <= A.NEAR else '_far')); dd.append(dz)
    return np.array(X, np.float16), np.array(Yg), lab, dd


WIN = 3.


def sheet(work):
    import cv2
    work = Path(work); R = json.loads((work / 'insert' / 'runs.json').read_text()); tiles = []
    for run in R[:12]:
        fr = run['run_frames'][len(run['run_frames']) * 2 // 3]
        im = cv2.imread(str(work / 'insert' / str(run['id']) / f'{fr:07d}.jpg'))
        im = cv2.resize(im, (240, 320)); cv2.putText(im, f"{run['id']} {run['kind']} s{run['scene']}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        tiles.append(im)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(work / 'insert' / 'sheet.jpg'), np.vstack([np.hstack(tiles[r:r + 6]) for r in range(0, len(tiles), 6)]))


# ---------------------------------------------------------------- D8 on all training scenes, scoring
def head(work, epochs=10):
    import torch
    work = Path(work); torch.manual_seed(2026)
    recs = {k: D.scene_records(work, k) for k in D.TRAIN}
    X = torch.from_numpy(np.concatenate([recs[k]['X'] for k in D.TRAIN]))
    Y = torch.from_numpy(np.concatenate([np.array([l == 'MISS_near' for l in recs[k]['lab']], np.float32) for k in D.TRAIN]))
    pw = min(float((len(Y) - Y.sum()) / max(Y.sum(), 1)), 100.)
    net = D.make_head(8).cuda(); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * (len(Y) // 256 + 1))
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device='cuda'))
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Y))
        for b in range(0, len(perm), 256):
            i = perm[b:b + 256]; loss = lossf(net(V.pad(X[i].cuda().float())), Y[i].cuda())
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    torch.save(net.state_dict(), work / 'decision_head8_all.pt'); print('trained on', len(Y), 'frames,', int(Y.sum()), 'MISS_near')


def score(work):
    import torch
    work = Path(work); R = json.loads((work / 'insert' / 'runs.json').read_text())
    net = D.make_head(8).cuda(); net.load_state_dict(torch.load(work / 'decision_head8_all.pt')); net.eval()
    per = []
    for run in R:
        g = np.load(work / 'insert' / str(run['id']) / 'grid.npz')
        with torch.no_grad():
            s = torch.sigmoid(net(V.pad(torch.from_numpy(g['X']).cuda().float()))).cpu().numpy()
        dec = lambda occ: (lambda r: bool(r[0] and not r[2]))(V.decide(occ))
        al = dict(D8=s > THETA, H=np.array([dec((x[0] > 0) | (x[1] > 0)) for x in g['X']]), G=np.array([dec(y) for y in g['Y']]))
        per.append(dict(id=run['id'], kind=run['kind'], lab=g['lab'].tolist(), d=g['d'].tolist(), s=s.tolist(),
                        **{a: v.tolist() for a, v in al.items()}))
    def summarize(alerts_of):
        out = {}
        for grp, kinds in (('miss', MISS_TYPES), ('control', set(TYPES) - MISS_TYPES)):
            rr = [p for p in per if p['kind'] in kinds]; caught = t07 = t10 = 0; disturbed = 0; nnear = 0
            for p in rr:
                a = alerts_of(p)
                near = [q for q, l in enumerate(p['lab']) if l.endswith('_near')]
                if not near:
                    continue
                nnear += 1
                hit = [q for q in near if a[q]]
                if grp == 'miss':
                    if hit:
                        caught += 1; t07 += p['d'][hit[0]] >= .7; t10 += p['d'][hit[0]] >= 1.
                else:
                    disturbed += bool(hit)
            out[grp] = dict(runs=nnear, caught=caught, timely_07=t07, timely_10=t10) if grp == 'miss' else dict(runs=nnear, disturbed=disturbed)
        return out
    rep = {a: summarize(lambda p, a=a: p[a]) for a in ('D8', 'H', 'G')}
    rep['curve_D8'] = {str(t): summarize(lambda p, t=t: np.array(p['s']) > t) for t in (.1, .2, .3, .4, .5, .6, .7, .8, .9)}
    rep['by_kind_D8'] = {k: summarize(lambda p: p['D8']) for k in TYPES}
    kinds = {}
    for p in per:
        kinds.setdefault(p['kind'], []).append(p)
    rep['by_kind'] = {}
    for kk, rr in kinds.items():
        near_runs = [p for p in rr if any(l.endswith('_near') for l in p['lab'])]
        rep['by_kind'][kk] = dict(runs=len(rr), near=len(near_runs),
                                  labels=dict(zip(*np.unique([l for p in rr for l in p['lab']], return_counts=True))),
                                  **{a: sum(any(p[a][q] for q, l in enumerate(p['lab']) if l.endswith('_near')) for p in near_runs) for a in ('D8', 'H', 'G')})
    (work / 'insert' / 'score.json').write_text(json.dumps(dict(report=rep, runs=per), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o)), encoding='utf8')
    print(json.dumps({a: rep[a] for a in ('D8', 'H', 'G')}, indent=1))
    print(json.dumps(rep['by_kind'], indent=1, default=lambda o: o.item()))
    for t, v in rep['curve_D8'].items():
        print('curve', t, v)


if __name__ == '__main__':
    dict(runs=runs, render=render, sheet=sheet, head=head, score=score)[sys.argv[1]](*sys.argv[2:])
