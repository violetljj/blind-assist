"""Re-insert with supports (posts for bar/sign, a backing wall for the shelf) and soft contact shadows, paired with the
first Spires insertion (same runs, sizes, colours, random stream). EXPLORE, 2026-11-17.
Plan: artifacts.local/work/spires-cane-reach-20261117/PLAN_INSERT_V2.md

  python spires_insert_v2_dev_20261117.py sheet  <spires> <work>             a few composites for a visual check
  python spires_insert_v2_dev_20261117.py render <spires> <work>
  python spires_insert_v2_dev_20261117.py score  <ase_work> <work>

Labels and the G grid use the main object only (supports sit outside the cane sweep or behind the cane window by
construction), so labels are identical to the first insertion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import spires_insert_dev_20261117 as SI  # noqa: E402
from spires_insert_dev_20261117 import A, V, I, W, H  # noqa: E402

POST, WALL = (70., 70., 70.), (165., 170., 175.)                       # BGR
OUT = 'insert_v2'


def supports(run):
    """Extra boxes (object frame: a along walking dir from the near face, b lateral, z above floor) and colours."""
    k, bx = run['kind'], run['boxes']
    if k == 'bar':
        zc = bx[0][0][2]
        return [((.025, s * .80, zc / 2), (.02, .02, zc / 2)) for s in (-1, 1)], [POST, POST]
    if k == 'sign':
        top = bx[0][0][2] + bx[0][1][2]; bot = bx[0][0][2] - bx[0][1][2]
        posts = [((.01, s * .40, top / 2), (.02, .02, top / 2)) for s in (-1, 1)]
        arms = [((.01, s * .35, top - .02), (.01, .05, .02)) for s in (-1, 1)]
        return posts + arms, [POST] * 4
    if k == 'shelf':
        return [((.40, 0., 1.1), (.05, .6, 1.1))], [WALL]
    return [], []


def raycast_idx(T, K, boxes):
    fx, fy, cx, cy = K
    vv, uu = np.mgrid[0:H, 0:W]
    dw = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(uu.shape)], -1) @ T[:3, :3].T; o = T[:3, 3]
    best = np.full((H, W), np.inf); nrm = np.zeros((H, W, 3)); idx = np.full((H, W), -1)
    for bi, (c, Ax, h) in enumerate(boxes):
        p = (o - c) @ Ax; d = dw @ Ax; d = np.where(np.abs(d) < 1e-9, 1e-9, d)
        t1, t2 = (-h - p) / d, (h - p) / d
        tlo, thi = np.minimum(t1, t2), np.maximum(t1, t2)
        tmin, tmax = tlo.max(-1), thi.min(-1)
        hit = (tmax >= np.maximum(tmin, 0)) & (tmin > .05) & (tmin < best)
        ax = tlo.argmax(-1)
        n = -np.sign(np.take_along_axis(d, ax[..., None], -1)) * Ax.T[ax]
        best = np.where(hit, tmin, best); nrm[hit] = n[hit]; idx[hit] = bi
    return best, nrm, idx


def shadow_factor(T, gt_mm, boxes, floor):
    """Soft ambient-occlusion style contact shadow on ground pixels (height above floor < 0.08 m)."""
    z = gt_mm / 1000.; m = z > 0
    vv, uu = np.mgrid[0:H, 0:W]
    P = (np.stack([(uu - SI.CX) / SI.F, (vv - SI.CY) / SI.F, np.ones(uu.shape)], -1)[m] * z[m, None]) @ T[:3, :3].T + T[:3, 3]
    g = (P[:, 2] - floor) < .08
    fac = np.ones(len(P))
    for c, Ax, h in boxes:
        loc = (P[g] - c) @ Ax
        da, db = np.maximum(np.abs(loc[:, 0]) - h[0], 0), np.maximum(np.abs(loc[:, 1]) - h[1], 0)
        w = np.clip(1 - np.hypot(da, db) / .15, 0, 1)
        a = .5 * np.exp(-max(c[2] - h[2] - floor, 0.) / .8)
        fac[g] *= 1 - a * w
    out = np.ones((H, W)); out[m] = fac
    return out


VARIANTS = dict(both=('insert_v2', True, True), sup=('insert_v2_sup', True, False), shd=('insert_v2_shd', False, True))


def composite(run, T, rgb, gt, noise, use_sup=True, use_shadow=True):
    """rgb, gt (mm) of the uninserted frame -> composite rgb, gt, main-object mask, main-object true depth (m)."""
    extra, ecol = supports(run) if use_sup else ([], [])
    main = I.world_boxes(run); sup = I.world_boxes(dict(run, boxes=extra)) if extra else []
    boxes = main + sup; cols = [np.array(I.COLORS[run['kind']], float)] * len(main) + [np.array(c) for c in ecol]
    Lg = np.array([.4, .3, .86]); Lg /= np.linalg.norm(Lg)
    rgb = rgb.astype(np.float32)
    if use_shadow:
        rgb = rgb * shadow_factor(T, gt, boxes, run['floor'])[..., None]
    t, n, bi = raycast_idx(T, (SI.F, SI.F, SI.CX, SI.CY), boxes)
    m = np.isfinite(t) & ((gt == 0) | (t * 1000 < gt)) & SI.undistort_map()[2]
    colmap = np.stack(cols)[np.clip(bi, 0, len(cols) - 1)]
    shade = (.35 + .65 * np.clip(n @ Lg, 0, 1))[..., None] * colmap + noise
    rgb[m] = shade[m]; gt = gt.copy(); gt[m] = t[m] * 1000
    mo = m & (bi < len(main))
    return np.clip(rgb, 0, 255).astype(np.uint8), gt, mo, np.where(mo, t, 0.)


def _frames(spires, work, R):
    prep = SI._prepare(spires, work, R)
    return prep


def sheet(spires, work):
    import cv2
    work = Path(work); R = json.loads((work / 'insert' / 'runs.json').read_text())
    pick = [r for r in R if r['kind'] in ('bar', 'shelf', 'sign', 'table')][:8]
    prep = SI._prepare(spires, work, pick); tiles = []
    for run in pick:
        p = prep[run['id']]; k = run['run_frames'][len(run['run_frames']) // 2]
        rgb = SI.pinhole_rgb(work / 'insert' / 'img' / run['seq'] / p['names'][k][5:])
        im, *_ = composite(run, p['T'][k], rgb, p['gts'][k].astype(np.float32), np.zeros((H, W, 1)))
        cv2.putText(im, f"{run['id']} {run['kind']}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        tiles.append(cv2.resize(im, (240, 320)))
    cv2.imwrite(str(work / OUT / 'sheet.jpg'), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


def render(spires, work, variant='both'):
    import torch
    import torch.nn.functional as Fn
    import cv2
    out_dir, use_sup, use_shadow = VARIANTS[variant]
    work = Path(work); R = json.loads((work / 'insert' / 'runs.json').read_text())
    meta = dict(projection=[SI.F, SI.F, SI.CX, SI.CY], size=[W, H])
    model = SI._dav2s(); rng = np.random.default_rng(7)               # same stream as the first insertion
    prep = SI._prepare(spires, work, R); px = []
    vv, uu = np.mgrid[0:H:V.STEP, 0:W:V.STEP]
    dp = np.stack([(uu - SI.CX) / SI.F, (vv - SI.CY) / SI.F, np.ones(uu.shape)], -1)
    for run in R:
        p = prep[run['id']]; od = work / out_dir / str(run['id']); od.mkdir(parents=True, exist_ok=True)
        ims, gts, masks, tts = [], [], [], []
        for k in run['frames']:
            noise = rng.normal(0, 6, (H, W, 1)); nm = p['names'][k]
            rgb = SI.pinhole_rgb(work / 'insert' / 'img' / run['seq'] / nm[5:]) if nm else np.zeros((H, W, 3), np.uint8)
            im, gt, mo, tt = composite(run, p['T'][k], rgb, p['gts'][k].astype(np.float32), noise, use_sup, use_shadow)
            if k == run['run_frames'][len(run['run_frames']) // 2]:
                cv2.imwrite(str(od / f'{k:03d}.jpg'), im, [cv2.IMWRITE_JPEG_QUALITY, 90])
            ims.append(im); gts.append(gt); masks.append(mo); tts.append(tt)
        zs = []
        for b in range(0, len(ims), 8):
            t_ = torch.cat([model.image2tensor(im, 518)[0] for im in ims[b:b + 8]])
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                dd = model(t_)
            zs += list(Fn.interpolate(dd[:, None].float(), (H, W), mode='bilinear', align_corners=True)[:, 0].cpu().numpy())
        frames = [(k, t) for k, t in zip(run['frames'], p['tim'])]; jx = {k: k for k in run['frames']}
        X, Yg, lab, dd_ = I.inputs_and_labels(run, frames, list(p['T']), dict(Y=p['Y'], frames=np.array(run['frames'])),
                                              jx, meta, gts, zs)
        np.savez_compressed(od / 'grid.npz', X=X, Y=Yg, d=np.array([np.nan if v is None else v for v in dd_]),
                            lab=np.array(lab), frames=np.array(run['run_frames']))
        if run['kind'] in I.MISS_TYPES:                                  # main-object depth metrics on near frames
            for q, l in enumerate(lab):
                if not l.endswith('_near'):
                    continue
                k = run['run_frames'][q]; m = masks[k]
                if not m.any():
                    continue
                g2, z2 = gts[k][::V.STEP, ::V.STEP], zs[k][::V.STEP, ::V.STEP] * 1000
                mm = (g2 > 0) & (g2 < A.MAX_MM) & (z2 > 0); s = float(np.median(g2[mm] / z2[mm])) if mm.sum() > 100 else 1.
                rel = (zs[k][m] * s - tts[k][m]) / tts[k][m]
                m2 = m[::V.STEP, ::V.STEP]; T = p['T'][k]; o = T[:3, 3]; f = I.heading(T); xr = np.cross(f, A.UP)
                Pm = (dp[m2] * (zs[k][::V.STEP, ::V.STEP] * s)[m2][:, None]) @ T[:3, :3].T + o; d = Pm - o
                body = (np.abs(d @ xr) <= V.X_HALF) & (d @ f >= 0) & (d @ f <= V.NF * V.CELL) & \
                       (Pm[:, 2] - run['floor'] >= .6) & (Pm[:, 2] - run['floor'] <= 1.9)
                px.append(dict(id=run['id'], kind=run['kind'], k=k, absrel=float(np.median(np.abs(rel))), in_body=float(body.mean())))
        print('run', run['id'], run['kind'], flush=True)
    (work / out_dir / 'pixels.json').write_text(json.dumps(px), encoding='utf8')


def score(ase_work, work):
    import torch
    ase_work, work = Path(ase_work), Path(work); R = json.loads((work / 'insert' / 'runs.json').read_text())
    net = I.D.make_head(8).cuda(); net.load_state_dict(torch.load(ase_work / 'decision_head8_all.pt')); net.eval()
    dec = lambda occ: (lambda r: bool(r[0] and not r[2]))(V.decide(occ))
    rows = {}
    for tag, sub in (('v1', 'insert'), ('v2', OUT)):
        per = []
        for run in R:
            g = np.load(work / sub / str(run['id']) / 'grid.npz')
            with torch.no_grad():
                s = torch.sigmoid(net(V.pad(torch.from_numpy(g['X']).cuda().float()))).cpu().numpy()
            al = dict(D8=s > I.THETA, H=np.array([dec((x[0] > 0) | (x[1] > 0)) for x in g['X']]))
            near = [q for q, l in enumerate(g['lab']) if str(l).endswith('_near')]
            r = dict(id=run['id'], kind=run['kind'])
            for a, v in al.items():
                hit = [q for q in near if v[q]]
                r[a] = dict(hit=bool(hit), first_d=None if not hit else float(g['d'][hit[0]]))
            r['max_s'] = float(max(s[q] for q in near)) if near else None
            per.append(r)
        rows[tag] = per

    def summ(per, arm, kinds):
        rr = [r for r in per if r['kind'] in kinds]
        hit = [r for r in rr if r[arm]['hit']]
        return dict(runs=len(rr), alerted=len(hit), timely_07=sum(r[arm]['first_d'] >= .7 for r in hit),
                    timely_10=sum(r[arm]['first_d'] >= 1. for r in hit))
    rep = {}
    for tag, per in rows.items():
        rep[tag] = {a: dict(miss=summ(per, a, I.MISS_TYPES), control=summ(per, a, set(I.TYPES) - I.MISS_TYPES),
                            thin=summ(per, a, {'bar', 'shelf'})) for a in ('D8', 'H')}
        rep[tag]['by_kind_D8'] = {k: summ(per, 'D8', {k})['alerted'] for k in I.TYPES}
    rep['paired_thin_D8'] = [(a['id'], a['kind'], a['D8']['hit'], b['D8']['hit'], round(a['max_s'], 2), round(b['max_s'], 2))
                             for a, b in zip(rows['v1'], rows['v2']) if a['kind'] in ('bar', 'shelf')]
    px1 = json.loads((work / 'insert' / 'thin_diag.json').read_text())
    v1 = [q for r in px1 if r['kind'] in ('bar', 'shelf') for q in r['pixels'] if q.get('obj_px')]
    v2 = [q for q in json.loads((work / OUT / 'pixels.json').read_text()) if q['kind'] in ('bar', 'shelf')]
    rep['thin_pixels'] = dict(v1_absrel=round(float(np.median([q['absrel'] for q in v1])), 3),
                              v2_absrel=round(float(np.median([q['absrel'] for q in v2])), 3),
                              v1_in_body=round(float(np.median([q['in_body'] for q in v1])), 3),
                              v2_in_body=round(float(np.median([q['in_body'] for q in v2])), 3),
                              v1_in_body_mean=round(float(np.mean([q['in_body'] for q in v1])), 3),
                              v2_in_body_mean=round(float(np.mean([q['in_body'] for q in v2])), 3), frames=(len(v1), len(v2)))
    (work / OUT / 'score.json').write_text(json.dumps(dict(report=rep, runs=rows), indent=1), encoding='utf8')
    print(json.dumps(rep, indent=1))



def score4(ase_work, work):
    """2x2 ablation: v1 (neither), sup only, shadow only, both."""
    import torch
    ase_work, work = Path(ase_work), Path(work); R = json.loads((work / 'insert' / 'runs.json').read_text())
    net = I.D.make_head(8).cuda(); net.load_state_dict(torch.load(ase_work / 'decision_head8_all.pt')); net.eval()
    arms = dict(neither='insert', sup='insert_v2_sup', shd='insert_v2_shd', both='insert_v2')
    rep, rows = {}, {}
    for tag, sub in arms.items():
        per = []
        for run in R:
            g = np.load(work / sub / str(run['id']) / 'grid.npz')
            with torch.no_grad():
                s = torch.sigmoid(net(V.pad(torch.from_numpy(g['X']).cuda().float()))).cpu().numpy()
            near = [q for q, l in enumerate(g['lab']) if str(l).endswith('_near')]
            hit = [q for q in near if s[q] > I.THETA]
            per.append(dict(id=run['id'], kind=run['kind'], hit=bool(hit), first_d=None if not hit else float(g['d'][hit[0]]),
                            max_s=float(max(s[q] for q in near)) if near else None))
        rows[tag] = per
        def summ(kinds):
            rr = [r for r in per if r['kind'] in kinds]; h = [r for r in rr if r['hit']]
            return dict(runs=len(rr), alerted=len(h), timely_07=sum(r['first_d'] >= .7 for r in h), timely_10=sum(r['first_d'] >= 1. for r in h))
        rep[tag] = dict(miss=summ(I.MISS_TYPES), control=summ(set(I.TYPES) - I.MISS_TYPES), thin=summ({'bar', 'shelf'}),
                        **{k: summ({k}) for k in I.TYPES})
        if tag == 'neither':
            px = [dict(q, kind=r['kind']) for r in json.loads((work / 'insert' / 'thin_diag.json').read_text())
                  if r['kind'] in ('bar', 'shelf') for q in r['pixels'] if q.get('obj_px')]
        else:
            px = [q for q in json.loads((work / sub / 'pixels.json').read_text()) if q['kind'] in ('bar', 'shelf')]
        rep[tag]['thin_pixels'] = dict(absrel_median=round(float(np.median([q['absrel'] for q in px])), 3),
                                       in_body_mean=round(float(np.mean([q['in_body'] for q in px])), 3),
                                       by_kind={k: round(float(np.median([q['absrel'] for q in px if q.get('kind', None) == k] or [np.nan])), 3)
                                                for k in ('bar', 'shelf')}, frames=len(px))
    (work / 'insert_v2' / 'ablation.json').write_text(json.dumps(dict(report=rep, runs=rows), indent=1), encoding='utf8')
    for tag in arms:
        r = rep[tag]
        print(tag, 'miss', r['miss'], 'control', r['control']['alerted'], 'thin', r['thin'])
        print('   bar', r['bar'], 'shelf', r['shelf'], 'sign', r['sign'], 'table', r['table'])
        print('   pixels', r['thin_pixels'])


if __name__ == '__main__':
    dict(sheet=sheet, render=render, score=score, score4=score4)[sys.argv[1]](*sys.argv[2:])
