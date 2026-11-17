"""Overhangs without ground support: hung from thin cables (hang) or cantilevered from a side wall (wall), paired with
the first Spires insertion and the supports-only arm (same runs, sizes, colours, noise stream). EXPLORE, 2026-11-17.
Plan: artifacts.local/work/spires-cane-reach-20261117/PLAN_INSERT_NOSUP.md

  python spires_insert_nosup_dev_20261117.py sheet  <spires> <work> hang|wall
  python spires_insert_nosup_dev_20261117.py render <spires> <work> hang|wall
  python spires_insert_nosup_dev_20261117.py score  <ase_work> <work>

Tables lose their legs in both arms. Labels and the G grid use the main object only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import spires_insert_dev_20261117 as SI  # noqa: E402
import spires_insert_v2_dev_20261117 as V2  # noqa: E402
from spires_insert_dev_20261117 import A, V, I, W, H  # noqa: E402

CABLE, WALLC = (60., 60., 60.), (150., 160., 170.)                     # BGR
OUT = dict(hang='insert_ns_hang', wall='insert_ns_wall', wallonly='insert_ns_wallonly',   # wallonly: post-hoc check
           empty='insert_ns_empty', twallonly='insert_ns_twallonly', twall='insert_ns_twall')  # recovery (plan addendum)
NOMAIN = ('wallonly', 'empty', 'twallonly')                           # no object drawn; cane-unreachable runs only
BRICK = 'brick'
_BR = np.random.default_rng(2026).uniform(-1, 1, (64, 64, 3))


def brick(u, v):
    """Procedural brick (21.5 x 6.5 cm bricks, 1 cm mortar) at wall-plane coordinates u (along) and v (up), metres."""
    row = np.floor(v / .075).astype(int); uu = u + (row % 2) * .1125; col = np.floor(uu / .225).astype(int)
    mortar = ((v - row * .075) < .01) | ((uu - col * .225) < .01)
    base = np.array([60., 80., 150.]) + _BR[row % 64, col % 64] * np.array([12., 15., 25.])
    return np.where(mortar[..., None], np.array([165., 172., 178.]), base)
TOPZ = 6.0


def main_run(run):
    """Tables keep only the top (legs are ground support)."""
    return dict(run, boxes=run['boxes'][:1]) if run['kind'] == 'table' else run


def wall_sides(R):
    rng = np.random.default_rng(1117)
    return {r['id']: float(rng.choice([-1., 1.])) for r in R}


def extras(run, arm, side):
    """Extra boxes in the object frame (a along walking dir from the near face, b lateral, z above floor)."""
    k, (c, h) = run['kind'], run['boxes'][0]
    top = c[2] + h[2]
    if arm == 'hang':
        if k not in I.MISS_TYPES:
            return [], []
        bs = dict(bar=.6, sign=.25, shelf=.25, table=.45)[k]
        cab = [((c[0], s * bs, (top + TOPZ) / 2), (.006, .006, (TOPZ - top) / 2)) for s in (-1, 1)]
        return cab, [CABLE] * 2
    s = side; wc = BRICK if arm.startswith('t') else WALLC
    if arm == 'empty':
        return [], []
    if arm in NOMAIN:
        return [((.25, s * .85, 1.25), (1.75, .05, 1.25))], [wc]
    out = [((.25, s * .85, 1.25), (1.75, .05, 1.25))]; col = [wc]
    if k in I.MISS_TYPES and k != 'bar':                               # the bar already reaches 0.80 m
        b0 = h[1]; zc = top - .02 if k == 'sign' else c[2]
        out.append(((c[0], s * (b0 + .80) / 2, zc), (.02, (.80 - b0) / 2, .02))); col.append(CABLE)
    return out, col


def composite(run, T, rgb, gt, noise, ex, ecol, draw_main=True):
    main = I.world_boxes(run) if draw_main else []; sup = I.world_boxes(dict(run, boxes=ex)) if ex else []
    boxes = main + sup; cols = [np.array(I.COLORS[run['kind']], float)] * len(main) + [np.array(WALLC if c == BRICK else c) for c in ecol]
    if not boxes:                                                       # empty baseline: the frame unchanged (noise is only drawn on inserted pixels)
        return rgb.astype(np.uint8), gt, np.zeros((H, W), bool), np.zeros((H, W))
    Lg = np.array([.4, .3, .86]); Lg /= np.linalg.norm(Lg)
    rgb = rgb.astype(np.float32)
    t, n, bi = V2.raycast_idx(T, (SI.F, SI.F, SI.CX, SI.CY), boxes)
    m = np.isfinite(t) & ((gt == 0) | (t * 1000 < gt)) & SI.undistort_map()[2]
    colmap = np.stack(cols)[np.clip(bi, 0, len(cols) - 1)]
    for j, c in enumerate(ecol):
        if c == BRICK:
            q = m & (bi == len(main) + j); fx, fy, cx, cy = SI.F, SI.F, SI.CX, SI.CY
            vv, uu = np.nonzero(q); dw = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones(len(uu))], -1) @ T[:3, :3].T
            cB, AxB, hB = boxes[len(main) + j]; loc = (T[:3, 3] + dw * t[q][:, None] - cB) @ AxB
            colmap[q] = brick(loc[:, 0] + hB[0], loc[:, 2] + hB[2])
    shade = (.35 + .65 * np.clip(n @ Lg, 0, 1))[..., None] * colmap + noise
    rgb[m] = shade[m]; gt = gt.copy(); gt[m] = t[m] * 1000
    mo = m & (bi < len(main))
    return np.clip(rgb, 0, 255).astype(np.uint8), gt, mo, np.where(mo, t, 0.)


def sheet(spires, work, arm):
    import cv2
    work = Path(work); R = json.loads((work / 'insert' / 'runs.json').read_text()); sides = wall_sides(R)
    pick = [r for r in R if r['kind'] in I.MISS_TYPES][:6] + [r for r in R if r['kind'] not in I.MISS_TYPES][:2]
    prep = SI._prepare(spires, work, pick); tiles = []
    for run0 in pick:
        run = main_run(run0); p = prep[run['id']]; ex, ec = extras(run, arm, sides[run['id']])
        for k in (run['run_frames'][0], run['run_frames'][-1]):
            rgb = SI.pinhole_rgb(work / 'insert' / 'img' / run['seq'] / p['names'][k][5:])
            im, *_ = composite(run, p['T'][k], rgb, p['gts'][k].astype(np.float32), np.zeros((H, W, 1)), ex, ec)
            cv2.putText(im, f"{run['id']} {run['kind']}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            tiles.append(cv2.resize(im, (240, 320)))
    (work / OUT[arm]).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(work / OUT[arm] / 'sheet.jpg'), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


def render(spires, work, arm):
    import torch
    import torch.nn.functional as Fn
    import cv2
    work = Path(work); R = json.loads((work / 'insert' / 'runs.json').read_text()); sides = wall_sides(R)
    out_dir = OUT[arm]; meta = dict(projection=[SI.F, SI.F, SI.CX, SI.CY], size=[W, H])
    model = SI._dav2s(); rng = np.random.default_rng(7)               # same noise stream as the other arms
    prep = SI._prepare(spires, work, R); px = []
    vv, uu = np.mgrid[0:H:V.STEP, 0:W:V.STEP]
    dp = np.stack([(uu - SI.CX) / SI.F, (vv - SI.CY) / SI.F, np.ones(uu.shape)], -1)
    for run0 in R:
        run = main_run(run0); ex, ec = extras(run, arm, sides[run['id']])
        if arm in NOMAIN and run['kind'] not in I.MISS_TYPES:
            for _ in run['frames']:
                rng.normal(0, 6, (H, W, 1))                            # keep the noise stream paired
            continue
        p = prep[run['id']]; od = work / out_dir / str(run['id']); od.mkdir(parents=True, exist_ok=True)
        ims, gts, masks, tts = [], [], [], []
        for k in run['frames']:
            noise = rng.normal(0, 6, (H, W, 1)); nm = p['names'][k]
            rgb = SI.pinhole_rgb(work / 'insert' / 'img' / run['seq'] / nm[5:]) if nm else np.zeros((H, W, 3), np.uint8)
            im, gt, mo, tt = composite(run, p['T'][k], rgb, p['gts'][k].astype(np.float32), noise, ex, ec, arm not in NOMAIN)
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
        if run['kind'] in I.MISS_TYPES:
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
    arms = dict(neither='insert', sup='insert_v2_sup', hang=OUT['hang'], wall=OUT['wall'], twall=OUT['twall'])
    rep, rows, labs = {}, {}, {}
    for tag, sub in arms.items():
        per = []
        for run in R:
            g = np.load(work / sub / str(run['id']) / 'grid.npz')
            with torch.no_grad():
                s = torch.sigmoid(net(V.pad(torch.from_numpy(g['X']).cuda().float()))).cpu().numpy()
            near = [q for q, l in enumerate(g['lab']) if str(l).endswith('_near')]
            hit = [q for q in near if s[q] > I.THETA]
            labs.setdefault(run['id'], {})[tag] = [str(x) for x in g['lab']]
            per.append(dict(id=run['id'], kind=run['kind'], hit=bool(hit), first_d=None if not hit else float(g['d'][hit[0]]),
                            max_s=float(max(s[q] for q in near)) if near else None, n_near=len(near)))
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
                                       in_body_mean=round(float(np.mean([q['in_body'] for q in px])), 3), frames=len(px))
    # label check: near labels of the non-table cane-unreachable runs must match v1
    rep['label_mismatch'] = [(i, t) for i, d in labs.items() for t in ('hang', 'wall', 'twall')
                             if [x.endswith('_near') for x in d[t]] != [x.endswith('_near') for x in d['neither']]]
    rep['sign_first_d'] = {t: [r['first_d'] for r in rows[t] if r['kind'] == 'sign'] for t in arms}
    rep['table_first_d'] = {t: [r['first_d'] for r in rows[t] if r['kind'] == 'table'] for t in arms}
    (work / OUT['hang'] / 'nosup.json').write_text(json.dumps(dict(report=rep, runs=rows), indent=1), encoding='utf8')
    for tag in arms:
        r = rep[tag]
        print(tag, 'miss', r['miss'], 'control', r['control']['alerted'], 'thin', r['thin'])
        print('   bar', r['bar'], 'shelf', r['shelf'], 'sign', r['sign'], 'table', r['table'])
        print('   pixels', r['thin_pixels'])
    print('label_mismatch', rep['label_mismatch'])
    print('sign_first_d', rep['sign_first_d'])
    print('table_first_d', rep['table_first_d'])


def score_wallonly(ase_work, work, arm='wallonly'):
    """Post-hoc: the wall without the object on the 24 cane-unreachable runs; any D8 alert on near frames is wall-driven."""
    import torch
    ase_work, work = Path(ase_work), Path(work); R = json.loads((work / 'insert' / 'runs.json').read_text())
    net = I.D.make_head(8).cuda(); net.load_state_dict(torch.load(ase_work / 'decision_head8_all.pt')); net.eval()
    per = []
    for run in R:
        if run['kind'] not in I.MISS_TYPES:
            continue
        g = np.load(work / OUT[arm] / str(run['id']) / 'grid.npz')
        with torch.no_grad():
            s = torch.sigmoid(net(V.pad(torch.from_numpy(g['X']).cuda().float()))).cpu().numpy()
        near = [q for q, l in enumerate(g['lab']) if str(l).endswith('_near')]
        hit = [q for q in near if s[q] > I.THETA]
        per.append(dict(id=run['id'], kind=run['kind'], hit=bool(hit), first_d=None if not hit else float(g['d'][hit[0]]),
                        max_s=round(float(max(s[q] for q in near)), 3), alert_frac_all=round(float((s > I.THETA).mean()), 3)))
    rep = dict(alerted=sum(r['hit'] for r in per), runs=len(per),
               by_kind={k: sum(r['hit'] for r in per if r['kind'] == k) for k in I.MISS_TYPES})
    (work / OUT[arm] / 'score.json').write_text(json.dumps(dict(report=rep, runs=per), indent=1), encoding='utf8')
    print(json.dumps(rep)); [print(r) for r in per]


if __name__ == '__main__':
    dict(sheet=sheet, render=render, score=score, score_wallonly=score_wallonly)[sys.argv[1]](*sys.argv[2:])
