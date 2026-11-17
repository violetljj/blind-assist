"""Why thin bars/shelves are missed on real-image insertion (EXPLORE, 2026-11-17).
Plan: artifacts.local/work/spires-cane-reach-20261117/PLAN_THIN.md

  python spires_thin_diag_dev_20261117.py run <spires> <work> <ase_work>

Replays the composites of spires_insert_dev_20261117.render exactly (same random stream), then for the 24 overhang
runs: object-pixel depth metrics on near frames, and D8 with three depth inputs (as-is, object pixels -> true depth,
all valid pixels -> true depth).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import spires_insert_dev_20261117 as SI  # noqa: E402
from spires_insert_dev_20261117 import A, V, I, W, H  # noqa: E402

ARMS = ('asis', 'obj_true', 'all_true')


def run(spires, work, ase_work):
    import torch
    import torch.nn.functional as Fn
    work = Path(work); R = json.loads((work / 'insert' / 'runs.json').read_text())
    miss = [r for r in R if r['kind'] in I.MISS_TYPES]
    meta = dict(projection=[SI.F, SI.F, SI.CX, SI.CY], size=[W, H])
    Lg = np.array([.4, .3, .86]); Lg /= np.linalg.norm(Lg)
    model = SI._dav2s(); rng = np.random.default_rng(7)
    net = I.D.make_head(8).cuda(); net.load_state_dict(torch.load(Path(ase_work) / 'decision_head8_all.pt')); net.eval()
    prep = SI._prepare(spires, work, miss)
    valid = SI.undistort_map()[2]
    vv, uu = np.mgrid[0:H:V.STEP, 0:W:V.STEP]
    dp = np.stack([(uu - SI.CX) / SI.F, (vv - SI.CY) / SI.F, np.ones(uu.shape)], -1)
    out = []
    for run_ in R:
        todo = run_['kind'] in I.MISS_TYPES
        p = prep.get(run_['id']) if todo else None
        boxes = I.world_boxes(run_); col = np.array(I.COLORS[run_['kind']], float)
        ims, gts, masks, tts = [], [], [], []
        for k in run_['frames']:
            noise = rng.normal(0, 6, (H, W, 1))                          # identical draw order to render()
            if not todo:
                continue
            nm = p['names'][k]
            rgb = SI.pinhole_rgb(work / 'insert' / 'img' / run_['seq'] / nm[5:]) if nm else np.zeros((H, W, 3), np.uint8)
            gt = p['gts'][k].astype(np.float32).copy()
            t, n = I.raycast(p['T'][k], meta['projection'], H, W, boxes)
            m = np.isfinite(t) & ((gt == 0) | (t * 1000 < gt)) & valid
            shade = (.35 + .65 * np.clip(n @ Lg, 0, 1))[..., None] * col + noise
            rgb[m] = np.clip(shade[m], 0, 255).astype(np.uint8); gt[m] = t[m] * 1000
            ims.append(rgb); gts.append(gt); masks.append(m); tts.append(np.where(m, t, 0.))
        if not todo:
            continue
        zs = []
        for b in range(0, len(ims), 8):
            t_ = torch.cat([model.image2tensor(im, 518)[0] for im in ims[b:b + 8]])
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                dd = model(t_)
            zs += list(Fn.interpolate(dd[:, None].float(), (H, W), mode='bilinear', align_corners=True)[:, 0].cpu().numpy())
        frames = [(k, t) for k, t in zip(run_['frames'], p['tim'])]; jx = {k: k for k in run_['frames']}
        T_up = list(p['T']); Z = dict(Y=p['Y'], frames=np.array(run_['frames']))
        scales = []
        for gt, z in zip(gts, zs):
            g2, z2 = gt[::V.STEP, ::V.STEP], z[::V.STEP, ::V.STEP] * 1000
            mm = (g2 > 0) & (g2 < A.MAX_MM) & (z2 > 0)
            scales.append(float(np.median(g2[mm] / z2[mm])) if mm.sum() > 100 else 1.)
        variants = dict(asis=zs,
                        obj_true=[np.where(m, tt / s, z) for z, m, tt, s in zip(zs, masks, tts, scales)],
                        all_true=[np.where(gt > 0, gt / 1000., z) for z, gt in zip(zs, gts)])
        rec = dict(id=run_['id'], kind=run_['kind'], seq=run_['seq'])
        lab = None
        for arm, zz in variants.items():
            X, _, lab_, d_ = I.inputs_and_labels(run_, frames, T_up, Z, jx, meta, gts, zz)
            with torch.no_grad():
                s = torch.sigmoid(net(V.pad(torch.from_numpy(X).cuda().float()))).cpu().numpy()
            if lab is None:
                lab, dd = lab_, d_
            near = [q for q, l in enumerate(lab) if l.endswith('_near')]
            hit = [q for q in near if s[q] > I.THETA]
            hdet = sum(bool(V.decide((X[q][0] > 0) | (X[q][1] > 0))[0]) for q in near)
            rec[arm] = dict(caught=bool(hit), first_d=None if not hit else round(dd[hit[0]], 2),
                            max_s=round(float(max(s[q] for q in near)), 3) if near else None, H_det_frames=hdet, near=len(near))
        # pixel metrics on near frames (as-is depth)
        near_fr = [run_['run_frames'][q] for q, l in enumerate(lab) if l.endswith('_near')]
        px = []
        for k in near_fr:
            m, tt, z, s, T = masks[k], tts[k], zs[k], scales[k], T_up[k]
            if m.sum() == 0:
                px.append(dict(k=k, obj_px=0)); continue
            zm = z[m] * s; tr = tt[m]; rel = (zm - tr) / tr
            m2 = m[::V.STEP, ::V.STEP]; zz2 = (zs[k][::V.STEP, ::V.STEP] * s)[m2]; tt2 = tt[::V.STEP, ::V.STEP][m2]
            o = T[:3, 3]; f = I.heading(T); xr = np.cross(f, A.UP)
            Pm = (dp[m2] * zz2[:, None]) @ T[:3, :3].T + o; Pt = (dp[m2] * tt2[:, None]) @ T[:3, :3].T + o
            dm, dt = Pm - o, Pt - o
            fw_m, la_m, h_m = dm @ f, dm @ xr, Pm[:, 2] - run_['floor']; fw_t, h_t = dt @ f, Pt[:, 2] - run_['floor']
            body = (np.abs(la_m) <= V.X_HALF) & (fw_m >= 0) & (fw_m <= V.NF * V.CELL) & (h_m >= .60) & (h_m <= 1.90)
            px.append(dict(k=k, obj_px=int(m.sum()), absrel=round(float(np.median(np.abs(rel))), 3),
                           within20=round(float((np.abs(rel) <= .2).mean()), 3), farther=round(float((rel > 0).mean()), 3),
                           in_body=round(float(body.mean()), 3), dfwd=round(float(np.median(fw_m - fw_t)), 3),
                           dh=round(float(np.median(h_m - h_t)), 3)))
        rec['pixels'] = px
        out.append(rec); print(run_['id'], run_['kind'], {a: rec[a]['caught'] for a in ARMS}, flush=True)
    (work / 'insert' / 'thin_diag.json').write_text(json.dumps(out, indent=1), encoding='utf8')
    summary(work)


def summary(work):
    out = json.loads((Path(work) / 'insert' / 'thin_diag.json').read_text())
    rep = {}
    for grp, kinds in (('thin', {'bar', 'shelf'}), ('table_sign', {'table', 'sign'})):
        rr = [r for r in out if r['kind'] in kinds]
        rep[grp] = {a: sum(r[a]['caught'] for r in rr) for a in ARMS} | dict(runs=len(rr))
        px = [q for r in rr for q in r['pixels'] if q.get('obj_px')]
        for key in ('obj_px', 'absrel', 'within20', 'farther', 'in_body', 'dfwd', 'dh'):
            rep[grp][key + '_median'] = round(float(np.median([q[key] for q in px])), 3) if px else None
        rep[grp]['near_frames_with_object_visible'] = f"{len(px)}/{sum(len(r['pixels']) for r in rr)}"
    for kind in ('bar', 'shelf', 'table', 'sign'):
        rr = [r for r in out if r['kind'] == kind]
        rep[kind] = {a: sum(r[a]['caught'] for r in rr) for a in ARMS}
    (Path(work) / 'insert' / 'thin_diag_summary.json').write_text(json.dumps(rep, indent=1), encoding='utf8')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    {'run': run, 'summary': lambda *a: summary(a[1])}[sys.argv[1]](*sys.argv[2:])
