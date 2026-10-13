"""Real consumed-capture check for the learned anchor pilot (EXPLORE; plan in learned-anchor-pilot-20261013/PLAN.md).

Per ready frame, one fp16 DAV2-S pass feeds three anchor chains:
  v3 : ToF fit -> robust replacement -> floor anchor -> last fit (as frozen v3)
  A  : ToF fit -> robust replacement -> LEARNED scale -> last fit (learned replaces the floor anchor)
  B  : LEARNED scale -> last fit
Scored with the v1 criteria and the v2 2-of-3 confirmation; first-half alert rate reported for the corridor check.
Usage: learned_anchor_real_dev_20261013.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from atom_camera import ROOT
import learned_anchor_pilot_dev_20261013 as L

BATCHES = {
    'm12': ('artifacts.local/work/phone-imu-capture-20261012/raw', 'artifacts.local/work/fusion-v3-dev-20261013/labels_morning_dev.json'),
    'a12': ('artifacts.local/work/rgb-real-feasibility-20261012/raw', 'artifacts.local/work/fusion-v3-dev-20261013/labels_afternoon_dev.json'),
    'm13': ('artifacts.local/work/frozen-eval-capture-20261013/raw', 'research/active/hardware-bringup/FROZEN_EVAL_LABELS_20261013.json'),
    'a13': ('artifacts.local/work/frozen-eval-capture-20261013b/raw', 'research/active/hardware-bringup/FROZEN_EVAL_LABELS_20261013B.json')}
ARMS = ('v3', 'A', 'B')


def zone_boxes(K, Rct, rays, h, w, hs=256, ws=192):
    rc = rays @ Rct.T
    u = K[0, 0] * rc[..., 0] / rc[..., 2] + K[0, 2]; v = K[1, 1] * rc[..., 1] / rc[..., 2] + K[1, 2]
    boxes = np.zeros((64, 4), int)
    for z in range(64):
        ins = (rc[z, :, 2] > .05) & (u[z] >= 0) & (u[z] < w) & (v[z] >= 0) & (v[z] < h)
        if ins.sum() >= 13:
            boxes[z] = (int(v[z][ins].min() * hs / h), int(u[z][ins].min() * ws / w),
                        int(np.ceil(v[z][ins].max() * hs / h)), int(np.ceil(u[z][ins].max() * ws / w)))
    return boxes


def main():
    import cv2
    import torch
    import torch.nn.functional as Fn
    from atom_imu import load_mounting
    import frozen_fusion_eval_v1 as V1
    import frozen_fusion_eval_v2 as V2
    import fusion_v3_dev_20261013 as D
    from fusion_anchor_candidates_dev_20261013 import fit_quality, robust_affine
    S, F, R, X, PE = V1.S, V1.F, V1.R, V1.X, V1.PE
    net = L.load_net(); model, _ = S.load_dav2_small()
    imu_rot = np.array(load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    bias = np.load(V1.P.OUT / 'instrument_arrays.npz')['bias'].reshape(64, 16); noise = V1.P.read('instrument_parameters.json')['noise']
    Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation'])
    cam = V1.AtomCamera(); K = cam.K_upright; rays = X.zone_rays(); w, h = cam.upright_size
    vv, uu = np.mgrid[0:h, 0:w]
    dirs = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1).reshape(-1, 3)
    boxes = zone_boxes(K, Rct, rays, h, w)
    engine = V1.ContractReplay(str(V1.MANIFEST)); out = {}; start = time.perf_counter()
    try:
        for bname, (raw, lab) in BATCHES.items():
            raw, labels = ROOT / raw, json.loads((ROOT / lab).read_text(encoding='utf8'))
            sessions = {d.name: d for d in raw.iterdir() if d.name in labels}
            height = V1.DEFAULT_HEIGHT
            for name, l in labels.items():
                if l['class'] == 'calibration' and name in sessions:
                    fit = R.floor_fit(R.load(sessions[name], imu_rot))
                    if fit and 1.0 <= fit[0] <= 1.9:
                        height = fit[0]
            rng = np.random.default_rng(20261013)
            for name in sorted(sessions):
                d = sessions[name]; x = R.load(d, imu_rot)
                grade, ready = V1.tof_grades(engine, d, x, bias, noise)
                gated, pose_ok = PE.apply(grade, x['pitch'], x['roll'])
                tof_t = [r['received_elapsed_ns'] for r in F.lines(d / 'tof.index.jsonl') if r.get('valid_frame')]
                rgb = list(F.lines(d / 'rgb.jsonl')); rgb_t = np.array([r['received_elapsed_ns'] for r in rgb])
                dist = np.where(np.isin(x['S'], (5, 6, 9, 10)), x['D'], np.nan).reshape(-1, 8, 8)[:, ::-1, ::-1].reshape(-1, 64) / 1000
                last = {k: None for k in ARMS}; rows = []
                for i in np.flatnonzero(ready):
                    j = int(np.argmin(np.abs(rgb_t - tof_t[i]))); alerts = {k: [False, False] for k in ARMS}; src = {}
                    if abs(rgb_t[j] - tof_t[i]) <= 1e8 and pose_ok[i]:
                        up = cam.undistort_upright(cv2.imread(str(d / rgb[j]['file']))[:, :, ::-1])
                        t, _ = model.image2tensor(np.ascontiguousarray(up[:, :, ::-1]), V1.SIZE)
                        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                            dd = model(t)
                        disp = Fn.interpolate(dd[:, None].float(), (h, w), mode='bilinear', align_corners=True)[0, 0].cpu().numpy()
                        dz, zfac = S.zone_disparity(disp, K, Rct, rays)
                        valid = np.isfinite(dist[i]) & (dist[i] > .3) & (dist[i] < 4.)
                        inv = np.where(valid, 1 / (dist[i] * zfac), np.nan)
                        tf = S.fit_affine(dz, inv); ts = 'tof' if tf else None
                        if not fit_quality(dz, inv, tf)['accepted']:
                            rb = robust_affine(dz, inv)['fit']
                            if rb is not None:
                                tf, ts = rb, 'robust'
                        small = cv2.resize(disp, (192, 256), interpolation=cv2.INTER_AREA)
                        lf = L.predict(net, small, np.where(valid, dist[i] * zfac, np.nan), boxes)
                        Rp = F.rot(x['pitch'][i], x['roll'][i]); vmask = up.reshape(-1, 3).sum(1) > 0
                        for k in ARMS:
                            fit, s_ = (tf, ts) if k != 'B' else (None, None)
                            if fit is None and k == 'v3':
                                fit = S.floor_anchor(disp, dirs, Rp, Rct, height, rng); s_ = 'floor' if fit else None
                            if fit is None and k in ('A', 'B'):
                                fit, s_ = lf, ('learned' if lf else None)
                            if fit is None and last[k] is not None:
                                fit, s_ = last[k], 'last'
                            if fit is not None:
                                last[k] = fit
                                alerts[k] = [n >= V1.MIN_PX for n in D.box_pixels(disp, vmask, fit, Rp, Rct, dirs)]
                            src[k] = s_
                    rows.append(dict(tof=[bool(gated[i, q] == 2) for q in range(2)], rgb=alerts, src=src))
                cls = labels[name]['class']; res = dict(cls=cls, ready=len(rows))
                for k in ARMS:
                    a = np.array([[r['tof'][q] or r['rgb'][k][q] for q in range(2)] for r in rows], bool).reshape(-1, 2)
                    a = np.stack([V2.confirm(a[:, 0]), V2.confirm(a[:, 1])], 1) if len(a) else a
                    sc = D.session_score(cls, a); half = len(a) // 2
                    sc['first_half'] = float(a[:half].any(1).mean()) if half else None
                    res[k] = sc
                res['sources_A'] = {str(s): sum(r['src'].get('A') == s for r in rows) for s in ('tof', 'robust', 'learned', 'last', None)}
                out[f'{bname}/{name[17:23]}'] = res
                print(bname, name[17:23], cls[:7], ' | '.join(
                    f"{k} {res[k].get('fused_any_alert') or 0:.2f}/{res[k].get('end_window_any_alert') or 0:.2f}/{str(res[k].get('pass'))[0]} fh {res[k].get('first_half') or 0:.2f}"
                    for k in ARMS), f'{time.perf_counter()-start:.0f}s', flush=True)
    finally:
        engine.close()
    (L.OUT / 'real.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
