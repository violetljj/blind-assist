"""Fusion v3 development harness on consumed captures (EXPLORE; frozen v1/v2 untouched).

One GPU pass per batch computes, for every ready ToF frame, the frozen v1 ToF strong grades and,
for the matched RGB frame plus up to two earlier RGB frames inside the same ToF interval, the
HEAD/BODY box pixel counts under two anchor arms:
  v1     : exactly the frozen v1 chain (ToF affine fit -> floor anchor -> last fit);
  robust : v1, but a ToF fit that fails the fixed quality contract is replaced by the exhaustive
           two-point consensus fit when that one passes (robust_replace, REAL_FUSION_REFINEMENT_DEV_20261013).
Readout (CPU, `score`) combines RGB temporal rules (single / maj3 / any3) with ToF strong, optionally
the frozen v2 2-of-3 confirmation, and scores with the unchanged v1 criteria.
Usage:
  fusion_v3_dev_20261013.py cache --raw DIR --labels L --out FILE.json
  fusion_v3_dev_20261013.py score FILE.json [FILE.json ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v1 as V1
import frozen_fusion_eval_v2 as V2

EXTRA_OFFSETS_NS = (67_000_000, 133_000_000)   # ~7.5 Hz RGB, the phone HTP rate (120 ms/frame)
ANCHORS = ('v1', 'robust')


def box_pixels(disp, valid, fit, Rp, Rct, dirs):
    a, b = fit
    inv = (disp.reshape(-1) - b) / a
    zz = np.where(inv > 1e-3, 1 / np.maximum(inv, 1e-3), np.inf)
    ok = np.isfinite(zz) & valid
    pts = (Rp @ (Rct.T @ (dirs[ok] * zz[ok, None]).T)).T
    out = []
    for lo, hi in V1.F.BOXES.values():
        inside = (np.abs(pts[:, 0]) <= .30) & (pts[:, 1] >= lo) & (pts[:, 1] <= hi) & (pts[:, 2] >= .30) & (pts[:, 2] <= V1.Z_MAX)
        out.append(int(inside.sum()))
    return out


def cache(raw, labels_path, out):
    import cv2
    import torch
    from atom_imu import load_mounting
    from fusion_anchor_candidates_dev_20261013 import robust_affine, fit_quality
    S, F, R, X, PE = V1.S, V1.F, V1.R, V1.X, V1.PE
    start = time.perf_counter()
    labels = json.loads(Path(labels_path).read_text(encoding='utf8'))
    imu_rot = np.array(load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    arrays = np.load(V1.P.OUT / 'instrument_arrays.npz'); bias = arrays['bias'].reshape(64, 16)
    noise = V1.P.read('instrument_parameters.json')['noise']
    Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation'])
    cam = V1.AtomCamera(); K = cam.K_upright; rays = X.zone_rays()
    w, h = cam.upright_size
    vv, uu = np.mgrid[0:h, 0:w]
    dirs = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1).reshape(-1, 3)
    sessions = {d.name: d for d in Path(raw).iterdir() if d.name in labels}
    height, height_source = V1.DEFAULT_HEIGHT, 'default'
    for name, lab in labels.items():
        if lab['class'] == 'calibration' and name in sessions:
            fit = R.floor_fit(R.load(sessions[name], imu_rot))
            if fit and 1.0 <= fit[0] <= 1.9:
                height, height_source = fit[0], name
    engine = V1.ContractReplay(str(V1.MANIFEST)); model, _ = S.load_dav2_small()
    rng = {'v1': np.random.default_rng(20261013), 'robust': np.random.default_rng(20261013)}
    rng_extra = np.random.default_rng(7)
    report = dict(evidence='consumed Development', raw=str(raw), labels=str(labels_path), head_height_m=height,
                  head_height_source=height_source, sessions={})
    try:
        for name in sorted(sessions):
            d = sessions[name]; lab = labels[name]
            x = R.load(d, imu_rot)
            grade, ready = V1.tof_grades(engine, d, x, bias, noise)
            gated, pose_ok = PE.apply(grade, x['pitch'], x['roll'])
            tof_t = [r['received_elapsed_ns'] for r in F.lines(d / 'tof.index.jsonl') if r.get('valid_frame')]
            rgb = list(F.lines(d / 'rgb.jsonl')); rgb_t = np.array([r['received_elapsed_ns'] for r in rgb])
            dist = np.where(np.isin(x['S'], (5, 6, 9, 10)), x['D'], np.nan).reshape(-1, 8, 8)[:, ::-1, ::-1].reshape(-1, 64) / 1000
            last = {a: None for a in ANCHORS}; frames = []
            for i in np.flatnonzero(ready):
                j = int(np.argmin(np.abs(rgb_t - tof_t[i])))
                row = dict(tof_index=int(i), pitch=float(x['pitch'][i]), roll=float(x['roll'][i]), pose_ok=bool(pose_ok[i]),
                           tof_strong=[bool(gated[i, q] == 2) for q in range(2)], rgb_index=[], anchor={a: [] for a in ANCHORS},
                           px={a: [] for a in ANCHORS}, quality=None)
                if abs(rgb_t[j] - tof_t[i]) <= 1e8 and pose_ok[i]:
                    picks = [j]
                    for off in EXTRA_OFFSETS_NS:
                        k = int(np.argmin(np.abs(rgb_t - (tof_t[i] - off))))
                        if k not in picks and abs(rgb_t[k] - (tof_t[i] - off)) <= 4e7:
                            picks.append(k)
                    valid = np.isfinite(dist[i]) & (dist[i] > .3) & (dist[i] < 4.)
                    Rp = F.rot(x['pitch'][i], x['roll'][i])
                    for n, k in enumerate(picks):
                        up = cam.undistort_upright(cv2.imread(str(d / rgb[k]['file']))[:, :, ::-1])
                        with torch.inference_mode():
                            disp = model.infer_image(np.ascontiguousarray(up[:, :, ::-1]), V1.SIZE)
                        vmask = up.reshape(-1, 3).sum(1) > 0
                        dz, zfac = S.zone_disparity(disp, K, Rct, rays)
                        inv = np.where(valid, 1 / (dist[i] * zfac), np.nan)
                        fit0 = S.fit_affine(dz, inv)
                        q = fit_quality(dz, inv, fit0)
                        rb = robust_affine(dz, inv)['fit'] if not q['accepted'] else None
                        if n == 0:
                            row['quality'] = dict(accepted=q['accepted'], reasons=q['reasons'], robust=rb is not None)
                        for arm in ANCHORS:
                            fit, src = fit0, ('tof' if fit0 else None)
                            if arm == 'robust' and rb is not None:
                                fit, src = rb, 'robust_tof'
                            if fit is None:
                                fit = S.floor_anchor(disp, dirs, Rp, Rct, height, rng[arm] if n == 0 else rng_extra)
                                src = 'floor' if fit else None
                            if fit is None and last[arm] is not None:
                                fit, src = last[arm], 'last'
                            if n == 0 and fit is not None:
                                last[arm] = fit
                            row['anchor'][arm].append(src)
                            row['px'][arm].append(box_pixels(disp, vmask, fit, Rp, Rct, dirs) if fit is not None else None)
                        row['rgb_index'].append(k)
                frames.append(row)
            report['sessions'][name] = dict(cls=lab['class'], scene=lab.get('scene'), frames=frames)
            print(f"{name[17:23]} {lab['class']:11s} n={len(frames)} rgb={sum(len(f['rgb_index']) for f in frames)} t={time.perf_counter()-start:.0f}s", flush=True)
    finally:
        engine.close()
    report['wall_s'] = time.perf_counter() - start
    Path(out).write_text(json.dumps(report), encoding='utf8')


def rgb_alerts(frames, arm, rule):
    out = np.zeros((len(frames), 2), bool)
    for t, f in enumerate(frames):
        px = [p for p in f['px'][arm] if p is not None] if f['px'][arm] else []
        if not px or f['px'][arm][0] is None:
            continue
        hit = np.array([[v >= V1.MIN_PX for v in p] for p in px])
        if rule == 'single':
            out[t] = hit[0]
        elif rule == 'any3':
            out[t] = hit.any(0)
        elif rule == 'maj3':
            out[t] = hit.sum(0) >= (len(hit) // 2 + 1)
    return out


def session_score(cls, alert):
    n = len(alert)
    if n == 0:
        return dict(ready_frames=0, passed=None)
    anyq = alert.any(1)
    res = dict(ready_frames=n, fused_any_alert=float(anyq.mean()), end_window_any_alert=float(anyq[-V1.END_WINDOW:].mean()),
               end_window_head_alert=float(alert[-V1.END_WINDOW:, 0].mean()),
               first_alert=int(np.argmax(anyq)) if anyq.any() else None)
    C = V1.CRITERIA
    res['pass'] = (None if cls in ('calibration', 'unknown') else
                   res['fused_any_alert'] <= C['quiet_max_alert'] if cls == 'quiet' else
                   res['fused_any_alert'] <= C['side_max_alert'] if cls == 'side' else
                   res['end_window_any_alert'] >= C['ahead_min_end_alert'] if cls == 'ahead' else
                   res['end_window_head_alert'] >= C['head_min_end_alert'] if cls == 'head' else None)
    return res


def fused(frames, arm, rule, confirm):
    tof = np.array([f['tof_strong'] for f in frames], bool).reshape(-1, 2)
    a = tof | rgb_alerts(frames, arm, rule)
    if confirm:
        a = np.stack([V2.confirm(a[:, 0]), V2.confirm(a[:, 1])], 1) if len(a) else a
    return a


def score(paths):
    combos = [(arm, rule, c) for arm in ANCHORS for rule in ('single', 'maj3', 'any3') for c in (False, True)]
    table = {}
    for p in paths:
        rep = json.loads(Path(p).read_text(encoding='utf8'))
        for name, s in rep['sessions'].items():
            for combo in combos:
                r = session_score(s['cls'], fused(s['frames'], *combo))
                table.setdefault(name, dict(cls=s['cls'], scene=s['scene']))['/'.join(map(str, combo))] = r
    return table


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('cache'); c.add_argument('--raw', required=True); c.add_argument('--labels', required=True); c.add_argument('--out', required=True)
    s = sub.add_parser('score'); s.add_argument('files', nargs='+'); s.add_argument('--out')
    a = ap.parse_args()
    if a.cmd == 'cache':
        cache(a.raw, a.labels, a.out)
    else:
        t = score(a.files)
        if a.out:
            Path(a.out).write_text(json.dumps(t, indent=1), encoding='utf8')
        keys = ['v1/single/False', 'v1/single/True', 'robust/single/True', 'robust/maj3/True', 'robust/any3/True']
        print(f"{'session':8s} {'class':11s} " + ' | '.join(k[:18].ljust(18) for k in keys))
        for name, r in t.items():
            cells = []
            for k in keys:
                v = r[k]
                cells.append((f"{v['fused_any_alert']:.2f}/{v['end_window_any_alert']:.2f}/{str(v['pass'])[0]}" if v['ready_frames'] else '-').ljust(18))
            print(f"{name[17:23]:8s} {r['cls']:11s} " + ' | '.join(cells))
