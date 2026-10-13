"""Fusion evaluation v6 = v3 + blob-depth rule for RGB alerts (protocol FROZEN_FUSION_EVAL_V6).

v6 changes only how RGB in-box pixels are counted: the in-box pixel mask of each query is split into 8-connected
image blobs; blobs under MIN_BLOB_PX are ignored and a blob counts only if the median depth z of its pixels is
<= BLOB_MAX_MEDIAN_Z (1.8 m); the query alerts if the counted pixels reach 150. Selected post hoc on consumed data
(BOX_BLOB_RULE_DEV_20261013): corridor side-wall/sign false alarms sit at the far box corner (median z ~1.9 m).
v3 description follows.
v3 = v2 + robust ToF anchor replacement (protocol FROZEN_FUSION_EVAL_V3).

Frozen 2026-10-13 before the new head-worn captures (see ../FROZEN_FUSION_EVAL_V3_PROTOCOL.md).
Identical to v1 (models, calibration, boxes, 150 px, pose gate, ToF strong, labels, criteria) except:
  * when the per-frame ToF affine fit fails the fixed quality contract
    (fusion_anchor_candidates_dev_20261013.QUALITY), the exhaustive two-point consensus fit replaces it
    if that fit passes the same contract; otherwise the v1 chain (fit -> floor anchor -> last fit) is kept;
  * the fused per-query alert is confirmed by the v2 rule (2 of the last 3 ready frames).
Usage: frozen_fusion_eval_v6.py --raw <dir of capture-* sessions> --labels <labels.json> --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v1 as V1
import frozen_fusion_eval_v2 as V2
import fusion_anchor_candidates_dev_20261013 as A

FROZEN_FILES = V1.FROZEN_FILES + [Path(A.__file__), Path(__file__)]
BLOB_MAX_MEDIAN_Z, MIN_BLOB_PX = 1.8, 10


def main(raw, labels_path, out):
    import cv2
    from atom_imu import load_mounting
    S, F, R, X, PE = V1.S, V1.F, V1.R, V1.X, V1.PE
    labels = json.loads(Path(labels_path).read_text(encoding='utf8'))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    imu_rot = np.array(load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    arrays = np.load(V1.P.OUT / 'instrument_arrays.npz'); bias = arrays['bias'].reshape(64, 16)
    noise = V1.P.read('instrument_parameters.json')['noise']
    Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation'])
    cam = V1.AtomCamera(); K = cam.K_upright; rays = X.zone_rays()
    h, w = cam.upright_size[1], cam.upright_size[0]
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
    rng = np.random.default_rng(20261013)
    report = dict(protocol='FROZEN_FUSION_EVAL_V6', blob_rule=dict(max_median_z_m=BLOB_MAX_MEDIAN_Z, min_blob_px=MIN_BLOB_PX), head_height_m=height, head_height_source=height_source,
                  frozen_sha256={str(Path(os.path.abspath(p)).relative_to(V1.ROOT)): V1.sha(p) for p in FROZEN_FILES},
                  quality_contract=A.QUALITY, confirm=f'{V2.K} of last {V2.N} ready frames per query', criteria=V1.CRITERIA, sessions={})
    try:
        for name in sorted(sessions):
            lab = labels[name]; d = sessions[name]
            x = R.load(d, imu_rot)
            grade, ready = V1.tof_grades(engine, d, x, bias, noise)
            gated, pose_ok = PE.apply(grade, x['pitch'], x['roll'])
            tof_t = [r['received_elapsed_ns'] for r in F.lines(d / 'tof.index.jsonl') if r.get('valid_frame')]
            rgb = list(F.lines(d / 'rgb.jsonl')); rgb_t = np.array([r['received_elapsed_ns'] for r in rgb])
            dist = np.where(np.isin(x['S'], (5, 6, 9, 10)), x['D'], np.nan).reshape(-1, 8, 8)[:, ::-1, ::-1].reshape(-1, 64) / 1000
            last = None; frames = []
            for i in np.flatnonzero(ready):
                j = int(np.argmin(np.abs(rgb_t - tof_t[i])))
                rgb_alert = [False, False]; anchor = None
                if abs(rgb_t[j] - tof_t[i]) <= 1e8 and pose_ok[i]:
                    up = cam.undistort_upright(cv2.imread(str(d / rgb[j]['file']))[:, :, ::-1])
                    disp = model.infer_image(np.ascontiguousarray(up[:, :, ::-1]), V1.SIZE)
                    dz, zfac = S.zone_disparity(disp, K, Rct, rays)
                    valid = np.isfinite(dist[i]) & (dist[i] > .3) & (dist[i] < 4.)
                    inv = np.where(valid, 1 / (dist[i] * zfac), np.nan)
                    fit = S.fit_affine(dz, inv); anchor = 'tof' if fit else None
                    if not A.fit_quality(dz, inv, fit)['accepted']:
                        robust = A.robust_affine(dz, inv)['fit']
                        if robust is not None:
                            fit, anchor = robust, 'robust_tof'
                    if fit is None:
                        fit = S.floor_anchor(disp, dirs, F.rot(x['pitch'][i], x['roll'][i]), Rct, height, rng); anchor = 'floor' if fit else None
                    if fit is None and last is not None:
                        fit, anchor = last, 'last'
                    if fit is not None:
                        last = fit; a, b = fit
                        inv_d = (disp.reshape(-1) - b) / a
                        zz = np.where(inv_d > 1e-3, 1 / np.maximum(inv_d, 1e-3), np.inf)
                        ok = np.isfinite(zz) & (up.reshape(-1, 3).sum(1) > 0)
                        pts = (F.rot(x['pitch'][i], x['roll'][i]) @ (Rct.T @ (dirs[ok] * zz[ok, None]).T)).T
                        for q, (lo, hi) in enumerate(F.BOXES.values()):
                            inside = (np.abs(pts[:, 0]) <= .30) & (pts[:, 1] >= lo) & (pts[:, 1] <= hi) & (pts[:, 2] >= .30) & (pts[:, 2] <= V1.Z_MAX)
                            mask = np.zeros(h * w, np.uint8); mask[np.flatnonzero(ok)[inside]] = 1
                            n_lab, cc = cv2.connectedComponents(mask.reshape(h, w), connectivity=8)
                            zfull = np.full(h * w, np.nan); zfull[np.flatnonzero(ok)] = pts[:, 2]
                            cc = cc.reshape(-1); counted = 0
                            for c in range(1, n_lab):
                                m = cc == c
                                if m.sum() >= MIN_BLOB_PX and np.median(zfull[m]) <= BLOB_MAX_MEDIAN_Z:
                                    counted += int(m.sum())
                            rgb_alert[q] = bool(counted >= V1.MIN_PX)
                tof_strong = [bool(gated[i, q] == 2) for q in range(2)]
                frames.append(dict(tof_index=int(i), pose_ok=bool(pose_ok[i]), anchor=anchor, tof_strong=tof_strong, rgb=rgb_alert,
                                   fused_raw=[tof_strong[q] or rgb_alert[q] for q in range(2)]))
            n = len(frames)
            if n:
                head = V2.confirm([f['fused_raw'][0] for f in frames]); body = V2.confirm([f['fused_raw'][1] for f in frames])
                for f, hq, bq in zip(frames, head, body):
                    f['fused'] = [bool(hq), bool(bq)]
                anyq = head | body
            res = dict(scene=lab.get('scene'), cls=lab['class'], ready_frames=n,
                       fused_any_alert=float(anyq.mean()) if n else None,
                       end_window_any_alert=float(anyq[-V1.END_WINDOW:].mean()) if n else None,
                       end_window_head_alert=float(head[-V1.END_WINDOW:].mean()) if n else None,
                       tof_strong_any=float(np.mean([f['tof_strong'][0] or f['tof_strong'][1] for f in frames])) if n else None,
                       rgb_any=float(np.mean([f['rgb'][0] or f['rgb'][1] for f in frames])) if n else None,
                       robust_anchor_frames=sum(f['anchor'] == 'robust_tof' for f in frames))
            c = lab['class']; C = V1.CRITERIA
            res['pass'] = (None if n == 0 or c in ('calibration', 'unknown') else
                           res['fused_any_alert'] <= C['quiet_max_alert'] if c == 'quiet' else
                           res['fused_any_alert'] <= C['side_max_alert'] if c == 'side' else
                           res['end_window_any_alert'] >= C['ahead_min_end_alert'] if c == 'ahead' else
                           res['end_window_head_alert'] >= C['head_min_end_alert'] if c == 'head' else None)
            report['sessions'][name] = dict(res, frames=frames)
            print(f"{name[17:23]} {c:11s} {str(lab.get('scene'))[:34]:34s} n={n:3d} fused {res['fused_any_alert']} end {res['end_window_any_alert']} "
                  f"robust {res['robust_anchor_frames']} -> {res['pass']}", flush=True)
    finally:
        engine.close()
    judged = [v for v in report['sessions'].values() if v['pass'] is not None]
    report['summary'] = dict(judged=len(judged), passed=sum(v['pass'] for v in judged),
                             all_ahead_detected=all(v['pass'] for v in judged if v['cls'] == 'ahead') if any(v['cls'] == 'ahead' for v in judged) else None,
                             all_quiet_pass=all(v['pass'] for v in judged if v['cls'] == 'quiet') if any(v['cls'] == 'quiet' for v in judged) else None)
    (out / 'frozen_fusion_eval_v6.json').write_text(json.dumps(report, indent=1), encoding='utf8')
    print(json.dumps(report['summary'], indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--raw', required=True); ap.add_argument('--labels', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); main(a.raw, a.labels, a.out)
