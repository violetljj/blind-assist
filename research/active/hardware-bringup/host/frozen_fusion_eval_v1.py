"""Frozen RGB-or-ToF fusion evaluation v1 for new head-worn phone captures (protocol FROZEN_FUSION_EVAL_V1).

Everything is fixed before the new data exist (see ../FROZEN_FUSION_EVAL_V1_PROTOCOL.md):
  ToF   : pose/floor/noise joint stack manifest (cnh-pose-floor-noise-dr-dev-20261012), segment-mean
          IMU pose, sensor at the head origin; strong grade only enters the fusion.
  Gate  : pose_envelope (pitch > +10 deg or |roll| > 25 deg -> no alert).
  RGB   : Atom fisheye calibration + fitted camera<-ToF rotation; official DAV2-Small (relative), short
          side 518; anchored per frame by ToF zones (>= 6) else floor plane with the session head
          height (ToF floor fit on the 'calibration' segment, else 1.35 m) else the last fit; pixels in
          HEAD/BODY boxes (x +/-0.30, y -0.20..0.42 / 0.42..0.90, z 0.30..2.00 m) >= 150 -> alert.
  Fusion: per query, ToF strong OR RGB alert (both gated).
Labels (scene class per session) must be written from RGB contact sheets before running.
Usage: frozen_fusion_eval_v1.py --raw <dir of capture-* sessions> --labels <labels.json> --out <dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from atom_camera import AtomCamera, ROOT
import atom_tof_extrinsic_dev_20261012 as X
import pose_envelope as PE
import real_pilot_calibration as P
import real_phone_imu_replay as R
import real_rgb_query_feasibility_dev_20261012 as F
import real_rgb_small_model_dev_20261012 as S
from local_signed_bias_adapter import ContractReplay
from real_contract_v5 import adapter_v5

MANIFEST = ROOT / 'artifacts.local/work/cnh-pose-floor-noise-dr-dev-20261012/real_manifest.json'
EXTRINSIC = ROOT / 'artifacts.local/work/atom-camera-calibration-20261012/camera_tof_extrinsic.json'
SIZE, MIN_PX, Z_MAX, DEFAULT_HEIGHT = 518, 150, 2.0, 1.35
END_WINDOW = 15            # last 15 ready ToF frames (~3 s) of an approach
CRITERIA = dict(quiet_max_alert=.10, side_max_alert=.30, ahead_min_end_alert=.50, head_min_end_alert=.50)
FROZEN_FILES = [MANIFEST, EXTRINSIC, ROOT / 'artifacts.local/work/atom-camera-calibration-20261012/calibration.json',
                ROOT / 'artifacts.local/models/depth-anything--Depth-Anything-V2-Small/depth_anything_v2_vits.pth']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tof_grades(engine, d, x, bias, noise):
    z = adapter_v5(x['H'], x['A'], bias, noise, None, True, False)
    engine.engine.configure_geometry(float(np.nanmean(x['pitch'])), float(np.nanmean(x['roll'])), 0.)
    r = engine.infer(z, x['seq'], x['ms'])
    return r['S_grade'], r['ready'].astype(bool)


def main(raw, labels_path, out):
    labels = json.loads(Path(labels_path).read_text(encoding='utf8'))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    import torch
    from atom_imu import load_mounting
    imu_rot = np.array(load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    arrays = np.load(P.OUT / 'instrument_arrays.npz'); bias = arrays['bias'].reshape(64, 16)
    noise = P.read('instrument_parameters.json')['noise']
    Rct = np.array(json.loads(EXTRINSIC.read_text(encoding='utf8'))['rotation'])
    cam = AtomCamera(); K = cam.K_upright; rays = X.zone_rays()
    h, w = cam.upright_size[1], cam.upright_size[0]
    vv, uu = np.mgrid[0:h, 0:w]
    dirs = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1).reshape(-1, 3)
    sessions = {d.name: d for d in Path(raw).iterdir() if d.name in labels}
    # Head height from the calibration segment (ToF floor fit), fixed rule.
    height, height_source = DEFAULT_HEIGHT, 'default'
    for name, lab in labels.items():
        if lab['class'] == 'calibration' and name in sessions:
            fit = R.floor_fit(R.load(sessions[name], imu_rot))
            if fit and 1.0 <= fit[0] <= 1.9:
                height, height_source = fit[0], name
    engine = ContractReplay(str(MANIFEST)); model, _ = S.load_dav2_small()
    rng = np.random.default_rng(20261013)
    report = dict(protocol='FROZEN_FUSION_EVAL_V1', head_height_m=height, head_height_source=height_source,
                  frozen_sha256={str(p.relative_to(ROOT)): sha(p) for p in FROZEN_FILES}, criteria=CRITERIA, sessions={})
    try:
        for name in sorted(sessions):
            lab = labels[name]; d = sessions[name]
            x = R.load(d, imu_rot)
            grade, ready = tof_grades(engine, d, x, bias, noise)
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
                    disp = model.infer_image(np.ascontiguousarray(up[:, :, ::-1]), SIZE)
                    dz, zfac = S.zone_disparity(disp, K, Rct, rays)
                    valid = np.isfinite(dist[i]) & (dist[i] > .3) & (dist[i] < 4.)
                    fit = S.fit_affine(dz, np.where(valid, 1 / (dist[i] * zfac), np.nan)); anchor = 'tof' if fit else None
                    if fit is None:
                        fit = S.floor_anchor(disp, dirs, F.rot(x['pitch'][i], x['roll'][i]), Rct, height, rng); anchor = 'floor' if fit else None
                    fit = fit or last
                    if fit is not None:
                        last = fit; a, b = fit
                        inv = (disp.reshape(-1) - b) / a
                        zz = np.where(inv > 1e-3, 1 / np.maximum(inv, 1e-3), np.inf)
                        ok = np.isfinite(zz) & (up.reshape(-1, 3).sum(1) > 0)
                        pts = (F.rot(x['pitch'][i], x['roll'][i]) @ (Rct.T @ (dirs[ok] * zz[ok, None]).T)).T
                        for q, (lo, hi) in enumerate(F.BOXES.values()):
                            inside = (np.abs(pts[:, 0]) <= .30) & (pts[:, 1] >= lo) & (pts[:, 1] <= hi) & (pts[:, 2] >= .30) & (pts[:, 2] <= Z_MAX)
                            rgb_alert[q] = bool(inside.sum() >= MIN_PX)
                tof_strong = [bool(gated[i, q] == 2) for q in range(2)]
                frames.append(dict(tof_index=int(i), pose_ok=bool(pose_ok[i]), anchor=anchor, tof_strong=tof_strong, rgb=rgb_alert,
                                   fused=[tof_strong[q] or rgb_alert[q] for q in range(2)]))
            n = len(frames); any_alert = np.array([f['fused'][0] or f['fused'][1] for f in frames]) if n else np.zeros(0, bool)
            head = np.array([f['fused'][0] for f in frames]) if n else np.zeros(0, bool)
            end = any_alert[-END_WINDOW:]; end_head = head[-END_WINDOW:]
            res = dict(scene=lab.get('scene'), cls=lab['class'], ready_frames=n, fused_any_alert=float(any_alert.mean()) if n else None,
                       tof_strong_any=float(np.mean([f['tof_strong'][0] or f['tof_strong'][1] for f in frames])) if n else None,
                       rgb_any=float(np.mean([f['rgb'][0] or f['rgb'][1] for f in frames])) if n else None,
                       end_window_any_alert=float(end.mean()) if len(end) else None, end_window_head_alert=float(end_head.mean()) if len(end_head) else None)
            c = lab['class']
            res['pass'] = (None if n == 0 or c in ('calibration', 'unknown') else
                           res['fused_any_alert'] <= CRITERIA['quiet_max_alert'] if c == 'quiet' else
                           res['fused_any_alert'] <= CRITERIA['side_max_alert'] if c == 'side' else
                           res['end_window_any_alert'] >= CRITERIA['ahead_min_end_alert'] if c == 'ahead' else
                           res['end_window_head_alert'] >= CRITERIA['head_min_end_alert'] if c == 'head' else None)
            report['sessions'][name] = dict(res, frames=frames)
            print(f"{name[17:23]} {c:11s} {str(lab.get('scene'))[:34]:34s} n={n:3d} fused {res['fused_any_alert']} end {res['end_window_any_alert']} "
                  f"ToF {res['tof_strong_any']} RGB {res['rgb_any']} -> {res['pass']}", flush=True)
    finally:
        engine.close()
    judged = [v for v in report['sessions'].values() if v['pass'] is not None]
    ahead = [v for v in judged if v['cls'] in ('ahead',)]
    report['summary'] = dict(judged=len(judged), passed=sum(v['pass'] for v in judged),
                             all_ahead_detected=all(v['pass'] for v in ahead) if ahead else None,
                             all_quiet_pass=all(v['pass'] for v in judged if v['cls'] == 'quiet') if any(v['cls'] == 'quiet' for v in judged) else None)
    (out / 'frozen_fusion_eval_v1.json').write_text(json.dumps(report, indent=1), encoding='utf8')
    print(json.dumps(report['summary'], indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--raw', required=True); ap.add_argument('--labels', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); main(a.raw, a.labels, a.out)
