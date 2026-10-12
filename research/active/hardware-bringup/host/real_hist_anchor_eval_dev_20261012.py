"""Compare firmware-distance vs histogram-consistent anchoring of DAV2-Small, and ToF confirmation (EXPLORE).

Afternoon phone batch (18 segments, already viewed; Claude's labels). Same camera model, extrinsic,
query boxes and MIN_PX as the frozen v1 readout; only the anchoring step differs:
  fw_floor        firmware zone distances (>= 6) -> floor plane (1.35 m) -> last fit   [= frozen v1]
  hist_only       histogram-consistent fit only -> last fit
  fw_hist_floor   firmware -> histogram -> floor -> last fit
For each RGB box alert, the fraction of in-box pixels confirmed by ToF evidence at their range bin is
recorded (cross-modal confirmation, tof_histogram_anchor.confirm_pixels).
"""
from __future__ import annotations

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
import real_rgb_query_feasibility_dev_20261012 as F
import real_rgb_small_model_dev_20261012 as S
import real_rgb_tof_fusion_dev_20261012 as U
import tof_histogram_anchor_dev_20261012 as HA
from real_contract_v5 import adapter_v5

OUT = ROOT / 'artifacts.local/work/hist-anchor-dev-20261012'
STRATEGIES = ('fw_floor', 'hist_only', 'fw_hist_floor')


def main():
    import real_phone_imu_replay as R
    from atom_imu import load_mounting
    imu_rot = np.array(load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    arrays = np.load(P.OUT / 'instrument_arrays.npz'); bias = arrays['bias'].reshape(64, 16)
    noise = P.read('instrument_parameters.json')['noise']
    Rct = np.array(json.loads(X.OUT.joinpath('camera_tof_extrinsic.json').read_text(encoding='utf8'))['rotation'])
    cam = AtomCamera(); K = cam.K_upright; rays = X.zone_rays()
    h, w = cam.upright_size[1], cam.upright_size[0]
    pzm = HA.PixelZoneMap(K, Rct, h, w, step=4)
    vv, uu = np.mgrid[0:h, 0:w]
    dirs = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu, float)], -1).reshape(-1, 3)
    model, _ = S.load_dav2_small(); rng = np.random.default_rng(20261012)
    OUT.mkdir(parents=True, exist_ok=True); report = {}
    for d in sorted(F.RAW.iterdir()):
        key = d.name[17:23]
        if key not in U.SCENES:
            continue
        x = R.load(d, imu_rot)
        zall = adapter_v5(x['H'], x['A'], bias, noise, None, True, False)
        tof_t = [r['received_elapsed_ns'] for r in F.lines(d / 'tof.index.jsonl') if r.get('valid_frame')]
        rgb = list(F.lines(d / 'rgb.jsonl')); rgb_t = np.array([r['received_elapsed_ns'] for r in rgb])
        dist = np.where(np.isin(x['S'], (5, 6, 9, 10)), x['D'], np.nan).reshape(-1, 8, 8)[:, ::-1, ::-1].reshape(-1, 64) / 1000
        cached = np.load(U.NEW / 'joint_replay' / f'{d.name}_imu.npz')
        ready = cached['ready'].astype(bool); _, pose_ok = PE.apply(cached['S_grade'], x['pitch'], x['roll'])
        last = {s: None for s in STRATEGIES}; rows = []
        for i in range(0, len(tof_t), U.STRIDE):
            if not ready[i] or not pose_ok[i]:
                continue
            j = int(np.argmin(np.abs(rgb_t - tof_t[i])))
            if abs(rgb_t[j] - tof_t[i]) > 1e8:
                continue
            up = cam.undistort_upright(cv2.imread(str(d / rgb[j]['file']))[:, :, ::-1])
            disp = model.infer_image(np.ascontiguousarray(up[:, :, ::-1]), 518)
            dz, zfac = S.zone_disparity(disp, K, Rct, rays)
            valid = np.isfinite(dist[i]) & (dist[i] > .3) & (dist[i] < 4.)
            fw = S.fit_affine(dz, np.where(valid, 1 / (dist[i] * zfac), np.nan))
            ha, hb, hinfo = HA.fit_histogram(disp, zall[i], pzm)
            hist = (ha, hb) if ha is not None else None
            floor = None
            row = dict(tof_index=int(i), fw=fw is not None, hist=hist is not None, hist_info=hinfo)
            if fw and hist:   # scale agreement at the median disparity
                dm = float(np.median(disp))
                zf = fw[0] / max(dm - fw[1], 1e-6); zh = hist[0] / max(dm - hist[1], 1e-6)
                row['median_depth_ratio_hist_over_fw'] = float(zh / zf) if zf > 0 and zh > 0 else None
            for s in STRATEGIES:
                order = {'fw_floor': [fw, 'floor'], 'hist_only': [hist], 'fw_hist_floor': [fw, hist, 'floor']}[s]
                fit = None
                for o in order:
                    if o == 'floor':
                        floor = floor or S.floor_anchor(disp, dirs, F.rot(x['pitch'][i], x['roll'][i]), Rct, 1.35, rng)
                        o = floor
                    if o is not None:
                        fit = o; break
                fit = fit or last[s]
                res = dict(anchored=fit is not None, alert=[False, False], confirm=[None, None])
                if fit is not None:
                    last[s] = fit; a, b = fit
                    inv = (disp - b) / a
                    zf_ = np.where(inv > 1e-3, 1 / np.maximum(inv, 1e-3), np.inf)
                    ok = (np.isfinite(zf_) & (up.sum(-1) > 0)).reshape(-1)
                    pts = np.full((h * w, 3), np.nan)
                    pts[ok] = (F.rot(x['pitch'][i], x['roll'][i]) @ (Rct.T @ (dirs[ok] * zf_.reshape(-1)[ok, None]).T)).T
                    for q, (lo, hi) in enumerate(F.BOXES.values()):
                        inside = (np.abs(pts[:, 0]) <= .30) & (pts[:, 1] >= lo) & (pts[:, 1] <= hi) & (pts[:, 2] >= .30) & (pts[:, 2] <= U.Z_MAX)
                        res['alert'][q] = bool(inside.sum() >= U.MIN_PX)
                        if res['alert'][q]:
                            res['confirm'][q] = HA.confirm_pixels(inside.reshape(h, w), zf_, pzm, zall[i])[0]
                row[s] = res
            rows.append(row)
        report[key] = dict(scene=U.SCENES[key], frames=rows)
        n = max(len(rows), 1)
        def f(fn): return sum(fn(r) for r in rows) / n
        ratios = [r['median_depth_ratio_hist_over_fw'] for r in rows if r.get('median_depth_ratio_hist_over_fw')]
        line = f"{key} {U.SCENES[key][:32]:32s} n={len(rows):3d} fw {f(lambda r: r['fw']):4.0%} hist {f(lambda r: r['hist']):4.0%}"
        line += f" ratio~{np.median(ratios):.2f}" if ratios else " ratio  -  "
        for s in STRATEGIES:
            al = [r[s] for r in rows]
            line += f" | {s} H/B {np.mean([a['alert'][0] for a in al]) if al else 0:4.0%}/{np.mean([a['alert'][1] for a in al]) if al else 0:4.0%}"
        conf = [c for r in rows for c in r['fw_hist_floor']['confirm'] if c is not None]
        line += f" | confirm>0.3 {np.mean(np.array(conf) > .3):4.0%} (n={len(conf)})" if conf else ' | confirm -'
        print(line, flush=True)
    (OUT / 'hist_anchor_eval.json').write_text(json.dumps(report, indent=1), encoding='utf8')


if __name__ == '__main__':
    main()
