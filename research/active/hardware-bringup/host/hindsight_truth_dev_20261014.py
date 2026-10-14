"""Hindsight distance truth for own broom-approach recordings (EXPLORE, plan in hindsight-truth-dev-20261014/PLAN.md).

Non-causal and independent of DAV2/ToF: the broom's red base contact pixel is intersected with the floor (frozen-chain camera
geometry + IMU pose, head height 1.33 m), then tied to every ToF frame with a step-count odometry fit z = d0 - L*s(t).
Usage: hindsight_truth_dev_20261014.py truth | alerts
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt, find_peaks

sys.path.insert(0, str(Path(__file__).parent))
from atom_camera import AtomCamera, ROOT
import real_phone_imu_replay as R
import real_rgb_query_feasibility_dev_20261012 as F
from atom_imu import load_mounting

RAW = ROOT / 'artifacts.local/work/frozen-eval-capture-20261013b/raw'
OUT = ROOT / 'artifacts.local/work/hindsight-truth-dev-20261014'
EXTRINSIC = ROOT / 'artifacts.local/work/atom-camera-calibration-20261012/camera_tof_extrinsic.json'
SEGMENTS = {'054159': 'ahead', '054301': 'ahead', '054334': 'ahead', '054405': 'ahead', '054437': 'retreat', '054511': 'retreat'}
VARIANTS = ((1.33, 14.5), (1.28, 14.5), (1.38, 14.5), (1.33, 13.0), (1.33, 16.0), (1.33, 0.0))  # (head height m, camera pitch fix deg)
EVALS = {'v1': ROOT / 'artifacts.local/work/frozen-eval-capture-20261013b/eval/frozen_fusion_eval_v1.json',
         'v2': ROOT / 'artifacts.local/work/frozen-eval-capture-20261013b/eval/frozen_fusion_eval_v2.json',
         'v3': ROOT / 'artifacts.local/work/frozen-eval-capture-20261013b/eval/frozen_fusion_eval_v3.json',
         'v6': ROOT / 'artifacts.local/work/fusion-v6-dev-20261013/a13/frozen_fusion_eval_v6.json'}
BINS = [(3.0, 99), (2.5, 3.0), (2.0, 2.5), (1.5, 2.0), (1.0, 1.5), (0.0, 1.0)]


def session_dir(tag):
    return next(d for d in RAW.iterdir() if d.name[17:23] == tag)


def red_base(up):
    hsv = cv2.cvtColor(up, cv2.COLOR_RGB2HSV)
    red = ((hsv[..., 0] <= 12) | (hsv[..., 0] >= 158)) & (hsv[..., 1] >= 22) & (hsv[..., 2] >= 40) & (hsv[..., 2] <= 200)
    red[:, :100] = False; red[:, 340:] = False; red[:300] = False
    m = cv2.morphologyEx(red.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lab == k)
    return dict(u=float(np.median(xs)), v=float(np.percentile(ys, 98)), area=int(st[k, cv2.CC_STAT_AREA]),
                trunc=bool(ys.max() >= up.shape[0] - 3))


def imu_series(d, rot):
    t, g, a, dev = [], [], [], []
    for r in F.lines(d / 'imu.index.jsonl'):
        if r.get('valid_frame') and r.get('gravity_unit') and r.get('accel_g'):
            t.append(r['received_elapsed_ns']); g.append(r['gravity_unit']); a.append(r['accel_g']); dev.append(r['device_timestamp_us'])
    t, g, a, dev = np.array(t, float), np.array(g), np.array(a), np.array(dev, float)
    o = np.argsort(dev); t, g, a, dev = t[o], g[o], a[o], dev[o]
    # device clock -> receipt clock: lower envelope (min latency) linear map
    p = np.polyfit(dev, t, 1); off = np.min(t - np.polyval(p, dev)); tt = np.polyval(p, dev) + off
    tv = (rot @ g.T).T; tv /= np.linalg.norm(tv, axis=1, keepdims=True)
    pitch = np.degrees(np.arcsin(tv[:, 2])); roll = np.degrees(np.arctan2(-tv[:, 0], -tv[:, 1]))
    return tt, pitch, roll, np.linalg.norm(a, axis=1)


def steps(tt, amag):
    fs = 100.
    grid = np.arange(tt[0], tt[-1], 1e9 / fs); x = np.interp(grid, tt, amag)
    b, a = butter(2, [0.8 / (fs / 2), 3.0 / (fs / 2)], 'band'); y = filtfilt(b, a, x)
    pk, _ = find_peaks(y, distance=int(.3 * fs), prominence=max(.03, .5 * np.std(y)))
    s = np.interp(grid, grid[pk], np.arange(len(pk)), left=0, right=len(pk) - 1) if len(pk) > 1 else np.zeros_like(grid)
    return grid, s, grid[pk]


def floor_point(u, v, pitch, roll, K, Rct, H, fix=0.):
    c = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.]); a = math.radians(fix)
    c = np.array([c[0], c[1] * math.cos(a) - c[2] * math.sin(a), c[1] * math.sin(a) + c[2] * math.cos(a)])  # rays up by fix
    ray = F.rot(pitch, roll) @ (Rct.T @ c)
    if ray[1] <= 1e-3:
        return None
    s = H / ray[1]
    return float(s * ray[2]), float(s * ray[0])


def truth():
    cam = AtomCamera(); K = cam.K_upright
    Rct = np.array(json.loads(EXTRINSIC.read_text(encoding='utf8'))['rotation'])
    rot = np.array(load_mounting(R.MOUNTING)['tof_from_imu_rotation'])
    out = {}
    for tag, kind in SEGMENTS.items():
        d = session_dir(tag)
        rgb = list(F.lines(d / 'rgb.jsonl'))
        tt, pitch, roll, amag = imu_series(d, rot)
        grid, s, pk_t = steps(tt, amag)
        dets = []
        for r in rgb:
            b = red_base(cam.undistort_upright(cv2.imread(str(d / r['file']))[:, :, ::-1]))
            dets.append(dict(t=r['received_elapsed_ns'], **(b or {})))
        us = np.array([x.get('u', np.nan) for x in dets]); med = np.nanmedian(us[:40]) if np.isfinite(us[:40]).any() else np.nan
        acc = []
        for x in dets:  # running-median continuity on u, ordered in time
            if 'u' not in x or x['trunc'] or not 300 <= x['area'] <= 4000 or not 450 <= x['v'] <= 635:
                continue
            if np.isfinite(med) and abs(x['u'] - med) > 60:
                continue
            med = .8 * med + .2 * x['u'] if np.isfinite(med) else x['u']
            j = int(np.argmin(np.abs(tt - x['t'])))
            acc.append(dict(x, pitch=float(pitch[j]), roll=float(roll[j]), s=float(np.interp(x['t'], grid, s))))
        tof_t = np.array([r['received_elapsed_ns'] for r in F.lines(d / 'tof.index.jsonl') if r.get('valid_frame')], float)
        seg = dict(kind=kind, n_rgb=len(rgb), n_contact=len(acc), n_steps=len(pk_t), duration_s=float((tt[-1] - tt[0]) / 1e9), heights={})
        if kind == 'retreat':  # use frames up to the closest contact only (largest v)
            if acc:
                k = int(np.argmax([a['v'] for a in acc])); acc = acc[:k + 1]
        for H, fix in VARIANTS:
            key = f'{H}/{fix}'
            pts = [(a, floor_point(a['u'], a['v'], a['pitch'], a['roll'], K, Rct, H, fix)) for a in acc]
            pts = [(a, p) for a, p in pts if p is not None]
            if len(pts) < 3:
                seg['heights'][key] = None; continue
            z = np.array([p[0] for _, p in pts]); xl = np.array([p[1] for _, p in pts])
            sv = np.array([a['s'] for a, _ in pts]); tv = np.array([a['t'] for a, _ in pts]) / 1e9
            f = least_squares(lambda q: q[0] - q[1] * sv - z, [z.max(), .6], loss='huber', f_scale=.1)
            res = f.x[0] - f.x[1] * sv - z
            lin = np.polyfit(tv - tv[0], z, 1); res_lin = np.polyval(lin, tv - tv[0]) - z
            s_tof = np.interp(tof_t, grid, s); zt = f.x[0] - f.x[1] * s_tof
            seg['heights'][key] = dict(d0=float(f.x[0]), L=float(f.x[1]), rms=float(np.sqrt(np.mean(res ** 2))),
                                         rms_linear_time=float(np.sqrt(np.mean(res_lin ** 2))), x_median=float(np.median(xl)),
                                         z_first=float(z[0]), z_last=float(z[-1]), end=float(zt[-1]),
                                         tof_truth=[float(v) for v in zt], extrapolated=[bool(t > tv[-1] * 1e9) for t in tof_t],
                                         contact=[dict(t=float(a['t']), u=a['u'], v=a['v'], z=float(p[0]), x=float(p[1])) for a, p in pts])
        h = seg['heights'].get('1.33/14.5')
        if h:
            seg['checks'] = dict(C1=bool(len(acc) >= 8 and h['rms'] <= .15), C3=bool(.2 <= h['end'] <= 1.2))
        out[d.name] = seg
        print(tag, kind, 'contacts', len(acc), 'steps', len(pk_t), {k: (None if v is None else {q: round(v[q], 3) for q in
              ('d0', 'L', 'rms', 'rms_linear_time', 'x_median', 'z_first', 'z_last', 'end')}) for k, v in seg['heights'].items()}, flush=True)
    Ls = {n: v['heights']['1.33/14.5']['L'] for n, v in out.items() if v['kind'] == 'ahead' and v['heights'].get('1.33/14.5')}
    medL = float(np.median(list(Ls.values()))) if Ls else None
    for n, v in out.items():
        if v.get('checks'):
            L = v['heights']['1.33/14.5']['L']; v['checks']['C2'] = bool(.45 <= L <= .85 and (medL is None or abs(L - medL) <= .2 * medL))
            v['valid'] = all(v['checks'].values())
    (OUT / 'truth.json').write_text(json.dumps(dict(median_L=medL, segments=out), indent=1), encoding='utf8')
    print('median L', medL, {n[17:23]: v.get('checks') for n, v in out.items()})


def alerts():
    T = json.loads((OUT / 'truth.json').read_text(encoding='utf8'))['segments']
    report = {}
    for ver, path in EVALS.items():
        ev = json.loads(path.read_text(encoding='utf8'))['sessions']
        rows = {k: dict(n=0, fused=0, tof=0, rgb=0) for k in map(str, BINS)}; per = {}
        for name, seg in T.items():
            if not seg.get('valid') or name not in ev:
                continue
            h = seg['heights']['1.33/14.5']; zt = np.array(h['tof_truth'])
            if seg['kind'] == 'retreat':
                last_t = h['contact'][-1]['t']; tof_t = None
            frames = ev[name]['frames']
            fused = np.array([f['fused'][0] or f['fused'][1] for f in frames])
            dist = np.array([zt[f['tof_index']] for f in frames])
            keep = np.ones(len(frames), bool)
            if seg['kind'] == 'retreat':
                kmin = int(np.argmin(dist)); keep[kmin + 1:] = False
            for f, z, k in zip(frames, dist, keep):
                if not k:
                    continue
                for lo, hi in BINS:
                    if lo <= z < hi:
                        r = rows[str((lo, hi))]; r['n'] += 1; r['fused'] += f['fused'][0] or f['fused'][1]
                        r['tof'] += f['tof_strong'][0] or f['tof_strong'][1]; r['rgb'] += f['rgb'][0] or f['rgb'][1]
            fk, dk = fused[keep], dist[keep]
            run = len(fk)
            while run > 0 and fk[run - 1]:
                run -= 1
            per[name[17:23]] = dict(frames=int(keep.sum()), alert_far_gt2p5=float(fk[dk > 2.5].mean()) if (dk > 2.5).any() else None,
                                    alert_0p5_2=float(fk[(dk >= .5) & (dk <= 2.)].mean()) if ((dk >= .5) & (dk <= 2.)).any() else None,
                                    final_run_start_m=float(dk[run]) if run < len(fk) else None, min_truth_m=float(dk.min()))
        report[ver] = dict(bins={k: dict(v, fused_rate=v['fused'] / v['n'] if v['n'] else None) for k, v in rows.items()}, segments=per)
        print(ver, {k: (v['n'], None if not v['n'] else round(v['fused'] / v['n'], 2), None if not v['n'] else round(v['rgb'] / v['n'], 2),
                        None if not v['n'] else round(v['tof'] / v['n'], 2)) for k, v in rows.items()})
        for s, v in per.items():
            print('   ', s, {k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items()})
    (OUT / 'alerts_by_truth.json').write_text(json.dumps(report, indent=1), encoding='utf8')


if __name__ == '__main__':
    {'truth': truth, 'alerts': alerts}[sys.argv[1]]()
