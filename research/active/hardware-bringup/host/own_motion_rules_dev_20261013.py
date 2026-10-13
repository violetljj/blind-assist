"""World-consistency (WC) rule on own head-worn captures with IMU-only ego-motion (EXPLORE; consumed Development data).

Input: own_blob_dump_dev_20261013 output. Ego-motion between ready frames i and j (~1 s earlier):
  * heading change: gyro rate about the measured up axis (gravity_unit = normalised accel, points up), integrated;
  * walking + speed: accel-magnitude cadence over a 2 s window (dominant 0.9-2.6 Hz, RMS >= MIN_RMS g),
    speed = STEP_LEN x cadence; displacement = speed x dt along the heading of frame i.
The WC rule is the ADT one (plan PLAN_BACKBONE_MOTION.md): a counted RGB blob (z50 <= 1.8 m) of a walking frame is kept
only if frame j's predicted-point grid (same height band) has >= 3 points within 0.2 + 0.1 z' m of where the blob was
then, or that spot was not observable. ToF strong alerts are unchanged. Session metrics follow the frozen criteria
(fused 2-of-3; quiet/side any-alert rate, ahead/head end-window rate).
Usage: own_motion_rules_dev_20261013.py <dump dir> [yaw sign +1|-1]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import frozen_fusion_eval_v1 as V1
import frozen_fusion_eval_v2 as V2

STEP_LEN, MIN_RMS, LOOK_S = .65, .015, 1.0   # MIN_RMS lowered from .04 after seeing own walks: 2 s accel-magnitude std .015-.04 g, cadence ~1.1-1.2 Hz
GXC, GZC = np.linspace(-1.45, 1.45, 30), np.linspace(.35, 5.25, 50)


def ego(imu, t_query):
    """Per query time: (heading deg integrated, walking bool, speed m/s)."""
    t = imu['t'] / 1e9; up = imu['acc'] / np.linalg.norm(imu['acc'], axis=1, keepdims=True)
    rate = np.einsum('ij,ij->i', imu['gyr'], up)                                  # deg/s about up
    head = np.concatenate([[0], np.cumsum(.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
    mag = np.linalg.norm(imu['acc'], axis=1)
    out = []
    for tq in t_query:
        hq = float(np.interp(tq, t, head))
        m = (t >= tq - 2) & (t <= tq)
        walking, speed = False, 0.
        if m.sum() > 100:
            tt = t[m]; g = np.interp(np.linspace(tt[0], tt[-1], 256), tt, mag[m]); g = g - g.mean()
            fs = 255 / (tt[-1] - tt[0]); F = np.abs(np.fft.rfft(g * np.hanning(256))); fr = np.fft.rfftfreq(256, 1 / fs)
            band = (fr >= .9) & (fr <= 2.6)
            if g.std() >= MIN_RMS and band.any():
                walking = True; speed = STEP_LEN * float(fr[band][np.argmax(F[band])])
        out.append((hq, walking, speed))
    return out


def wc_keep(c, q, dpsi, disp, grid_j, Rcam_j, K, wh):
    a = np.radians(dpsi)                                                          # heading of i relative to j (left +)
    right, fwd = np.array([np.cos(a), 0, -np.sin(a)]), np.array([np.sin(a), 0, np.cos(a)])
    cj = c[0] * right + np.array([0, c[1], 0]) + c[2] * fwd + disp * fwd
    pc = Rcam_j.T @ cj
    if pc[2] <= .3:
        return True
    u, v = K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
    if not (0 <= u < wh[0] and 0 <= v < wh[1]) or cj[2] > 5.3 or abs(cj[0]) > 1.5:
        return True
    r = .2 + .1 * abs(cj[2])
    near = ((GXC[:, None] - cj[0]) ** 2 + (GZC[None, :] - cj[2]) ** 2) <= r * r
    return int(grid_j[q][near].sum()) >= 3


def main(dump, sign='1'):
    sign = float(sign); dump = Path(dump)
    F_, X = V1.F, V1.X
    Rct = np.array(json.loads(V1.EXTRINSIC.read_text(encoding='utf8'))['rotation'])
    cam = V1.AtomCamera(); K = cam.K_upright; wh = cam.upright_size
    C = V1.CRITERIA; rows = []
    for jf in sorted(dump.glob('capture-*.json')):
        J = json.loads(jf.read_text(encoding='utf8')); Z = np.load(jf.with_suffix('.npz'))
        fr = J['frames']; lab = J['label']; n = len(fr)
        if n == 0:
            continue
        imu = {k[4:]: Z[k] for k in Z.files if k.startswith('imu_')}; G = Z['grid']
        tq = np.array([f['t_ns'] for f in fr]) / 1e9
        E = ego(imu, tq)
        res = {}
        for rule in ('v6', 'wc'):
            raw = np.zeros((n, 2), bool); vetoed = 0
            for i, f in enumerate(fr):
                rgb = [False, False]
                if f['blobs'] is not None:
                    j = int(np.argmin(np.abs(tq - (tq[i] - LOOK_S))))
                    jok = .6 <= tq[i] - tq[j] <= 1.5 and fr[j]['blobs'] is not None
                    for q in (0, 1):
                        cnt = 0
                        for b in f['blobs'][q][1]:
                            if b[1] > 1.8:
                                continue
                            if rule == 'wc' and E[i][1] and jok:
                                dpsi = sign * (E[i][0] - E[j][0]); disp = E[i][2] * (tq[i] - tq[j])
                                Rcam_j = F_.rot(fr[j]['pitch'], fr[j]['roll']) @ Rct.T          # camera -> corridor_j
                                if not wc_keep(np.array(b[2:5]), q, dpsi, disp, G[j], Rcam_j, K, wh):
                                    vetoed += 1; continue
                            cnt += b[0]
                        rgb[q] = cnt >= V1.MIN_PX
                raw[i] = [f['tof_strong'][q] or rgb[q] for q in (0, 1)]
            hq, bq = V2.confirm(raw[:, 0]), V2.confirm(raw[:, 1]); anyq = hq | bq
            res[rule] = dict(any=float(anyq.mean()), end=float(anyq[-V1.END_WINDOW:].mean()), end_head=float(hq[-V1.END_WINDOW:].mean()), vetoed=vetoed)
        walk = float(np.mean([e[1] for e in E])); spd = float(np.median([e[2] for e in E if e[1]] or [0]))
        c = lab['class']
        def ok(r):
            return (None if c in ('calibration', 'unknown') else r['any'] <= C['quiet_max_alert'] if c == 'quiet' else
                    r['any'] <= C['side_max_alert'] if c == 'side' else r['end'] >= C['ahead_min_end_alert'] if c == 'ahead' else
                    r['end_head'] >= C['head_min_end_alert'] if c == 'head' else None)
        rows.append(dict(session=jf.stem, cls=c, scene=lab.get('scene'), n=n, walking=walk, speed=spd, **{f'{k}_{r}': v for r in res for k, v in res[r].items()},
                         pass_v6=ok(res['v6']), pass_wc=ok(res['wc'])))
        x = rows[-1]
        print(f"{jf.stem[17:23]} {c:11s} {str(lab.get('scene'))[:30]:30s} n={n:3d} walk {walk:.2f} {spd:.2f}m/s | any {x['any_v6']:.2f}->{x['any_wc']:.2f} "
              f"end {x['end_v6']:.2f}->{x['end_wc']:.2f} head {x['end_head_v6']:.2f}->{x['end_head_wc']:.2f} veto {x['vetoed_wc']} | pass {x['pass_v6']}->{x['pass_wc']}", flush=True)
    (dump / f'motion_rules_sign{int(sign)}.json').write_text(json.dumps(rows, indent=1), encoding='utf8')


if __name__ == '__main__':
    main(*sys.argv[1:3])
