"""Travel-direction estimation error and torso offset on BlindWays (11 blind/low-vision walkers).

BlindWays (Kim et al., NeurIPS 2024): Xsens joint positions only (no orientations), 60 Hz,
10 s clips (600 frames x 24 joints x 3, z up). Head yaw is not recoverable, so this checks the
heading-estimate side (P3 input) across participants, plus torso yaw from the shoulder line.
Definitions mirror cnh_heads_up_heading.py on the head-position trajectory:
  truth T1   centred 1 s chord of the head position (primary); T2 centred 2 s; P1 pelvis centred 1 s
  E1         direction of head p(t) - p(t-1 s)
  EMA        adaptive EMA of 60 Hz head velocity (tau 0.5 s, 0.25 s when own 0.5 s rate >= 10 deg/s)
  walking    centred 1 s horizontal head speed >= 0.3 m/s; turning |T1(t+1 s)-T1(t-1 s)|/2 s >= 10 deg/s
  torso      forward = (L shoulder - R shoulder) x up, offset vs T1
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT/'artifacts.local/downloads/blindways-20261007/Motion'
OUT = ROOT/'artifacts.local/work/cnh-blindways-heading-20261007'
HZ = 60
J = dict(pelvis=0, head=6, rsh=7, lsh=11)


def wrap(a):
    return (np.asarray(a)+180.) % 360.-180.


def yaw(d):
    return np.degrees(np.arctan2(d[..., 1], d[..., 0]))


def ema(p, tau_s=.5, tau_t=.25, switch=10.):
    v = np.diff(p[:, :2], axis=0)*HZ; m = v[0].copy(); th = np.full(len(p), np.nan); tau = tau_s
    for i in range(1, len(p)):
        a = 1-np.exp(-1/(HZ*tau)); m = (1-a)*m+a*v[i-1]; th[i] = np.degrees(np.arctan2(m[1], m[0]))
        if i >= 31 and np.isfinite(th[i-30]):
            tau = tau_t if abs(wrap(th[i]-th[i-30]))/.5 >= switch else tau_s
    return th


def clip(x):
    h, pv = x[:, J['head']], x[:, J['pelvis']]
    t = np.arange(90, len(x)-90)  # needs t-90 (turn rate) .. t+90
    T1 = yaw(h[t+30]-h[t-30]); T2 = yaw(h[t+60]-h[t-60]); P1 = yaw(pv[t+30]-pv[t-30])
    E1 = yaw(h[t]-h[t-60]); EM = ema(h)[t]
    speed = np.linalg.norm((h[t+30]-h[t-30])[:, :2], axis=1)
    rate = wrap(yaw(h[t+90]-h[t+30])-yaw(h[t-30]-h[t-90]))/2.  # |T1(t+1 s)-T1(t-1 s)|/2 s
    sh = x[:, J['lsh']]-x[:, J['rsh']]; fwd = np.stack([sh[:, 1], -sh[:, 0]], 1)  # (L-R) x z
    torso = np.degrees(np.arctan2(fwd[t, 1], fwd[t, 0]))
    ok = speed >= .3
    return dict(ok=ok, turn=np.abs(rate) >= 10., e1=wrap(E1-T1), e1_T2=wrap(E1-T2), e1_P1=wrap(E1-P1), ema=wrap(EM-T1),
                torso=wrap(torso-T1), speed=speed)


def stats(e):
    a = np.abs(e)
    return None if len(e) == 0 else dict(n_s=round(len(e)/HZ, 1), rms=float(np.sqrt(np.mean(e**2))), p50=float(np.median(a)),
                                         p90=float(np.percentile(a, 90)), gt2=float(np.mean(a > 2)), gt4=float(np.mean(a > 4)))


def main():
    files = sorted(glob.glob(str(SRC/'*.npy'))); per = {}
    for f in files:
        pid = Path(f).stem.split('_')[0]; d = clip(np.load(f))
        per.setdefault(pid, []).append(d)
    keys = ('e1', 'e1_T2', 'e1_P1', 'ema', 'torso', 'turn', 'ok', 'speed')
    out = {}
    def summarise(ds):
        c = {k: np.concatenate([d[k] for d in ds]) for k in keys}; ok = c['ok']; st = ok & ~c['turn']; tu = ok & c['turn']
        o = c['torso'][ok]; med = float(np.median(o))
        return dict(clips=len(ds), walking_s=round(ok.sum()/HZ, 1), turning_share=float(tu.sum()/max(ok.sum(), 1)),
                    median_speed=float(np.median(c['speed'][ok]/1.)),
                    E1=dict(all=stats(c['e1'][ok]), straight=stats(c['e1'][st]), turning=stats(c['e1'][tu])),
                    E1_vs_T2_straight=stats(c['e1_T2'][st]), E1_vs_pelvis_straight=stats(c['e1_P1'][st]),
                    EMA=dict(all=stats(c['ema'][ok]), straight=stats(c['ema'][st]), turning=stats(c['ema'][tu])),
                    torso=dict(median=med, sd=float(np.std(o)), **{f'centred_gt{k}': float(np.mean(np.abs(o-med) > k)) for k in (10, 15, 20, 30)}))
    for pid, ds in sorted(per.items()):
        out[pid] = summarise(ds)
    out['pooled'] = summarise([d for ds in per.values() for d in ds])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'result.json').write_text(json.dumps(out, indent=1)+'\n', encoding='utf8')
    f = lambda x: '%5.2f' % x
    print('pid  clips walk_s  turn%  E1 all/str/turn RMS   E1 str p50  EMA all/str/turn RMS   torso sd  >15c')
    for pid, o in out.items():
        print(f"{pid:6s} {o['clips']:4d} {o['walking_s']:7.1f} {100*o['turning_share']:5.1f}  "
              f"{f(o['E1']['all']['rms'])}/{f(o['E1']['straight']['rms'])}/{f(o['E1']['turning']['rms'] if o['E1']['turning'] else float('nan'))}  "
              f"{f(o['E1']['straight']['p50'])}   {f(o['EMA']['all']['rms'])}/{f(o['EMA']['straight']['rms'])}/{f(o['EMA']['turning']['rms'] if o['EMA']['turning'] else float('nan'))}  "
              f"{f(o['torso']['sd'])}  {100*o['torso']['centred_gt15']:.1f}%")
    p = out['pooled']
    print('pooled E1 straight vs T2', p['E1_vs_T2_straight'], '\n vs pelvis', p['E1_vs_pelvis_straight'])


if __name__ == '__main__':
    main()
