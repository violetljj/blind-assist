"""Body-reference travel-direction estimates on BlindWays (EXPLORE, descriptive).

Truth: direction from the pelvis to the pelvis position 1.5 m further along its path (FUT, primary;
where the body goes inside the alarm horizon) and the centred 2 s pelvis chord (C2). Causal estimates:
head E1 (1 s head displacement, current method), pelvis E1 (1 s pelvis displacement; body-worn
device), torso (shoulder-line normal), pelvis EMA (tau 0.5/0.25 s). Walking: centred 1 s pelvis speed
>= 0.3 m/s; split by that speed.
"""
import glob
import json
from pathlib import Path

import numpy as np

import cnh_blindways_heading as B

wrap, yaw, HZ = B.wrap, B.yaw, 60


def clip(x):
    h, pv = x[:, 6, :2], x[:, 0, :2]; n = len(x)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pv, axis=0), axis=1))]
    t = np.arange(60, n-60); fwd = np.searchsorted(s, s[t]+1.5, side='left'); v = fwd < n; t, fwd = t[v], fwd[v]
    FUT = yaw(pv[fwd]-pv[t]); C2 = yaw(pv[np.minimum(t+60, n-1)]-pv[t-60])
    sh = x[:, 11, :2]-x[:, 7, :2]; torso = np.degrees(np.arctan2(-sh[:, 0], sh[:, 1]))[t]
    est = dict(head_E1=yaw(h[t]-h[t-60]), pelvis_E1=yaw(pv[t]-pv[t-60]), torso=torso,
               pelvis_EMA=B.ema(np.c_[pv, np.zeros(n)])[t])
    speed = np.linalg.norm(pv[np.minimum(t+30, n-1)]-pv[t-30], axis=1)
    return dict(est=est, FUT=FUT, C2=C2, speed=speed, ok=speed >= .3)


def main():
    acc = {}
    for f in sorted(glob.glob(str(B.SRC/'*.npy'))):
        d = clip(np.load(f))
        for k, e in d['est'].items():
            for tn in ('FUT', 'C2'):
                acc.setdefault((k, tn), []).append(wrap(e-d[tn])[d['ok']])
        acc.setdefault('speed', []).append(d['speed'][d['ok']])
    sp = np.concatenate(acc.pop('speed')); out = {}
    print('estimate      truth  ALL rms/med     v<0.6       0.6-0.9      >=0.9')
    for (k, tn), L in acc.items():
        e = np.concatenate(L); row = {}
        for name, m in (('all', np.ones(len(e), bool)), ('slow', sp < .6), ('mid', (sp >= .6) & (sp < .9)), ('fast', sp >= .9)):
            row[name] = dict(rms=float(np.sqrt(np.mean(e[m]**2))), p50=float(np.median(np.abs(e[m]))), seconds=float(m.sum()/HZ))
        out[f'{k}/{tn}'] = row
        print(f"{k:12s} {tn:4s}  " + '   '.join('%5.1f/%4.1f' % (row[n]['rms'], row[n]['p50']) for n in ('all', 'slow', 'mid', 'fast')))
    (B.OUT/'body_reference.json').write_text(json.dumps(out, indent=1)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
