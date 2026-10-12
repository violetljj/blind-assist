"""Descriptive check: time-based vs distance-based travel-direction estimates against the future
path direction (head position 1.5 m ahead along the path), on BlindWays (60 Hz) and HEADS-UP
unconstrained (30 Hz). Causal estimates: E1 = p(t)-p(t-1 s); D0.8 = p(t)-p(back point 0.8 m of path
earlier); EMA as in cnh_blindways_heading. Walking = centred 1 s speed >= 0.3 m/s."""
import glob
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cnh_blindways_heading as B  # noqa: E402
import cnh_heads_up_heading as H  # noqa: E402

wrap = B.wrap


def analyse(h, hz, ok_extra=None, L_back=.8, L_fwd=1.5):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(h, axis=0), axis=1))]
    n = len(h); t = np.arange(hz, n-hz)
    back = np.searchsorted(s, s[t]-L_back, side='right')-1; fwd = np.searchsorted(s, s[t]+L_fwd, side='left')
    valid = (back >= 0) & (fwd < n)
    t, back, fwd = t[valid], back[valid], fwd[valid]
    F = B.yaw(h[fwd]-h[t])
    E1 = B.yaw(h[t]-h[t-hz]); D = B.yaw(h[t]-h[back])
    speed = np.linalg.norm(h[np.minimum(t+hz//2, n-1)]-h[t-hz//2], axis=1)
    ok = speed >= .3
    if ok_extra is not None:
        ok &= ok_extra[t]
    return dict(e1=wrap(E1-F)[ok], d=wrap(D-F)[ok], speed=speed[ok], t=t[ok])


def rms(e):
    return np.sqrt(np.mean(e**2)), np.median(np.abs(e))


def main():
    rows = {}
    for f in sorted(glob.glob(str(B.SRC/'*.npy'))):
        x = np.load(f); h = x[:, 6, :2]; pid = Path(f).stem.split('_')[0]
        r = analyse(h, 60); em = B.ema(np.c_[h, np.zeros(len(h))])
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(h, axis=0), axis=1))]
        rows.setdefault(pid, []).append(r)
    print('dataset/pid   n_s   speed  E1 vs future RMS/med   D0.8 vs future RMS/med')
    allr = []
    for pid, L in sorted(rows.items()):
        e1 = np.concatenate([r['e1'] for r in L]); d = np.concatenate([r['d'] for r in L]); sp = np.concatenate([r['speed'] for r in L])
        allr.append((e1, d, sp))
        print(f'BW {pid}  {len(e1)/60:7.1f}  {np.median(sp):.2f}   %5.2f / %5.2f        %5.2f / %5.2f' % (*rms(e1), *rms(d)))
    e1 = np.concatenate([a[0] for a in allr]); d = np.concatenate([a[1] for a in allr]); sp = np.concatenate([a[2] for a in allr])
    print(f'BW pooled {len(e1)/60:7.1f}  {np.median(sp):.2f}   %5.2f / %5.2f        %5.2f / %5.2f' % (*rms(e1), *rms(d)))
    for lo, hi in ((.3, .6), (.6, .9), (.9, 3)):
        m = (sp >= lo) & (sp < hi)
        print(f'   speed {lo}-{hi}: {m.sum()/60:7.1f}s  E1 %5.2f/%5.2f   D0.8 %5.2f/%5.2f' % (*rms(e1[m]), *rms(d[m])))
    for sub in ('unconstrained', 'hard'):
        p, _ = H.load(sub); step = np.linalg.norm(np.diff(p, axis=0), axis=1); bad = np.r_[False, step > .2]
        seg = np.cumsum(bad); res = []
        for k in np.unique(seg):
            idx = np.flatnonzero(seg == k)
            if len(idx) > 300:
                res.append(analyse(p[idx, :2], 30))
        e1 = np.concatenate([r['e1'] for r in res]); d = np.concatenate([r['d'] for r in res]); sp = np.concatenate([r['speed'] for r in res])
        print(f'HU {sub:13s} {len(e1)/30:7.1f}  {np.median(sp):.2f}   %5.2f / %5.2f        %5.2f / %5.2f' % (*rms(e1), *rms(d)))


if __name__ == '__main__':
    main()
