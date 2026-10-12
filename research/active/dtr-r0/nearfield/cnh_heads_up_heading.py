"""Real head-motion check of the CNH travel-direction estimators (EXPLORE, public data).

HEADS-UP (Haghighi et al. 2024, arXiv 2409.20324): ZED Mini on a cap worn by a blind walker,
outdoor, ZED SDK VIO camera poses at 30 Hz (raw_labels.zip/*_camera_poses.csv; quaternion
order x,y,z,w; camera z = optical axis; world z ~ up, verified from data). Definitions follow
artifacts.local/work/cnh-heads-up-heading-20261007/PLAN.json, frozen before the data were read:

  truth T      centred 1 s chord direction p(t+0.5 s)-p(t-0.5 s)   (secondary: p(t+1.5 s)-p(t))
  E1           sim deployable: direction of p(t)-p(t-1 s)
  E2           sim lag-compensated: E1 + 2.5*(E1(t)-E1(t-0.2 s))
  offset       head yaw (camera optical axis, horizontal) minus T
  walking      centred 1 s horizontal speed >= 0.3 m/s, no tracking jump inside the windows used
  turning      |T(t+1 s)-T(t-1 s)|/2 s >= 10 deg/s (2 s window chosen after the 0.2 s rate proved
               gait-contaminated; the 0.2 s split is kept in step1.json as turning_fast)
The 5 Hz sim rate is reproduced by 6-frame (0.2 s) lags on the 30 Hz grid. The released easy and
hard pose files are one recording in frames rotated 180 deg about x (step lengths equal to 0.04 mm
median, head yaw-rate correlation -0.997, both 14,038 frames = the paper's Hard count), so easy is
reported only as a duplicate check and pooled = hard + unconstrained.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT/'artifacts.local/work/cnh-heads-up-heading-20261007'
RAW = WORK/'raw'
SUBSETS = ('easy', 'hard', 'unconstrained')
POOLED = ('hard', 'unconstrained')  # easy duplicates hard
HZ = 30
JUMP_M, JUMP_DEG = 0.2, 90.  # per 30 Hz frame: 6 m/s or 2700 deg/s is not head motion
OFFSETS = (5, 10, 15, 20, 30)
ERR_THR = (2, 4, 8)
TURN = 10.  # deg/s


def wrap(a):
    return (a+180.) % 360.-180.


def load(subset):
    rows = list(csv.DictReader(open(RAW/f'{subset}_camera_poses.csv')))
    p = np.array([[float(r[k]) for k in 'xyz'] for r in rows])
    x, y, z, w = np.array([[float(r[k]) for k in ('q1', 'q2', 'q3', 'q4')] for r in rows]).T
    fwd = np.stack([2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)], -1)  # third column of R
    return p, fwd


def chord_yaw(p, a, b):
    d = p[b]-p[a]
    return np.degrees(np.arctan2(d[:, 1], d[:, 0])), np.linalg.norm(d[:, :2], axis=1)


def series(subset):
    p, fwd = load(subset)
    n = len(p)
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    fa = fwd/np.linalg.norm(fwd, axis=1, keepdims=True)
    rot = np.degrees(np.arccos(np.clip(np.sum(fa[1:]*fa[:-1], 1), -1, 1)))
    bad = np.r_[False, (step > JUMP_M) | (rot > JUMP_DEG)]  # bad[i]: jump between i-1 and i
    cum = np.cumsum(bad)

    def clean(a, b):  # no jump in (a, b]
        return cum[b]-cum[a] == 0

    t = np.arange(45, n-45)  # needs t-45 (2 s turn rate) .. t+45 (secondary truth)
    T, sp = chord_yaw(p, t-15, t+15)
    T2, _ = chord_yaw(p, t, t+45)
    E1, _ = chord_yaw(p, t-30, t)
    E1p, _ = chord_yaw(p, t-36, t-6)
    E2 = E1+2.5*wrap(E1-E1p)
    head = np.degrees(np.arctan2(fwd[t, 1], fwd[t, 0]))
    Tf, _ = chord_yaw(p, t-12, t+18)
    Tb, _ = chord_yaw(p, t-18, t+12)
    fast = wrap(Tf-Tb)/0.2
    Tf, _ = chord_yaw(p, t+15, t+45)
    Tb, _ = chord_yaw(p, t-45, t-15)
    rate = wrap(Tf-Tb)/2.
    ok = clean(t-45, t+45) & (sp >= 0.3)
    return dict(t=t, ok=ok, T=T, T2=T2, E1=E1, E2=E2, head=head, rate=rate, fast=fast, speed=sp,
                n=n, jumps=int(bad.sum()))


def lowpass(e, ok):
    """Centred 1 s moving average of e inside each valid run (gait-frequency content removed)."""
    lo = np.full(len(e), np.nan)
    for a, b in segments(ok):
        x = e[a:b]; k = min(30, b-a)
        lo[a:b] = np.convolve(x, np.ones(k)/k, 'same')
    return lo[ok]


def stats(e, lo):
    a = np.abs(e)
    if len(e) == 0:
        return None
    return dict(n=int(len(e)), seconds=round(len(e)/HZ, 1), rms=float(np.sqrt(np.mean(e**2))), mean=float(np.mean(e)),
                p50=float(np.percentile(a, 50)), p90=float(np.percentile(a, 90)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), **{f'share_gt{k}': float(np.mean(a > k)) for k in ERR_THR},
                rms_low=float(np.sqrt(np.mean(lo**2))), rms_high=float(np.sqrt(np.mean((e-lo)**2))))


def offset_stats(o):
    a = np.abs(o)
    return dict(median=float(np.median(o)), mean=float(np.mean(o)), sd=float(np.std(o)),
                **{f'share_gt{k}': float(np.mean(a > k)) for k in OFFSETS},
                centred={f'share_gt{k}': float(np.mean(np.abs(o-np.median(o)) > k)) for k in OFFSETS})


def segments(ok):
    """Contiguous runs of valid samples (indices into t)."""
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    return list(zip(edges[::2], edges[1::2]))


def summarise(e1, e2, e1s, e2s, lo1, lo2, rate, fast, off):
    st, sf = np.abs(rate) < TURN, np.abs(fast) < TURN
    part = lambda e, lo: dict(all=stats(e, lo), straight=stats(e[st], lo[st]), turning=stats(e[~st], lo[~st]),
                              straight_fast=stats(e[sf], lo[sf]), turning_fast=stats(e[~sf], lo[~sf]))
    return dict(walking_s=round(len(e1)/HZ, 1), turning_share=float(np.mean(~st)), turning_fast_share=float(np.mean(~sf)),
                E1=part(e1, lo1), E2=part(e2, lo2), E1_vs_future=stats(e1s, lo1), E2_vs_future=stats(e2s, lo2),
                head_offset=offset_stats(off))


def main():
    out, series_out = {}, {}
    keys = ('e1', 'e2', 'e1s', 'e2s', 'lo1', 'lo2', 'rate', 'fast', 'off')
    pooled = {k: [] for k in keys}
    for s in SUBSETS:
        d = series(s); ok = d['ok']
        E1, E2 = wrap(d['E1']-d['T']), wrap(d['E2']-d['T'])
        v = dict(e1=E1[ok], e2=E2[ok], e1s=wrap(d['E1']-d['T2'])[ok], e2s=wrap(d['E2']-d['T2'])[ok],
                 lo1=lowpass(E1, ok), lo2=lowpass(E2, ok), rate=d['rate'][ok], fast=d['fast'][ok],
                 off=wrap(d['head']-d['T'])[ok])
        out[s] = dict(frames=d['n'], duration_s=round(d['n']/HZ, 1), jumps=d['jumps'], walking_segments=len(segments(ok)),
                      median_speed=float(np.median(d['speed'][ok])), duplicate_of_hard=(s == 'easy'), **summarise(**v))
        if s in POOLED:
            for k in keys:
                pooled[k].append(v[k])
        series_out[s] = dict(ok=ok, e1=E1, e2=E2, rate=d['rate'], off=wrap(d['head']-d['T']))
    out['pooled'] = dict(subsets=list(POOLED), **summarise(**{k: np.concatenate(v) for k, v in pooled.items()}))
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK/'step1.json').write_text(json.dumps(out, indent=1))
    np.savez_compressed(WORK/'series.npz', **{f'{s}_{k}': v for s, d in series_out.items() for k, v in d.items()})
    show(out)


def show(out):
    f = lambda x: '%.2f' % x
    for s in (*SUBSETS, 'pooled'):
        o = out[s]
        print(f"== {s}{' (duplicate of hard)' if o.get('duplicate_of_hard') else ''}: walking {o['walking_s']} s"
              + (f", jumps {o['jumps']}, segments {o['walking_segments']}, speed {o['median_speed']:.2f} m/s" if s != 'pooled' else '')
              + f", turning share 2 s {o['turning_share']:.3f} (0.2 s {o['turning_fast_share']:.3f})")
        for est in ('E1', 'E2'):
            for part in ('all', 'straight', 'turning'):
                x = o[est][part]
                print(f"  {est} {part:8s} {x['seconds']:6.1f}s RMS {f(x['rms'])} mean {f(x['mean'])} p50/p90/p99 {f(x['p50'])}/{f(x['p90'])}/{f(x['p99'])}"
                      f" >2/4/8: {x['share_gt2']:.3f}/{x['share_gt4']:.3f}/{x['share_gt8']:.3f} RMS low/high {f(x['rms_low'])}/{f(x['rms_high'])}")
            x = o[f'{est}_vs_future']
            print(f"  {est} vs future-1.5s RMS {f(x['rms'])} p50 {f(x['p50'])} p90 {f(x['p90'])}")
        h = o['head_offset']
        print(f"  head offset median {f(h['median'])} sd {f(h['sd'])} |o|>5/10/15/20/30: " + '/'.join('%.3f' % h[f'share_gt{k}'] for k in OFFSETS)
              + ' centred: ' + '/'.join('%.3f' % h['centred'][f'share_gt{k}'] for k in OFFSETS))


if __name__ == '__main__':
    main()
