"""Causal travel-direction estimators from the head trajectory vs E1 on real head motion (EXPLORE, descriptive).

HEADS-UP poses (ZED Mini VIO on a blind walker's cap, 30 Hz) via cnh_heads_up_heading (imported, unchanged).
Only hard and unconstrained are used (easy duplicates hard). Every estimator uses data at or before t only;
its history is cut at the last detected tracking jump (same jump rule as the helper), which is causal and
deployable. Evaluation reuses the helper's walking mask and 2 s turning split (both use future data, eval only).

Estimators (all tried variants are reported, including grid points that were not selected):
  B0  E1      direction of p(t)-p(t-1 s)
  B1  F1.5    direction of p(t)-p(t-1.5 s)
  V1  SS-k    stride-synchronous chord over k strides; the stride period is estimated causally from the
              autocorrelation of the linearly detrended vertical head position over the last 4 s
              (step lag 0.33-0.9 s, stride = refined peak near 2 steps, ac >= 0.2, else hold last value)
              grid k in {1, 2}
  V1f SS-fix  ablation of V1: fixed window = median causal stride of the selection subset (no tracking)
  V2  SS-LC   1-stride chord C1 plus linear lag compensation from the previous stride chord C0:
              C1 + g*wrap(C1-C0); grid g in {0.25, 0.5, 0.75} (0.5 = constant-turn-rate projection to t)
  V3  EMA-ad  exponential low-pass of the 30 Hz horizontal velocity vector, time constant tau_s when the
              causal rate of its own direction over 0.5 s is < 10 deg/s, else 0.25 s; grid tau_s in {0.5, 0.75, 1.0} s
Fixed a priori (not tuned): 4 s autocorrelation history, 0.33-0.9 s step range, ac >= 0.2, 0.25 s turn tau,
10 deg/s switch. Selection: lowest all-walking RMS vs the primary truth on the selection subset (hard);
unconstrained is held out; the reverse selection is reported as a check.
Truths: T1 centred 1 s chord (primary), T2c centred 2 s chord, Tf future 1.5 s chord p(t+1.5 s)-p(t).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cnh_heads_up_heading as H  # noqa: E402

OUT = H.WORK/'gait_estimator'
HZ = H.HZ
SUBS = ('hard', 'unconstrained')
AC_L, AC_MIN_LEN = 120, 90
STEP_LAGS = (10, 27)
AC_MIN = 0.2
DEFAULT_P = 30.
TAU_TURN, SWITCH = 0.25, 10.
GRID = dict(V1=[1, 2], V2=[0.25, 0.5, 0.75], V3=[0.5, 0.75, 1.0])
TRUTHS = ('T1', 'T2c', 'Tf')
PARTS = ('straight', 'turning', 'all')
BLOCK, NBOOT = 150, 2000
TRACKER = sys.argv[1] if len(sys.argv) > 1 else 'localmax'  # 'argmax' reproduces run 1
RESULT = 'result.json' if TRACKER == 'localmax' else 'result_run1_argmax_tracker.json'
wrap = H.wrap


def jump_mask(p, fwd):
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    fa = fwd/np.linalg.norm(fwd, axis=1, keepdims=True)
    rot = np.degrees(np.arccos(np.clip(np.sum(fa[1:]*fa[:-1], 1), -1, 1)))
    return np.r_[False, (step > H.JUMP_M) | (rot > H.JUMP_DEG)]


def stride_period(z, j):
    """Causal stride period (frames) from vertical autocorrelation; returns P and a validity flag."""
    n = len(z)
    P, valid = np.full(n, DEFAULT_P), np.zeros(n, bool)
    last = DEFAULT_P
    for t in range(n):
        a = max(t-AC_L+1, j[t])
        m = t-a+1
        if m >= AC_MIN_LEN:
            x = z[a:t+1]
            u = np.arange(m)-(m-1)/2
            x = x-x.mean()-u*(u @ x)/(u @ u)
            c = np.correlate(x, x, 'full')[m-1:]
            e = np.cumsum(x*x)
            k = np.arange(1, 61)
            ac = np.r_[1., c[k]/np.sqrt(e[m-1-k]*(e[m-1]-e[k-1]))]
            if TRACKER == 'argmax':  # run 1: boundary maxima of a smooth ac were accepted as steps
                ks = STEP_LAGS[0]+int(np.argmax(ac[STEP_LAGS[0]:STEP_LAGS[1]+1]))
            else:  # run 2: highest local maximum inside the step range
                r = np.arange(STEP_LAGS[0], STEP_LAGS[1]+1)
                pk = r[(ac[r] > ac[r-1]) & (ac[r] >= ac[r+1])]
                ks = int(pk[np.argmax(ac[pk])]) if len(pk) else STEP_LAGS[0]
                if not len(pk):
                    P[t] = last
                    continue
            if ac[ks] >= AC_MIN:
                lo, hi = max(2*ks-4, 20), min(2*ks+4, 59)
                kk = lo+int(np.argmax(ac[lo:hi+1]))
                y0, y1, y2 = ac[kk-1], ac[kk], ac[kk+1]
                den = y0-2*y1+y2
                d = 0.5*(y0-y2)/den if den < 0 else 0.
                if TRACKER != 'argmax' and not (y1 > y0 and y1 >= y2):
                    kk, d, y1 = 2*ks, 0., ac[2*ks]
                if y1 >= AC_MIN:
                    last = kk+float(np.clip(d, -0.5, 0.5))
                    valid[t] = True
        P[t] = last
    return P, valid


def pos(p, s):
    f = np.floor(s).astype(int)
    a = (s-f)[:, None]
    return p[f]*(1-a)+p[np.minimum(f+1, len(p)-1)]*a


def yaw(d):
    return np.degrees(np.arctan2(d[:, 1], d[:, 0]))


def chord_back(p, t, W, j):
    """Direction of p(t)-p(t-W), W in frames (fractional), history cut at the last jump."""
    W = np.minimum(W, t-j[t])
    return yaw(p[t]-pos(p, t-W)), W


def ema_adaptive(p, bad, tau_s):
    n = len(p)
    th = np.full(n, np.nan)
    m = np.zeros(2)
    start, tau = 0, tau_s
    for i in range(1, n):
        if bad[i]:
            m[:] = 0; start = i; tau = tau_s
            continue
        a = 1-np.exp(-1/(HZ*tau))
        m = (1-a)*m+a*(p[i, :2]-p[i-1, :2])
        th[i] = np.degrees(np.arctan2(m[1], m[0]))
        if i-15 > start and np.isfinite(th[i-15]):
            tau = TAU_TURN if abs(wrap(th[i]-th[i-15]))/0.5 >= SWITCH else tau_s
    return th


def estimators(s, sel_fixed=None):
    p, fwd = H.load(s)
    d = H.series(s)
    t, ok = d['t'], d['ok']
    bad = jump_mask(p, fwd)
    j = np.maximum.accumulate(np.where(bad, np.arange(len(p)), 0))
    P, valid = stride_period(p[:, 2], j)
    est = {}
    est['B0 E1'], _ = chord_back(p, t, np.full(len(t), 30.), j)
    est['B1 F1.5'], _ = chord_back(p, t, np.full(len(t), 45.), j)
    fallback = {}
    for k in GRID['V1']:
        est[f'V1 SS-{k}'], W = chord_back(p, t, k*P[t], j)
        fallback[f'V1 SS-{k}'] = float(np.mean((W < k*P[t])[ok]))
    W1 = np.minimum(P[t], t-j[t])
    C1 = yaw(p[t]-pos(p, t-W1))
    has0 = t-2*W1 >= j[t]
    C0 = yaw(pos(p, t-W1)-pos(p, np.maximum(t-2*W1, j[t])))
    for g in GRID['V2']:
        est[f'V2 SS-LC g={g}'] = C1+np.where(has0, g, 0.)*wrap(C1-C0)
    fallback['V2'] = float(np.mean(~has0[ok]))
    for tau in GRID['V3']:
        est[f'V3 EMA-ad tau={tau}'] = ema_adaptive(p, bad, tau)[t]
    if sel_fixed is not None:
        est[f'V1f SS-fix {sel_fixed/HZ:.3f}s'], _ = chord_back(p, t, np.full(len(t), sel_fixed), j)
    truth = dict(T1=d['T'], T2c=H.chord_yaw(p, t-30, t+30)[0], Tf=d['T2'])
    diag = dict(stride_valid_share_walking=float(np.mean(valid[t][ok])),
                stride_s_walking_p10_p50_p90=[float(x)/HZ for x in np.percentile(P[t][ok], [10, 50, 90])],
                v1_window_truncated_share=fallback, v2_no_previous_chord_share=fallback['V2'])
    return dict(t=t, ok=ok, rate=d['rate'], est=est, truth=truth, P=P, diag=diag)


def stats(e, t):
    a = np.abs(e)
    f = (t % 6) == 0
    return dict(n=int(len(e)), seconds=round(len(e)/HZ, 1), rms=float(np.sqrt(np.mean(e**2))), mean=float(np.mean(e)),
                p50=float(np.median(a)), p90=float(np.percentile(a, 90)), share_gt2=float(np.mean(a > 2)),
                share_gt4=float(np.mean(a > 4)), rms_5hz=float(np.sqrt(np.mean(e[f]**2))), n_5hz=int(f.sum()))


def evaluate(r):
    ok, t = r['ok'], r['t'][r['ok']]
    st = np.abs(r['rate'][ok]) < H.TURN
    masks = dict(straight=st, turning=~st, all=np.ones_like(st))
    res, err = {}, {}
    for name, e in r['est'].items():
        res[name], err[name] = {}, {}
        for T in TRUTHS:
            x = wrap(e-r['truth'][T])[ok]
            err[name][T] = x
            res[name][T] = {pt: stats(x[m], t[m]) for pt, m in masks.items()}
    ref = {}
    for a, b in (('T1', 'T2c'), ('T1', 'Tf')):
        x = wrap(r['truth'][a]-r['truth'][b])[ok]
        ref[f'{a}_minus_{b}'] = {pt: stats(x[m], t[m]) for pt, m in masks.items()}
    return res, err, masks, ref


def select(res, fam):
    names = [n for n in res if n.startswith(fam+' ')]
    return min(names, key=lambda n: res[n]['T1']['all']['rms'])


def boot_delta(ev, eb, seed):
    """Moving-block bootstrap (5 s blocks) of RMS(ev)-RMS(eb), paired samples in time order."""
    rng = np.random.default_rng(seed)
    n = len(ev)
    nb = int(np.ceil(n/BLOCK))
    st = rng.integers(0, max(n-BLOCK, 1), size=(NBOOT, nb))
    idx = (st[:, :, None]+np.arange(BLOCK)).reshape(NBOOT, -1)[:, :n]
    d = np.sqrt(np.mean(ev[idx]**2, 1))-np.sqrt(np.mean(eb[idx]**2, 1))
    return dict(delta=float(np.sqrt(np.mean(ev**2))-np.sqrt(np.mean(eb**2))),
                ci95=[float(x) for x in np.percentile(d, [2.5, 97.5])])


def run_direction(sel_s, held_s, cache):
    res_sel = cache[sel_s]['res']
    chosen = {f: select(res_sel, f) for f in ('V1', 'V2', 'V3')}
    fixed = float(np.median(cache[sel_s]['r']['P'][cache[sel_s]['r']['t'][cache[sel_s]['r']['ok']]]))
    r = estimators(held_s, sel_fixed=fixed)
    res, err, masks, _ = evaluate(r)
    fix_name = [n for n in res if n.startswith('V1f')][0]
    rows = ['B0 E1', 'B1 F1.5', chosen['V1'], fix_name, chosen['V2'], chosen['V3']]
    boot = {}
    for nm in rows[1:]:
        boot[nm] = {T: {pt: boot_delta(err[nm][T][masks[pt]], err['B0 E1'][T][masks[pt]], seed=7)
                        for pt in PARTS} for T in TRUTHS}
    return dict(selection_subset=sel_s, held_out_subset=held_s, chosen=chosen,
                v1f_fixed_window_s=fixed/HZ, rows=rows, held_out={nm: res[nm] for nm in rows},
                delta_rms_vs_E1_block_bootstrap=boot)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    for s in SUBS:
        r = estimators(s)
        res, _, masks, ref = evaluate(r)
        cache[s] = dict(r=r, res=res, ref=ref, diag=r['diag'], walking_s=round(int(r['ok'].sum())/HZ, 1),
                        turning_share=float(np.mean(~masks['straight'])))
    out = dict(task='CNH_HEADS_UP_GAIT_ESTIMATOR_20261007', lane='EXPLORE descriptive; public HEADS-UP VIO poses; CPU only',
               definitions=__doc__, subsets=list(SUBS), excluded=['easy (duplicate of hard)'],
               selection_criterion='lowest all-walking RMS vs T1 on the selection subset, per family',
               grid=GRID, fixed=dict(ac_history_s=AC_L/HZ, step_range_s=[x/HZ for x in STEP_LAGS], ac_min=AC_MIN,
                                     tau_turn_s=TAU_TURN, switch_deg_s=SWITCH, block_s=BLOCK/HZ, n_boot=NBOOT),
               per_subset_all_tried={s: dict(walking_s=c['walking_s'], turning_share=c['turning_share'], stride=c['diag'],
                                             truth_disagreement=c['ref'], results=c['res']) for s, c in cache.items()})
    out['primary'] = run_direction('hard', 'unconstrained', cache)
    out['reverse_check'] = run_direction('unconstrained', 'hard', cache)
    out['stride_tracker'] = TRACKER
    out['runtime_s'] = round(time.time()-t0, 1)
    (OUT/RESULT).write_text(json.dumps(out, indent=1))
    show(out, cache)


def show(out, cache):
    for s, c in cache.items():
        print(f"== {s}: walking {c['walking_s']} s, turning share {c['turning_share']:.3f}, stride {c['diag']}")
        print(f"   truth T1-T2c straight RMS {c['ref']['T1_minus_T2c']['straight']['rms']:.2f}, T1-Tf straight RMS {c['ref']['T1_minus_Tf']['straight']['rms']:.2f}")
        for nm, v in c['res'].items():
            print('   %-22s' % nm + ' | '.join(f"{T} " + ' '.join(f"{v[T][pt]['rms']:.2f}/{v[T][pt]['p50']:.2f}" for pt in PARTS) for T in TRUTHS))
    for key in ('primary', 'reverse_check'):
        o = out[key]
        print(f"== {key}: select on {o['selection_subset']} -> {o['chosen']}, V1f {o['v1f_fixed_window_s']:.3f}s; held-out {o['held_out_subset']}")
        for nm in o['rows']:
            v = o['held_out'][nm]
            line = '   %-22s' % nm + ' | '.join(f"{T} " + ' '.join(f"{v[T][pt]['rms']:.2f}/{v[T][pt]['p50']:.2f}" for pt in PARTS) for T in TRUTHS)
            if nm in o['delta_rms_vs_E1_block_bootstrap']:
                b = o['delta_rms_vs_E1_block_bootstrap'][nm]['T1']
                line += '  dT1 st %+.2f [%+.2f,%+.2f] all %+.2f [%+.2f,%+.2f]' % (
                    b['straight']['delta'], *b['straight']['ci95'], b['all']['delta'], *b['all']['ci95'])
            print(line)
    print('runtime', out['runtime_s'], 's')


if __name__ == '__main__':
    main()
