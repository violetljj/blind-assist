"""walk-alert-r0 policy feasibility on SANPO-Synthetic (EXPLORE, 2026-11-16). Protocol: POLICY_PROTOCOL_20261116.md.

Stages
  prepare <work>  per-session cache (labels from fused GT static map; perception from each frame's own depth, no classes)
  count   <work>  denominator check on the frozen split (labels only; no policy run)
  run     <work>  thresholds chosen on TRAIN sessions only, then one EVAL pass + session bootstrap
  cv      <work> v2  v3: session-level 5-fold CV on the v2 cache (POLICY_PROTOCOL_V3_CV_20261116.md)
Conventions (checked in hardware-bringup/host/sanpo_cane_reach_dev_20261015.py): depth = optical z metres after 2 header
values, pose quaternion camera->world, world Z up, class id in PNG R channel. SANPO time base is unreliable, so all timing
is re-derived from walker speed over path distance.
"""
from __future__ import annotations

import csv
import glob
import gzip
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LOCAL = ROOT / 'artifacts.local/datasets/sanpo-synthetic-ba-nfo'
NEW = ROOT / 'artifacts.local/downloads/sanpo-synthetic-walk-alert-20261116'
OLDSEL = ROOT / 'artifacts.local/work/ba-nfo-20260919/sanpo-selection.json'
BLINDWAYS = ROOT / 'artifacts.local/downloads/blindways-20261007/Motion'
WALK = [1, 2, 3, 5, 6, 17, 30]; SKY = 27; DYN = [12, 13, 14, 21]
S, CELL, VOX = 8, .25, .05
X_HALF, FWD0, H, HGT = .30, .30, 4.0, (.15, 1.90)
MIN_PTS, MIN_VOX = 21, 3
DECEL, BODY_MARGIN, ALERT_LEAD, REFRACTORY_S = 2.0, .30, 1.0, 2.0
RT_RANGE = (1.16, 2.0)
K_WALKERS = 6
LATE_CAP = .05
SPLIT_SALT = 'walk-alert-r0-split:'
EVAL_FRAC = .40
SIGHTED = (1.35, .12, 1.0, 1.7)
# v2 (POLICY_PROTOCOL_V2_20261116.md): centred truth heading, >= 8 truth voxels, events >= 2 hazard frames,
# late cap = best rule's TRAIN late-rate floor + 5 pp.
V2 = False
MIN_VOX_V2, MIN_EVENT_FRAMES_V2, CAP_MARGIN_V2 = 8, 2, .05


def quat(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def sessions():
    old = [(str(LOCAL), s['scene']) for s in json.loads(OLDSEL.read_text())['sessions'] if s['split'] in ('train', 'val')]
    new = [(str(NEW), s) for s in json.loads((NEW / 'selection.json').read_text())['sessions']]
    return [x for x in old + new if (Path(x[0]) / x[1]).is_dir()]


def split_of(scene):
    h = int(hashlib.sha256((SPLIT_SALT + scene).encode()).hexdigest()[:8], 16) / 16 ** 8
    return 'eval' if h < EVAL_FRAC else 'train'


def corridor_d(z, n_min):
    """Nearest forward distance with >= n_min support within 0.30 m behind it; inf if none."""
    if len(z) < n_min:
        return np.inf
    z = np.sort(z)
    cnt = np.searchsorted(z, z + .30, side='right') - np.arange(len(z))
    ok = np.nonzero(cnt >= n_min)[0]
    return float(z[ok[0]]) if len(ok) else np.inf


def prepare_one(arg):
    root, scene, out = Path(arg[0]), arg[1], Path(arg[2]); v2 = len(arg) > 3 and arg[3]
    dst = out / f'{scene}.npz'
    if dst.exists():
        return scene
    cam = next(root.joinpath(scene).glob('camera_*'))
    meta = json.loads((root / scene / 'description.json').read_text())['session_camera_details'][0]
    p = meta['left_camera_params']; Hh, Ww = p['image_height'], p['image_width']
    vv, uu = np.mgrid[0:Hh:S, 0:Ww:S]
    dirs = np.stack([(uu - p['cx']) / p['fx'], (vv - p['cy']) / p['fy'], np.ones(uu.shape)], -1).reshape(-1, 3)
    poses = list(csv.DictReader(open(cam / 'camera_poses.csv')))
    frames = sorted((cam / 'left/depth_maps').glob('*.float16.gz'))
    P_all, L_all, ps, cams = [], [], [], []
    rng = np.random.default_rng(int(hashlib.sha256(scene.encode()).hexdigest()[:8], 16))
    for f in frames:
        k = int(f.name[:6]); r = poses[k]
        d = np.frombuffer(gzip.decompress(f.read_bytes()), '<f2')[2:].astype(np.float32).reshape(Hh, Ww)[::S, ::S].reshape(-1)
        seg = cv2.imread(str(cam / 'left/segmentation_masks' / f'{k:06d}.png'), cv2.IMREAD_UNCHANGED)[::S, ::S, 2].reshape(-1)
        R = quat(*[float(r[c]) for c in ('q_x', 'q_y', 'q_z', 'q_w')])
        t = np.array([float(r[c]) for c in ('pos_x', 'pos_y', 'pos_z')])
        ok = np.isfinite(d) & (d > .1) & (d < 30)
        ray = dirs[ok] * d[ok, None]
        # noisy-depth proxy (descriptive arm): per-frame scale exp(N(0,.10)) and 3% per-point noise
        dn = ray * (np.exp(rng.normal(0, .10)) * (1 + rng.normal(0, .03, len(ray))))[:, None]
        P_all.append(((R @ ray.T).T + t, (R @ dn.T).T + t)); L_all.append(seg[ok]); ps.append(t); cams.append(R)
    n = len(ps); ps = np.array(ps)
    if n < 5:
        np.savez_compressed(dst, n=0); return scene
    # ---- truth: GT ground grid + fused static obstacle voxels (evaluator only)
    G = np.concatenate([P[0][np.isin(L, WALK)] for P, L in zip(P_all, L_all)])
    from scipy.spatial import cKDTree
    keys, inv = np.unique(np.floor(G[:, :2] / CELL).astype(np.int64), axis=0, return_inverse=True); inv = inv.ravel()
    gz = np.array([np.median(G[inv == i, 2]) for i in range(len(keys))]); gtree = cKDTree((keys + .5) * CELL)
    cellset = {tuple(k) for k in keys}

    def height(Q):
        dist, idx = gtree.query(Q[:, :2], distance_upper_bound=1.0)
        h = np.full(len(Q), np.nan); okg = np.isfinite(dist); h[okg] = Q[okg, 2] - gz[idx[okg]]
        return h
    stat = np.concatenate([P[0][~np.isin(L, WALK + [SKY] + DYN)] for P, L in zip(P_all, L_all)])
    hs = height(stat); stat = stat[(hs >= HGT[0]) & (hs <= HGT[1])]
    vox = np.unique(np.floor(stat / VOX).astype(np.int64), axis=0); vc = (vox + .5) * VOX
    for k in np.unique(np.floor(vc[:, :2] / CELL).astype(np.int64), axis=0):
        cellset.add(tuple(k))
    seg_len = np.r_[0, np.linalg.norm(np.diff(ps[:, :2], axis=0), axis=1)]
    rows = []
    for i in range(n):
        a = max(0, i - 2); b = i if i >= 2 else min(n - 1, 2)
        v = (ps[b] - ps[a])[:2]
        vt = (ps[min(n - 1, i + 2)] - ps[max(0, i - 2)])[:2] if v2 else v   # truth heading (evaluator only)
        if np.linalg.norm(v) < .2 or np.linalg.norm(vt) < .2:   # standing/turning in place
            rows.append([np.nan] * 16); continue
        fwd = np.array([v[0], v[1], 0.]) / np.linalg.norm(v); right = np.array([fwd[1], -fwd[0], 0.])
        tf = np.array([vt[0], vt[1], 0.]) / np.linalg.norm(vt); tr_ = np.array([tf[1], -tf[0], 0.])
        rel = vc - ps[i]; x, z = rel @ tr_, rel @ tf
        inc = (np.abs(x) <= X_HALF) & (z >= FWD0) & (z <= H)
        D = corridor_d(z[inc], MIN_VOX_V2 if v2 else MIN_VOX)
        reach = min(D, H)
        samp = ps[i][:2] + np.outer(np.arange(FWD0, reach + 1e-6, CELL), tf[:2])
        cover = np.mean([tuple(np.floor(s_ / CELL).astype(np.int64)) in cellset for s_ in samp]) if len(samp) else 1.0
        P0, Lc = P_all[i][0], L_all[i]
        h0 = height(P0); rel0 = P0 - ps[i]; x0, z0 = rel0 @ tr_, rel0 @ tf
        dyn = bool(((np.abs(x0) <= X_HALF) & (z0 >= FWD0) & (z0 <= H) & np.isin(Lc, DYN) & (h0 >= HGT[0]) & (h0 <= HGT[1])).sum() >= MIN_PTS)
        feats = []
        for arm in (0, 1):   # 0 = true depth no semantics, 1 = noisy depth proxy
            Q = P_all[i][arm]; rq = Q - ps[i]; xq, zq = rq @ right, rq @ fwd
            strip = (np.abs(xq) <= 1.0) & (zq >= 1.0) & (zq <= H)
            g = np.percentile(Q[strip, 2], 5) if strip.sum() >= 50 else ps[i][2] - 1.5
            hq = Q[:, 2] - g
            c = (np.abs(xq) <= X_HALF) & (zq >= FWD0) & (zq <= H) & (hq >= HGT[0]) & (hq <= HGT[1])
            dh = corridor_d(zq[c], MIN_PTS)
            near = c & (zq <= (dh + .5 if np.isfinite(dh) else -1))
            hn = hq[near]
            feats += [dh, near.sum(), c.sum(), np.mean(hn < .5) if len(hn) else 0, np.mean(hn > 1.2) if len(hn) else 0,
                      np.mean(np.abs(xq[near])) if near.any() else 0]
        rows.append([D, cover, float(dyn), seg_len[i]] + feats)
    np.savez_compressed(dst, n=n, rows=np.array(rows, dtype=np.float64), cam=str(cam.name))
    return scene


def walkers(kind, seed, k):
    rng = np.random.default_rng(seed)
    if kind == 'sighted':
        m, s, lo, hi = SIGHTED
        v = np.clip(rng.normal(m, s, k), lo, hi)
    else:
        v = np.array([BW_SPEEDS[p][rng.integers(len(BW_SPEEDS[p]))] for p in rng.choice(sorted(BW_SPEEDS), k)])
    rt = rng.uniform(*RT_RANGE, k)
    return v, rt


def bw_speeds():
    out = {}
    for f in sorted(glob.glob(str(BLINDWAYS / '*.npy'))):
        a = np.load(f); c = a[:, :, :2].mean(1)[::30]
        v = float(np.median(np.linalg.norm(np.diff(c, axis=0), axis=1) / .5))
        if v >= .3:
            out.setdefault(os.path.basename(f)[:3], []).append(v)
    return out


BW_SPEEDS: dict = {}


def load(work):
    data = {}
    for root, scene in sessions():
        z = np.load(Path(work) / ('cache_v2' if V2 else 'cache') / f'{scene}.npz')
        if int(z['n']) >= 5:
            data[scene] = z['rows']
    return data


def episodes(scene, rows, kind):
    """Expand one session into K walker episodes with per-frame arrays."""
    seed = int(hashlib.sha256(f'{kind}:{scene}'.encode()).hexdigest()[:8], 16)
    v, rt = walkers(kind, seed, K_WALKERS)
    out = []
    valid = np.isfinite(rows[:, 0]) | np.isinf(rows[:, 0])
    valid &= ~np.isnan(rows[:, 1]) & (rows[:, 1] >= .6)
    for j in range(K_WALKERS):
        dreq = v[j] * rt[j] + v[j] ** 2 / (2 * DECEL) + BODY_MARGIN
        out.append(dict(scene=scene, v=v[j], rt=rt[j], dreq=dreq, rows=rows, valid=valid))
    return out


def feature_matrix(ep, arm):
    r = ep['rows']; o = 4 + 6 * arm
    dh = np.minimum(r[:, o], 6.0); prev = np.r_[6.0, dh[:-1]]
    v, rt, dreq = ep['v'], ep['rt'], ep['dreq']
    return np.column_stack([dh, r[:, o + 1], r[:, o + 2], r[:, o + 3], r[:, o + 4], r[:, o + 5], prev, prev - dh,
                            np.full(len(r), v), np.full(len(r), rt), dh / v, dh - dreq])


def labels(ep):
    D = ep['rows'][:, 0]; dyn = ep['rows'][:, 2] > .5
    y = np.full(len(D), -1)
    y[D <= ep['dreq'] + ALERT_LEAD] = 1
    y[np.isinf(D) & ~dyn] = 0
    if V2:
        y[weak_mask(ep)] = -1
    y[~ep['valid']] = -1
    return y


def score(ep, alerts):
    """Return (late events, avoidable events, unavoidable events, FA episodes, km)."""
    r = ep['rows']; D, dyn, ds = r[:, 0], r[:, 2] > .5, r[:, 3]
    valid = ep['valid']; haz = valid & (D <= H)
    alerts = alerts & valid
    late = avoid = unavoid = 0
    i, n = 0, len(D)
    while i < n:
        if haz[i]:
            j = i
            while j + 1 < n and (haz[j + 1] or (j + 2 < n and haz[j + 2] and valid[j + 1])):
                j += 1
            seg = slice(i, j + 1); ok = D[seg] >= ep['dreq']
            if V2 and haz[seg].sum() < MIN_EVENT_FRAMES_V2:
                i = j + 1; continue
            if ok.any():
                avoid += 1; late += int(not (alerts[seg] & ok).any())
            else:
                unavoid += 1
            i = j + 1
        else:
            i += 1
    weak = weak_mask(ep) if V2 else None
    fa = 0; idx = np.nonzero(alerts)[0]; refr = ep['v'] * REFRACTORY_S
    start = 0
    while start < len(idx):
        end = start
        while end + 1 < len(idx) and ds[idx[end] + 1:idx[end + 1] + 1].sum() <= refr:
            end += 1
        fr = idx[start:end + 1]
        if not haz[fr].any() and not dyn[fr].any() and not (V2 and weak[fr].any()):
            fa += 1
        start = end + 1
    km = ds[valid].sum() / 1000
    return late, avoid, unavoid, fa, km


def policy_alerts(name, thr, ep, arm, model=None):
    r = ep['rows']; dh = r[:, 4 + 6 * arm]
    if name == 'B1':
        return dh <= thr
    if name == 'B2':
        return dh / ep['v'] <= thr
    if name == 'B3':
        c = dh / ep['v'] <= thr
        return c & np.r_[False, c[:-1]]
    key = f'p{arm}'
    if key not in ep:
        ep[key] = model.predict_proba(np.nan_to_num(feature_matrix(ep, arm), posinf=6.0))[:, 1]
    return ep[key] >= thr


GRIDS = dict(B1=np.round(np.arange(.5, 5.01, .1), 2), B2=np.round(np.arange(.5, 5.01, .1), 2),
             B3=np.round(np.arange(.5, 5.01, .1), 2), P1=np.round(np.arange(.01, 1.0, .01), 2))


def totals(eps, name, thr, arm, model=None):
    t = np.zeros(5)
    for ep in eps:
        t += score(ep, policy_alerts(name, thr, ep, arm, model))
    late, avoid, unavoid, fa, km = t
    return dict(late=int(late), avoidable=int(avoid), unavoidable=int(unavoid), fa=int(fa), km=round(km, 4),
                late_rate=late / avoid if avoid else None, fa_per_km=fa / km if km else None)


def fit_model(eps, arm):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, y = [], []
    for ep in eps:
        yy = labels(ep); m = yy >= 0
        X.append(np.nan_to_num(feature_matrix(ep, arm), posinf=6.0)[m]); y.append(yy[m])
    X, y = np.concatenate(X), np.concatenate(y)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_leaf_nodes=15, random_state=0)
    clf.fit(X, y)
    return clf, dict(n=int(len(y)), pos=int(y.sum()))


def choose(eps, name, arm, model=None, cap=LATE_CAP):
    best = None
    for thr in GRIDS[name]:
        t = totals(eps, name, thr, arm, model)
        if t['late_rate'] is not None and t['late_rate'] <= cap + 1e-12:
            if best is None or t['fa_per_km'] < best[1]['fa_per_km']:
                best = (float(thr), t)
    return best


def cmd_prepare(work):
    out = Path(work) / ('cache_v2' if V2 else 'cache'); out.mkdir(parents=True, exist_ok=True)
    args = [(r, s, str(out), V2) for r, s in sessions()]
    t0 = time.time()
    with ProcessPoolExecutor(6) as ex:
        for k, _ in enumerate(ex.map(prepare_one, args), 1):
            if k % 25 == 0:
                print(f'{k}/{len(args)} {time.time()-t0:.0f}s', flush=True)
    print(f'prepared {len(args)} in {time.time()-t0:.0f}s')


def cmd_count(work):
    BW_SPEEDS.update(bw_speeds()); data = load(work); res = {}
    for kind in ('blind', 'sighted'):
        for sp in ('train', 'eval'):
            eps = [e for s, r in data.items() if split_of(s) == sp for e in episodes(s, r, kind)]
            z = totals(eps, 'B1', -1.0, 0)
            res[f'{kind}/{sp}'] = dict(sessions=sum(split_of(s) == sp for s in data), avoidable=z['avoidable'] // K_WALKERS,
                                       avoidable_walker_events=z['avoidable'], unavoidable_walker_events=z['unavoidable'],
                                       km_per_walker=round(z['km'] / K_WALKERS, 3))
    res['bw_participants'] = {k: len(v) for k, v in BW_SPEEDS.items()}
    (Path(work) / ('count_v2.json' if V2 else 'count.json')).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def cmd_run(work):
    BW_SPEEDS.update(bw_speeds()); data = load(work); report = dict(protocol='POLICY_PROTOCOL_20261116.md', k_walkers=K_WALKERS, late_cap=LATE_CAP)
    for kind in ('blind', 'sighted'):
        for arm, arm_name in ((0, 'true_depth_no_semantics'), (1, 'noisy_depth_proxy')):
            tr = [e for s, r in data.items() if split_of(s) == 'train' for e in episodes(s, r, kind)]
            ev = [e for s, r in data.items() if split_of(s) == 'eval' for e in episodes(s, r, kind)]
            model, minfo = fit_model(tr, arm)
            cap, floors = LATE_CAP, None
            if V2:
                floors = {n: min(totals(tr, n, t, arm)['late_rate'] for t in GRIDS[n]) for n in ('B1', 'B2', 'B3')}
                cap = min(floors.values()) + CAP_MARGIN_V2
            sel = {n: choose(tr, n, arm, model if n == 'P1' else None, cap) for n in ('B1', 'B2', 'B3', 'P1')}
            rules = {n: s for n, s in sel.items() if n != 'P1' and s}
            best_rule = min(rules, key=lambda n: rules[n][1]['fa_per_km']) if rules else None
            evals = {n: (totals(ev, n, s[0], arm, model if n == 'P1' else None) if s else None) for n, s in sel.items()}
            boot = None
            if best_rule and sel['P1']:
                by = {}
                for e in ev:
                    a = score(e, policy_alerts(best_rule, sel[best_rule][0], e, arm))
                    b = score(e, policy_alerts('P1', sel['P1'][0], e, arm, model))
                    by.setdefault(e['scene'], np.zeros(10))[:] += np.r_[a, b]
                M = np.array(list(by.values())); rng = np.random.default_rng(20261116); red = []
                for _ in range(2000):
                    s_ = M[rng.integers(len(M), size=len(M))].sum(0)
                    if s_[3] > 0:
                        red.append(1 - (s_[8] / s_[9]) / (s_[3] / s_[4]))
                red = np.array(red)
                boot = dict(best_rule=best_rule, reduction=1 - evals['P1']['fa_per_km'] / evals[best_rule]['fa_per_km']
                            if evals[best_rule]['fa'] else None,
                            ci95=[float(np.percentile(red, 2.5)), float(np.percentile(red, 97.5))] if len(red) else None,
                            sessions=len(M))
            curves = {n: [dict(thr=float(t), **totals(ev, n, t, arm, model if n == 'P1' else None)) for t in GRIDS[n][::5]]
                      for n in ('B1', 'B2', 'B3', 'P1')}
            report[f'{kind}/{arm_name}'] = dict(late_cap=cap, train_rule_floors=floors, model=minfo, train_selected=sel, eval=evals, bootstrap=boot,
                                                eval_curves_descriptive=curves)
            print(kind, arm_name, json.dumps(dict(sel={n: s[0] if s else None for n, s in sel.items()},
                                                  eval={n: (e['late'], e['avoidable'], e['fa'], round(e['fa_per_km'], 2)) if e else None
                                                        for n, e in evals.items()}, boot=boot), default=str), flush=True)
    (Path(work) / ('report_v2.json' if V2 else 'report.json')).write_text(json.dumps(report, indent=1, default=float))


# v3 (POLICY_PROTOCOL_V3_CV_20261116.md): session-level 5-fold CV on the v2 cache; selection per fold on the other folds.
CV_K, CV_SALT = 5, 'walk-alert-r0-cv:'


def fold_of(scene):
    return int(int(hashlib.sha256((CV_SALT + scene).encode()).hexdigest()[:8], 16) / 16 ** 8 * CV_K)


def score_table(eps_by, name, arm, model=None):
    """scene -> (len(grid), 5) summed score() over that scene's walker episodes."""
    out = {}
    for s, eps in eps_by.items():
        if name == 'P1':
            for e in eps:
                e.pop(f'p{arm}', None)
        out[s] = np.array([np.sum([score(e, policy_alerts(name, t, e, arm, model)) for e in eps], 0) for t in GRIDS[name]],
                          dtype=float)
    return out


def _rates(T):
    with np.errstate(divide='ignore', invalid='ignore'):
        return T[..., 0] / T[..., 1], T[..., 3] / T[..., 4]


def _pick(T, cap):
    lr, fk = _rates(T); idx = np.nonzero(lr <= cap + 1e-12)[0]
    return int(idx[np.argmin(fk[idx])]) if len(idx) else None


def _summ(T):
    late, avoid, unavoid, fa, km = T
    return dict(late=int(late), avoidable=int(avoid), unavoidable=int(unavoid), fa=int(fa), km=round(float(km), 4),
                late_rate=late / avoid if avoid else None, fa_per_km=fa / km if km else None)


def cv_one(data, kind, arm):
    eps_by = {s: episodes(s, r, kind) for s, r in data.items()}
    folds = {s: fold_of(s) for s in data}
    rules = {n: score_table(eps_by, n, arm) for n in ('B1', 'B2', 'B3')}
    pooled = {n: np.zeros(5) for n in ('B1', 'B2', 'B3', 'P1', 'best_rule')}
    per_session, fold_info, ok = {}, [], True
    for f in range(CV_K):
        tr = [s for s in data if folds[s] != f]; ho = [s for s in data if folds[s] == f]
        model, minfo = fit_model([e for s in tr for e in eps_by[s]], arm)
        p1 = score_table({s: eps_by[s] for s in data}, 'P1', arm, model)
        tabs = dict(rules, P1=p1)
        trT = {n: sum(t[s] for s in tr) for n, t in tabs.items()}
        floors = {n: float(np.nanmin(_rates(trT[n])[0])) for n in ('B1', 'B2', 'B3')}
        cap = min(floors.values()) + CAP_MARGIN_V2
        sel = {n: _pick(trT[n], cap) for n in tabs}
        sr = {n: sel[n] for n in ('B1', 'B2', 'B3') if sel[n] is not None}
        best = min(sr, key=lambda n: _rates(trT[n][sr[n]])[1]) if sr else None
        fold_info.append(dict(fold=f, train_sessions=len(tr), holdout_sessions=len(ho), cap=cap, train_rule_floors=floors,
                              model=minfo, best_rule=best,
                              thr={n: (float(GRIDS[n][i]) if i is not None else None) for n, i in sel.items()},
                              train={n: (_summ(trT[n][i]) if i is not None else None) for n, i in sel.items()},
                              holdout={n: (_summ(sum(tabs[n][s][i] for s in ho)) if i is not None else None) for n, i in sel.items()}))
        if best is None or sel['P1'] is None:
            ok = False; continue
        for n, i in sel.items():
            if i is not None:
                pooled[n] += sum(tabs[n][s][i] for s in ho)
        for s in ho:
            a = tabs[best][s][sr[best]]; b = p1[s][sel['P1']]
            pooled['best_rule'] += a; per_session[s] = np.r_[a, b]
    res = dict(folds=fold_info, evaluable=ok)
    if not ok:
        return res
    res['pooled'] = {n: _summ(t) for n, t in pooled.items()}
    pb, pp = res['pooled']['best_rule'], res['pooled']['P1']
    M = np.array(list(per_session.values())); rng = np.random.default_rng(20261116); red = []
    for _ in range(2000):
        s_ = M[rng.integers(len(M), size=len(M))].sum(0)
        if s_[3] > 0:
            red.append(1 - (s_[8] / s_[9]) / (s_[3] / s_[4]))
    red = np.array(red)
    res['bootstrap'] = dict(reduction=1 - pp['fa_per_km'] / pb['fa_per_km'] if pb['fa'] else None,
                            ci95=[float(np.percentile(red, 2.5)), float(np.percentile(red, 97.5))] if len(red) else None,
                            late_excess=pp['late'] - pb['late'], sessions=len(M))
    return res


def cmd_cv(work):
    assert V2, 'v3 CV reuses the v2 cache and v2 rules; run with the v2 flag'
    BW_SPEEDS.update(bw_speeds()); data = load(work); t0 = time.time()
    folds = {s: fold_of(s) for s in data}
    den = {}
    for kind in ('blind', 'sighted'):
        z = totals([e for s, r in data.items() for e in episodes(s, r, kind)], 'B1', -1.0, 0)
        den[kind] = dict(sessions=len(data), avoidable_per_walker=z['avoidable'] // K_WALKERS,
                         unavoidable_walker_events=z['unavoidable'], km_per_walker=round(z['km'] / K_WALKERS, 3))
    den['fold_sessions'] = [sum(v == f for v in folds.values()) for f in range(CV_K)]
    report = dict(protocol='POLICY_PROTOCOL_V3_CV_20261116.md', denominator=den)
    print(json.dumps(den), flush=True)
    if den['blind']['avoidable_per_walker'] < 30:
        report['status'] = 'NOT_EVALUABLE_DENOMINATOR'
    else:
        for kind in ('blind', 'sighted'):
            for arm, arm_name in ((0, 'true_depth_no_semantics'), (1, 'noisy_depth_proxy')):
                r = cv_one(data, kind, arm); report[f'{kind}/{arm_name}'] = r
                print(kind, arm_name, f'{time.time()-t0:.0f}s', json.dumps(dict(
                    evaluable=r['evaluable'], best=[f_['best_rule'] for f_ in r['folds']],
                    pooled={n: (p['late'], p['avoidable'], p['fa'], round(p['fa_per_km'], 2)) for n, p in r.get('pooled', {}).items()},
                    boot=r.get('bootstrap')), default=float), flush=True)
    (Path(work) / 'report_v3_cv.json').write_text(json.dumps(report, indent=1, default=float))


def weak_mask(ep):
    """v2: hazard frames belonging to runs with < MIN_EVENT_FRAMES_V2 hazard frames (excluded from events and FA)."""
    r = ep['rows']; valid = ep['valid']; haz = valid & (r[:, 0] <= H); w = np.zeros(len(haz), bool)
    i, n = 0, len(haz)
    while i < n:
        if haz[i]:
            j = i
            while j + 1 < n and (haz[j + 1] or (j + 2 < n and haz[j + 2] and valid[j + 1])):
                j += 1
            if haz[i:j + 1].sum() < MIN_EVENT_FRAMES_V2:
                w[i:j + 1] = haz[i:j + 1]
            i = j + 1
        else:
            i += 1
    return w


if __name__ == '__main__':
    V2 = len(sys.argv) > 3 and sys.argv[3] == 'v2'
    {'prepare': cmd_prepare, 'count': cmd_count, 'run': cmd_run, 'cv': cmd_cv}[sys.argv[1]](sys.argv[2])
