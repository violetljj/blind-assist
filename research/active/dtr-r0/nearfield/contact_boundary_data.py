"""Continuous swept-cross-section queries; privileged labels stay outside models."""
from __future__ import annotations

import numpy as np

TRAIN_WIDTHS = (.36, .60, .84, 1.08)
TRAIN_HORIZONS = (.6, .9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.)
NEW_WIDTHS = (.48, .72, .96)
NEW_HORIZONS = (.75, 1.05, 1.35, 1.65, 1.95, 2.25, 2.55, 2.85)
LAYERS = ((.42, .9), (-.2, .42))


def grid(widths, horizons):
    return np.asarray([(w, h, layer) for layer in range(2)
                       for w in widths for h in horizons], dtype=np.float32)


def queries():
    return dict(seen=grid(TRAIN_WIDTHS, TRAIN_HORIZONS),
        width=grid(NEW_WIDTHS, TRAIN_HORIZONS),
        horizon=grid(TRAIN_WIDTHS, NEW_HORIZONS),
        both=grid(NEW_WIDTHS, NEW_HORIZONS),
        extrapolation=grid((.24, 1.20), NEW_HORIZONS),
        width_curve=grid(np.linspace(.2, 1.2, 51), (3.,)),
        horizon_curve=grid((.6,), np.linspace(.3, 3., 55)))


def camera_boxes(geometry):
    c = geometry['declared_camera']
    if any(abs(c[k]) > 1e-8 for k in ('pitch', 'yaw', 'roll')):
        raise ValueError('Only the declared level-camera source is admitted')
    actual = geometry['actual_camera_location_m']
    if max(abs(actual[j]-c[k]) for j, k in enumerate(('x', 'y', 'z'))) >= .002:
        raise ValueError('Rendered camera disagrees with declared source')
    boxes = []
    for obj in geometry['objects']:
        centre = np.asarray(obj['render_bounds_center_m'], dtype=np.float64)
        extent = np.asarray(obj['render_bounds_extent_m'], dtype=np.float64)
        xyz = np.array([centre[1]-c['y'], c['z']-centre[2], centre[0]-c['x']])
        half = extent[[1, 2, 0]]
        boxes.append((xyz-half, xyz+half))
    return boxes


def contact_labels(boxes, query):
    query = np.asarray(query, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 3 or not np.isfinite(query).all():
        raise ValueError('Expected finite Qx3 queries')
    w, h, layer = query.T
    if not ((w > 0).all() and (w <= 1.2000001).all() and (h >= .3-1e-7).all()
            and (h <= 3.0000001).all() and np.isin(layer, [0, 1]).all()):
        raise ValueError('Query outside source-audited domain')
    bands = np.asarray(LAYERS)[layer.astype(int)]
    first = np.full(len(query), np.inf)
    for lo, hi in boxes:
        overlap = ((hi[0] >= -w/2) & (lo[0] <= w/2)
                   & (hi[1] >= bands[:, 0]) & (lo[1] <= bands[:, 1])
                   & (hi[2] >= .3) & (lo[2] <= h))
        first = np.minimum(first, np.where(overlap, max(float(lo[2]), .3), np.inf))
    return np.isfinite(first), first-.3


def critical_values(boxes, kind):
    """Exact boundaries, including out-of-grid infinities and left censoring."""
    values = []
    for yl, yh in LAYERS:
        candidates = []
        for lo, hi in boxes:
            if hi[1] < yl or lo[1] > yh or hi[2] < .3:
                continue
            if kind == 'width':
                if lo[2] <= 3:
                    candidates.append(2*max(0., float(lo[0]), float(-hi[0])))
            elif kind == 'horizon':
                if hi[0] >= -.3 and lo[0] <= .3:
                    candidates.append(max(.3, float(lo[2])))
            else:
                raise ValueError(kind)
        values.append(min(candidates, default=np.inf))
    return np.array(values, np.float64)


def counts(score, truth, threshold):
    p, y = np.asarray(score, np.float64) >= threshold, np.asarray(truth, bool)
    tp, fp, fn, tn = (int(v.sum()) for v in (p & y, p & ~y, ~p & y, ~p & ~y))
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, recall=tp/(tp+fn) if tp+fn else None,
        precision=tp/(tp+fp) if tp+fp else None, FPR=fp/(fp+tn) if fp+tn else None,
        Brier=float(np.mean((np.asarray(score, np.float64)-y)**2)))


def choose_threshold(score, truth):
    """One atomic dev-only threshold, including the all-negative point."""
    s, y = np.asarray(score, np.float64).ravel(), np.asarray(truth, bool).ravel()
    if not np.isfinite(s).all() or not y.any() or y.all():
        raise ValueError('Need finite scores and both dev classes')
    order = np.argsort(-s, kind='stable')
    ss, yy = s[order], y[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss)-1]
    tp, fp = yy.cumsum()[ends], (~yy).cumsum()[ends]
    feasible = fp <= .05*(~y).sum()+1e-10
    if not feasible.any():
        return float(np.nextafter(s.max(), np.inf))
    options = np.flatnonzero(feasible)
    best = max(options, key=lambda i: (int(tp[i]), -int(fp[i]), float(ss[ends[i]])))
    if tp[best] == 0:
        return float(np.nextafter(s.max(), np.inf))
    return float(ss[ends[best]])


def boundary_metrics(probability, threshold, truth, kind):
    axis = np.linspace(.2, 1.2, 51) if kind == 'width' else np.linspace(.3, 3., 55)
    p = np.asarray(probability, np.float64).reshape(-1, 2, len(axis))
    on = p >= threshold
    predicted = np.where(on.any(-1), axis[on.argmax(-1)], np.inf)
    finite = np.isfinite(truth) & (truth > axis[0]+1e-6) & (truth <= axis[-1]+1e-6)
    resolved = finite & np.isfinite(predicted)
    error = np.full(truth.shape, np.inf)
    error[resolved] = np.abs(predicted[resolved]-truth[resolved])
    n = int(finite.sum())
    return dict(interior_true_boundaries=n, left_censored_truth=int((truth <= axis[0]+1e-6).sum()),
        right_censored_truth=int((truth > axis[-1]+1e-6).sum()),
        predicted_boundary_coverage=float(resolved.sum()/n) if n else None,
        conditional_MAE_m=float(error[resolved].mean()) if resolved.any() else None,
        within_5cm=int((finite & (error <= .05+1e-6)).sum()),
        joint_within_5cm=float((finite & (error <= .05+1e-6)).sum()/n) if n else None,
        wrong_crossings_on_right_censored=int(((truth > axis[-1]+1e-6) & np.isfinite(predicted)).sum()),
        probability_monotonic_violations=int((np.diff(p, axis=-1) < -1e-6).sum()),
        binary_reversals=int((on[..., :-1] & ~on[..., 1:]).sum()))
