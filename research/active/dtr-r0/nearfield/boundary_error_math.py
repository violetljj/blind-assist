"""Pure diagnostic geometry/constraint functions; no model or data access."""
from __future__ import annotations

import numpy as np

LAYERS = ((.42, .9), (-.2, .42))


def label_bracket(query, labels, kind, layer):
    """Sharp lower-open/upper-closed crossing constraints from monotonicity.

    Width has infimum0; horizon has lower domain endpoint.3. Lower endpoint
    inclusivity only matters for exact endpoint labels, outside primary truths.
    """
    # These declared decimal knots have float32 encoding noise. Reconcile only
    # the known query-grid coordinates to micrometre precision, not truths.
    w, h, band = np.round(np.asarray(query, float), 6).T
    y = np.asarray(labels, bool)
    mask = band == layer
    if kind == 'width':
        coordinate, fixed, at, domain = w, h, 3., 0.
    elif kind == 'horizon':
        coordinate, fixed, at, domain = h, w, .6, .3
    else:
        raise ValueError(kind)
    negative = mask & ~y & (fixed >= at)
    positive = mask & y & (fixed <= at)
    lo = float(coordinate[negative].max()) if negative.any() else domain
    hi = float(coordinate[positive].min()) if positive.any() else float('inf')
    if lo > hi+1e-6:
        raise ValueError('Inconsistent monotonic labels')
    return lo, hi


def curve_crossing(axis, probability, threshold):
    """Sampled crossing interval, point and censor class. Never invent a root."""
    axis = np.round(np.asarray(axis, float), 6)
    hit = np.flatnonzero(np.asarray(probability, float) >= threshold)
    if not len(hit):
        return float(axis[-1]), float('inf'), float('inf'), 'right'
    j = int(hit[0])
    return (float(axis[j-1]) if j else -float('inf'), float(axis[j]),
            float(axis[j]), 'interior' if j else 'left')


def disjoint_crossing(bracket, crossing):
    # Both upper ends are closed, lower bounds from negative queries are open.
    # Declared knots/curve axes are canonicalized at their input boundary.
    lo, hi = max(bracket[0], crossing[0]), min(bracket[1], crossing[1])
    return bool(lo >= hi)


def point_boundary(points, valid, kind, layer, target=None):
    x, y, z = np.asarray(points, float).T
    yl, yh = LAYERS[layer]
    m = np.asarray(valid, bool) & (y >= yl) & (y <= yh) & (z >= .3)
    if kind == 'width':
        m &= z <= 3.
        value = 2*np.abs(x)
    elif kind == 'horizon':
        m &= np.abs(x) <= .3
        value = z
    else:
        raise ValueError(kind)
    indices = np.flatnonzero(m)
    if not len(indices):
        return float('inf'), None
    objective = value[indices] if target is None else np.abs(value[indices]-target)
    i = int(indices[np.argmin(objective)])
    return float(value[i]), i


def support_span(ax, ay, depth, kind, layer):
    """Projection of a zone/ray x public-Z interval onto a contact coordinate.

    Linear inequalities clip feasible Z; every retained Z permits some ray
    in each independent angular interval. Width is full cross-section width.
    """
    low, high = max(.3, float(depth[0])), float(depth[1])
    if kind == 'width':
        high = min(high, 3.)
    elif kind != 'horizon':
        raise ValueError(kind)
    yl, yh = LAYERS[layer]
    constraints = [(ay[1], yl), (-ay[0], -yh)]  # slope*Z >= face
    if kind == 'horizon':
        constraints += [(ax[1], -.3), (-ax[0], -.3)]
    for slope, face in constraints:
        if abs(slope) < 1e-15:
            if face > 0:
                return None
        elif slope > 0:
            low = max(low, face/slope)
        else:
            high = min(high, face/slope)
    if low > high:
        return None
    if kind == 'horizon':
        return float(low), float(high)
    a = 0. if ax[0] <= 0 <= ax[1] else min(abs(ax[0]), abs(ax[1]))
    b = max(abs(ax[0]), abs(ax[1]))
    return float(2*a*low), float(2*b*high)


def stats(values):
    a = np.asarray(values, float)
    a = a[np.isfinite(a)]
    return dict(n=int(len(a)), minimum=float(a.min()) if len(a) else None,
                median=float(np.median(a)) if len(a) else None,
                p90=float(np.quantile(a, .9)) if len(a) else None,
                maximum=float(a.max()) if len(a) else None)
