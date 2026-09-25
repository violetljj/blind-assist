"""Certified fast path for signed box margins (exact labels, fewer LPs).

signed_margin(S, B) = max over surface points of the minimum signed box slack.
Over the object's axis-aligned bounding box the slack is separable per axis, so
min_a min(max_a - lo_a, hi_a - min_a) is an upper bound on the exact value. When
that bound is already below -(boundary band) the pair is certified non-contact
and non-boundary, and the bound is returned instead of solving any LP; labels
(m >= 0), boundary flags (|m| < .05) and contributors are identical to the exact
computation. Witness requests and every near pair use the exact LP.
"""
import numpy as np
from cnh_track_a_geometry import signed_margin as exact_margin

BAND = .05
STATS = dict(certified=0, exact=0)


def fast_margin(triangles, low, high, return_witness=False):
    if return_witness:
        return exact_margin(triangles, low, high, return_witness=True)
    t = np.asarray(triangles, float).reshape(-1, 3)
    low, high = np.asarray(low, float), np.asarray(high, float)
    bound = float(np.minimum(t.max(0)-low, high-t.min(0)).min())
    if bound < -BAND-1e-9:
        STATS['certified'] += 1
        return bound
    STATS['exact'] += 1
    return exact_margin(triangles, low, high)
