"""Train-geometry-only continuous contact-query sampling.

This pure label-side module opens no files and never receives dev/evaluation
metadata. The caller must supply train geometry only. Its only model-facing
output is the float32 [72,3] width/horizon/layer query array; labels and the
sampling receipt stay separate. Tiny scalar/array work: TASK_NOT_GPU_SUITABLE.
"""
from __future__ import annotations

import numpy as np

from contact_boundary_data import LAYERS, contact_labels

WIDTH_DOMAIN = (.36, 1.08)
HORIZON_DOMAIN = (.6, 3.)
OFFSETS = (.01, .02, .05)
MAX_ATTEMPTS = 8
GLOBAL_PER_LAYER = 18
PAIRS_PER_LAYER = 9


def _bounds(boxes):
    values = np.asarray(boxes, np.float64)
    if values.size == 0:
        return np.empty((0, 2, 3), np.float64)
    if (values.ndim != 3 or values.shape[1:] != (2, 3)
            or not np.isfinite(values).all() or (values[:, 0] > values[:, 1]).any()):
        raise ValueError('Expected finite ordered camera-space AABB pairs')
    return values


def _slice_boundaries(bounds, layer, kind, fixed):
    """Vectorized exact union boundary at each fixed orthogonal coordinate."""
    fixed = np.asarray(fixed, np.float64)
    if layer not in (0, 1) or kind not in ('width', 'horizon') or not np.isfinite(fixed).all():
        raise ValueError('Expected BODY/HEAD, width/horizon and finite slices')
    shape = fixed.shape
    fixed = fixed.ravel()
    if not len(bounds):
        return np.full(shape, np.inf)
    low, high = bounds[:, 0], bounds[:, 1]
    yl, yh = LAYERS[layer]
    eligible = (high[:, 1] >= yl) & (low[:, 1] <= yh) & (high[:, 2] >= .3)
    if kind == 'width':
        overlap = eligible[:, None] & (low[:, 2, None] <= fixed)
        candidates = 2 * np.maximum.reduce((np.zeros(len(bounds)), low[:, 0], -high[:, 0]))
    else:
        overlap = (eligible[:, None] & (high[:, 0, None] >= -fixed / 2)
                   & (low[:, 0, None] <= fixed / 2))
        candidates = np.maximum(.3, low[:, 2])
    return np.min(np.where(overlap, candidates[:, None], np.inf), axis=0).reshape(shape)


def critical_slice(boxes, layer, kind, fixed_value):
    """Exact critical width at fixed horizon, or horizon at fixed width.

    Infinity means no declared object intersects that slice. A zero width or
    .3 horizon can be a real boundary outside the admissible training interior;
    the sampler reports uniform fallback instead of moving such a boundary.
    """
    return float(_slice_boundaries(_bounds(boxes), layer, kind, fixed_value))


def sample_queries(boxes, seed):
    """Return exactly 72 queries, boolean labels and a label-side receipt.

    First 36 rows are uniform (18 per layer). Remaining rows are 9 consecutive
    pairs per layer; failed pair slots contain 2 independent uniform queries.
    Pair kinds alternate by seed and layer; offsets cycle through .01/.02/.05.
    Eight candidate slices are drawn per pair in one bounded vectorized batch.
    Accepted rows are strictly inside the training domain after float32 cast.
    """
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('Expected a nonnegative integer seed')
    bounds = _bounds(boxes)
    rng = np.random.default_rng(int(seed))
    lower = np.array([WIDTH_DOMAIN[0], HORIZON_DOMAIN[0]])
    upper = np.array([WIDTH_DOMAIN[1], HORIZON_DOMAIN[1]])
    q = np.empty((72, 3), np.float32)
    for layer in (0, 1):
        start = layer * GLOBAL_PER_LAYER
        q[start:start+GLOBAL_PER_LAYER, :2] = rng.uniform(lower, upper, (GLOBAL_PER_LAYER, 2))
        q[start:start+GLOBAL_PER_LAYER, 2] = layer
    records = []
    for layer in (0, 1):
        kinds = ['width' if (int(seed) + layer + pair) % 2 == 0 else 'horizon'
                 for pair in range(PAIRS_PER_LAYER)]
        # Float32 slices are used in both the analytic oracle and returned query.
        slices = np.empty((PAIRS_PER_LAYER, MAX_ATTEMPTS), np.float32)
        for pair, kind in enumerate(kinds):
            domain = HORIZON_DOMAIN if kind == 'width' else WIDTH_DOMAIN
            slices[pair] = rng.uniform(*domain, MAX_ATTEMPTS)
        width_critical = _slice_boundaries(bounds, layer, 'width', slices)
        horizon_critical = _slice_boundaries(bounds, layer, 'horizon', slices)
        fallbacks = rng.uniform(lower, upper, (PAIRS_PER_LAYER, 2, 2)).astype(np.float32)
        for pair, kind in enumerate(kinds):
            row = 36 + layer * PAIRS_PER_LAYER * 2 + pair * 2
            offset = OFFSETS[(int(seed) + pair) % len(OFFSETS)]
            critical = width_critical[pair] if kind == 'width' else horizon_critical[pair]
            candidates = np.empty((MAX_ATTEMPTS, 2, 3), np.float32)
            candidates[:, :, 2] = layer
            axis = 0 if kind == 'width' else 1
            candidates[:, :, 1-axis] = slices[pair, :, None]
            candidates[:, :, axis] = np.column_stack((critical-offset, critical+offset))
            valid = (np.isfinite(candidates).all((1, 2))
                     & (candidates[:, :, :2] > lower).all((1, 2))
                     & (candidates[:, :, :2] < upper).all((1, 2)))
            found = np.flatnonzero(valid)
            accepted = bool(len(found))
            attempt = int(found[0]) if accepted else MAX_ATTEMPTS-1
            if accepted:
                q[row:row+2] = candidates[attempt]
            else:
                q[row:row+2, :2] = fallbacks[pair]
                q[row:row+2, 2] = layer
            records.append(dict(rows=[row, row+1], layer=layer, kind=kind, offset_m=offset,
                accepted=accepted, attempts=attempt+1,
                critical_value=float(critical[attempt]) if accepted else None,
                fixed_value=float(slices[pair, attempt]) if accepted else None,
                fallback_reason=None if accepted else 'NO_ADMISSIBLE_SLICE_WITHIN_8_ATTEMPTS'))
    labels, _ = contact_labels(bounds, q)
    for record in records:
        if record['accepted']:
            before, after = record['rows']
            if labels[before] or not labels[after]:
                raise RuntimeError('Analytic slice did not produce an opposite-label pair')
    per_layer = {}
    for layer in (0, 1):
        selected = [r for r in records if r['layer'] == layer]
        accepted = sum(r['accepted'] for r in selected)
        per_layer[str(layer)] = dict(global_queries=18, boundary_pairs=accepted,
            fallback_pairs=PAIRS_PER_LAYER-accepted,
            width_pairs=sum(r['accepted'] and r['kind'] == 'width' for r in selected),
            horizon_pairs=sum(r['accepted'] and r['kind'] == 'horizon' for r in selected))
    accepted = sum(r['accepted'] for r in records)
    receipt = dict(schema='contact-boundary-training-sampler-v1', seed=int(seed),
        queries=72, global_queries=36, requested_boundary_pairs=18,
        boundary_pairs=accepted, boundary_queries=2*accepted,
        fallback_pairs=18-accepted, fallback_queries=36-2*accepted,
        candidate_slices_drawn=18*MAX_ATTEMPTS,
        attempted_slices=sum(r['attempts'] for r in records), max_attempts_per_pair=MAX_ATTEMPTS,
        per_layer=per_layer, pairs=records, backend='TASK_NOT_GPU_SUITABLE',
        authority='TRAIN_GEOMETRY_LABEL_GENERATION_ONLY_NOT_MODEL_FEATURES')
    return q, labels, receipt
