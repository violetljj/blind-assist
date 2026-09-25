"""NumPy R1 readouts. No labels, scene geometry, or future samples consumed."""
import numpy as np
from cnh_route_sensor import angular_rays, RAW_BIN_M, RAW_BINS
from cnh_street_e2e_materialize import BOXES


def train_median_bias(raw, train_indices):
    """[N,64,128] -> [64,128]; explicit integer train IDs or Boolean mask."""
    raw = np.asarray(raw)
    if raw.ndim != 3 or raw.shape[1:] != (64, RAW_BINS):
        raise ValueError('Expected [N,64,128] raw response')
    ids = np.asarray(train_indices)
    if ids.ndim != 1:
        raise ValueError('Explicit one-dimensional training selection required')
    if ids.dtype == np.bool_:
        if len(ids) != len(raw):
            raise ValueError('Training mask length differs')
        ids = np.flatnonzero(ids)
    elif not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('Training indices must be integers or Boolean mask')
    if len(ids) == 0 or len(np.unique(ids)) != len(ids) or np.any((ids < 0) | (ids >= len(raw))):
        raise ValueError('Nonempty unique in-range train indices required')
    selected = raw[ids]
    if not np.isfinite(selected).all():
        raise ValueError('Nonfinite training response')
    return np.median(selected, axis=0)


def known_xtalk(signal_counts, pulse_matrix, xtalk_bin, fraction=.02):
    """Known assumed response component [128], not hardware-estimated bias."""
    matrix = np.asarray(pulse_matrix)
    if matrix.shape != (RAW_BINS, RAW_BINS) or not np.isfinite(matrix).all():
        raise ValueError('Finite 128x128 pulse matrix required')
    if int(xtalk_bin) != xtalk_bin or not 0 <= xtalk_bin < RAW_BINS:
        raise ValueError('Crosstalk input bin outside window')
    if not np.isfinite([signal_counts, fraction]).all() or min(signal_counts, fraction) < 0:
        raise ValueError('Nonnegative finite signal/fraction required')
    return signal_counts * fraction * matrix[int(xtalk_bin)]


def raw_query_weights(boxes=BOXES, samples=16):
    """[Q,64,128] public ray/AABB overlap; same quadrature as H3 rule."""
    boxes = np.asarray(boxes, dtype=float)
    if boxes.ndim != 3 or boxes.shape[1:] != (2, 3) or not np.isfinite(boxes).all() or np.any(boxes[:, 1] <= boxes[:, 0]):
        raise ValueError('Nondegenerate [Q,2,3] boxes required')
    rays, area = angular_rays(samples)
    rays, area = rays.reshape(64, -1, 3), area.reshape(64, -1)
    area /= area.sum(-1, keepdims=True)
    edges = np.arange(RAW_BINS+1)*RAW_BIN_M
    result = []
    for low, high in boxes:
        with np.errstate(divide='ignore', invalid='ignore'):
            a, b = low/rays, high/rays
        near = np.maximum(np.minimum(a, b).max(-1), 0)
        far = np.maximum(np.maximum(a, b).min(-1), 0)
        overlap = np.maximum(0, np.minimum(far[..., None], edges[1:])-
                             np.maximum(near[..., None], edges[:-1]))/RAW_BIN_M
        result.append(np.einsum('zr,zrb->zb', area, overlap))
    return np.clip(np.asarray(result), 0, 1)


def matched_correlation(raw, pulse_matrix):
    """Signed matched amplitude: raw @ templates.T / each template energy.

    A pure amplitude-A impulse convolved with template k recovers A at k;
    overlapping templates remain correlated, not deconvolved.
    """
    raw, matrix = np.asarray(raw), np.asarray(pulse_matrix)
    if raw.shape[-1] != RAW_BINS or matrix.shape != (RAW_BINS, RAW_BINS):
        raise ValueError('128-bin response and pulse templates required')
    energy = np.sum(matrix*matrix, axis=1)
    if not np.isfinite(raw).all() or not np.isfinite(matrix).all() or np.any(energy <= 0):
        raise ValueError('Finite response and nonzero template energy required')
    return (raw @ matrix.T)/energy


def query_scores(raw, weights):
    """[...,64,128] and [Q,64,128] -> [...,Q], without clipping."""
    raw, weights = np.asarray(raw), np.asarray(weights)
    if raw.shape[-2:] != (64, RAW_BINS) or weights.ndim != 3 or weights.shape[1:] != (64, RAW_BINS):
        raise ValueError('Raw response/geometry dimensions differ')
    return np.einsum('...zb,qzb->...q', raw, weights)


def zone_cosines(samples=16):
    """64 solid-angle-weighted means of ray z (forward radial projection)."""
    rays, area = angular_rays(samples)
    return ((rays[..., 2]*area).sum(-1)/area.sum(-1)).reshape(64)


def shift_toward_near(past, delta_m):
    """out[z,b] = past[z,b+delta_m[z]/RAW_BIN_M], linear, outside zero."""
    past = np.asarray(past)
    delta = np.broadcast_to(np.asarray(delta_m, dtype=float), (64,))
    if past.shape != (64, RAW_BINS) or not np.isfinite(delta).all() or not np.isfinite(past).all():
        raise ValueError('Finite [64,128] past and per-zone shift required')
    bins = np.arange(RAW_BINS)
    return np.stack([np.interp(bins+delta[z]/RAW_BIN_M, bins, past[z], left=0, right=0)
                     for z in range(64)])


def temporal_mean(raw, clip_ids, steps, K, *, nominal_step_m=0., compensate=False):
    """Mean current plus up to K-1 earlier available steps of the same clip.

    IDs must uniquely identify clips across layouts; integer steps are unique
    within each clip. Input row order need not be sorted. No future is used.
    K=1 returns a bitwise copy preserving dtype, without arithmetic.
    """
    raw, clips, steps = np.asarray(raw), np.asarray(clip_ids), np.asarray(steps)
    if raw.ndim != 3 or raw.shape[1:] != (64, RAW_BINS):
        raise ValueError('Expected [N,64,128] response')
    if clips.shape != (len(raw),) or steps.shape != (len(raw),) or not np.issubdtype(steps.dtype, np.integer):
        raise ValueError('One clip ID and integer step per frame required')
    if int(K) != K or K < 1 or not np.isfinite(nominal_step_m) or nominal_step_m < 0:
        raise ValueError('Positive integer K and nonnegative nominal step required')
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite raw response')
    if len(set(zip(clips.tolist(), steps.tolist()))) != len(raw):
        raise ValueError('Duplicate clip/step identity')
    if K == 1:
        return raw.copy()
    cosine = zone_cosines()
    result = np.empty(raw.shape, dtype=np.result_type(raw.dtype, np.float64))
    for i in range(len(raw)):
        ids = np.flatnonzero((clips == clips[i]) & (steps <= steps[i]))
        ids = ids[np.argsort(steps[ids], kind='stable')][-int(K):]
        values = [shift_toward_near(raw[j], (steps[i]-steps[j])*nominal_step_m*cosine)
                  if compensate and j != i else raw[j] for j in ids]
        result[i] = np.mean(values, axis=0)
    return result
