"""Fixed public-only RGB motion/range consistency, with explicit missingness."""
import cv2
import numpy as np
from inherit_spatial_model import canonical_tof

CENTER = np.array([159.5, 89.5])
MISSING_REASONS = {0: 'AVAILABLE', 1: 'OUTSIDE_CURRENT_FOOTPRINT',
                   2: 'TRACKING_FAILED', 3: 'CURRENT_RETURN_MISSING',
                   4: 'PREVIOUS_FOOTPRINT_MISSING', 5: 'PREVIOUS_RETURN_MISSING'}


def projection_residual(current_xy, previous_xy, current_z, previous_z, valid):
    q, p = np.asarray(current_xy, float), np.asarray(previous_xy, float)
    z, prior = np.asarray(current_z, float), np.asarray(previous_z, float)
    usable = (np.asarray(valid, bool) & np.isfinite(q).all(axis=1)
              & np.isfinite(p).all(axis=1) & np.isfinite(z) & np.isfinite(prior)
              & (z >= .1) & (z < 8) & (prior >= .1) & (prior < 8))
    result = np.full(len(q), np.nan)
    prediction = CENTER + (p[usable]-CENTER)*(prior[usable]/z[usable])[:, None]
    result[usable] = np.linalg.norm(q[usable]-prediction, axis=1)
    return result


def sample_grid():
    yy, xx = np.meshgrid(np.arange(0, 192, 4), np.arange(0, 256, 4), indexing='ij')
    nx = np.floor((xx.ravel()+.5)*640/256)
    ny = np.floor((yy.ravel()+.5)*360/192)
    return (yy*256+xx).ravel(), np.stack(((nx+.5)/2-.5, (ny+.5)/2-.5), axis=1).astype(np.float32)


def zones_for_points(points, boxes):
    normalized = (np.asarray(points)+.5)/[320, 180]
    assigned = np.full(len(points), -1, np.int32)
    for zone, (y0, x0, y1, x1) in enumerate(boxes):
        inside = (np.isfinite(normalized).all(axis=1) & (normalized[:, 0] >= x0)
                  & (normalized[:, 0] < x1) & (normalized[:, 1] >= y0)
                  & (normalized[:, 1] < y1))
        if np.any(inside & (assigned >= 0)):
            raise ValueError('Overlapping public boxes')
        assigned[inside] = zone
    return assigned


def extract(current_rgb, previous_rgb, saved_tof, prev_tof):
    """Return all 3072 grid rows; no truth/identity/native input is accepted.

    Missing reason is the first failure, while independent Boolean flags retain
    overlapping failures. Static control substitutes p=q in the projection on
    the same matched population/ranges. Wrong-zone uses prior row, column+4 mod8.
    """
    cv2.setNumThreads(1)
    images = []
    for image in (current_rgb, previous_rgb):
        if np.asarray(image).shape != (3, 180, 320) or image.dtype != np.uint8:
            raise ValueError('Public RGB must be uint8 [3,180,320]')
        images.append(cv2.cvtColor(np.ascontiguousarray(image.transpose(1, 2, 0)), cv2.COLOR_RGB2GRAY))
    z, valid, boxes, _, _ = canonical_tof(saved_tof)
    pz, pvalid, pboxes, _, _ = canonical_tof(prev_tof)
    flat, q = sample_grid(); n = len(q)
    cz = zones_for_points(q, boxes); candidate = cz >= 0
    p = np.full((n, 2), np.nan, np.float32)
    reverse = np.full_like(p, np.nan)
    raw = np.zeros(n, bool); fb = np.full(n, np.nan)
    lk = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    indices = np.flatnonzero(candidate)
    if len(indices):
        forward, status, _ = cv2.calcOpticalFlowPyrLK(images[0], images[1], q[indices, None], None, **lk)
        if forward is not None:
            p[indices] = forward[:, 0]
            finite = np.isfinite(forward[:, 0]).all(axis=1)
            back_indices = indices[finite]
            if len(back_indices):
                back, backstatus, _ = cv2.calcOpticalFlowPyrLK(images[1], images[0], p[back_indices, None], None, **lk)
                if back is not None:
                    reverse[back_indices] = back[:, 0]
                    raw[back_indices] = status[:, 0][finite].astype(bool) & backstatus[:, 0].astype(bool)
                    fb[back_indices] = np.linalg.norm(reverse[back_indices]-q[back_indices], axis=1)
    inimage = lambda xy: np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) & (xy[:, 0] <= 319) & (xy[:, 1] >= 0) & (xy[:, 1] <= 179)
    tracked = candidate & raw & inimage(p) & inimage(reverse) & (fb <= 1.)
    priorzone = zones_for_points(p, pboxes)
    wrongzone = np.where(priorzone >= 0, (priorzone//8)*8+(priorzone%8+4)%8, -1)
    current_valid = candidate & valid[np.maximum(cz, 0)]
    previous_valid = (priorzone >= 0) & pvalid[np.maximum(priorzone, 0)]
    wrong_valid = (wrongzone >= 0) & pvalid[np.maximum(wrongzone, 0)]
    matched = tracked & current_valid & previous_valid
    wrong_matched = tracked & current_valid & wrong_valid
    current_z = z[np.maximum(cz, 0)]; previous_z = pz[np.maximum(priorzone, 0)]
    residual = projection_residual(q, p, current_z, previous_z, matched)
    wrong = projection_residual(q, p, current_z, pz[np.maximum(wrongzone, 0)], wrong_matched)
    static = projection_residual(q, q, current_z, previous_z, matched)
    reason = np.zeros(n, np.int8)
    for flag, code in ((~candidate, 1), (~tracked, 2), (~current_valid, 3),
                       (priorzone < 0, 4), (~previous_valid, 5)):
        reason[(reason == 0) & flag] = code
    return dict(low_flat_index=flat, current_xy=q, previous_xy=p, current_zone=cz,
                previous_zone=priorzone, wrong_previous_zone=wrongzone, candidate=candidate,
                tracking_raw=raw, tracked=tracked, current_return_valid=current_valid,
                previous_return_valid=previous_valid, matched=matched, wrong_matched=wrong_matched,
                fb_error=fb, residual=residual, static_residual=static,
                wrong_residual=wrong, missing_reason=reason)
