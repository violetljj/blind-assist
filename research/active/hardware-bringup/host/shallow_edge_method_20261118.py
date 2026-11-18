"""Test whether an observed edge supports a connected, locally raised shallow surface."""

import cv2
import numpy as np

from ground_geometry_filter_20261118 import MIN_PROBABILITY, down_from_ground
from shallow_surface_evidence_20261118 import _points, shallow_masks, core


VARIANTS = ("rgb_relief", "relief", "rgb")
MIN_AREA = .005


def largest_area(labels, area, roi=None):
    """A single connected surface must supply the area, including in an object ROI."""
    selected = labels > 0
    if roi is not None:
        selected &= roi
    if not np.any(selected):
        return 0.
    return float(np.bincount(labels[selected], weights=area[selected]).max())


def long_enough(run, u, v, z, k):
    """At least 5 cm of uninterrupted image contour, with local depth giving scale."""
    scale = .5 * (z + np.roll(z, 1))
    length = np.hypot((u - np.roll(u, 1)) / k[0, 0],
                      (v - np.roll(v, 1)) / k[1, 1]) * scale
    segment = run & np.roll(run, 1)
    if np.all(segment):
        return length.sum() >= .05
    start = (np.flatnonzero(~segment)[0] + 1) % len(u)
    lengths = np.roll(np.where(segment, length, 0), -start)
    cum = np.cumsum(lengths)
    longest = (cum - np.maximum.accumulate(np.where(lengths == 0, cum, 0))).max()
    return longest >= .05


def edge_variants(bgr, z, k, probability, view):
    """Inputs share a calibrated pixel grid. Return per-mechanism component IDs and pixel area."""
    h, w = z.shape
    if bgr.shape != (h, w, 3) or bgr.dtype != np.uint8 or probability.shape != z.shape or view.shape != z.shape:
        raise ValueError("Need uint8 BGR, metric optical depth, ground probability and same-grid camera support")
    z = np.where(view, z, np.nan)
    probability = np.where(view, probability, 0.)
    _, shallow, _ = shallow_masks(z, k, probability)
    area = core.pixel_area(k, h, w, z.ravel()).reshape(h, w)
    n, labels = cv2.connectedComponents(shallow.astype(np.uint8), connectivity=8)
    accepted = {v: np.zeros(n, bool) for v in VARIANTS}
    totals = np.bincount(labels[shallow], weights=area[shallow], minlength=n)
    if not np.any(totals >= MIN_AREA):
        return {v: np.zeros_like(labels) for v in VARIANTS}, area

    # Use the same observed ground direction as the surface proposal. Physical depth
    # and image brightness are measured on opposite sides of its outer contour.
    step = max(1, max(h, w) // 256)
    v, u = np.mgrid[0:h:step, 0:w:step]
    sampled = z[::step, ::step]
    good = (np.isfinite(sampled) & (sampled >= .25) & (sampled <= 6)
            & (probability[::step, ::step] >= MIN_PROBABILITY) & (v >= .5*h))
    down = down_from_ground(_points(u[good], v[good], sampled[good], k))
    if down is None:
        raise RuntimeError("Shallow proposals must reproduce their supporting ground direction")
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255., cv2.COLOR_BGR2LAB)
    lab = cv2.GaussianBlur(lab, (3, 3), 0.)
    smooth = cv2.GaussianBlur(shallow.astype(np.float32), (5, 5), 0.)
    nx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
    ny = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    contours, _ = cv2.findContours(shallow.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        u, v = contour.reshape(-1, 2).T
        component = labels[v[0], u[0]]
        if len(u) < 3 or totals[component] < MIN_AREA:
            continue
        depth = z[v, u]
        dx, dy = nx[v, u], ny[v, u]
        norm = np.maximum(np.hypot(dx, dy), 1e-6)
        dx, dy = dx / norm, dy / norm
        witnesses = {name: np.zeros(len(u), bool) for name in VARIANTS}
        for separation in (.02, .04, .07):
            shift = np.maximum(1., np.sqrt(k[0, 0] * k[1, 1]) * separation / depth)
            su = np.rint(shift * dx).astype(int)
            sv = np.rint(shift * dy).astype(int)
            inside_u, inside_v, outside_u, outside_v = u+su, v+sv, u-su, v-sv
            usable = ((inside_u >= 0) & (inside_u < w) & (inside_v >= 0) & (inside_v < h)
                      & (outside_u >= 0) & (outside_u < w) & (outside_v >= 0) & (outside_v < h))
            idx = np.flatnonzero(usable)
            a, b = inside_v[idx], inside_u[idx]
            c, d = outside_v[idx], outside_u[idx]
            valid = (np.isfinite(z[a, b]) & (z[a, b] > .05) & np.isfinite(z[c, d]) & (z[c, d] > .05)
                     & (labels[a, b] == component) & (labels[c, d] == 0))
            delta = _points(d, c, z[c, d], k) - _points(b, a, z[a, b], k)
            drop = delta @ down
            valid &= np.linalg.norm(delta - drop[:, None]*down, axis=1) <= .25
            relief = valid & (drop >= .025) & (drop <= .20)
            color = valid & (np.linalg.norm(lab[a,b] - lab[c,d], axis=1) >= 12.)
            witnesses["relief"][idx] |= relief
            witnesses["rgb"][idx] |= color
            witnesses["rgb_relief"][idx] |= color & relief
        for name in VARIANTS:
            accepted[name][component] |= long_enough(witnesses[name], u, v, depth, k)
    return {v: np.where(keep[labels], labels, 0) for v, keep in accepted.items()}, area


def edge_components(bgr, z, K, probability, view, variant):
    """Frozen candidate extension: OR its largest component with the full direct-5-cm arm."""
    masks, area = edge_variants(bgr, z, K, probability, view)
    return masks[variant], area
