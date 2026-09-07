"""Bounded predicted-only ground fit; native depth is never an input.

Development assumption: a broad lower-central visible surface is local ground.
Passing geometric gates does not establish ground identity (e.g. a tabletop).
"""
from dataclasses import asdict, dataclass
import math

import numpy as np

from near_field import Camera


@dataclass(frozen=True)
class Fit:
    status: str
    reason: str
    plane: tuple | None = None
    scale: float | None = None
    samples: int = 0
    inliers: int = 0
    covered_cells: int = 0
    median_residual_m: float | None = None

    def summary(self):
        return asdict(self)


def fit_ground(depth: np.ndarray, camera: Camera) -> Fit:
    if depth.shape != (camera.height, camera.width):
        raise ValueError("depth/calibration dimensions disagree")
    # Fixed image ROI and stride, independent of native masks and outcomes.
    top, bottom = int(.62*camera.height), int(.95*camera.height)
    left, right = int(.20*camera.width), int(.80*camera.width)
    v, u = np.mgrid[top:bottom:8, left:right:8]
    d = np.asarray(depth[v, u], dtype=np.float64).ravel()
    f = camera.width/(2*math.tan(math.radians(camera.hfov_deg/2)))
    up = -(v.ravel()-(camera.height-1)/2)/f
    y = (u.ravel()-(camera.width-1)/2)/f
    pitch = math.radians(camera.pitch_deg)
    x = math.cos(pitch)-math.sin(pitch)*up
    z = math.sin(pitch)+math.cos(pitch)*up
    cells = (np.minimum(2, (v.ravel()-top)*3//max(1,bottom-top))*3
             + np.minimum(2, (u.ravel()-left)*3//max(1,right-left))).astype(int)
    valid = np.isfinite(d) & (d > .08) & (d < 12) & (z < -.12) & (x > 0)
    ids = np.flatnonzero(valid)
    if len(ids) > 1000:
        ids = ids[np.linspace(0, len(ids)-1, 1000).astype(int)]
    n = len(ids)
    if n < 30:
        return Fit("UNKNOWN", "insufficient_downward_samples", samples=n)
    design = np.column_stack((d[ids]*x[ids], d[ids]*y[ids], np.ones(n)))
    target, cells = d[ids]*z[ids], cells[ids]
    rng = np.random.default_rng(1707)
    best, best_count = None, 0
    max_slope = math.tan(math.radians(20))
    for _ in range(64):
        picked = rng.choice(n, 3, replace=False)
        small = design[picked]
        if abs(np.linalg.det(small)) < 1e-6:
            continue
        plane = np.linalg.solve(small, target[picked])
        if np.hypot(*plane[:2]) > max_slope or plane[2] >= 0:
            continue
        count = int((np.abs(design@plane-target) <= .08).sum())
        if count > best_count:
            best, best_count = plane, count
    if best is None:
        return Fit("UNKNOWN", "no_plausible_plane", samples=n)
    for _ in range(3):
        inside = np.abs(design@best-target) <= .08
        if inside.sum() < 30:
            return Fit("UNKNOWN", "insufficient_inliers", samples=n, inliers=int(inside.sum()))
        best = np.linalg.lstsq(design[inside], target[inside], rcond=None)[0]
    residual = np.abs(design@best-target)
    inside = residual <= .08
    covered = int((np.bincount(cells[inside], minlength=9) >= 3).sum())
    count = int(inside.sum())
    if count/n < .50 or covered < 6:
        return Fit("UNKNOWN", "insufficient_ground_coverage", samples=n, inliers=count, covered_cells=covered)
    if best[2] >= 0 or np.hypot(*best[:2]) > max_slope:
        return Fit("UNKNOWN", "implausible_refined_plane", samples=n, inliers=count, covered_cells=covered)
    scale = camera.camera_height_m / -best[2]
    if not .5 <= scale <= 2:
        return Fit("UNKNOWN", "scale_outside_fixed_range", samples=n, inliers=count, covered_cells=covered)
    return Fit("ACCEPTED", "geometric_gates_only", tuple(float(v) for v in best),
               float(scale), n, count, covered, float(np.median(residual[inside])))


def arm_inputs(depth: np.ndarray, fit: Fit, arm: str):
    """None means UNKNOWN for every cell, never silent raw/clear fallback."""
    if arm == "raw":
        return depth, None
    if arm not in ("scale_only", "ground_only", "scale_and_ground"):
        raise ValueError("unknown arm")
    if fit.status != "ACCEPTED":
        return None
    scaled = arm in ("scale_only", "scale_and_ground")
    k = fit.scale if scaled else 1.
    output = depth*k if scaled else depth
    plane = None
    if arm in ("ground_only", "scale_and_ground"):
        a, b, c = fit.plane
        plane = (a, b, k*c)
    return output, plane
