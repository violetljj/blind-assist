"""Read a shallow raised surface from predicted optical depth, intrinsics and RGB ground probability.

The global and local branches share the vertical axis, ground patches, corridor and
pixel readout. Run this file directly for a flat-plane and 10 cm step geometry check.
"""

from pathlib import Path
import sys

import cv2
import numpy as np

from ground_geometry_filter_20261118 import CELL, MIN_PROBABILITY, RADIUS, down_from_ground

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evalkit'))
import core


def _points(u, v, z, K):
    return np.column_stack(((u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z)).astype(np.float32)


def _ground_grid(points, down, right, forward):
    # Same origin, grid extent and upper-middle cell median as the existing protection.
    columns, rows = 240, 160
    x = np.floor((points @ right + 6.) / CELL).astype(int)
    y = np.floor((points @ forward + 1.) / CELL).astype(int)
    inside = (x >= 0) & (x < columns) & (y >= 0) & (y < rows)
    keys = y[inside] * columns + x[inside]
    levels = points[inside] @ down
    order = np.lexsort((levels, keys))
    cells, first, counts = np.unique(keys[order], return_index=True, return_counts=True)
    dense = counts >= 3
    grid = np.full((rows, columns), np.nan, np.float32)
    grid.flat[cells[dense]] = levels[order[first[dense] + counts[dense] // 2]]

    corners = np.stack((grid[:-1, :-1], grid[1:, :-1], grid[:-1, 1:], grid[1:, 1:]))
    good = np.isfinite(corners)
    upper = np.where(good, corners, np.inf).min(axis=0)
    lower = np.where(good, corners, -np.inf).max(axis=0)
    stable = (good.sum(axis=0) >= 3) & (lower - upper <= .03)
    # Patches are anchored at the upper-left cell, as in the legacy neighborhood lookup.
    patches = np.full(grid.shape, -np.inf, np.float32)
    patches[:-1, :-1][stable] = (lower[stable] + upper[stable]) / 2
    return patches


def shallow_masks(z, K, probability):
    """Return two opened boolean masks plus per-frame ground availability diagnostics.

    A local anchor may exist even when fewer than three patches support the global
    floor. A missing anchor yields an empty branch mask, recorded in diagnostics.
    """
    h, w = z.shape
    if probability.shape != z.shape:
        raise ValueError('Depth and ground probability must share a camera grid')
    gm = np.zeros((h, w), dtype=bool)
    lm = np.zeros_like(gm)
    diagnostics = dict(down_available=False, stable_patches=0, global_patches=0,
                       global_floor_m=None, local_anchored_corridor_px=0,
                       corridor_px=0, global_area_m2=0., local_area_m2=0.)
    stride = max(1, max(h, w) // 256)
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    sampled = z[::stride, ::stride]
    accepted = (np.isfinite(sampled) & (sampled >= .25) & (sampled <= 6.)
                & (probability[::stride, ::stride] >= MIN_PROBABILITY))
    points = _points(u[accepted], v[accepted], sampled[accepted], K)
    down = down_from_ground(points[v[accepted] >= .5 * h])
    if down is None:
        return gm, lm, diagnostics
    diagnostics['down_available'] = True
    right = np.cross(down, [0., 0., 1.]).astype(np.float32)
    right /= np.linalg.norm(right)
    forward = np.cross(right, down)
    patches = _ground_grid(points, down, right, forward)
    gy, gx = np.nonzero(np.isfinite(patches))
    diagnostics['stable_patches'] = int(len(gx))
    # The geometric center of a 2x2 patch is one cell width from its grid origin.
    px, py = (gx + 1) * CELL - 6., (gy + 1) * CELL - 1.
    global_levels = patches[gy[(abs(px) <= 1.25) & (py >= .30) & (py <= 3.)],
                            gx[(abs(px) <= 1.25) & (py >= .30) & (py <= 3.)]]
    diagnostics['global_patches'] = int(len(global_levels))
    if len(global_levels) >= 3:
        diagnostics['global_floor_m'] = float(np.median(global_levels))
    if not len(gx):
        return gm, lm, diagnostics

    # Only ground anchors use RGB evidence; all valid pixels may be shallow targets.
    v, u = np.nonzero(np.isfinite(z) & (z > .05))
    targets = _points(u, v, z[v, u], K)
    across, along = targets @ right, targets @ forward
    corridor = (abs(across) <= core.X_HALF) & (along >= core.Z_NEAR) & (along <= 1.50)
    v, u = v[corridor], u[corridor]
    levels = targets[corridor] @ down
    diagnostics['corridor_px'] = int(len(u))
    if diagnostics['global_floor_m'] is not None:
        rise = diagnostics['global_floor_m'] - levels
        gm[v, u] = (rise >= .05) & (rise < .15)
    radius = round(RADIUS / CELL)
    dy, dx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    kernel = (dx * dx + dy * dy <= radius * radius).astype(np.uint8)
    local_floor = cv2.dilate(patches, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=float('-inf'))
    xx = np.floor((across[corridor] + 6.) / CELL).astype(int)
    yy = np.floor((along[corridor] + 1.) / CELL).astype(int)
    anchor = local_floor[yy, xx]
    diagnostics['local_anchored_corridor_px'] = int(np.isfinite(anchor).sum())
    rise = anchor - levels
    lm[v, u] = (rise >= .05) & (rise < .15)
    gm = core._open(gm, h, w).reshape(h, w)
    lm = core._open(lm, h, w).reshape(h, w)
    area = core.pixel_area(K, h, w, z.ravel()).reshape(h, w)
    diagnostics['global_area_m2'] = float(area[gm].sum())
    diagnostics['local_area_m2'] = float(area[lm].sum())
    return gm, lm, diagnostics


def geometry_check():
    """Analytic oblique pinhole view of a flat floor or a raised, non-ground-colored top."""
    h, w = 240, 320
    K = np.array([[175., 0., 159.5], [0., 175., 119.5], [0., 0., 1.]])
    pitch = np.deg2rad(28.)
    down = np.array([0., np.cos(pitch), np.sin(pitch)])
    forward = np.array([0., -np.sin(pitch), np.cos(pitch)])
    rays = core.rays(K, h, w)
    toward_floor = rays @ down
    floor = np.full(h * w, np.nan)
    np.divide(1.25, toward_floor, out=floor, where=toward_floor > .15)
    top = floor * (1.15 / 1.25)
    p = rays * top[:, None]
    is_block = (abs(p[:, 0]) <= .15) & (p @ forward >= 1.0) & (p @ forward <= 1.25)
    for block in [False, True]:
        z = np.where(is_block & block, top, floor).astype(np.float32).reshape(h, w)
        prob = np.where(is_block & block, 0., 1.).astype(np.float32).reshape(h, w)
        gm, lm, d = shallow_masks(z, K, prob)
        assert d['global_patches'] >= 3 and d['local_anchored_corridor_px'] > 0
        if block:
            assert not np.any((gm | lm).ravel() & ~is_block)
            assert min(d['global_area_m2'], d['local_area_m2']) >= core.MIN_AREA
        else:
            assert not gm.any() and not lm.any()
        print('10 cm step' if block else 'flat plane', d)


if __name__ == '__main__':
    geometry_check()
