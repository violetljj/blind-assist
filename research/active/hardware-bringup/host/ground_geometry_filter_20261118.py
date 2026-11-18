"""Preserve raised surfaces using current predicted metric depth and RGB ground probability."""
import cv2
import numpy as np

from rgb_ground_mask_20261118 import GroundMask as GroundEvidence, MIN_PROBABILITY

CONTROL_THRESHOLDS = (.80, .85, .90, .93, .95, .97, .98, .99, .995, .999, 1.01)
CELL = .05
RADIUS = .45


def down_from_ground(points):
    """Estimate only the vertical direction from a broad, near-horizontal patch."""
    if len(points) < 48:
        return None
    rng = np.random.default_rng(11018)
    if len(points) > 2048:
        points = points[rng.choice(len(points), 2048, replace=False)]
    a, b, c = points[rng.integers(len(points), size=(3, 128))].astype(np.float64)
    normals = np.cross(b - a, c - a)
    norms = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(norms[:, None], 1e-12)
    normals *= np.where(normals[:, 1:2] < 0, -1., 1.)
    usable = (norms > .005) & (normals[:, 1] >= np.cos(np.deg2rad(35)))
    if not usable.any():
        return None
    normals = normals[usable]
    offsets = np.sum(a[usable] * normals, axis=1)
    residual = np.abs(points @ normals.T - offsets)
    counts = (residual <= .05).sum(axis=0)
    best = counts.argmax()
    if counts[best] < max(32, .15 * len(points)):
        return None
    return normals[best].astype(np.float32)


def protection(depth_m_z, K, probability):
    """A high-confidence ground pixel can still be measurably above adjacent level ground.

    Grid axes are estimated from the predicted point cloud. Within shared horizontal
    cells, distances along 'down' increase towards the floor. No reference plane,
    labels, evaluation corridor, mount height or future frames enter this computation.
    """
    h, w = depth_m_z.shape
    keep = np.zeros((h, w), dtype=bool)
    step = max(1, max(h, w) // 256)
    v, u = np.mgrid[0:h:step, 0:w:step]
    z = depth_m_z[::step, ::step]
    valid = np.isfinite(z) & (z >= .25) & (z <= 6) & (probability[::step, ::step] >= MIN_PROBABILITY)
    v, u, z = v[valid], u[valid], z[valid]
    p = np.column_stack(((u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z)).astype(np.float32)
    down = down_from_ground(p[v >= .5 * h])
    if down is None:
        return keep
    right = np.cross(down, [0., 0., 1.]).astype(np.float32)
    right /= np.linalg.norm(right)
    forward = np.cross(right, down)

    # Median in each 5 cm cell. Require at least three observed samples per cell.
    columns, rows = 240, 160
    def project(points):
        x = np.floor((points @ right + 6.) / CELL).astype(np.int32)
        y = np.floor((points @ forward + 1.) / CELL).astype(np.int32)
        return x, y, (x >= 0) & (x < columns) & (y >= 0) & (y < rows)
    x, y, inside = project(p)
    key = y[inside] * columns + x[inside]
    levels = p[inside] @ down
    order = np.lexsort((levels, key))
    keys, starts, counts = np.unique(key[order], return_index=True, return_counts=True)
    dense = counts >= 3
    grid = np.full((rows, columns), np.nan, np.float32)
    grid.flat[keys[dense]] = levels[order[starts[dense] + counts[dense] // 2]]

    # A reference patch needs three contiguous cells at approximately the same level.
    corners = np.stack([grid[:-1, :-1], grid[1:, :-1], grid[:-1, 1:], grid[1:, 1:]])
    good = np.isfinite(corners)
    lowest = np.where(good, corners, -1e4).max(axis=0)
    highest = np.where(good, corners, 1e4).min(axis=0)
    stable = (good.sum(axis=0) >= 3) & ((lowest - highest) <= .06)
    reference = np.full_like(grid, -1e4)
    reference[:-1, :-1] = np.where(stable, (lowest + highest) / 2, -1e4)
    r = round(RADIUS / CELL)
    dy, dx = np.ogrid[-r:r+1, -r:r+1]
    reference = cv2.dilate(reference, ((dy*dy + dx*dx) <= r*r).astype(np.uint8))

    vv, uu = np.nonzero(np.isfinite(depth_m_z) & (depth_m_z >= .25) & (depth_m_z <= 6)
                        & (probability >= MIN_PROBABILITY))
    zz = depth_m_z[vv, uu]
    pts = np.column_stack(((uu - K[0, 2]) * zz / K[0, 0],
                           (vv - K[1, 2]) * zz / K[1, 1], zz)).astype(np.float32)
    xx, yy, use = project(pts)
    rise = reference[yy[use], xx[use]] - pts[use] @ down
    keep[vv[use], uu[use]] = (.10 <= rise) & (rise <= .60)
    return keep
