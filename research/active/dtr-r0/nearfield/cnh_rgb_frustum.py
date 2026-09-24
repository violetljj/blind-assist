"""Public-calibration support of ToF zones on an RGB feature grid.

The returned weights describe possible angular overlap, not an obstacle mask or
pixel-level CNH ownership. Camera depth is sampled only as a geometric nuisance
variable for an offset camera/ToF pair; no scene depth or evaluator truth enters.
"""
from __future__ import annotations

import numpy as np


def zone_rgb_support(
    camera_k,
    tof_from_camera,
    image_size,
    feature_size,
    zones=(8, 8),
    tof_fov_deg=45.0,
    range_m=(0.1, 8.0),
    edge_uncertainty_deg=3.0,
    depth_samples=24,
):
    """Return [rows*cols, feature_h, feature_w] float32 support in [0, 1].

    Coordinates use x right, y down, z forward. ``tof_from_camera`` maps
    camera-frame metres to the ToF frame. The grid samples RGB pixel centres.
    For nonzero baseline, a fixed range grid approximates the soft weight.
    An analytic line/zone interval check prevents a sampled zero from excluding
    a ray that intersects the uncertainty-expanded zone at an unsampled depth.
    The soft edge is a declared angular uncertainty, not a learned correction.
    """
    k = np.asarray(camera_k, dtype=np.float64)
    transform = np.asarray(tof_from_camera, dtype=np.float64)
    width, height = map(int, image_size)
    feature_h, feature_w = map(int, feature_size)
    rows, cols = map(int, zones)
    near, far = map(float, range_m)
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Finite positive camera intrinsics required')
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('Finite 4x4 ToF-from-camera transform required')
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError('Rigid homogeneous transform required')
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0.999:
        raise ValueError('ToF-from-camera rotation must be proper and rigid')
    if min(width, height, feature_h, feature_w, rows, cols) <= 0 or feature_w > width or feature_h > height:
        raise ValueError('Invalid image, feature, or zone dimensions')
    if not 0 < near < far or not np.isfinite([near, far]).all() or int(depth_samples) != depth_samples or depth_samples < 2:
        raise ValueError('Invalid projection range samples')
    if not 0 < tof_fov_deg < 180 or not 0 < edge_uncertainty_deg < 45:
        raise ValueError('Invalid angular support')

    pixel_x = (np.arange(feature_w) + .5) * width / feature_w - .5
    pixel_y = (np.arange(feature_h) + .5) * height / feature_h - .5
    xx, yy = np.meshgrid(pixel_x, pixel_y)
    rays = np.stack(((xx - k[0, 2]) / k[0, 0], (yy - k[1, 2]) / k[1, 1], np.ones_like(xx)), -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    depths = np.geomspace(near, far, int(depth_samples))
    points = depths[:, None, None, None] * rays[None]
    tof = points @ rotation.T + transform[:3, 3]
    slopes = rays @ rotation.T
    front = tof[..., 2] > 0
    x = np.divide(tof[..., 0], tof[..., 2], out=np.full_like(tof[..., 0], np.nan), where=front)
    y = np.divide(tof[..., 1], tof[..., 2], out=np.full_like(tof[..., 1], np.nan), where=front)
    edge = np.tan(np.deg2rad(tof_fov_deg / 2))
    width_tan = np.tan(np.deg2rad(edge_uncertainty_deg))
    x_bounds = np.linspace(-edge, edge, cols + 1)
    y_bounds = np.linspace(-edge, edge, rows + 1)
    result = np.empty((rows * cols, feature_h, feature_w), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            margin = np.minimum.reduce((
                x - x_bounds[col], x_bounds[col + 1] - x,
                y - y_bounds[row], y_bounds[row + 1] - y,
            ))
            # A point just outside a zone may belong to it under calibration
            # uncertainty; far outside the margin contributes exactly zero.
            weights = np.clip(.5 + margin / (2 * width_tan), 0, 1)
            weights = np.where(front, weights, 0)
            sampled = np.nanmax(weights, axis=0)
            # A transformed camera ray is p(d)=slope*d+offset. Each tangent
            # zone boundary is a linear inequality in d while ToF z is front.
            # Intersect the five allowed intervals exactly over [near, far].
            lo = np.full((feature_h, feature_w), near)
            hi = np.full((feature_h, feature_w), far)
            feasible = np.ones_like(lo, dtype=bool)
            left, right = x_bounds[col] - width_tan, x_bounds[col + 1] + width_tan
            top, bottom = y_bounds[row] - width_tan, y_bounds[row + 1] + width_tan
            constraints = (
                (slopes[..., 0] - left * slopes[..., 2], transform[0, 3] - left * transform[2, 3]),
                (right * slopes[..., 2] - slopes[..., 0], right * transform[2, 3] - transform[0, 3]),
                (slopes[..., 1] - top * slopes[..., 2], transform[1, 3] - top * transform[2, 3]),
                (bottom * slopes[..., 2] - slopes[..., 1], bottom * transform[2, 3] - transform[1, 3]),
                (slopes[..., 2], transform[2, 3] - 1e-9),
            )
            for slope, intercept in constraints:
                boundary = np.divide(-intercept, slope, out=np.zeros_like(lo), where=np.abs(slope) > 1e-12)
                lo = np.where(slope > 1e-12, np.maximum(lo, boundary), lo)
                hi = np.where(slope < -1e-12, np.minimum(hi, boundary), hi)
                feasible &= ~((np.abs(slope) <= 1e-12) & (intercept < 0))
            possible = feasible & (lo <= hi)
            result[row * cols + col] = np.where(possible, np.maximum(sampled, 1e-6), 0).astype(np.float32)
    return result
