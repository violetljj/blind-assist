"""Fixed public ray locations and evaluator-only observed-return teacher.

Native winning-bin membership describes the unchanged simulator, not physical
ownership. Native coordinates/depth are never returned as public features.
Materialization is deferred until the root's fixed support ceiling is evaluated.
"""
from __future__ import annotations

import numpy as np

from ba_camera_corridor import sample_indices, sample_native
from inherit_spatial_model import QUERIES
from query_occupancy_data import FOCAL as FOCAL640, observation_tokens
from tof_fov45_core import boxes45, simulate

RAYS = 1024


def ray_layout():
    """16 integer sensor points per zone; row-major zones and 4x4 points."""
    zone, sy, sx = [], [], []
    for zi, (y0, x0, y1, x1) in enumerate(boxes45()):
        yy = np.linspace(y0, y1-1, 4).round().astype(np.int64)
        xx = np.linspace(x0, x1-1, 4).round().astype(np.int64)
        if len(np.unique(yy)) != 4 or len(np.unique(xx)) != 4:
            raise ValueError('Each fixed zone must support four unique rows/columns')
        y, x = np.meshgrid(yy, xx, indexing='ij')
        zone.extend([zi]*16)
        sy.extend(y.ravel())
        sx.extend(x.ravel())
    sy, sx = np.asarray(sy), np.asarray(sx)
    sensor = sy*256+sx
    native_y, native_x = sample_indices()
    ny, nx = native_y[sy], native_x[sx]
    native = ny*640+nx
    if len(sensor) != RAYS or len(np.unique(sensor)) != RAYS or len(np.unique(native)) != RAYS:
        raise ValueError('Fixed ray mapping must contain 1024 unique point samples')
    return dict(zone=np.asarray(zone, np.int64), sensor_indices=sensor,
        native_indices=native, rgb_indices=(ny//2)*320+nx//2,
        ax=(nx+.5-320)/FOCAL640, ay=(ny+.5-180)/FOCAL640)


def teacher(depth, tof, noise_identity):
    """Return fixed-ray labels after exact original public-ToF parity.

    Eligibility depends only on the public zone validity. A native-invalid ray
    in an observed zone is eligible with y=False. query_inside is a privileged
    native-point diagnostic, never a model input or a public depth assignment.
    """
    layout = ray_layout()
    values, traces = simulate(sample_native(depth), noise_identity, boxes45())
    if not np.array_equal(observation_tokens(values, boxes45()), tof):
        raise ValueError('Exact original public ToF parity failed')
    observed = [np.asarray(t['pixel_indices'], np.int64) for t in traces if t['observed']]
    observed = np.concatenate(observed) if observed else np.empty(0, np.int64)
    y = np.isin(layout['sensor_indices'], observed)
    eligible = np.asarray(tof)[layout['zone'], 1] == 1
    z = np.asarray(depth).ravel()[layout['native_indices']]
    native_valid = np.isfinite(z) & (z > 0)
    safe_z = np.where(native_valid, z, 0).astype(np.float64)
    x, yy = layout['ax']*safe_z, layout['ay']*safe_z
    query_inside = np.stack([native_valid & (safe_z >= .3) & (safe_z <= 3)
        & (x >= xl) & (x <= xh) & (yy >= yl) & (yy <= yh)
        for xl, xh, yl, yh in QUERIES], axis=1)
    if np.any(y & (~eligible | ~native_valid)):
        raise ValueError('Observed contributor has missing public/native support')
    return dict(y=y, eligible=eligible, query_inside=query_inside,
        native_valid=native_valid, has_native=native_valid,
        public_tof_exact_parity=True)
