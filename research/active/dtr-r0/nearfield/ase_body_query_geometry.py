"""ASE ray-distance coverage in a camera-anchored proxy wearer frame.

No floor is inferred: the camera is assumed 1.70 m above a proxy floor.
Counts are native source pixels, not resolution-equivalent V1 model labels.
"""
from __future__ import annotations

import numpy as np
from contact_retina_spec import BODY_BOXES


def horizontal_frame(T_world_camera, gravity_world, camera_forward=(0., 0., 1.)):
    """Return world-space forward/right/up columns; gravity points downward.

    T maps camera coordinates to world. The optical axis defaults to camera +Z;
    callers with another convention must provide it explicitly. Pitch and roll
    affect ray geometry, but the proxy wearer heading is horizontal.
    """
    transform = np.asarray(T_world_camera, dtype=np.float64)
    gravity = np.asarray(gravity_world, dtype=np.float64)
    optical = np.asarray(camera_forward, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('Expected finite 4x4 T_world_camera')
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError('Expected rigid homogeneous transform')
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6):
        raise ValueError('Camera transform rotation must be proper orthonormal')
    if gravity.shape != (3,) or not np.isfinite(gravity).all() or np.linalg.norm(gravity) < 1e-8:
        raise ValueError('Expected nonzero finite downward world gravity')
    if optical.shape != (3,) or not np.isfinite(optical).all() or np.linalg.norm(optical) < 1e-8:
        raise ValueError('Expected nonzero finite camera forward axis')
    up = -gravity / np.linalg.norm(gravity)
    forward = rotation @ (optical / np.linalg.norm(optical))
    forward -= up * np.dot(forward, up)
    if np.linalg.norm(forward) < 1e-6:
        raise ValueError('Vertical optical axis has no defined horizontal heading')
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    return np.column_stack((forward, right, up))


def query_boxes():
    """Same twelve-cell ordering and bounds as the frozen BODY-QUERY model."""
    boxes = []
    for low, high in BODY_BOXES:
        for distance in range(2):
            for lateral in range(3):
                dy = (high[1] - low[1]) / 3
                boxes.append(((high[0] + 1.5 * distance, low[1] + dy * lateral, low[2]),
                              (high[0] + 1.5 * (distance + 1), low[1] + dy * (lateral + 1), high[2])))
    return np.asarray(boxes, dtype=np.float64)


def count_query_points(points_proxy, valid=None):
    """Count visible endpoints; invalid endpoints never become clear-space truth."""
    points = np.asarray(points_proxy, dtype=np.float64)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError('Expected ...x3 points')
    finite = np.isfinite(points).all(axis=-1)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        if valid.shape != points.shape[:-1]:
            raise ValueError('Validity shape mismatch')
        finite &= valid
    masks = []
    for i, (low, high) in enumerate(query_boxes()):
        inside = finite & (points >= low).all(axis=-1) & (points <= high).all(axis=-1)
        if i % 6 // 3 == 0:
            inside &= points[..., 0] < high[0]
        if i % 3 != 2:
            inside &= points[..., 1] < high[1]
        masks.append(inside)
    membership = np.stack(masks)
    counts = membership.reshape(12, -1).sum(axis=1)
    capped = np.minimum(counts, 3)
    near = capped.reshape(2, 6).sum(axis=1) >= 3
    body, head = map(bool, near)
    return dict(raw_counts=counts, capped_counts=capped, near=near,
                four_way='BOTH' if body and head else 'BODY_ONLY' if body else 'HEAD_ONLY' if head else 'NEITHER_DETECTED',
                four_way_flags=dict(BOTH=body and head, BODY_ONLY=body and not head,
                                    HEAD_ONLY=head and not body, NEITHER_DETECTED=not body and not head),
                unknown_count=int((~finite).sum()), valid_count=int(finite.sum()),
                membership=membership,
                semantics='Native visible-pixel counts; zero is not certified free space')


def ray_depth_coverage(unit_rays_camera, ray_depth_m, T_world_camera, gravity_world,
                       camera_height_m=1.70, camera_forward=(0., 0., 1.)):
    """Multiply unit rays by Euclidean distance, then form horizontal proxy points.

    Nonfinite/zero/nonunit rays and nonfinite distances outside (0,100) are
    UNKNOWN. Source calibration invalidity can be represented with NaN rays.
    The input calibration is never rotated or modified.
    """
    rays = np.asarray(unit_rays_camera, dtype=np.float64)
    depth = np.asarray(ray_depth_m, dtype=np.float64)
    if rays.shape != depth.shape + (3,):
        raise ValueError('Expected rays shape depth.shape + (3,)')
    if not np.isfinite(camera_height_m) or camera_height_m <= 0:
        raise ValueError('Expected positive explicit proxy camera height')
    transform = np.asarray(T_world_camera, dtype=np.float64)
    basis = horizontal_frame(transform, gravity_world, camera_forward)
    valid = (np.isfinite(depth) & (depth > 0) & (depth < 100)
             & np.isfinite(rays).all(axis=-1)
             & np.isclose(np.linalg.norm(rays, axis=-1), 1., atol=1e-5, rtol=0))
    # Subtracting the camera origin cancels world translation exactly. Avoid
    # adding/subtracting large world positions and losing boundary precision.
    safe_depth = np.where(valid, depth, 0.)
    safe_rays = np.where(valid[..., None], rays, 0.)
    points = (safe_rays * safe_depth[..., None]) @ transform[:3, :3].T @ basis
    points[..., 2] += camera_height_m
    points[~valid] = np.nan
    result = count_query_points(points, valid)
    result.update(points_proxy=points, frame_world=basis,
                  camera_height_m=float(camera_height_m),
                  height_authority='ASSUMED_CAMERA_ANCHORED_PROXY_NOT_MEASURED_FLOOR',
                  depth_semantics='EUCLIDEAN_UNIT_RAY_DISTANCE_METERS')
    return result
