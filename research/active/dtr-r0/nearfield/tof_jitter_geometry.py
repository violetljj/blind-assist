"""Fixed-origin rotational sensitivity geometry, NOT a walking/sensor simulator.

Body axes are forward X, right Y, up Z. Positive yaw points right and positive
pitch points up. A static camera depth map is reused; no translation, changing
occlusion, RGB motion, inertial estimation, or real temporal evidence is implied.
"""
import math

import numpy as np
import torch

from contact_retina_spec import BODY_BOXES

DIAGONAL_FOV_DEG = 15.


def trajectory():
    """25 inclusive samples over two seconds; frozen seed23 angular perturbation."""
    times = np.linspace(0., 2., 25)
    perturbation = np.random.default_rng(23).uniform(-.25, .25, (25, 2))
    pitch = 2. * np.sin(2. * np.pi * 2. * times) + perturbation[:, 0]
    yaw = np.sin(2. * np.pi * times) + perturbation[:, 1]
    return [dict(step=i, time_s=float(t), pitch_deg=float(p), yaw_deg=float(y))
            for i, (t, p, y) in enumerate(zip(times, pitch, yaw))]


def rotation(pitch_deg, yaw_deg):
    """Sensor-to-body matrix Rz(yaw) @ Ry(-pitch); sensor origin is unchanged."""
    if not all(math.isfinite(v) for v in (pitch_deg, yaw_deg)):
        raise ValueError('Finite orientation required')
    p, y = map(math.radians, (pitch_deg, yaw_deg))
    cp, sp, cy, sy = math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, -sy, -cy*sp],
                     [sy*cp, cy, -sy*sp], [sp, 0., cp]], dtype=np.float64)


def footprint(device, pitch_deg, yaw_deg):
    """Rotated square footprint in ORIGINAL 640x360,100deg-HFoV camera depth.

    Returns the same mask/solid-angle weights/radial factors contract as the
    static helper. Refuses clipped footprints; missing off-camera rays must not
    be silently removed from the full-footprint support denominator.
    """
    r = rotation(pitch_deg, yaw_deg)
    slope = math.tan(math.radians(DIAGONAL_FOV_DEG/2.))/math.sqrt(2.)
    f = 320./math.tan(math.radians(50.))
    corners = np.array([[1., y, z] for y in (-slope, slope)
                        for z in (-slope, slope)]) @ r.T
    if (np.any(corners[:, 0] <= 0.)
            or np.any(np.abs(corners[:, 1]/corners[:, 0]) > 320./f)
            or np.any(np.abs(corners[:, 2]/corners[:, 0]) > 180./f)):
        raise ValueError('Rotated footprint leaves native camera view; not evaluable')
    yy, xx = torch.meshgrid(torch.arange(360, device=device),
                            torch.arange(640, device=device), indexing='ij')
    rx, ry = (xx-319.5)/f, (yy-179.5)/f
    rays = torch.stack((torch.ones_like(rx), rx, -ry), dim=-1)
    sensor = rays @ torch.as_tensor(r, device=device, dtype=rays.dtype)
    mask = ((sensor[..., 0] > 0.)
            & (sensor[..., 1].abs() <= slope*sensor[..., 0])
            & (sensor[..., 2].abs() <= slope*sensor[..., 0]))
    norms = (1.+rx.square()+ry.square()).sqrt()
    return mask, norms.pow(-3)[mask], norms[mask]


def support(range_m, valid, uncertainty_m, pitch_deg, yaw_deg, eye_height_m=1.7):
    """Containment-only positive evidence using an orientation-aware shell AABB.

    Rotate the exact aligned shell AABB and bound its eight corners. This encloses
    the rotated shell conservatively, though it may abstain more than tight shell
    extrema. No unsupported return, invalid value, or failed containment is CLEAR.
    """
    if not valid:
        return dict(status='UNKNOWN', supported=[], bounds=None)
    values = (range_m, uncertainty_m, eye_height_m)
    if any(v is None or not math.isfinite(v) for v in values):
        raise ValueError('Finite calibrated range inputs required')
    if range_m <= 0. or uncertainty_m < 0.:
        raise ValueError('Invalid range uncertainty')
    r = rotation(pitch_deg, yaw_deg)
    lo, hi = max(0., range_m-uncertainty_m), range_m+uncertainty_m
    slope = math.tan(math.radians(DIAGONAL_FOV_DEG/2.))/math.sqrt(2.)
    lateral = hi*slope/math.sqrt(1.+slope*slope)
    low = np.array([lo/math.sqrt(1.+2.*slope*slope), -lateral, -lateral])
    high = np.array([hi, lateral, lateral])
    # Interval multiplication bounds every point in the enclosing box.
    lower = np.minimum(r*low, r*high).sum(axis=1)
    upper = np.maximum(r*low, r*high).sum(axis=1)
    lower[2] += eye_height_m
    upper[2] += eye_height_m
    supported = []
    for kind, (body_low, body_high) in zip(('BODY', 'HEAD'), BODY_BOXES):
        for half, (start, end) in zip(('NEAR', 'FAR'),
                ((body_high[0], body_high[0]+1.5), (body_high[0]+1.5, body_high[0]+3.))):
            if (lower[0] >= start and upper[0] <= end
                    and lower[1] >= body_low[1] and upper[1] <= body_high[1]
                    and lower[2] >= body_low[2] and upper[2] <= body_high[2]):
                supported.append(kind+'_'+half)
    return dict(status='POSITIVE_SUPPORT' if supported else 'UNKNOWN',
                supported=supported, bounds=[lower.tolist(), upper.tolist()])
