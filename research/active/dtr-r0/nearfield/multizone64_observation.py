"""MZ0 clean geometric observations; no hardware noise, labels or learned fusion.

Native is Bx360x640 axial depth on CUDA. Camera HFoV100deg, eye1.7m; body
coordinates X forward, Y right, Z up. Angular crop45x45deg uses independent
atan(Y/X), atan(Z/X) coordinates, not solid-angle-equal cells. Zone order is
image row-major (top to bottom, left to right). No invalid-to-CLEAR conversion.
"""
import math
import torch

from contact_retina_spec import BODY_BOXES

EVENT_ORDER = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')


def _check(depth):
    if depth.ndim != 3 or tuple(depth.shape[1:]) != (360, 640):
        raise ValueError('Expected Bx360x640 native axial depth')
    if not depth.is_cuda or not depth.is_floating_point():
        raise ValueError('Floating CUDA native depth required')


def geometry(device, zones_per_axis=8):
    """Input-independent camera rays, centered crop, zone IDs and center rays."""
    if zones_per_axis not in (1, 8):
        raise ValueError('Only frozen1x1 and8x8 partitions supported')
    yy, xx = torch.meshgrid(torch.arange(360, device=device, dtype=torch.float64),
                            torch.arange(640, device=device, dtype=torch.float64), indexing='ij')
    f = 320./math.tan(math.radians(50.))
    right, up = (xx-319.5)/f, -(yy-179.5)/f
    rays = torch.stack((torch.ones_like(right), right, up), -1)
    norm = rays.norm(dim=-1)
    azimuth, elevation = torch.rad2deg(torch.atan(right)), torch.rad2deg(torch.atan(up))
    crop = (azimuth.abs() <= 22.5) & (elevation.abs() <= 22.5)
    width = 45./zones_per_axis
    column = ((azimuth+22.5)/width).floor().long().clamp(0, zones_per_axis-1)
    row = ((22.5-elevation)/width).floor().long().clamp(0, zones_per_axis-1)
    zone = (row*zones_per_axis+column).masked_fill(~crop, -1)
    angles = torch.arange(zones_per_axis, device=device, dtype=torch.float64)*width+width/2.
    elev, az = torch.meshgrid(22.5-angles, angles-22.5, indexing='ij')
    centers = torch.stack((torch.ones_like(az), torch.tan(torch.deg2rad(az)),
                           torch.tan(torch.deg2rad(elev))), -1).reshape(-1, 3)
    centers /= centers.norm(dim=-1, keepdim=True)
    return dict(rays=rays, radial_factor=norm, crop=crop, zone_ids=zone, center_rays=centers)


def observe(depth, zones_per_axis=8, readout='nearest'):
    """Model-visible ranges[B,Z,K] and valid flags, K=2 only for multi_surface.

    Nearest is the true minimum valid radial sample. Median weights each valid
    pixel equally, averaging the two middle samples for an even count.
    Multi-surface uses40 radial bins[0,4]m, internal boundaries assigned upward;
    exactly4m belongs to the final bin. Each supported bin needs >=3 valid pixels.
    Return actual sample means in first/last supported bins; repeated bin yields
    only one valid target. This is a clean readout hypothesis, not device emulation.
    """
    _check(depth)
    if readout not in ('nearest', 'median', 'multi_surface'):
        raise ValueError('Unknown readout')
    g = geometry(depth.device, zones_per_axis)
    radial = depth.double()*g['radial_factor']
    known = torch.isfinite(depth) & (depth > 0.) & (depth < 100.) & (radial <= 4.)
    values, flags = [], []
    for zone in range(zones_per_axis**2):
        mask = g['zone_ids'] == zone
        r, valid = radial[:, mask], known[:, mask]
        if readout == 'nearest':
            v = r.masked_fill(~valid, torch.inf).min(dim=1).values[:, None]
            good = valid.any(dim=1)[:, None]
        elif readout == 'median':
            ordered = r.masked_fill(~valid, torch.inf).sort(dim=1).values
            count = valid.sum(dim=1)
            left = ((count-1)//2).clamp(min=0)
            right = (count//2).clamp(min=0)
            v = (ordered.gather(1, left[:, None])+ordered.gather(1, right[:, None]))/2.
            good = (count > 0)[:, None]
        else:
            edges = torch.linspace(0., 4., 41, device=depth.device, dtype=torch.float64)
            bins = torch.bucketize(r.contiguous(), edges[1:-1], right=True)
            counts, means = [], []
            for index in range(40):
                member = valid & (bins == index)
                count = member.sum(dim=1)
                counts.append(count)
                means.append(r.masked_fill(~member, 0.).sum(dim=1)/count.clamp(min=1))
            counts, means = torch.stack(counts, 1), torch.stack(means, 1)
            supported = counts >= 3
            first = supported.int().argmax(dim=1)
            last = 39-supported.flip(1).int().argmax(dim=1)
            any_bin = supported.any(dim=1)
            good = torch.stack((any_bin, any_bin & (last != first)), 1)
            v = means.gather(1, torch.stack((first, last), 1))
        values.append(v.masked_fill(~good, torch.nan))
        flags.append(good)
    return dict(range_m=torch.stack(values, 1), valid=torch.stack(flags, 1),
                center_rays=g['center_rays'], zones_per_axis=zones_per_axis,
                readout=readout, eye_height_m=1.7, crop_degrees=(45., 45.),
                source='MZ0_CLEAN_GEOMETRIC_READOUT')


def _event_counts(points, valid):
    counts = []
    for low, high in BODY_BOXES:
        for half, start in enumerate((high[0], high[0]+1.5)):
            # Near is half-open, far includes the outer corridor endpoint.
            end_ok = points[..., 0] < start+1.5 if half == 0 else points[..., 0] <= start+1.5
            inside = ((points[..., 0] >= start) & end_ok
                      & (points[..., 1] >= low[1]) & (points[..., 1] <= high[1])
                      & (points[..., 2] >= low[2]) & (points[..., 2] <= high[2]))
            counts.append((inside & valid).sum(dim=1))
    return torch.stack(counts, 1)


def native_events(depth, crop=True):
    """Full pixel 3D oracle, >=3 valid pixels per event, same <=4m radial domain.

    Returned events are positive support only. observation_valid=False explicitly
    preserves all-invalid input as UNKNOWN; false event bits are not safety CLEAR.
    """
    _check(depth)
    g = geometry(depth.device)
    radial = depth.double()*g['radial_factor']
    valid = torch.isfinite(depth) & (depth > 0.) & (depth < 100.) & (radial <= 4.)
    if crop:
        valid &= g['crop']
    points = depth.double()[..., None]*g['rays']
    points[..., 2] += 1.7
    counts = _event_counts(points.flatten(1, 2), valid.flatten(1))
    return dict(counts=counts, events=counts >= 3, observation_valid=valid.flatten(1).any(1),
                valid_pixel_count=valid.flatten(1).sum(1), event_order=EVENT_ORDER,
                interpretation='POSITIVE_PIXEL_SUPPORT_ONLY')


def center_events(packet):
    """Simple readout: place each valid return along its zone center ray.

    >=1 projected point asserts an event (not the oracle's3-pixel criterion).
    This deliberately discards within-zone direction and is not information-
    theoretically optimal. It does not imply the return is actually at its center.
    """
    ranges, valid, rays = packet['range_m'], packet['valid'], packet['center_rays']
    if ranges.ndim != 3 or valid.shape != ranges.shape or rays.shape != (ranges.shape[1], 3):
        raise ValueError('Expected ranges/valid[B,Z,K] and center_rays[Z,3]')
    valid = valid & torch.isfinite(ranges) & (ranges > 0.) & (ranges <= 4.)
    points = ranges[..., None]*rays[None, :, None, :]
    points[..., 2] += packet['eye_height_m']
    counts = _event_counts(points.flatten(1, 2), valid.flatten(1))
    return dict(counts=counts, events=counts >= 1, observation_valid=valid.flatten(1).any(1),
                valid_point_count=valid.flatten(1).sum(1), event_order=EVENT_ORDER,
                interpretation='ZONE_CENTER_POINT_APPROXIMATION')
