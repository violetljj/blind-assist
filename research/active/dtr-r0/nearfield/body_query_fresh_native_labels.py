"""Frozen native admission functions packaged without unrelated collection WIP.

Exact function snapshot from the executed admission overlay; no changed math.
"""
import torch
from body_query_model import CALIBRATION, query_boxes
from worlds_verify import camera_points, corridor_masks


def labels_batch(native, cameras, floors):
    """Same arithmetic/partition as body_query_labels.labels, with a batch axis."""
    if native.ndim != 3 or tuple(native.shape[1:]) != (360, 640):
        raise ValueError('Expected Bx360x640 native axial depth')
    if native.dtype != torch.float32 or len(cameras) != len(native) or len(floors) != len(native):
        raise ValueError('Native dtype or metadata count mismatch')
    for c, floor in zip(cameras, floors, strict=True):
        if (abs(float(c['pitch'])) > 1e-5 or abs(float(c.get('roll', 0))) > 1e-5
                or abs(float(c['z'])-float(floor)-CALIBRATION['height_m']) > 1e-4):
            raise ValueError('Frame violates fixed optical calibration')
    origin = dict(x=0., y=0., z=0., pitch=0., yaw=0., roll=0.)
    rays = camera_points(torch.ones_like(native[0]), origin)
    points = native[..., None]*rays
    points[..., 2] += CALIBRATION['height_m']
    valid = torch.isfinite(native) & (native > 0) & (native < 100)
    masks = corridor_masks(points, valid, origin, 3.).movedim(0, 1)
    membership = torch.zeros_like(masks, dtype=torch.int16)
    counts = []
    for i, (low, high) in enumerate(query_boxes()):
        head, distance, lateral = i//6, (i % 6)//3, i % 3
        inside = masks[:, head].clone()
        inside &= (points[..., 0] >= low[0]) & ((points[..., 0] <= high[0]) if distance == 1 else (points[..., 0] < high[0]))
        inside &= (points[..., 1] >= low[1]) & ((points[..., 1] <= high[1]) if lateral == 2 else (points[..., 1] < high[1]))
        counts.append(inside.sum((-2, -1)))
        membership[:, head] += inside
    if not torch.equal(membership, masks.to(torch.int16)):
        raise ValueError('Query partition overlaps or omits native points')
    raw = torch.stack(counts, 1)
    near = (masks.sum((-2, -1)) >= 3).long()
    capped = raw.clamp_max(3)
    if not torch.equal((capped.reshape(-1, 2, 6).sum(-1) >= 3).long(), near):
        raise ValueError('Count aggregation changed native near label')
    return dict(near=near, counts=capped, raw_counts=raw,
                support=masks.to(torch.int8).masked_fill(~valid[:, None], -1))


def endpoint_reasons(near, counts, raw, endpoint, floor_ok):
    """Pure scalar admission; opposite-half subthreshold pixels still reject."""
    reasons = []
    if list(near) != [0, 1]:
        reasons.append('NATIVE_BODY_HEAD_NOT_0_1')
    if endpoint not in ('near', 'far'):
        return reasons+['UNKNOWN_ENDPOINT']
    intended = slice(6, 9) if endpoint == 'near' else slice(9, 12)
    opposite = slice(9, 12) if endpoint == 'near' else slice(6, 9)
    if sum(counts[intended]) < 3:
        reasons.append('INTENDED_HEAD_HALF_BELOW_THREE_CAPPED_COUNTS')
    if sum(raw[opposite]) != 0:
        reasons.append('OPPOSITE_HEAD_HALF_HAS_NATIVE_PIXELS')
    if not floor_ok:
        reasons.append('CAMERA_FLOOR_PROBE_FAILED')
    return reasons
