"""Delayed query interaction with native angular/radial observations.

Only native observations, their public rigid transforms and fixed HEAD/BODY
query boxes enter this module. Extent is angular midpoint quadrature at each
radial bin center, not a reconstruction or an exact finite-volume overlap.
Center and extent produce the same feature shape and use the identical model.
"""
import math

import torch
from torch import nn

from cnh_cvr_projection import EDGE, WIDTH


QUERY_BOXES = ((-.30, -.20, .30, .30, .42, 3.),
               (-.30, .42, .30, .30, .90, 3.))
FEATURE_CHANNELS = 128
SOFTNESS = .05


def _valid_lengths(lengths, batch, device):
    lengths = torch.as_tensor(lengths, device=device)
    if lengths.shape != (batch,) or bool(((lengths < 1) | (lengths > 8)
                                       | (lengths != lengths.round())).any()):
        raise ValueError('Expected integral history lengths [B] in 1..8')
    return lengths


def _valid_mask(lengths):
    return torch.arange(8, device=lengths.device)[None] >= 8 - lengths[:, None]


def _sensor_points(device, mode):
    if mode not in ('center', 'extent'):
        raise ValueError('Geometry mode must be center or extent')
    offsets = torch.tensor([.5] if mode == 'center' else [1/6, .5, 5/6],
                           dtype=torch.float32, device=device)
    slopes = -EDGE + (torch.arange(8, device=device)[:, None] + offsets) * (2*EDGE/8)
    # Axes are zone_y, zone_x, node_y, node_x, xyz.
    y = slopes[:, None, :, None].expand(8, 8, len(offsets), len(offsets))
    x = slopes[None, :, None, :].expand_as(y)
    rays = torch.stack((x, y, torch.ones_like(x)), -1).reshape(8, 8, -1, 3)
    rays = rays / torch.linalg.vector_norm(rays, dim=-1, keepdim=True)
    radius = (torch.arange(16, device=device) + .5) * WIDTH
    return rays[:, :, None] * radius[None, None, :, None, None]


@torch.no_grad()
def feature_geometry(transforms, lengths, mode, *, softness=SOFTNESS):
    """Return memberships [B,2,8,8,8,16,3] on the transforms' device.

    Transforms [B,8,4,4] map each historical sensor frame to the current
    public query frame. The final three channels are node mean/min/max.
    Center duplicates its single value across the three channels. Membership
    is exp(-L1 distance outside the query box / softness), with value 1 inside.
    Left padding is excluded even when its transforms contain NaN.
    Call in short batches or once per distinct public transform history; cache
    build_features output rather than keeping this larger geometry resident.
    """
    transforms = torch.as_tensor(transforms)
    if transforms.ndim != 4 or transforms.shape[1:] != (8, 4, 4):
        raise ValueError('Expected public transforms [B,8,4,4]')
    if not math.isfinite(softness) or softness <= 0:
        raise ValueError('softness must be positive and finite')
    lengths = _valid_lengths(lengths, len(transforms), transforms.device)
    valid = _valid_mask(lengths)
    t = torch.where(valid[:, :, None, None], transforms.float(), 0.)
    points = _sensor_points(t.device, mode)
    xyz = torch.einsum('btij,yxrnj->btyxrni', t[:, :, :3, :3], points)
    xyz = xyz + t[:, :, None, None, None, None, :3, 3]
    boxes = xyz.new_tensor(QUERY_BOXES)
    memberships = []
    for box in boxes:
        distance = (box[:3] - xyz).clamp_min(0) + (xyz - box[3:]).clamp_min(0)
        value = torch.exp(-distance.sum(-1) / softness)
        memberships.append(torch.stack((value.mean(-1), value.amin(-1), value.amax(-1)), -1))
    geometry = torch.stack(memberships, 1)
    return torch.where(valid[:, None, :, None, None, None, None], geometry, 0.)


def build_features(histories, geometry, lengths):
    """Return [B,2,128,8,8] fixed features; no learned preprocessing.

    Signed-log evidence interacts with query membership before pooling time.
    All 16 radial bins remain separate feature channels. Channel groups are
    gated history mean (16x3), gated current (16x3), ungated history mean (16),
    and ungated current (16). The mean includes all valid exposures, including
    zero observations. NaN left padding in observations/geometry is ignored.
    Features can be cached as FP16; computation uses FP32.
    """
    if histories.ndim != 5 or histories.shape[1:] != (8, 8, 8, 16):
        raise ValueError('Expected native histories [B,8,8,8,16]')
    batch = len(histories)
    if geometry.shape != (batch, 2, 8, 8, 8, 16, 3):
        raise ValueError('Expected query geometry [B,2,8,8,8,16,3]')
    if histories.device != geometry.device:
        raise ValueError('Histories and geometry must use the same device')
    lengths = _valid_lengths(lengths, batch, histories.device)
    valid = _valid_mask(lengths)
    z = torch.where(valid[:, :, None, None, None], histories.float(), 0.)
    z = z.sign() * z.abs().log1p()
    membership = torch.where(valid[:, None, :, None, None, None, None], geometry.float(), 0.)
    gated = z[:, None, ..., None] * membership
    denominator = lengths.float()[:, None, None, None, None, None]
    gated_mean = (gated.sum(2) / denominator).flatten(-2)
    gated_current = gated[:, :, -1].flatten(-2)
    native_mean = (z.sum(1) / lengths.float()[:, None, None, None])[:, None].expand(-1, 2, -1, -1, -1)
    native_current = z[:, None, -1].expand(-1, 2, -1, -1, -1)
    features = torch.cat((gated_mean, gated_current, native_mean, native_current), -1)
    return features.permute(0, 1, 4, 2, 3).contiguous()


class DelayedQueryReadout(nn.Module):
    """15,713 parameter neighborhood readout shared by both geometry modes."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(FEATURE_CHANNELS, 32, 1), nn.GELU(),
                                     nn.Conv2d(32, 32, 3, padding=1), nn.GELU())
        self.head = nn.Sequential(nn.Linear(71, 32), nn.GELU(), nn.Linear(32, 1))
        self.register_buffer('query_boxes', torch.tensor(QUERY_BOXES) /
                             torch.tensor([.6, 1., 3., .6, 1., 3.]))

    def forward(self, features, lengths):
        if features.ndim != 5 or features.shape[1:] != (2, FEATURE_CHANNELS, 8, 8):
            raise ValueError('Expected delayed features [B,2,128,8,8]')
        batch = len(features)
        lengths = _valid_lengths(lengths, batch, features.device)
        hidden = self.encoder(features.float().reshape(batch*2, FEATURE_CHANNELS, 8, 8))
        pooled = torch.cat((hidden.mean((-2, -1)), hidden.amax((-2, -1))), -1).reshape(batch, 2, 64)
        boxes = self.query_boxes[None].expand(batch, -1, -1)
        length = (lengths.float()/8)[:, None, None].expand(-1, 2, -1)
        return self.head(torch.cat((pooled, boxes, length), -1)).squeeze(-1)

    def forward_observations(self, histories, geometry, lengths):
        return self(build_features(histories, geometry, lengths), lengths)


def parameter_count():
    return sum(parameter.numel() for parameter in DelayedQueryReadout().parameters())
