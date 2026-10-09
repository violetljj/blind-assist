"""Learn radial/query boundary interactions before compressing native bins.

Only observed histories, public transforms, fixed query boxes and radial/angle
calibration are inputs. A signed margin identifies each query face separately;
positive means inside that face, negative means outside. Extent describes nine
angular hypotheses at a radial bin center, not recovered object locations.
"""
import math

import torch
from torch import nn

from cnh_cvr_projection import EDGE, WIDTH


QUERY_BOXES = ((-.30, -.20, .30, .30, .42, 3.),
               (-.30, .42, .30, .30, .90, 3.))
GEOMETRY_CHANNELS = 11
FEATURE_CHANNELS = 15
SOFTNESS = .05


def _lengths(lengths, batch, device):
    lengths = torch.as_tensor(lengths, device=device)
    if lengths.shape != (batch,) or bool(((lengths < 1) | (lengths > 8)
                                       | (lengths != lengths.round())).any()):
        raise ValueError('Expected integral history lengths [B] in 1..8')
    return lengths


def _valid(lengths):
    return torch.arange(8, device=lengths.device)[None] >= 8-lengths[:, None]


def _points(device, mode):
    if mode not in ('center', 'extent'):
        raise ValueError('Geometry mode must be center or extent')
    offsets = torch.tensor([.5] if mode == 'center' else [1/6, .5, 5/6],
                           device=device, dtype=torch.float32)
    slopes = -EDGE + (torch.arange(8, device=device)[:, None]+offsets)*(2*EDGE/8)
    y = slopes[:, None, :, None].expand(8, 8, len(offsets), len(offsets))
    x = slopes[None, :, None, :].expand_as(y)
    rays = torch.stack((x, y, torch.ones_like(x)), -1).reshape(8, 8, -1, 3)
    rays /= torch.linalg.vector_norm(rays, dim=-1, keepdim=True)
    radius = (torch.arange(16, device=device)+.5)*WIDTH
    return rays[:, :, None]*radius[None, None, :, None, None]


@torch.no_grad()
def feature_geometry(transforms, lengths, mode, *, softness=SOFTNESS):
    """[B,2,time8,zone_y8,zone_x8,bin16,11] geometry on input device.

    Channels: membership mean/min/max (3), per-face signed margin min across
    angular nodes (6), and outside-L1-distance min/max (2). Face order is
    x-low, x-high, y-low, y-high, z-low, z-high; fixed margin scales are
    .6,.6,1,1,3,3 meters. Outside distances use .6 meters. Extent's per-face
    minima retain which side the angular footprint reaches; each face may
    attain its minimum at a different node. These are hypothesis-envelope
    features, not a joint point attribution. Center uses one node.
    Left padding is zero even for NaN transforms. Precompute in short batches.
    """
    transforms = torch.as_tensor(transforms)
    if transforms.ndim != 4 or transforms.shape[1:] != (8, 4, 4):
        raise ValueError('Expected public transforms [B,8,4,4]')
    if not math.isfinite(softness) or softness <= 0:
        raise ValueError('softness must be positive and finite')
    lengths = _lengths(lengths, len(transforms), transforms.device)
    valid = _valid(lengths)
    t = torch.where(valid[:, :, None, None], transforms.float(), 0.)
    xyz = torch.einsum('btij,yxrnj->btyxrni', t[:, :, :3, :3], _points(t.device, mode))
    xyz += t[:, :, None, None, None, None, :3, 3]
    boxes = xyz.new_tensor(QUERY_BOXES)
    margin_scale = xyz.new_tensor([.6, .6, 1., 1., 3., 3.])
    output = []
    for box in boxes:
        lower, upper = xyz-box[:3], box[3:]-xyz
        # Stack xyz low/high pairs to retain a separate signed value per face.
        margins = torch.stack((lower, upper), -1).flatten(-2)
        outside = (-margins).clamp_min(0).sum(-1)
        membership = torch.exp(-outside/softness)
        stats = torch.stack((membership.mean(-1), membership.amin(-1), membership.amax(-1)), -1)
        signed_faces = margins.amin(-2)/margin_scale
        outside_stats = torch.stack((outside.amin(-1), outside.amax(-1)), -1)/.6
        output.append(torch.cat((stats, signed_faces, outside_stats), -1))
    geometry = torch.stack(output, 1)
    return torch.where(valid[:, None, :, None, None, None, None], geometry, 0.)


def build_features(histories, geometry, lengths):
    """[B,2,bin16,15,zone_y8,zone_x8] fixed FP32 token features.

    Channels 0..2: signed-log current, valid-history mean, membership-gated
    history mean. Channels 3..13: current geometry's eleven channels in the
    order documented by feature_geometry. Channel 14: radial bin center/3m.
    No bin or zone compression occurs here. Temporal mean includes all valid
    exposures and ignores left padding, including NaN padding. Features may
    be cached FP16; the learned token network sees per-bin geometry/evidence.
    """
    if histories.ndim != 5 or histories.shape[1:] != (8, 8, 8, 16):
        raise ValueError('Expected native histories [B,8,8,8,16]')
    batch = len(histories)
    if geometry.shape != (batch, 2, 8, 8, 8, 16, GEOMETRY_CHANNELS):
        raise ValueError('Expected boundary geometry [B,2,8,8,8,16,11]')
    if histories.device != geometry.device:
        raise ValueError('Histories and geometry must use the same device')
    lengths = _lengths(lengths, batch, histories.device)
    valid = _valid(lengths)
    z = torch.where(valid[:, :, None, None, None], histories.float(), 0.)
    z = z.sign()*z.abs().log1p()
    g = torch.where(valid[:, None, :, None, None, None, None], geometry.float(), 0.)
    current = z[:, None, -1].expand(-1, 2, -1, -1, -1)
    mean = (z.sum(1)/lengths.float()[:, None, None, None])[:, None].expand_as(current)
    gated_mean = (z[:, None]*g[..., 0]).sum(2)/lengths.float()[:, None, None, None, None]
    radius = ((torch.arange(16, device=z.device)+.5)*WIDTH/3)[None, None, None, None].expand_as(current)
    features = torch.cat((torch.stack((current, mean, gated_mean), -1),
                          g[:, :, -1], radius[..., None]), -1)
    return features.permute(0, 1, 4, 5, 2, 3).contiguous()


class BoundaryTokenReadout(nn.Module):
    """8,833 parameters; encode each radial token before bin/zone pooling."""
    def __init__(self):
        super().__init__()
        self.token_encoder = nn.Sequential(nn.Conv2d(FEATURE_CHANNELS, 16, 1), nn.GELU(),
                                           nn.Conv2d(16, 16, 1), nn.GELU())
        self.spatial = nn.Sequential(nn.Conv2d(32, 24, 3, padding=1), nn.GELU())
        self.head = nn.Sequential(nn.Linear(55, 24), nn.GELU(), nn.Linear(24, 1))
        self.register_buffer('query_boxes', torch.tensor(QUERY_BOXES)/
                             torch.tensor([.6, 1., 3., .6, 1., 3.]))

    def forward(self, features, lengths):
        if features.ndim != 6 or features.shape[1:] != (2, 16, FEATURE_CHANNELS, 8, 8):
            raise ValueError('Expected boundary features [B,2,16,15,8,8]')
        batch = len(features)
        lengths = _lengths(lengths, batch, features.device)
        tokens = self.token_encoder(features.float().reshape(batch*2*16, FEATURE_CHANNELS, 8, 8))
        tokens = tokens.reshape(batch*2, 16, 16, 8, 8)
        bins = torch.cat((tokens.mean(1), tokens.amax(1)), 1)
        zones = self.spatial(bins)
        pooled = torch.cat((zones.mean((-2, -1)), zones.amax((-2, -1))), -1).reshape(batch, 2, 48)
        boxes = self.query_boxes[None].expand(batch, -1, -1)
        le = (lengths.float()/8)[:, None, None].expand(-1, 2, -1)
        return self.head(torch.cat((pooled, boxes, le), -1)).squeeze(-1)

    def forward_observations(self, histories, geometry, lengths):
        return self(build_features(histories, geometry, lengths), lengths)


def parameter_count():
    return sum(p.numel() for p in BoundaryTokenReadout().parameters())
