"""Small single-frame angular-return readout; eligibility is not occupancy.

Inputs are frozen full-image features and measured radial returns only. Angular
midpoint rays approximate each zone's angular footprint; they are hypotheses,
not measured return locations or an exhaustive intersection test. This assumes
the existing level-camera calibration (HFOV100, 640x360, eye1.7m).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from contact_retina_spec import BODY_BOXES
from multizone64_observation import EVENT_ORDER


def angular_hypotheses(samples_per_axis=7, visual_shift_columns=0):
    """Return radial unit rays[64,S,3] and full-image grid[64,S,2]."""
    if samples_per_axis < 1:
        raise ValueError('samples_per_axis must be positive')
    offsets = (torch.arange(samples_per_axis, dtype=torch.float64)+.5)/samples_per_axis
    rr, cc = torch.meshgrid(offsets, offsets, indexing='ij')
    row = torch.arange(64, dtype=torch.float64).div(8, rounding_mode='floor')[:, None]
    col = torch.arange(64, dtype=torch.float64).remainder(8)[:, None]
    az = torch.deg2rad(-22.5+(col+cc.flatten())*45/8)
    el = torch.deg2rad(22.5-(row+rr.flatten())*45/8)
    rays = torch.stack((torch.ones_like(az), az.tan(), el.tan()), -1)
    rays = rays/rays.norm(dim=-1, keepdim=True)
    # Shift visual correspondence alone, keeping the ray's within-zone offset.
    shifted_az = torch.deg2rad(-22.5+((col+visual_shift_columns)%8+cc.flatten())*45/8)
    f = 320/math.tan(math.radians(50))
    # align_corners=False: source pixel319.5 maps to normalized coordinate0.
    grid = torch.stack((shifted_az.tan()*f/320, -el.tan()*f/180), -1)
    return rays.float(), grid.float()


def query_membership(points):
    """Boolean [...,4] membership of candidate XYZ body-frame positions."""
    masks = []
    for low, high in BODY_BOXES:
        for half, start in enumerate((high[0], high[0]+1.5)):
            end = points[..., 0] < start+1.5 if half == 0 else points[..., 0] <= start+1.5
            masks.append((points[..., 0] >= start) & end
                         & (points[..., 1] >= low[1]) & (points[..., 1] <= high[1])
                         & (points[..., 2] >= low[2]) & (points[..., 2] <= high[2]))
    return torch.stack(masks, -1)


class AngularReturnReadout(nn.Module):
    """One shared visual/range evidence MLP, then eligible-candidate max.

    forward returns logits[B,4]. inspect also exposes candidate support and an
    observation_valid flag. Finite negative outputs without support must not be
    interpreted as CLEAR. Neither query identity nor ground truth enters the
    evidence MLP. Query biases are four learned readout thresholds.
    """
    event_order = EVENT_ORDER

    def __init__(self, in_channels=64, hidden=32, samples_per_axis=7):
        super().__init__()
        self.in_channels = in_channels
        rays, grid = angular_hypotheses(samples_per_axis)
        _, shifted = angular_hypotheses(samples_per_axis, visual_shift_columns=4)
        self.register_buffer('rays', rays)
        self.register_buffer('grid', grid)
        self.register_buffer('wrong_grid', shifted)
        self.evidence = nn.Sequential(nn.Linear(in_channels+1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.query_bias = nn.Parameter(torch.zeros(4))

    def eligibility(self, ranges, valid):
        if ranges.ndim != 3 or tuple(ranges.shape[1:]) != (64, 2) or valid.shape != ranges.shape:
            raise ValueError('Expected ranges/valid[B,64,2]')
        good = valid.bool() & torch.isfinite(ranges) & (ranges > 0) & (ranges <= 4)
        clean = torch.where(good, ranges, torch.zeros_like(ranges))
        points = clean[..., None, None]*self.rays[None, :, None, :, :]
        points = points + points.new_tensor([0., 0., 1.7])
        eligible = query_membership(points) & good[..., None, None]
        return clean, good, eligible

    def sample_visual(self, features, wrong_zone=False):
        if features.ndim != 4 or features.shape[1] != self.in_channels:
            raise ValueError('Expected dense full-image features[B,in_channels,H,W]')
        grid = self.wrong_grid if wrong_zone else self.grid
        sampled = F.grid_sample(features, grid[None].expand(features.shape[0], -1, -1, -1),
                                mode='bilinear', padding_mode='zeros', align_corners=False)
        return sampled.permute(0, 2, 3, 1)  # B,64,S,C

    def inspect(self, features, ranges, valid, wrong_zone=False):
        if features.shape[0] != ranges.shape[0]:
            raise ValueError('Feature and packet batch sizes differ')
        tokens = self.sample_visual(features, wrong_zone)
        return self.forward_tokens(tokens, ranges, valid, return_details=True)

    def forward_tokens(self, tokens, ranges, valid, wrong_zone=False, return_details=False):
        """Use cached sampled tokens[B,64,S*S,C]; control shifts four columns."""
        if (tokens.ndim != 4 or tokens.shape[0] != ranges.shape[0]
                or tuple(tokens.shape[1:]) != (64, self.rays.shape[1], self.in_channels)):
            raise ValueError('Expected sampled tokens[B,64,S*S,in_channels]')
        if wrong_zone:
            tokens = tokens.reshape(tokens.shape[0], 8, 8, *tokens.shape[2:]).roll(4, dims=2).flatten(1, 2)
        clean, good, eligible = self.eligibility(ranges, valid)
        tokens = tokens[:, :, None].expand(-1, -1, 2, -1, -1)
        r = (clean/4)[..., None, None].expand(*tokens.shape[:-1], 1)
        evidence = self.evidence(torch.cat((tokens, r.to(tokens.dtype)), -1)).squeeze(-1)
        support = eligible.flatten(1, 3).any(dim=1)
        score = evidence[..., None].expand_as(eligible).masked_fill(~eligible, -torch.inf)
        pooled = score.flatten(1, 3).amax(dim=1)
        logits = torch.where(support, pooled+self.query_bias, pooled.new_full(pooled.shape, -20.))
        if return_details:
            return dict(logits=logits, support=support, eligible=eligible,
                        candidate_logits=evidence, candidate_evidence=evidence,
                        observation_valid=good.flatten(1).any(1),
                        interpretation='ANGULAR_CANDIDATE_SUPPORT_NOT_CLEAR')
        return logits

    def forward(self, features, ranges, valid, wrong_zone=False):
        return self.inspect(features, ranges, valid, wrong_zone)['logits']
