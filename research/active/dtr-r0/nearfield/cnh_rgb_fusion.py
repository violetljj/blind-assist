"""Small, untrained RGB/CNH fusion candidate for Development experiments.

This module is an architecture interface, not an evaluated model. The same
``FrustumFusion`` can use signed CNH bins or the corresponding scalar readout.
Neither target geometry nor labels are accepted in ``forward``.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class FrustumFusion(nn.Module):
    """Decode six corridor queries from locally aligned RGB and zone tokens.

    Inputs: RGB [B,3,H,W], signed histogram [B,Z,L], ambient [B,Z],
    scalar metres [B,Z], scalar-valid [B,Z], age seconds [B,Z], and public
    calibration support [Z,H/4,W/4] or [B,Z,H/4,W/4]. ``mode`` chooses a
    paired CNH/scalar/RGB-only arm. Invalid scalar values remain invalid.
    """

    def __init__(self, zones: int, bins: int, width: int = 64, queries: int = 6):
        super().__init__()
        if zones < 1 or bins < 1 or width < 8 or queries < 1:
            raise ValueError('Positive model dimensions required')
        side = math.isqrt(zones)
        if side * side != zones:
            raise ValueError('Declared ToF zones must form a square grid')
        self.zones, self.bins = zones, bins
        self.rgb = nn.Sequential(
            # Nonoverlapping patches keep pixels outside a frustum from entering
            # its local feature through a neighbouring convolution footprint.
            nn.Conv2d(3, width, 4, stride=4), nn.GELU(),
            nn.Conv2d(width, width, 1), nn.GELU(),
        )
        # Shared public metadata is held fixed across CNH and scalar arms.
        self.metadata = nn.Linear(4, width)
        self.cnh = nn.Linear(bins, width)
        self.scalar = nn.Linear(1, width)
        self.fuse = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
        coordinates = torch.stack(torch.meshgrid(
            (torch.arange(side, dtype=torch.float32) + .5) * 2 / side - 1,
            (torch.arange(side, dtype=torch.float32) + .5) * 2 / side - 1,
            indexing='ij'), -1).reshape(zones, 2)
        self.register_buffer('zone_coordinates', coordinates)
        self.zone_position = nn.Linear(2, width)
        self.query = nn.Parameter(torch.randn(queries, width) * .02)
        self.occupancy = nn.Linear(width, 1)
        self.distance = nn.Linear(width, 1)

    def forward(self, rgb, histogram, ambient, scalar_m, scalar_valid, age_s, support, mode='cnh'):
        if mode not in ('cnh', 'scalar', 'rgb'):
            raise ValueError('mode must be cnh, scalar, or rgb')
        if rgb.ndim != 4 or rgb.shape[1] != 3 or rgb.shape[2] % 4 or rgb.shape[3] % 4:
            raise ValueError('RGB must be [B,3,H,W] with dimensions divisible by four')
        batch = rgb.shape[0]
        if histogram.shape != (batch, self.zones, self.bins):
            raise ValueError('Histogram shape differs from declared configuration')
        shape = (batch, self.zones)
        if any(t.shape != shape for t in (ambient, scalar_m, scalar_valid, age_s)):
            raise ValueError('Zone metadata shape mismatch')
        if not all(torch.isfinite(t).all() for t in (rgb, histogram, ambient, age_s)):
            raise ValueError('Nonfinite observable input')
        if not torch.isfinite(scalar_m[scalar_valid.bool()]).all():
            raise ValueError('Nonfinite valid scalar distance')
        if torch.any(age_s < 0):
            raise ValueError('Negative observation age')
        features = self.rgb(rgb)
        grid = features.shape[-2:]
        if support.ndim == 3:
            support = support.unsqueeze(0).expand(batch, -1, -1, -1)
        if support.shape != (batch, self.zones, *grid):
            raise ValueError('Frustum support and RGB feature grid mismatch')
        if not torch.isfinite(support).all() or torch.any((support < 0) | (support > 1)):
            raise ValueError('Frustum support must be finite and in [0,1]')
        total = support.sum((-1, -2))
        if torch.any(total <= 0):
            raise ValueError('Each zone needs RGB support; inspect calibration/field of view')
        rgb_zone = torch.einsum('bchw,bzhw->bzc', features, support) / total[..., None]
        valid = scalar_valid.bool()
        safe_scalar = torch.where(valid, scalar_m, torch.zeros_like(scalar_m))
        public = torch.stack((
            torch.log1p(torch.clamp_min(ambient, 0)),
            safe_scalar,
            valid.to(rgb.dtype),
            torch.log1p(age_s),
        ), -1)
        token = self.metadata(public)
        if mode == 'cnh':
            # Background-corrected bins can be negative; keep their sign.
            token = token + self.cnh(torch.sign(histogram) * torch.log1p(histogram.abs()))
        elif mode == 'scalar':
            token = token + self.scalar(safe_scalar[..., None])
        else:
            token = torch.zeros_like(token)
        zones = self.fuse(torch.cat((rgb_zone, token), -1)) + self.zone_position(self.zone_coordinates)[None]
        weights = torch.softmax(torch.einsum('qd,bzd->bqz', self.query, zones) / zones.shape[-1] ** .5, -1)
        decoded = torch.einsum('bqz,bzd->bqd', weights, zones)
        return {'occupancy_logits': self.occupancy(decoded).squeeze(-1),
                'distance_m': F.softplus(self.distance(decoded).squeeze(-1)),
                'query_zone_weights': weights}
