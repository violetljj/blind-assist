"""Matched carriers for direct and monotone contact-boundary supervision.

Inputs are externally extracted, train-only standardized observation features:
frozen ImageNet MobileNetV3-small features[:7], pooled to 8 x 14 (4,480),
then ordered canonical ToF 64 x 6 (384). No images or evaluator truth enter
this module. The geometry arm is a nonnegative latent parameterization, not
a claim of calibrated density, physical geometry, or independent events.
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ContactBoundaryModel(nn.Module):
    """Same state carrier and shared trunk; mode selects the effective head."""

    FEATURE_DIM = 4864
    WIDTH_CELLS = 20
    DEPTH_CELLS = 30

    def __init__(self, mode: Literal["direct", "geometry"] = "direct") -> None:
        super().__init__()
        if mode not in ("direct", "geometry"):
            raise ValueError("mode must be 'direct' or 'geometry'")
        self.mode = mode
        self.trunk = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
        )
        self.direct_head = nn.Sequential(
            nn.Linear(132, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.geometry_head = nn.Linear(128, 2 * self.WIDTH_CELLS * self.DEPTH_CELLS)
        nn.init.constant_(self.geometry_head.bias, -7.0)
        self.register_buffer("width_edges", torch.linspace(0.0, 0.6, 21))
        self.register_buffer("depth_edges", torch.linspace(0.0, 3.0, 31))

    def encode_scene(self, features: Tensor) -> Tensor:
        if features.ndim != 2 or features.shape[1] != self.FEATURE_DIM:
            raise ValueError("features must have shape [B,4864]")
        if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
            raise ValueError("features must be finite floating-point observations")
        return self.trunk(features)

    @staticmethod
    def _validate_queries(queries: Tensor) -> None:
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError("queries must have shape [Q,3]")
        if not queries.is_floating_point() or not bool(torch.isfinite(queries).all()):
            raise ValueError("queries must be finite floating-point values")
        width, horizon, layer = queries.unbind(dim=1)
        valid = ((width > 0) & (width <= 1.2) & (horizon >= 0.3)
                 & (horizon <= 3.0) & ((layer == 0) | (layer == 1)))
        if not bool(valid.all()):
            raise ValueError("query width must be in (0,1.2], horizon in [0.3,3], layer 0 or 1")

    def query_overlap(self, queries: Tensor) -> Tensor:
        """Fraction of each latent cell within |X| <= width/2, .3 <= Z <= H.

        The folded |X| grid represents both sides of the corridor; there is no
        extra factor of two or conversion to a physical density.
        """
        self._validate_queries(queries)
        x = self.width_edges.to(queries)
        z = self.depth_edges.to(queries)
        x_overlap = (torch.minimum(queries[:, 0, None] / 2, x[1:]) - x[:-1]).clamp_min(0)
        z_overlap = (torch.minimum(queries[:, 1, None], z[1:])
                     - torch.maximum(z[:-1], queries.new_tensor(0.3))).clamp_min(0)
        fraction = (x_overlap / (x[1:] - x[:-1]))[:, :, None] * (
            z_overlap / (z[1:] - z[:-1]))[:, None, :]
        layer = F.one_hot(queries[:, 2].long(), num_classes=2).to(queries.dtype)
        return layer[:, :, None, None] * fraction[:, None, :, :]

    def geometry_intensity(self, features: Tensor) -> Tensor:
        """Return nonnegative latent intensities [B, BODY/HEAD, 20, 30]."""
        return self._geometry_intensity_from_scene(self.encode_scene(features))

    def _geometry_intensity_from_scene(self, scene: Tensor) -> Tensor:
        return F.softplus(self.geometry_head(scene)).reshape(-1, 2, 20, 30)

    def forward(self, features: Tensor, queries: Tensor) -> Tensor:
        self._validate_queries(queries)
        scene = self.encode_scene(features)
        queries = queries.to(device=scene.device, dtype=scene.dtype)
        if self.mode == "direct":
            condition = torch.cat((queries[:, :1] / 1.2, queries[:, 1:2] / 3.0,
                                   F.one_hot(queries[:, 2].long(), 2).to(scene.dtype)), dim=1)
            batch, count = scene.shape[0], queries.shape[0]
            pair = torch.cat((scene[:, None, :].expand(batch, count, 128),
                              condition[None, :, :].expand(batch, count, 4)), dim=2)
            return self.direct_head(pair).squeeze(-1)
        intensity = self._geometry_intensity_from_scene(scene)
        mass = intensity.flatten(1) @ self.query_overlap(queries).flatten(1).T
        # logit(1-exp(-mass)) = mass + log(-expm1(-mass)). The floor only
        # gives the exactly empty sweep a finite logit; no large-mass clipping.
        return mass + torch.log((-torch.expm1(-mass)).clamp_min(torch.finfo(mass.dtype).tiny))

    def parameter_counts(self) -> dict[str, int]:
        """Carrier size is matched; effective head sizes explicitly differ."""
        count = lambda module: sum(p.numel() for p in module.parameters())
        shared = count(self.trunk)
        direct, geometry = count(self.direct_head), count(self.geometry_head)
        active = direct if self.mode == "direct" else geometry
        return {"carrier": shared + direct + geometry, "shared": shared,
                "direct_head": direct, "geometry_head": geometry,
                "effective": shared + active, "unused": direct + geometry - active}
