"""Matched native-raster RGB readout; ToF remains the observed 45-degree field.

Static masks are [45,80], rays are unit [45,80,3] vectors in camera forward,
right, up coordinates. Labels and native depth are never prediction inputs.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from mz15_train import balanced_local


class RasterQuery(nn.Module):
    """CROP_RASTER accepts [B,64,28,28]; FULL_RASTER [B,64,45,80].

    ``inspect`` returns masked field[B,45,80,4], static candidate_mask,
    OPEN={raw[B,4], support[B,4]}, and CROP with the same logits restricted
    to crop_mask. Support denotes an RGB candidate, never measured clearance.
    """

    def __init__(self, initial_state, arm):
        super().__init__()
        if arm not in ('CROP_RASTER', 'FULL_RASTER'):
            raise ValueError(f'Unknown raster arm: {arm}')
        self.arm = arm
        self.local = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(41, 32), nn.ReLU(), nn.Linear(32, 4))
        self.local.load_state_dict({k: initial_state['local.' + k] for k in self.local.state_dict()})
        if initial_state['head.0.weight'].shape != (32, 40):
            raise ValueError('Expected the MZ50 40-input spatial head')
        with torch.no_grad():
            self.head[0].weight[:, :40].copy_(initial_state['head.0.weight'])
            self.head[0].weight[:, 40].zero_()
            self.head[0].bias.copy_(initial_state['head.0.bias'])
            self.head[2].load_state_dict({k: initial_state['head.2.' + k] for k in ('weight', 'bias')})

        row, col = torch.meshgrid(torch.arange(45, dtype=torch.float64),
                                  torch.arange(80, dtype=torch.float64), indexing='ij')
        u, v = 8 * col + 3.5, 8 * row + 3.5
        focal = 320 / math.tan(math.radians(50))
        right, up = (u - 319.5) / focal, (179.5 - v) / focal
        rays = torch.stack((torch.ones_like(right), right, up), -1)
        rays = rays / rays.norm(dim=-1, keepdim=True)
        az, el = torch.rad2deg(torch.atan(right)), torch.rad2deg(torch.atan(up))
        angular_col, angular_row = (az + 22.5) / 5.625, (22.5 - el) / 5.625
        bin_col, bin_row = angular_col.floor().long(), angular_row.floor().long()
        coverage = (az.abs() <= 22.5) & (el.abs() <= 22.5)
        zone = torch.where(coverage, bin_row * 8 + bin_col, -1)
        assert ((zone[coverage] >= 0) & (zone[coverage] < 64)).all()
        crop = (col >= 26) & (col <= 53) & (row >= 9) & (row <= 35)
        candidate = crop if arm == 'CROP_RASTER' else torch.ones_like(crop)
        # These unbounded bin IDs group RGB context; only zone0..63 has ToF.
        groups, context_ids = torch.unique(torch.stack((bin_row, bin_col), -1).reshape(-1, 2),
                                           dim=0, return_inverse=True)
        counts = torch.zeros(len(groups), dtype=torch.float64)
        counts.scatter_add_(0, context_ids, candidate.flatten().double())
        if arm == 'CROP_RASTER':
            grid = torch.stack((2 * (u + .5 - 208) / 224 - 1,
                                2 * (v + .5 - 68) / 224 - 1), -1)
        else:
            grid = torch.stack((2 * (u + .5) / 640 - 1,
                                2 * (v + .5) / 360 - 1), -1)
        absolute = torch.stack(((u - 319.5) / 112, (v - 179.5) / 112), -1)
        relative = torch.stack((2 * (angular_col - bin_col) - 1,
                                2 * (angular_row - bin_row) - 1), -1)
        for name, value in dict(grid=grid, absolute=absolute, relative=relative, rays=rays,
                                context_counts=counts).items():
            self.register_buffer(name, value.float())
        for name, value in dict(candidate_mask=candidate.clone(), crop_mask=crop,
                                sensor_coverage=coverage, sensor_zone=zone,
                                context_ids=context_ids).items():
            self.register_buffer(name, value)

    def packet_features(self, ranges, valid):
        """Observed range/4, range/4, valid, valid per raster center.

        Outside-ToF cells address a separate zero sentinel, not an edge zone.
        A zero range with validity false is an inert missing-value encoding.
        """
        if ranges.ndim != 3 or ranges.shape[1:] != (64, 2) or valid.shape != ranges.shape:
            raise ValueError('Expected matching ranges/valid [B,64,2]')
        good = valid.bool() & torch.isfinite(ranges) & (ranges > 0) & (ranges <= 4)
        clean = torch.where(good, ranges, torch.zeros_like(ranges))
        observed = torch.cat((clean / 4, good.to(ranges.dtype)), -1)
        observed = torch.cat((observed, observed.new_zeros((len(ranges), 1, 4))), 1)
        index = torch.where(self.sensor_coverage, self.sensor_zone, 64)
        return observed[:, index], good.flatten(1).any(1)

    @staticmethod
    def pool(field, mask):
        eligible = mask[None, :, :, None].expand_as(field)
        support = eligible.flatten(1, 2).any(1)
        raw = field.masked_fill(~eligible, -torch.inf).flatten(1, 2).amax(1)
        return dict(raw=torch.where(support, raw, torch.full_like(raw, -20.)), support=support)

    def inspect(self, normalized_features, ranges, valid):
        expected = (64, 28, 28) if self.arm == 'CROP_RASTER' else (64, 45, 80)
        if normalized_features.ndim != 4 or tuple(normalized_features.shape[1:]) != expected:
            raise ValueError(f'{self.arm} expects features [B,{expected}]')
        batch = len(normalized_features)
        if len(ranges) != batch:
            raise ValueError('Feature and packet batch sizes differ')
        local = self.local(normalized_features)
        grid = self.grid[None].expand(batch, -1, -1, -1)
        tokens = F.grid_sample(local, grid, align_corners=False).permute(0, 2, 3, 1)
        tokens = tokens.masked_fill(~self.candidate_mask[None, :, :, None], 0.)
        sums = tokens.new_zeros((batch, len(self.context_counts), 16))
        ids = self.context_ids[None, :, None].expand(batch, -1, 16)
        sums.scatter_add_(1, ids, tokens.reshape(batch, -1, 16))
        means = sums / self.context_counts.clamp_min(1)[None, :, None]
        context = means[:, self.context_ids].reshape(batch, 45, 80, 16)
        packet, available = self.packet_features(ranges, valid)
        static = torch.cat((self.absolute, self.relative), -1)[None].expand(batch, -1, -1, -1)
        coverage = self.sensor_coverage[None, :, :, None].expand(batch, -1, -1, -1)
        field = self.head(torch.cat((tokens, context, static, packet,
                                     coverage.to(tokens.dtype)), -1))
        field = field.masked_fill(~self.candidate_mask[None, :, :, None], 0.)
        return dict(field=field, candidate_mask=self.candidate_mask,
                    OPEN=self.pool(field, self.candidate_mask),
                    CROP=self.pool(field, self.candidate_mask & self.crop_mask),
                    sensor_coverage=self.sensor_coverage, packet_available=available,
                    interpretation='PREDICTED_LOCAL_INTRUSION_NOT_CLEAR')


def loss_for(out, cell_truth, cell_known, frame_truth, frame_known):
    """MZ50 local/query objective, restricted to known arm-visible cells."""
    known = cell_known.bool() & out['candidate_mask'][None]
    truth = cell_truth.bool()
    local = balanced_local(out['field'], truth, known[..., None])
    witness = (truth & known[..., None]).flatten(1, 2).any(1)
    mask = frame_known.bool() & known.flatten(1).any(1)[:, None] & (~frame_truth.bool() | witness)
    values = F.binary_cross_entropy_with_logits(out['OPEN']['raw'], frame_truth.float(), reduction='none')
    query = (values * mask).sum() / mask.sum().clamp_min(1)
    return local + .25 * query, dict(local=local, query=query,
        skipped_positive=(frame_known.bool() & frame_truth.bool() & ~witness).sum())
