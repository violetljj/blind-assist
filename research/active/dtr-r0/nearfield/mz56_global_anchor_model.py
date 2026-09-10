"""Global measured-zone context without extending local ToF coverage."""
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import mz54_full_rgb_model as inherited_model
from mz54_full_rgb_model import RasterQuery, loss_for


INHERITED_MODEL_SHA256 = '54c4dbd5932eda4636eec495b7d235ec32069fcf5f4f49100df3bec06ba0b9ce'


class AnchorQuery(RasterQuery):
    """Two matched FULL raster arms; predictor inputs remain features/packets.

    In addition to the MZ54 output contract, inspect returns:
      observed_anchor[B,8]: masked mean of valid-zone encodings;
      anchor_vector[B,8]: actual head input, zero for LOCAL_ONLY/suppression;
      anchor_available[B]: at least one valid measured return.
    These are scene context, not an outside-object measurement or uncertainty.
    """

    def __init__(self, initial_mz54_state, arm):
        if arm not in ('LOCAL_ONLY', 'GLOBAL_ANCHOR'):
            raise ValueError(f'Unknown anchor arm: {arm}')
        actual = hashlib.sha256(Path(inherited_model.__file__).read_bytes()).hexdigest()
        if actual != INHERITED_MODEL_SHA256:
            raise ValueError('Inherited MZ54 source does not match the frozen binding')
        if initial_mz54_state['head.0.weight'].shape != (32, 41):
            raise ValueError('Expected the completed MZ54 41-input head state')
        bridge = dict(initial_mz54_state)
        bridge['head.0.weight'] = initial_mz54_state['head.0.weight'][:, :40]
        super().__init__(bridge, 'FULL_RASTER')
        # Reject a crop or otherwise changed geometry, rather than loading it
        # under a FULL name. No source tensor is modified by this constructor.
        for name, value in self.named_buffers():
            if not torch.equal(initial_mz54_state[name].cpu(), value.cpu()):
                raise ValueError(f'MZ54 FULL geometry mismatch: {name}')
        self.load_state_dict(initial_mz54_state, strict=True)
        old_head = self.head[0]
        expanded = nn.Linear(49, 32)
        with torch.no_grad():
            expanded.weight[:, :41].copy_(old_head.weight)
            expanded.weight[:, 41:].zero_()
            expanded.bias.copy_(old_head.bias)
        self.head[0] = expanded
        self.packet_encoder = nn.Sequential(nn.Linear(6, 8), nn.ReLU())
        self.arm = arm
        row, col = torch.meshgrid(torch.arange(8, dtype=torch.float32),
                                  torch.arange(8, dtype=torch.float32), indexing='ij')
        angles = torch.stack((-22.5 + (col + .5) * 5.625,
                             22.5 - (row + .5) * 5.625), -1).reshape(64, 2) / 22.5
        self.register_buffer('zone_angles', angles)

    def encode_anchor(self, ranges, valid):
        """Average only real zones with at least one valid, finite 0<r<=4 slot."""
        if ranges.ndim != 3 or ranges.shape[1:] != (64, 2) or valid.shape != ranges.shape:
            raise ValueError('Expected matching ranges/valid [B,64,2]')
        good = valid.bool() & torch.isfinite(ranges) & (ranges > 0) & (ranges <= 4)
        clean = torch.where(good, ranges, torch.zeros_like(ranges))
        inputs = torch.cat((clean / 4, good.to(ranges.dtype),
                            self.zone_angles[None].expand(len(ranges), -1, -1)), -1)
        encoded = self.packet_encoder(inputs)
        zone_valid = good.any(-1)
        encoded = encoded.masked_fill(~zone_valid[..., None], 0.)
        anchor = encoded.sum(1) / zone_valid.sum(1).clamp_min(1)[:, None]
        available = zone_valid.any(1)
        anchor = anchor.masked_fill(~available[:, None], 0.)
        return anchor, available

    def inspect(self, normalized_features, ranges, valid, suppress_global=False):
        if normalized_features.ndim != 4 or tuple(normalized_features.shape[1:]) != (64, 45, 80):
            raise ValueError('AnchorQuery expects FULL features [B,64,45,80]')
        batch = len(normalized_features)
        if len(ranges) != batch:
            raise ValueError('Feature and packet batch sizes differ')
        # Preserve the exact MZ54 token/context/geometry/packet concatenation.
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
        observed, anchor_available = self.encode_anchor(ranges, valid)
        anchor = observed if self.arm == 'GLOBAL_ANCHOR' and not suppress_global else torch.zeros_like(observed)
        expanded = anchor[:, None, None].expand(-1, 45, 80, -1)
        field = self.head(torch.cat((tokens, context, static, packet,
                                     coverage.to(tokens.dtype), expanded), -1))
        field = field.masked_fill(~self.candidate_mask[None, :, :, None], 0.)
        return dict(field=field, candidate_mask=self.candidate_mask,
                    OPEN=self.pool(field, self.candidate_mask),
                    CROP=self.pool(field, self.candidate_mask & self.crop_mask),
                    sensor_coverage=self.sensor_coverage, packet_available=available,
                    observed_anchor=observed, anchor_vector=anchor, anchor_available=anchor_available,
                    interpretation='PREDICTED_LOCAL_INTRUSION_WITH_INFIELD_CONTEXT_NOT_CLEAR')
