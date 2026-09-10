"""Spatial query logits; identical weights with optional echo eligibility."""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from mz8_attribution import AngularReturnReadout
from mz15_train import balanced_local


class SpatialQuery(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.local = copy.deepcopy(rank.local).requires_grad_(True)
        self.head = nn.Sequential(nn.Linear(40, 32), nn.ReLU(), nn.Linear(32, 4))
        with torch.no_grad():
            self.head[0].weight.copy_(rank.head[0].weight[:, :40].cpu())
            self.head[0].bias.copy_(rank.head[0].bias.cpu())
            self.head[2].load_state_dict({k: v.cpu() for k, v in rank.head[2].state_dict().items()})
        for name in ('grid', 'rays', 'relative'):
            self.register_buffer(name, getattr(rank, name).detach().clone())

    eligibility = AngularReturnReadout.eligibility

    @staticmethod
    def pool(field, eligible):
        support = eligible.flatten(1, 2).any(1)
        pooled = field.masked_fill(~eligible, -torch.inf).flatten(1, 2).amax(1)
        return dict(raw=torch.where(support, pooled, torch.full_like(pooled, -20.)), support=support)

    def inspect(self, features, ranges, valid):
        # Only normalized RGB features, observed ranges and validity enter here.
        clean, good, gated = self.eligibility(ranges, valid)
        local = self.local(features)
        grid = self.grid[None].expand(len(features), -1, -1, -1)
        tokens = F.grid_sample(local, grid, align_corners=False).permute(0, 2, 3, 1)
        zone = tokens.mean(2, keepdim=True).expand_as(tokens)
        absolute = grid
        relative = self.relative[None, None].expand(len(features), 64, -1, -1)
        packet = torch.cat([clean / 4, good.float()], -1)[:, :, None].expand(-1, -1, 49, -1)
        field = self.head(torch.cat([tokens, zone, absolute, relative, packet], -1))
        gated = gated.any(2)
        opened = torch.ones_like(gated)
        return dict(field=field, eligible=gated, OPEN=self.pool(field, opened),
                    GATED=self.pool(field, gated),
                    packet_available=good.flatten(1).any(1),
                    interpretation='PREDICTED_LOCAL_INTRUSION_NOT_CLEAR')


def loss_for(out, cell_truth, cell_known, frame_truth, frame_known):
    local = balanced_local(out['field'], cell_truth, cell_known[..., None])
    witness = cell_truth.flatten(1, 2).any(1)
    # Unknown crop cells are not negatives; a full-image positive can be outside
    # this crop and therefore cannot supervise a local max as a witnessed event.
    mask = frame_known & cell_known.flatten(1).any(1)[:, None] & (~frame_truth | witness)
    values = F.binary_cross_entropy_with_logits(out['OPEN']['raw'], frame_truth.float(), reduction='none')
    query = (values * mask).sum() / mask.sum().clamp_min(1)
    return local + .25 * query, dict(local=local, query=query,
                                   skipped_positive=(frame_known & frame_truth & ~witness).sum())
