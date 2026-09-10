"""Observable local return support, followed by fixed body-query geometry."""
import torch
from torch import nn
from mz8_attribution import AngularReturnReadout


class LocalSupportReadout(AngularReturnReadout):
    def __init__(self, shared=True):
        super().__init__()
        del self.evidence
        del self.query_bias
        self.shared = shared
        self.local = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(42, 32), nn.ReLU(), nn.Linear(32, 1 if shared else 4))
        offsets = (torch.arange(7).float()+.5)/7*2-1
        yy, xx = torch.meshgrid(offsets, offsets, indexing='ij')
        self.register_buffer('relative', torch.stack([xx.flatten(), yy.flatten()], -1))

    def inspect(self, features, ranges, valid, wrong_zone=False):
        clean, good, eligible = self.eligibility(ranges, valid)
        local = self.local(features)
        grid = self.grid[None].expand(features.shape[0], -1, -1, -1)
        tokens = torch.nn.functional.grid_sample(local, grid, align_corners=False).permute(0, 2, 3, 1)
        if wrong_zone:
            tokens = tokens.reshape(len(tokens), 8, 8, 49, 16).roll(4, dims=2).flatten(1, 2)
        zone = tokens.mean(2, keepdim=True).expand_as(tokens)
        visual = torch.cat([tokens, zone], -1)[:, :, None].expand(-1, -1, 2, -1, -1)
        batch = len(features)
        absolute = self.grid[None, :, None].expand(batch, -1, 2, -1, -1)
        relative = self.relative[None, None, None].expand(batch, 64, 2, -1, -1)
        packet = torch.cat([clean/4, good.float()], -1)[:, :, None, None].expand(-1, -1, 2, 49, -1)
        current = (clean/4)[..., None, None].expand(-1, -1, -1, 49, -1)
        echo = torch.arange(2, device=features.device, dtype=features.dtype)[None, None, :, None, None].expand(batch, 64, -1, 49, -1)
        field = self.head(torch.cat([visual, absolute, relative, packet, current, echo], -1))
        candidates = field.expand(-1, -1, -1, -1, 4) if self.shared else field
        support = eligible.flatten(1, 3).any(1)
        pooled = candidates.masked_fill(~eligible, -torch.inf).flatten(1, 3).amax(1)
        logits = torch.where(support, pooled, pooled.new_full(pooled.shape, -20.))
        return dict(logits=logits, field=field, candidate_logits=candidates, eligible=eligible,
            support=support, observation_valid=good.flatten(1).any(1),
            interpretation='PREDICTED_RETURN_SUPPORT_WITH_GEOMETRY_NOT_CLEAR')


def matched_pair():
    """Equal initial local features and per-query predictions before learning."""
    shared = LocalSupportReadout(shared=True)
    query = LocalSupportReadout(shared=False)
    query.local.load_state_dict(shared.local.state_dict())
    query.head[0].load_state_dict(shared.head[0].state_dict())
    with torch.no_grad():
        query.head[2].weight.copy_(shared.head[2].weight.expand(4, -1))
        query.head[2].bias.copy_(shared.head[2].bias.expand(4))
    return shared, query
