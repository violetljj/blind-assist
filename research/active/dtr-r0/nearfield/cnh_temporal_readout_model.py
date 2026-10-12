"""Pilot V/T models. Runtime inputs contain observations and query geometry only.

No labels, scene generator, target geometry, background or oracle templates enter
the token builder. Transforms map each past sensor frame to the current query
frame using the same public current-query transform and estimated relative poses
as V. Query boxes are the retained HEAD/BODY boxes.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from cnh_cvr_pilot import CVR
from cnh_cvr_projection import WIDTH, EDGE, query_masks


QUERY_BOXES = np.array([[-.30, -.20, .30, .30, .42, 3.],
                        [-.30, .42, .30, .30, .90, 3.]], np.float32)


def zone_bin_centers():
    """Radial bin centers on central zone unit rays, zone axes y,x."""
    slopes = -EDGE + (np.arange(8) + .5) * (2 * EDGE / 8)
    yy, xx = np.meshgrid(slopes, slopes, indexing='ij')
    rays = np.stack((xx, yy, np.ones_like(xx)), -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return (rays[:, :, None] * ((np.arange(16)+.5)*WIDTH)[None, None, :, None]).astype(np.float32)


def observation_tokens(histories, transforms, lengths, ambient, centers=None):
    """Construct [B,8,64,66] tokens, keeping 16 native bins per zone.

    Histories are native z, not already signed-log transformed. Left padding is
    excluded by lengths; valid frames may contain all-zero observations. Ambient
    is the fixed sensor ambient or an explicitly supplied observed [B,8,8,8].
    """
    if histories.ndim != 5 or histories.shape[1:] != (8, 8, 8, 16):
        raise ValueError('Expected native history [B,8,8,8,16]')
    b = len(histories)
    if transforms.shape != (b, 8, 4, 4) or lengths.shape != (b,):
        raise ValueError('Transform/valid-history length axes differ')
    if bool(((lengths < 1) | (lengths > 8)).any()):
        raise ValueError('History lengths must be in 1..8')
    z = histories.float()
    z = z.sign() * z.abs().log1p()
    points = torch.as_tensor(zone_bin_centers(), device=z.device) if centers is None else centers
    points = points.to(device=z.device, dtype=torch.float32)
    t = transforms.float()
    geom = torch.einsum('btij,yxrj->btyxri', t[:, :, :3, :3], points)
    geom = geom + t[:, :, None, None, None, :3, 3]
    # Fixed unit scaling; no sample/scene-dependent normalization.
    geom = geom / geom.new_tensor([.6, 1., 3.])
    a = torch.as_tensor(ambient, device=z.device, dtype=torch.float32)
    if a.ndim == 0:
        a = a.expand(b, 8, 8, 8)
    if a.shape != (b, 8, 8, 8) or bool((a < 0).any()):
        raise ValueError('Observed/fixed ambient must be nonnegative and match zones')
    age = (torch.arange(7, -1, -1, device=z.device, dtype=torch.float32)/7)
    age = age[None, :, None, None, None].expand(b, -1, 8, 8, -1)
    tokens = torch.cat((z, a.log1p()[..., None], geom.flatten(-2), age), -1)
    valid = torch.arange(8, device=z.device)[None] >= 8-lengths[:, None]
    return tokens.reshape(b, 8, 64, 66), valid


class TemporalReadout(nn.Module):
    """Frame-zone evidence, query-conditioned spatial pooling, causal GRU."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(66, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())
        self.query_embedding = nn.Embedding(2, 8)
        self.attention = nn.Sequential(nn.Linear(78, 32), nn.GELU(), nn.Linear(32, 1))
        self.frame = nn.Sequential(nn.Linear(206, 64), nn.GELU())
        self.temporal = nn.GRU(64, 64, batch_first=True)
        self.head = nn.Sequential(nn.Linear(78, 32), nn.GELU(), nn.Linear(32, 1))
        self.register_buffer('bin_centers', torch.from_numpy(zone_bin_centers()))
        boxes = torch.from_numpy(QUERY_BOXES.copy()) / torch.tensor([.6, 1., 3., .6, 1., 3.])
        self.register_buffer('query_boxes', boxes)

    def forward(self, histories, transforms, lengths, ambient):
        tokens, valid = observation_tokens(histories, transforms, lengths, ambient, self.bin_centers)
        h = self.encoder(tokens)
        b = len(h)
        q = torch.cat((self.query_boxes, self.query_embedding.weight), -1)
        hq = h[:, None].expand(-1, 2, -1, -1, -1)
        qq = q[None, :, None, None].expand(b, -1, 8, 64, -1)
        attention = self.attention(torch.cat((hq, qq), -1)).squeeze(-1).softmax(-1)
        weighted = (hq * attention[..., None]).sum(-2)
        pooled = torch.cat((weighted, hq.mean(-2), hq.amax(-2), q[None, :, None].expand(b, -1, 8, -1)), -1)
        per_frame = self.frame(pooled)
        # Eight cheap steps preserve left-padding masks without updating hidden
        # state on a padding frame. Avoid packed_sequence CPU length transfers.
        state = per_frame.new_zeros((1, b*2, 64))
        for f in range(8):
            _, candidate = self.temporal(per_frame[:, :, f].reshape(b*2, 1, 64), state)
            mask = valid[:, f].repeat_interleave(2)[None, :, None]
            state = torch.where(mask, candidate, state)
        last = state.squeeze(0).reshape(b, 2, 64)
        return self.head(torch.cat((last, q[None].expand(b, -1, -1)), -1)).squeeze(-1)


def prepare_voxels(voxels, masks=None):
    """Same native three-channel scaling and public query masks as M3."""
    if voxels.ndim != 5 or voxels.shape[1:] != (3, 24, 17, 33):
        raise ValueError('Expected V voxels [B,3,24,17,33]')
    x = voxels.float()
    evidence = x[:, 0].sign()*x[:, 0].abs().log1p()
    current = x[:, 2].sign()*x[:, 2].abs().log1p()
    x = torch.stack((evidence, x[:, 1]/8, current), 1)
    if masks is None:
        masks = torch.as_tensor(query_masks(), device=x.device)
    return torch.cat((x, masks[None].expand(len(x), -1, -1, -1, -1)), 1)


def parameter_counts():
    return {arm: sum(p.numel() for p in model.parameters())
            for arm, model in [('V', CVR()), ('T', TemporalReadout())]}
