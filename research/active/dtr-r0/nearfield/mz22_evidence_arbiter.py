"""Observable global/local arbitration; no evaluator labels enter features."""
import torch
from torch import nn


def observable_features(out, baseline, ranges, valid, rays):
    e = out['eligible']; f = out['candidate_logits']
    count = e.sum((1,2,3)); den = count.clamp_min(1)
    selected = f.masked_fill(~e, 0.)
    mean = selected.sum((1,2,3)) / den
    var = ((f - mean[:,None,None,None,:]).square().masked_fill(~e,0.)).sum((1,2,3)) / den
    flat = f.masked_fill(~e,-1e6).flatten(1,3)
    top = flat.topk(3,dim=1).values
    topmask = torch.arange(3,device=f.device)[None,:,None] < count[:,None,:]
    top3 = top.masked_fill(~topmask,0.).sum(1) / count.clamp(1,3)
    winner = flat.argmax(1)
    positions = ranges[...,None,None] * rays[None,:,None,:,:]
    positions = positions + positions.new_tensor([0.,0.,1.7])
    points = positions.flatten(1,3).gather(1,winner[...,None].expand(-1,-1,3))
    points = torch.where(out['support'][...,None],points,torch.zeros_like(points))
    raw = torch.where(out['support'],out['logits'],torch.zeros_like(out['logits']))
    global_scores = torch.cat([baseline,raw],-1)[:,None,:].expand(-1,4,-1)
    scalars = torch.stack([top3,mean,var.sqrt(),count.float().log1p(),
        e.any(3).any(2).sum(1).float()/8,e.any(3).sum((1,2)).float()/8,
        valid.float().mean((1,2))[:,None].expand(-1,4)],-1)
    return torch.cat([global_scores,scalars,points],-1)


class EvidenceArbiter(nn.Module):
    def __init__(self, feature_count=18):
        super().__init__()
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(feature_count,32),nn.ReLU(),nn.Linear(32,1)) for _ in range(4)])
        for h in self.heads:
            nn.init.zeros_(h[-1].weight);nn.init.zeros_(h[-1].bias)

    def forward(self, features, baseline, support):
        residual=torch.cat([h(features[:,q]) for q,h in enumerate(self.heads)],1)
        return torch.where(support,baseline+residual,baseline)
