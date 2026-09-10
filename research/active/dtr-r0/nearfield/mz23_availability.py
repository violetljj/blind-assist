"""Predict angular availability of native-equivalent <=4m depth evidence.

Availability is not occupancy, FREE_RAY, a device-validity guarantee or CLEAR.
Its supervision is separate from unchanged unknown query/contributor labels.
"""
import torch
from torch import nn
from torch.nn import functional as F


class AngularAvailability(nn.Module):
    def __init__(self, grid):
        super().__init__()
        self.register_buffer('grid',grid.detach().clone())
        offsets=(torch.arange(7).float()+.5)/7*2-1
        yy,xx=torch.meshgrid(offsets,offsets,indexing='ij')
        self.register_buffer('relative',torch.stack([xx.flatten(),yy.flatten()],-1))
        self.local=nn.Sequential(nn.Conv2d(64,16,3,padding=1),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(40,32),nn.ReLU(),nn.Linear(32,1))

    def forward(self,features,ranges,valid,wrong_zone=False):
        good=valid.bool()&torch.isfinite(ranges)&(ranges>0)&(ranges<=4)
        clean=torch.where(good,ranges,torch.zeros_like(ranges))
        x=self.local(features)
        grid=self.grid[None].expand(len(x),-1,-1,-1)
        tokens=F.grid_sample(x,grid,align_corners=False).permute(0,2,3,1)
        if wrong_zone:tokens=tokens.reshape(len(x),8,8,49,16).roll(4,dims=2).flatten(1,2)
        zone=tokens.mean(2,keepdim=True).expand_as(tokens)
        relative=self.relative[None,None].expand(len(x),64,-1,-1)
        packet=torch.cat([clean/4,good.float()],-1)[:,:,None].expand(-1,-1,49,-1)
        return self.head(torch.cat([tokens,zone,grid,relative,packet],-1)).squeeze(-1)


def restrict_candidates(out, availability):
    """Mask candidate eligibility at fixed availability logit0; never assert clear."""
    eligible=out['eligible']&(availability[:,:,None,:,None]>=0)
    support=eligible.flatten(1,3).any(1)
    pooled=out['candidate_logits'].masked_fill(~eligible,-1e6).flatten(1,3).amax(1)
    logits=torch.where(support,pooled,pooled.new_full(pooled.shape,-20.))
    return dict(logits=logits,support=support,eligible=eligible,
        candidate_logits=out['candidate_logits'],availability=availability,
        interpretation='PREDICTED_ANGULAR_EVIDENCE_AVAILABILITY_NOT_CLEAR')
