"""Query-specific source supervision; unchanged MZ8 eligibility/max pooling."""
import torch
from torch import nn
from mz8_attribution import AngularReturnReadout


class SourceReadout(AngularReturnReadout):
    def __init__(self,no_rgb=False):
        super().__init__()
        self.evidence=nn.Sequential(nn.Linear(65,32),nn.ReLU(),nn.Linear(32,4))
        self.no_rgb=no_rgb

    def forward_tokens(self,tokens,ranges,valid,wrong_zone=False,return_details=False):
        if wrong_zone:
            tokens=tokens.reshape(tokens.shape[0],8,8,*tokens.shape[2:]).roll(4,dims=2).flatten(1,2)
        if self.no_rgb:tokens=torch.zeros_like(tokens)
        clean,good,eligible=self.eligibility(ranges,valid)
        tokens=tokens[:,:,None].expand(-1,-1,2,-1,-1)
        r=(clean/4)[...,None,None].expand(*tokens.shape[:-1],1)
        evidence=self.evidence(torch.cat([tokens,r.to(tokens.dtype)],-1))
        support=eligible.flatten(1,3).any(1)
        pooled=evidence.masked_fill(~eligible,-torch.inf).flatten(1,3).amax(1)
        logits=torch.where(support,pooled+self.query_bias,pooled.new_full(pooled.shape,-20))
        if return_details:return dict(logits=logits,support=support,candidate_logits=evidence,eligible=eligible,
            observation_valid=good.flatten(1).any(1))
        return logits
