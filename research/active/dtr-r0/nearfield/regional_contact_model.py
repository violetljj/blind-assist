"""Regional RGB/ToF contact head; no flattened visual or evaluator input path."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from contact_boundary_model import ContactBoundaryModel


class RegionContactModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_encoder=nn.Sequential(nn.Linear(46,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU())
        self.key=nn.Linear(64,64,bias=False)
        self.query_encoder=nn.Sequential(nn.Linear(3,64),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(131,64),nn.SiLU(),nn.Linear(64,3))

    def _inputs(self,features,queries):
        if (not isinstance(features,torch.Tensor) or not features.is_floating_point() or
                features.ndim!=3 or tuple(features.shape[1:])!=(64,46) or features.shape[0]==0 or
                not bool(torch.isfinite(features).all())):
            raise ValueError('Expected finite floating regional features [B,64,46]')
        if queries.ndim==2:
            ContactBoundaryModel._validate_queries(queries)
            queries=queries[None].expand(len(features),-1,-1)
        elif queries.ndim==3 and len(queries)==len(features):
            ContactBoundaryModel._validate_queries(queries.reshape(-1,3))
        else:raise ValueError('Expected Qx3 or BxQx3 queries')
        return queries.to(features)

    def components(self,features,queries):
        queries=self._inputs(features,queries)
        tokens=self.token_encoder(features)
        condition=torch.cat((queries[...,:1]/1.2,F.one_hot(queries[...,2].long(),2).to(features)),dim=-1)
        query=self.query_encoder(condition)
        weights=torch.einsum('bqc,bzc->bqz',query,self.key(tokens))/math.sqrt(64.)
        pooled=torch.einsum('bqz,bzc->bqc',weights.softmax(-1),tokens)
        mean=tokens.mean(1)[:,None].expand(-1,queries.shape[1],-1)
        a,b,c=self.head(torch.cat((pooled,mean,condition),dim=-1)).unbind(-1)
        return a.sigmoid(),.3+2.7*b.sigmoid(),.01+F.softplus(c)

    def forward(self,features,queries):
        queries=self._inputs(features,queries)
        q,mu,s=self.components(features,queries)
        logp=q.clamp_min(torch.finfo(q.dtype).tiny).log()+F.logsigmoid((queries[...,1]-mu)/s)-F.logsigmoid((3-mu)/s)
        logp=logp.clamp_max(-torch.finfo(logp.dtype).eps)
        return logp-torch.log(-torch.expm1(logp))

    def continuous_crossing(self,features,queries,threshold):
        q,mu,s=self.components(features,queries)
        ratio=threshold*torch.sigmoid((3-mu)/s)/q.clamp_min(torch.finfo(q.dtype).tiny)
        z=mu+s*torch.logit(ratio.clamp(torch.finfo(q.dtype).eps,1-torch.finfo(q.dtype).eps))
        return torch.where(q>=threshold,z.clamp(.3,3.),torch.full_like(z,float('inf')))

    def parameter_counts(self):
        counts={name:sum(p.numel() for p in module.parameters()) for name,module in
                [('token_encoder',self.token_encoder),('key',self.key),('query_encoder',self.query_encoder),('head',self.head)]}
        counts['effective']=sum(counts.values())
        return counts
