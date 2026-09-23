"""Current-status first-contact CDF; no exact-distance targets or clear authority."""
import torch
from torch import nn
from torch.nn import functional as F
from contact_boundary_model import ContactBoundaryModel


class MetricContactModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = ContactBoundaryModel().trunk
        self.head = nn.Sequential(nn.Linear(131,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),nn.Linear(64,3))

    def _queries(self, features, queries):
        if queries.ndim == 2:
            ContactBoundaryModel._validate_queries(queries)
            queries = queries[None].expand(len(features),-1,-1)
        elif queries.ndim == 3 and len(queries)==len(features):
            ContactBoundaryModel._validate_queries(queries.reshape(-1,3))
        else:
            raise ValueError('Expected Qx3 or BxQx3 queries')
        return queries.to(features)

    def components(self, features, queries):
        queries = self._queries(features,queries)
        scene=self.trunk(features)[:,None].expand(-1,queries.shape[1],-1)
        condition=torch.cat((queries[...,:1]/1.2,F.one_hot(queries[...,2].long(),2).to(features)),dim=-1)
        a,b,c=self.head(torch.cat((scene,condition),dim=-1)).unbind(-1)
        return a.sigmoid(), .3+2.7*b.sigmoid(), .01+F.softplus(c)

    def forward(self, features, queries):
        queries=self._queries(features,queries)
        q,mu,s=self.components(features,queries)
        logp=q.clamp_min(torch.finfo(q.dtype).tiny).log()+F.logsigmoid((queries[...,1]-mu)/s)-F.logsigmoid((3-mu)/s)
        logp=logp.clamp_max(-torch.finfo(logp.dtype).eps)
        return logp-torch.log(-torch.expm1(logp))

    def continuous_crossing(self, features, queries, threshold):
        q,mu,s=self.components(features,queries)
        ratio=threshold*torch.sigmoid((3-mu)/s)/q.clamp_min(torch.finfo(q.dtype).tiny)
        z=mu+s*torch.logit(ratio.clamp(torch.finfo(q.dtype).eps,1-torch.finfo(q.dtype).eps))
        return torch.where(q>=threshold,z.clamp(.3,3.),torch.full_like(z,float('inf')))

    def parameter_counts(self):
        return dict(shared=sum(p.numel() for p in self.trunk.parameters()),head=sum(p.numel() for p in self.head.parameters()),effective=sum(p.numel() for p in self.parameters()))
