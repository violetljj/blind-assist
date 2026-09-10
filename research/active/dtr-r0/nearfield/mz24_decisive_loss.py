"""Train availability extremes without relabeling unknown occupancy/query truth."""
import torch
from torch.nn import functional as F
from mz15_train import balanced_local


def decisive_terms(availability, known, eligible, query_contributor, teacher):
    """Negative: worst unavailable geometric candidate; positive: real witness.

    Native labels enter training only. The teacher selects a positive location
    within actual query contributors, independently of the availability head.
    No calibrated task cutoff or DEV example participates in selection.
    """
    values=availability[:,:,None,:,None].expand_as(teacher)
    negative=eligible & ~known[:,:,None,:,None]
    nvalid=negative.flatten(1,3).any(1)
    nmax=values.masked_fill(~negative,-1e6).flatten(1,3).amax(1)
    nloss=F.softplus(nmax[nvalid]).mean() if nvalid.any() else availability.sum()*0
    positive=eligible & query_contributor
    pvalid=positive.flatten(1,3).any(1)
    selected=teacher.detach().masked_fill(~positive,-torch.inf).flatten(1,3).argmax(1)
    witness=values.flatten(1,3).gather(1,selected[:,None,:]).squeeze(1)
    ploss=F.softplus(-witness[pvalid]).mean() if pvalid.any() else availability.sum()*0
    return nloss,ploss,dict(negative_bags=int(nvalid.sum()),positive_bags=int(pvalid.sum()))


def availability_objective(availability,known,eligible,query_contributor,teacher):
    dense=balanced_local(availability[...,None],known[...,None],torch.ones_like(known[...,None]))
    negative,positive,counts=decisive_terms(availability,known,eligible,query_contributor,teacher)
    total=dense+.25*negative+.25*positive
    return total,dict(dense=float(dense.detach()),negative=float(negative.detach()),positive=float(positive.detach()),**counts)
