"""Censored 5cm interval supervision for the unchanged first-contact CDF."""
import torch
from torch.nn import functional as F


def interval_nll(q, mu, scale, targets):
    """Finite z supervises [z-.025,z+.025]; inf means no crossing by3m.

    Left-censored z=.3 includes all modeled mass before the upper bin endpoint.
    Stable logistic differences avoid subtracting two nearly equal CDF values.
    """
    finite=torch.isfinite(targets)
    z=torch.where(finite,targets,torch.full_like(targets,3.))
    lo=(z-.025).clamp_min(.3);hi=(z+.025).clamp_max(3.)
    eps=torch.finfo(q.dtype).eps
    logq=q.clamp(eps,1-eps).log()
    normalizer=F.logsigmoid((3-mu)/scale)
    a=(lo-mu)/scale;b=(hi-mu)/scale
    log_interval=F.logsigmoid(b)+F.logsigmoid(-a)+torch.log(-torch.expm1(-(hi-lo)/scale))
    log_mass=logq+torch.where(z<=.300001,F.logsigmoid(b),log_interval)-normalizer
    log_survival=torch.log1p(-q.clamp(eps,1-eps))
    return -torch.where(finite,log_mass,log_survival).mean()


def supervised_loss(model, features, queries, binary, targets, positive_weight):
    logits=model(features,queries)
    binary_loss=F.binary_cross_entropy_with_logits(logits,binary,pos_weight=positive_weight)
    q,mu,scale=model.components(features,queries)
    metric_loss=interval_nll(q,mu,scale,targets)
    return binary_loss+metric_loss,binary_loss,metric_loss
