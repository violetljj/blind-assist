"""Same capped-count event as final near, restricted to each three-cell range."""
import torch


def range_from_counts(logits):
    if logits.ndim != 3 or logits.shape[1:] != (12,4):
        raise ValueError('Expected Bx12x4 counts')
    p=logits.softmax(-1).reshape(-1,2,2,3,4)
    d0=torch.ones_like(p[:,:,:,0,0]);d1=torch.zeros_like(d0);d2=torch.zeros_like(d0)
    for j in range(3):
        q0,q1,q2=p[:,:,:,j,0],p[:,:,:,j,1],p[:,:,:,j,2]
        d0,d1,d2=d0*q0,d1*q0+d0*q1,d2*q0+d1*q1+d0*q2
    below=(d0+d1+d2).clamp(1e-7,1-1e-7)
    return torch.log1p(-below)-below.log()
