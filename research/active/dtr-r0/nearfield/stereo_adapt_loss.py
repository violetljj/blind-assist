"""Fixed supervised targets and two paired disparity losses; no evaluator input."""
import math
import numpy as np

F=320/math.tan(math.radians(35.))

def targets(native):
    z=np.asarray(native,dtype=np.float32)
    assert z.shape==(360,640)
    yy,xx=np.indices(z.shape)
    finite=np.isfinite(z)&(z>0)
    disp=np.zeros(z.shape,np.float32);disp[finite]=F*.1/z[finite]
    valid=finite&(disp>0)&(disp<416)&(xx-disp>=0)
    # Observed near pixels on either side of a discontinuity, never missing centers.
    padded=np.pad(z,1,constant_values=np.nan);boundary=np.zeros(z.shape,bool)
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            other=padded[1+dy:361+dy,1+dx:641+dx]
            boundary|=~np.isfinite(other)|(np.abs(z-other)>.05)
    near=valid&(z<=4)
    y=(xx-320)*z/F;up=-(yy-180)*z/F
    inside=((np.abs(y)<=.28)&(up>=-1.05)&(up<=-.3))|((np.abs(y)<=.18)&(up>=-.3)&(up<=.15))
    regions=np.full(z.shape,-1,np.int8)
    regions[near&boundary]=0
    regions[near&~boundary&inside]=1
    regions[near&~boundary&~inside]=2
    regions[valid&(z>4)]=3
    assert np.array_equal(regions>=0,valid)
    return disp,valid,regions

def loss_weights(valid,regions,mode):
    assert mode in ('ordinary','balanced')
    n=int(valid.sum());assert n>0
    weights=valid.astype(np.float32)/n
    if mode=='balanced':
        present=[k for k in range(4) if np.any(regions==k)]
        balanced=np.zeros(valid.shape,np.float32)
        for k in present:
            mask=regions==k;balanced[mask]=1/(len(present)*int(mask.sum()))
        weights=.5*weights+.5*balanced
    assert np.isclose(weights.sum(),1,rtol=1e-5)
    return weights

def sequence_loss(predictions,target,weights):
    import torch
    assert predictions
    factors=[.9**(len(predictions)-1-i) for i in range(len(predictions))]
    scale=sum(factors)
    return sum(f/scale*((p.float()-target).abs()*weights).sum() for f,p in zip(factors,predictions))
