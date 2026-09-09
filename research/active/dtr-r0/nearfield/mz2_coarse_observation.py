"""MZ2 clean4x4 observation from original native pixels, not8x8 return means."""
import torch
from multizone64_observation import geometry


def observe4(depth):
    if not depth.is_cuda or depth.ndim!=3 or tuple(depth.shape[1:])!=(360,640):
        raise ValueError('CUDA Bx360x640 native depth required')
    g=geometry(depth.device,8);ids=g['zone_ids']
    merged=((ids//8)//2)*4+(ids%8)//2
    merged=merged.masked_fill(ids<0,-1)
    radial=depth.double()*g['radial_factor']
    valid=torch.isfinite(depth)&(depth>0)&(depth<100)&(radial<=4)
    edges=torch.linspace(0,4,41,device=depth.device,dtype=torch.float64)
    ranges=[];flags=[]
    for zone in range(16):
        mask=merged==zone;r=radial[:,mask];known=valid[:,mask]
        bins=torch.bucketize(r.contiguous(),edges[1:-1],right=True)
        counts=[];means=[]
        for b in range(40):
            member=known&(bins==b);count=member.sum(1)
            counts.append(count);means.append(r.masked_fill(~member,0).sum(1)/count.clamp(min=1))
        support=torch.stack(counts,1)>=3;means=torch.stack(means,1)
        first=support.int().argmax(1);last=39-support.flip(1).int().argmax(1)
        anybin=support.any(1);good=torch.stack((anybin,anybin&(last!=first)),1)
        values=means.gather(1,torch.stack((first,last),1)).masked_fill(~good,torch.nan)
        ranges.append(values);flags.append(good)
    return dict(range_m=torch.stack(ranges,1),valid=torch.stack(flags,1))


def encode(packet,size):
    if size not in (1,4):raise ValueError('Only fixed coarse resolutions')
    n=len(packet['range_m']);repeat=8//size
    ranges=torch.nan_to_num(packet['range_m'],nan=0).float().reshape(n,size,size,2)/4
    valid=packet['valid'].float().reshape(n,size,size,2)
    ranges=ranges.repeat_interleave(repeat,1).repeat_interleave(repeat,2)
    valid=valid.repeat_interleave(repeat,1).repeat_interleave(repeat,2)
    return torch.cat((ranges.flatten(1),valid.flatten(1)),1)
