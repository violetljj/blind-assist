"""T2: explicit finite-volume query support plus ungated observed context.

Inputs are observations, estimated sensor-to-query transforms, valid history
lengths and precomputed geometric query weights. No scene/background/teacher
artifact is opened by this module.
"""
import numpy as np
import torch
from torch import nn

from cnh_cvr_projection import WIDTH, EDGE
from cnh_temporal_readout_model import QUERY_BOXES


MODEL_CONFIG = dict(name='T2', bin_features=10, shared_bin_widths=[32,32],
    query_frame_features=143, frame_width=64, gru_width=64, query_embedding=8,
    head_width=32, parameters=38161,
    pooling='signed-log weighted sum32, normalized weighted mean32, log support1, ungated context mean/max64, query14',
    geometry='SUB3 original query-mask integral; full-cell volume centroid xyz and signed lateral clearance',
    padding='left slots excluded from GRU state; query weights0 in padding',
    precision='float32 model; geometry cached float32 from float64 integration')


def volume_bin_centers(samples=32):
    """Uniform volume centroid of each angular sector x radial shell.

    Radial moment is analytic. Angular solid-angle moment uses fixed midpoint
    quadrature, separately from query-overlap's voxel SUB3 integration.
    """
    n=int(samples)
    s=-EDGE+(np.arange(8*n)+.5)*(2*EDGE/(8*n))
    yy,xx=np.meshgrid(s,s,indexing='ij')
    rays=np.stack((xx,yy,np.ones_like(xx)),-1)
    norm=np.linalg.norm(rays,axis=-1)
    solid=(2*EDGE/(8*n))**2/norm**3
    rays=rays/norm[...,None]
    rays=rays.reshape(8,n,8,n,3).transpose(0,2,1,3,4)
    solid=solid.reshape(8,n,8,n).transpose(0,2,1,3)
    direction=(rays*solid[...,None]).sum((2,3))/solid.sum((2,3))[...,None]
    a=np.arange(16)*WIDTH;b=a+WIDTH
    radial=.75*(b**4-a**4)/(b**3-a**3)
    return (direction[:,:,None]*radial[None,None,:,None]).astype(np.float32)


class T2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(10,32),nn.GELU(),nn.Linear(32,32),nn.GELU())
        self.embedding=nn.Embedding(2,8)
        self.frame=nn.Sequential(nn.Linear(143,64),nn.GELU())
        self.temporal=nn.GRU(64,64,batch_first=True)
        self.head=nn.Sequential(nn.Linear(78,32),nn.GELU(),nn.Linear(32,1))
        self.register_buffer('centers',torch.from_numpy(volume_bin_centers()))
        boxes=torch.from_numpy(QUERY_BOXES.copy())/torch.tensor([.6,1.,3.,.6,1.,3.])
        self.register_buffer('query_boxes',boxes)

    def forward(self,histories,transforms,length,ambient,query_weights):
        if histories.ndim!=5 or histories.shape[1:]!=(8,8,8,16):
            raise ValueError('histories[B,8,8,8,16] required')
        b=len(histories)
        if transforms.shape!=(b,8,4,4) or length.shape!=(b,) or ambient.shape!=(b,8,8,8):
            raise ValueError('Observation/pose/history axes mismatch')
        if query_weights.shape!=(b,8,2,8,8,16):
            raise ValueError('SUB3 query_weights[B,8,2,8,8,16] required')
        if bool(((length<1)|(length>8)).any()): raise ValueError('Invalid history length')
        z=histories.float();z=z.sign()*z.abs().log1p()
        t=transforms.float()
        geom=torch.einsum('btij,yxrj->btyxri',t[:,:,:3,:3],self.centers)
        geom=geom+t[:,:,None,None,None,:3,3]
        clearance=(.3-geom[...,0]) # signed right boundary
        # min left/right signed distance to lateral corridor boundaries.
        clearance=torch.minimum(clearance,.3+geom[...,0])/.3
        xyz=geom/geom.new_tensor([.6,1.,3.])
        ages=torch.arange(7,-1,-1,device=z.device,dtype=torch.float32)/7
        ages=ages[None,:,None,None,None,None].expand(b,-1,8,8,16,-1)
        amb=ambient.float().log1p()[...,None,None].expand(-1,-1,-1,-1,16,-1)
        w=query_weights.float().permute(0,1,3,4,5,2)
        valid=torch.arange(8,device=z.device)[None]>=8-length[:,None]
        w=w*valid[:,:,None,None,None,None]
        feature=torch.cat((z[...,None],amb,xyz,clearance[...,None],ages,w,
                           torch.full_like(z[...,None],WIDTH)), -1)
        h=self.encoder(feature).reshape(b,8,1024,32)
        w=w.reshape(b,8,1024,2).permute(0,3,1,2)
        weighted=torch.einsum('bqtr,btrd->bqtd',w,h)
        support=w.sum(-1,keepdim=True)
        mean=weighted/support.clamp_min(1e-8)
        logged=weighted.sign()*weighted.abs().log1p()
        context=torch.cat((h.mean(-2),h.amax(-2)),-1)
        q=torch.cat((self.query_boxes,self.embedding.weight),-1)
        f=torch.cat((logged,mean,support.log1p(),context[:,None].expand(-1,2,-1,-1),
                     q[None,:,None].expand(b,-1,8,-1)),-1)
        frames=self.frame(f)
        state=frames.new_zeros((1,b*2,64))
        for i in range(8):
            _,candidate=self.temporal(frames[:,:,i].reshape(b*2,1,64),state)
            state=torch.where(valid[:,i].repeat_interleave(2)[None,:,None],candidate,state)
        last=state.squeeze(0).reshape(b,2,64)
        return self.head(torch.cat((last,q[None].expand(b,-1,-1)),-1)).squeeze(-1)


def forward(net,batch):
    return net(*(batch[k] for k in ('histories','transforms','length','ambient','query_weights')))


def parameter_count():return sum(p.numel() for p in T2().parameters())
