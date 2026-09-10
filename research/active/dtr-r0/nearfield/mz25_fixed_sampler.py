"""Fixed-grid bilinear sampling with a dense deterministic CUDA adjoint.

The constant sampling matrix replaces index-scatter backward. This is for the
fixed28x28 feature/64x49 angular layout, not a general grid_sample replacement.
"""
import torch
from torch import nn
from mz23_availability import AngularAvailability


class FixedBilinearSampler(nn.Module):
    def __init__(self,grid,height=28,width=28):
        super().__init__();self.height=height;self.width=width;self.output_shape=grid.shape[:-1]
        g=grid.detach().float().cpu().reshape(-1,2)
        # Same align_corners=False coordinate map and zero padding.
        x=((g[:,0]+1)*width-1)/2;y=((g[:,1]+1)*height-1)/2
        x0=x.floor().long();y0=y.floor().long();dx=x-x0;dy=y-y0
        matrix=torch.zeros(len(g),height*width,dtype=torch.float32)
        for ox,oy,weight in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
            xx=x0+ox;yy=y0+oy;keep=(xx>=0)&(xx<width)&(yy>=0)&(yy<height)
            rows=keep.nonzero().flatten();matrix[rows,yy[keep]*width+xx[keep]]=weight[keep]
        self.register_buffer('matrix',matrix,persistent=False)

    def forward(self,x):
        assert tuple(x.shape[-2:])==(self.height,self.width)
        return (x.flatten(2)@self.matrix.t()).reshape(len(x),x.shape[1],*self.output_shape)


class StableAngularAvailability(AngularAvailability):
    def __init__(self,grid):
        super().__init__(grid);self.sampler=FixedBilinearSampler(grid)

    def load_state_dict(self,state_dict,strict=True,assign=False):
        # The derived constant matrix must match the saved angular layout.
        torch.testing.assert_close(state_dict['grid'].cpu(),self.grid.cpu(),rtol=0,atol=0)
        return super().load_state_dict(state_dict,strict=strict,assign=assign)

    def forward(self,features,ranges,valid,wrong_zone=False):
        good=valid.bool()&torch.isfinite(ranges)&(ranges>0)&(ranges<=4)
        clean=torch.where(good,ranges,torch.zeros_like(ranges));x=self.local(features)
        tokens=self.sampler(x).permute(0,2,3,1)
        if wrong_zone:tokens=tokens.reshape(len(x),8,8,49,16).roll(4,dims=2).flatten(1,2)
        zone=tokens.mean(2,keepdim=True).expand_as(tokens)
        relative=self.relative[None,None].expand(len(x),64,-1,-1)
        packet=torch.cat([clean/4,good.float()],-1)[:,:,None].expand(-1,-1,49,-1)
        grid=self.grid[None].expand(len(x),-1,-1,-1)
        return self.head(torch.cat([tokens,zone,grid,relative,packet],-1)).squeeze(-1)
