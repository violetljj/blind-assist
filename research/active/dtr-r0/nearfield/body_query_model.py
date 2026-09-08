"""Matched BODY/HEAD visible-evidence models, with a count-preserving bottleneck.

The model input is RGB only. Calibration and wearer envelopes are fixed buffers.
The twelve outputs describe observed native-pixel counts, NOT hidden occupancy.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F
from contact_retina_spec import BODY_BOXES
from decoupled_model import DecoupledModel
from whisker_model import IMAGE_SIZE, SUPPORT_SIZE


CALIBRATION = dict(height_m=1.70, pitch_degrees=0., hfov_degrees=100.,
                   width=640, height=360, range_m=3.)


def query_boxes():
    """Disjoint within each head; retain closed outer bounds and shared Z edge."""
    result = []
    for low, high in BODY_BOXES:
        for distance in range(2):
            for lateral in range(3):
                dy = (high[1]-low[1])/3
                result.append(((high[0]+1.5*distance, low[1]+dy*lateral, low[2]),
                               (high[0]+1.5*(distance+1), low[1]+dy*(lateral+1), high[2])))
    return result


def fixed_projection():
    """Project a 3x3x3 sampling lattice per cell, not just twelve centroids.

    Samples outside the FOV are masked. No per-image simulator metadata is used.
    Normalized image coordinates use pixel centers and align_corners=False.
    """
    grids, visible, coordinates = [], [], []
    aspect = CALIBRATION['height']/CALIBRATION['width']
    tangent = math.tan(math.radians(CALIBRATION['hfov_degrees']/2))
    fractions = (.15, .5, .85)
    for low, high in query_boxes():
        cell, mask, xyz = [], [], []
        for fx in fractions:
            for fy in fractions:
                for fz in fractions:
                    x, y, z = [a+(b-a)*f for a,b,f in zip(low,high,(fx,fy,fz))]
                    u = y/(x*tangent)
                    v = -(z-CALIBRATION['height_m'])/(x*tangent*aspect)
                    cell.append((u,v)); mask.append(abs(u)<=1 and abs(v)<=1)
                    xyz.append((x/3.18,y/.28,z/1.85))
        grids.append(cell); visible.append(mask); coordinates.append(xyz)
    return torch.tensor(grids), torch.tensor(visible), torch.tensor(coordinates)


def projection_weights(grid):
    """Static bilinear sampling as matmul, avoiding CUDA grid_sample atomics."""
    h,w=SUPPORT_SIZE
    weights=torch.zeros(grid.shape[0]*grid.shape[1],h*w)
    for i,(u,v) in enumerate(grid.reshape(-1,2).tolist()):
        x,y=((u+1)*w-1)/2,((v+1)*h-1)/2
        x0,y0=math.floor(x),math.floor(y)
        for xx in (x0,x0+1):
            for yy in (y0,y0+1):
                if 0<=xx<w and 0<=yy<h:
                    weights[i,yy*w+xx]=(1-abs(x-xx))*(1-abs(y-yy))
    return weights


def near_from_counts(count_logits):
    """P(sum six capped counts >=3), using a conditional-independence readout.

    For deterministic count labels this EXACTLY reproduces the >=3 native-pixel
    rule, including three single pixels in three separate cells. The probabilistic
    independence assumption is an architectural choice, not a calibrated guarantee.
    """
    if count_logits.ndim != 3 or count_logits.shape[1:] != (12,4):
        raise ValueError('Expected Bx12x4 capped native-count logits')
    p = count_logits.softmax(-1).reshape(-1,2,6,4)
    z = torch.zeros_like(p[:,:,0,0]); d0=z+1; d1=z; d2=z
    for j in range(6):
        q0,q1,q2 = p[:,:,j,0],p[:,:,j,1],p[:,:,j,2]
        d0,d1,d2 = d0*q0, d1*q0+d0*q1, d2*q0+d1*q1+d0*q2
    # Computing logits of the complement avoids cancelling small positive tails.
    below = (d0+d1+d2).clamp(1e-7,1-1e-7)
    return torch.log1p(-below)-below.log()


class BodyQueryModel(DecoupledModel):
    """A retains G13 support gating. B's near output has no feature bypass.

    Both arms instantiate exactly the same trainable modules and receive the
    same support, count and near losses. Only the final near computation differs.
    The unused legacy near parameters in B are reported, not counted as active.
    """
    def __init__(self, pretrained_root, arm='A'):
        super().__init__(pretrained_root)
        if arm not in ('A','B'):
            raise ValueError('arm must be A or B')
        self.arm = arm
        grid, valid, xyz = fixed_projection()
        self.register_buffer('query_grid',grid)
        self.register_buffer('query_valid',valid)
        self.register_buffer('query_xyz',xyz)
        self.register_buffer('query_projection',projection_weights(grid))
        self.query_point = nn.Sequential(nn.Linear(67,48),nn.GELU(),nn.Linear(48,32),nn.GELU())
        self.query_readout = nn.Linear(32,4)
        # A conservative common initial count prior prevents B starting at P=1.
        nn.init.zeros_(self.query_readout.weight)
        with torch.no_grad():
            self.query_readout.bias.copy_(torch.tensor([.94,.02,.02,.02]).log())

    def initialize_g13(self, state):
        missing, unexpected = self.load_state_dict(state,strict=False)
        allowed = ('query_point.','query_readout.','query_grid','query_valid','query_xyz','query_projection')
        if unexpected or any(not name.startswith(allowed) for name in missing):
            raise ValueError(f'G13 initialization mismatch: {missing}, {unexpected}')

    def forward(self,rgb):
        if rgb.ndim!=4 or tuple(rgb.shape[1:])!=(3,*IMAGE_SIZE):
            raise ValueError('Expected RGB Bx3x144x256')
        deep,shallow = self.extract((rgb-self.image_mean)/self.image_std)
        deep = F.interpolate(self.deep_projection(deep),size=SUPPORT_SIZE,mode='bilinear',align_corners=False)
        detail = self.detail(shallow)
        support = self.support(deep+detail)
        sampled = torch.matmul(torch.cat((deep,detail),1).flatten(2),self.query_projection.T)
        sampled = sampled.transpose(1,2).reshape(rgb.shape[0],12,27,64)
        xyz = self.query_xyz[None].expand(rgb.shape[0],-1,-1,-1)
        feature = self.query_point(torch.cat((sampled,xyz),-1))
        mask = self.query_valid[None,:,:,None]
        feature = (feature*mask).sum(2)/mask.sum(2).clamp_min(1)
        counts = self.query_readout(feature)
        if self.arm=='B':
            near = near_from_counts(counts)
        else:
            gated = deep[:,None]*support.sigmoid()[:,:,None]
            b,heads,channels,h,w = gated.shape
            pooled = self.near[0](gated.reshape(b*heads,channels,h,w)).reshape(b,heads,-1)
            near = (pooled*self.near[2].weight[None]).sum(-1)+self.near[2].bias
        return near,support,counts
