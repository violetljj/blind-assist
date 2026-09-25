"""QG-1: query-local spatial image features and geometrically weighted CNH."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from cnh_qg1_geometry import project_boxes, query_descriptors, tof_query_weights


class QueryGeometryModel(nn.Module):
    def __init__(self, camera_k, image_size):
        super().__init__()
        self.image_size = tuple(image_size)
        bounds = project_boxes(camera_k, self.image_size)
        self.register_buffer('roi_bounds', torch.tensor(bounds))
        self.register_buffer('query_geometry', torch.tensor(query_descriptors()))
        weights = torch.tensor(tof_query_weights(), dtype=torch.float32)
        self.register_buffer('tof_weights', weights)
        zone = weights.sum(-1)
        self.register_buffer('zone_weights', zone / zone.amax(-1, keepdim=True).clamp_min(1e-8))
        self.image_encoder = nn.Sequential(nn.Conv2d(5, 24, 3, stride=2, padding=1), nn.GELU(),
                                           nn.Conv2d(24, 24, 3, padding=1), nn.GELU())
        self.image_project = nn.Sequential(nn.Linear(24*8*8, 64), nn.GELU())
        self.tof_project = nn.Sequential(nn.Linear(64*16+64*4, 64), nn.GELU(), nn.Linear(64,64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(64+64+10,64), nn.GELU(), nn.Linear(64,1))

    def roi_pool(self, features):
        """Include all feature centres in each projected box; retain 8x8 bins."""
        height,width=features.shape[-2:]
        native_w,native_h=self.image_size
        values=[]
        for bound in self.roi_bounds:
            x0,y0,x1,y1=bound.tolist()
            if x1<=x0 or y1<=y0:
                values.append(features.new_zeros((len(features),24,8,8)))
                continue
            # Stride-2 padded convolutions have centres 0,2,4,... in input pixels.
            left=max(0, min(width, math.ceil(x0/2)))
            right=max(0, min(width, math.floor(x1/2)+1))
            top=max(0, min(height, math.ceil(y0/2)))
            bottom=max(0, min(height, math.floor(y1/2)+1))
            if right<=left or bottom<=top:
                values.append(features.new_zeros((len(features),24,8,8)))
            else:
                roi=features[...,top:bottom,left:right]
                kh,kw=math.ceil(roi.shape[-2]/8),math.ceil(roi.shape[-1]/8)
                padded=F.pad(roi,(0,kw*8-roi.shape[-1],0,kh*8-roi.shape[-2]),value=-float('inf'))
                pooled=F.max_pool2d(padded,(kh,kw),stride=(kh,kw))
                values.append(torch.where(torch.isfinite(pooled),pooled,torch.zeros_like(pooled)))
        return torch.stack(values,1)

    def forward(self, image, histogram, ambient, scalar, valid, *, arm):
        batch=len(histogram)
        if arm not in ('full_depth','tof_sim','tof_rgb'):
            raise ValueError(arm)
        if arm=='tof_sim':
            im=histogram.new_zeros((batch,6,64))
        else:
            im=self.image_project(self.roi_pool(self.image_encoder(image)).flatten(2))
        if arm=='full_depth':
            tof=histogram.new_zeros((batch,6,64))
        else:
            h=torch.sign(histogram)*torch.log1p(histogram.abs())
            h=(h[:,None]*self.tof_weights[None]).flatten(2)
            meta=torch.stack((torch.where(valid,scalar,torch.zeros_like(scalar))/5,
                              valid.float(),torch.log1p(ambient.clamp_min(0))/5,
                              torch.zeros_like(ambient)), -1)
            meta=(meta[:,None]*self.zone_weights[None,:,:,None]).flatten(2)
            tof=self.tof_project(torch.cat((h,meta),-1))
        desc=self.query_geometry[None].expand(batch,-1,-1)
        return self.head(torch.cat((im,tof,desc),-1)).squeeze(-1)


def depth_features(depth, ray_coordinates):
    """Observable XYZ + validity/inverse range; no box-hit oracle or labels."""
    valid=torch.isfinite(depth)&(depth>0)&(depth<100)
    z=torch.where(valid,depth,torch.zeros_like(depth))
    xyz=torch.stack((z*ray_coordinates[0],z*ray_coordinates[1],z),1)
    xyz=xyz.clamp(-10,10)/10
    return torch.cat((xyz,valid[:,None].float(),(torch.where(valid,1/(1+z),0))[:,None]),1)
