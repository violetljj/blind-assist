"""Evaluator-side native visible-count labels; no learned or hidden truth."""
from __future__ import annotations
import torch
from body_query_model import CALIBRATION, query_boxes
from worlds_verify import camera_points, corridor_masks


def labels(native,camera,floor_z_m):
    if abs(float(camera['pitch'])-CALIBRATION['pitch_degrees'])>1e-5:
        raise ValueError('Frame violates fixed pitch; do not pass GT pose to model')
    if abs(float(camera['z'])-float(floor_z_m)-CALIBRATION['height_m'])>1e-4:
        raise ValueError('Frame violates fixed optical height')
    if abs(float(camera.get('roll',0)))>1e-5:
        raise ValueError('Frame violates fixed roll')
    local_camera = dict(x=0.,y=0.,z=0.,pitch=0.,yaw=0.,roll=0.)
    points = camera_points(native,local_camera)
    points[...,2] += CALIBRATION['height_m']
    valid = torch.isfinite(native)&(native>0)&(native<100)
    masks = corridor_masks(points,valid,dict(x=0.,y=0.,z=0.),3.)
    counts=[]; membership=torch.zeros_like(masks,dtype=torch.int16)
    for i,(low,high) in enumerate(query_boxes()):
        head=i//6; distance=(i%6)//3; lateral=i%3
        inside=masks[head].clone()
        inside &= (points[...,0]>=low[0]) & ((points[...,0]<=high[0]) if distance==1 else (points[...,0]<high[0]))
        inside &= (points[...,1]>=low[1]) & ((points[...,1]<=high[1]) if lateral==2 else (points[...,1]<high[1]))
        counts.append(inside.sum()); membership[head]+=inside
    if not torch.equal(membership,masks.to(torch.int16)):
        raise ValueError('Query partition overlaps or omits visible native points')
    counts=torch.stack(counts)
    near=(masks.sum((-2,-1))>=3).to(torch.int64)
    capped=counts.clamp_max(3)
    if not torch.equal((capped.reshape(2,6).sum(1)>=3).long(),near):
        raise ValueError('Count aggregation changed original near label')
    encoded=masks.to(torch.int8).masked_fill(~valid[None],-1)
    return dict(near=near,counts=capped,raw_counts=counts,support=encoded)
