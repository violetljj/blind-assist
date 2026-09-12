"""Stereo query-local surface extent; optional disparity-perturbation contrast.

These are deterministic support qualifications, not calibrated probabilities or
proof of object identity. The ToF expert and alert lifecycle remain unchanged.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import mz101_spatial as base

MIN_IMAGE_PLANE_SPAN_M=.10
MAX_NEIGHBOR_DISPARITY_STEP_PX=1.
PERTURBATION_PX=1.


def qualify(depth,pose,*,perturbation=False):
    primary_masks=qualify(depth,pose)[2] if perturbation else None
    yy,xx=np.nonzero(np.isfinite(depth));z=depth[yy,xx]
    f=base.RIG['width']/(2*np.tan(np.radians(base.RIG['hfov_deg']/2)))
    fb=f*base.RIG['baseline_m'];disp=fb/z
    camera=np.stack([z,(xx-320)*z/f,-(yy-180)*z/f],1)
    rotation=base.rotation(pose['yaw'],pose['pitch'],pose.get('roll',0.))
    origin=np.asarray(pose['camera_in_body_m']);body=camera@rotation.T+origin
    common=(np.abs(np.degrees(np.arctan2(camera[:,1],camera[:,0])))<=22.5)&(np.abs(np.degrees(np.arctan2(camera[:,2],camera[:,0])))<=20)
    if perturbation:
        valid_interval=disp>PERTURBATION_PX
        near=camera*(disp/(disp+PERTURBATION_PX))[:,None]
        far=camera*(disp/np.maximum(disp-PERTURBATION_PX,.001))[:,None]
        near_body=near@rotation.T+origin;far_body=far@rotation.T+origin
    counts=[];diagnostics=[];masks=[]
    for part,(low,high) in enumerate(base.BOXES):
        inside=((body>=low)&(body<=high)).all(1)&common
        if perturbation:
            inside &= primary_masks[part,yy,xx] & valid_interval & ((near_body>=low)&(near_body<=high)).all(1)&((far_body>=low)&(far_body<=high)).all(1)
        indices=np.flatnonzero(inside);n=len(indices)
        accepted=np.zeros(depth.shape,bool)
        if n:
            # 4-neighbor topology with explicit disparity continuity. A large
            # surface outside this query cannot sponsor a tiny inside fragment.
            node=np.full(depth.shape,-1,np.int32);node[yy[indices],xx[indices]]=np.arange(n)
            disparity=np.zeros(depth.shape,np.float32);disparity[yy,xx]=disp
            a,b=node[:,:-1],node[:,1:]
            h=(a>=0)&(b>=0)&(np.abs(disparity[:,:-1]-disparity[:,1:])<=MAX_NEIGHBOR_DISPARITY_STEP_PX)
            c,d=node[:-1,:],node[1:,:]
            v=(c>=0)&(d>=0)&(np.abs(disparity[:-1,:]-disparity[1:,:])<=MAX_NEIGHBOR_DISPARITY_STEP_PX)
            rows=np.concatenate([a[h],c[v]]);cols=np.concatenate([b[h],d[v]])
            graph=coo_matrix((np.ones(len(rows),np.uint8),(rows,cols)),shape=(n,n)).tocsr()
            components,labels=connected_components(graph,directed=False)
            min_u=np.full(components,np.inf);max_u=np.full(components,-np.inf)
            min_v=min_u.copy();max_v=max_u.copy();min_z=min_u.copy()
            np.minimum.at(min_u,labels,xx[indices]);np.maximum.at(max_u,labels,xx[indices])
            np.minimum.at(min_v,labels,yy[indices]);np.maximum.at(max_v,labels,yy[indices])
            np.minimum.at(min_z,labels,z[indices])
            # Projected lateral/vertical span only: disparity noise along depth
            # cannot supply its own qualifying length. No minimum pole width.
            span=np.maximum(max_u-min_u,max_v-min_v)*min_z/f
            keep=span>=MIN_IMAGE_PLANE_SPAN_M
            chosen=indices[keep[labels]]
            accepted[yy[chosen],xx[chosen]]=True
            count=len(np.unique(np.floor(body[chosen]/.05).astype(np.int32),axis=0))
            diagnostics.append(dict(input_pixels=n,components=components,accepted_components=int(keep.sum()),
                accepted_pixels=len(chosen),max_span_m=float(span.max())))
        else:
            count=0;diagnostics.append(dict(input_pixels=0,components=0,accepted_components=0,accepted_pixels=0,max_span_m=0.))
        counts.append(count);masks.append(accepted)
    return np.asarray(counts,np.int32),diagnostics,np.asarray(masks)
