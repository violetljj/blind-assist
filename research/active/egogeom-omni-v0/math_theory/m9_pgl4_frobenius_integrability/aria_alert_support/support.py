"""Ray-domain opportunities and a single fixed history-normalized obstacle score."""
from collections import deque

import numpy as np

STARTS = np.round(np.arange(.3, 3.71, .2), 2)
WIDTH = .30
MIN_HITS = 18
EPS = 1.e-4


def axes(rotation, heading):
    fw = np.array([heading[0], heading[1]],float)
    if np.linalg.norm(fw)<1.e-4:
        fw = rotation[:2,2].astype(float)
    fw /= max(np.linalg.norm(fw),1.e-6)
    return fw, np.array([fw[1],-fw[0]])


def world_rays(shape, K, R):
    v,u=np.indices(shape,dtype=float)
    cam=np.stack(((4*u+2-K[0,2])/K[0,0],(4*v+2-K[1,2])/K[1,1],np.ones_like(u)),axis=-1)
    return cam.reshape(-1,3)@R.T


def intervals(rays, camera_offset, fw, side):
    # At axial depth z, lateral and forward coordinates are affine in z.
    lat = rays[:,:2]@side
    front = rays[:,:2]@fw
    lateral_offset = camera_offset[:2]@side
    forward_offset = camera_offset[:2]@fw
    nonzero = np.abs(lat)>1.e-10
    # Slack in metres gives an outer approximation despite float32 source poses.
    pad = .30+EPS
    low=(-pad-lateral_offset)/np.where(nonzero,lat,1.)
    high=(pad-lateral_offset)/np.where(nonzero,lat,1.)
    zlo=np.where(nonzero,np.maximum(.25-EPS,np.minimum(low,high)),.25-EPS)
    zhi=np.where(nonzero,np.minimum(10.+EPS,np.maximum(low,high)),10.+EPS)
    can=(nonzero|(abs(lateral_offset)<=pad+1.e-9))&(zlo<=zhi)
    left = np.maximum(.30-EPS,forward_offset+np.minimum(zlo*front,zhi*front)-EPS)
    right = np.minimum(4.+EPS,forward_offset+np.maximum(zlo*front,zhi*front)+EPS)
    keep=can&(left<=right)
    return np.sort(left[keep]-WIDTH-EPS),np.sort(right[keep]+EPS)


def at(starts, ends, s):
    return np.searchsorted(starts,s,side="right")-np.searchsorted(ends,s,side="left")


def best(starts, ends, target):
    lo,hi=max(.30,target-.55)-EPS,min(4.,target+.55)+EPS
    pos=starts[np.searchsorted(starts,lo):np.searchsorted(starts,hi,side="right")]
    at_left=int(at(starts,ends,lo))
    return max(at_left,int(at(starts,ends,pos).max()) if len(pos) else 0)


class CausalRayReadout:
    def __init__(self):
        self.history=deque(maxlen=6)

    def step(self, depth, K, R, t, heading):
        fw,side=axes(R,heading)
        rays=world_rays(depth.shape,K,R)
        z=depth.ravel()
        points=rays*z[:,None]
        valid=np.isfinite(z)&(z>.25)&(z<10.)
        f,lat=points[:,:2]@fw,points[:,:2]@side
        strip=valid&(np.abs(lat)<=1.2)&(f>=.8)&(f<=4.5)
        ground=np.percentile(points[strip,2],8) if strip.sum()>=40 else -1.50
        obstacle=valid&(points[:,2]>=ground+.15)&(points[:,2]<=ground+1.90)
        self.history.append((rays,t.copy(),points[obstacle,:2]+t[:2]))
        mass=np.zeros(len(STARTS))
        capacity=np.zeros_like(mass)
        prior=None
        for age,(old_rays,old_pos,old_points) in enumerate(reversed(self.history)):
            left,right=intervals(old_rays,old_pos-t,fw,side)
            available=at(left,right,STARTS)
            rel=old_points-t[:2]
            front=rel@fw
            eligible=np.abs(rel@side)<=.30
            zs=np.sort(front[eligible])
            hits=np.searchsorted(zs,STARTS+WIDTH,side="right")-np.searchsorted(zs,STARTS,side="left")
            # A bin must be supported by at least 18 predicted obstacle returns
            # from one real frame. Repeated views cannot manufacture sole support.
            this=hits>=MIN_HITS
            prior=this if prior is None else (prior|this)
            if age==0:
                current=available.copy()
                live_hits=bool(np.any(this & (available>=MIN_HITS)))
            mass+=(.7**age)*hits
            capacity+=(.7**age)*available
        usable=capacity>=MIN_HITS
        score=np.max(np.where(usable & prior,mass/np.maximum(capacity,MIN_HITS),0.))
        # Zero evidence and unobserved geometry have distinct published states.
        return float(score),bool(usable.any()),int(current.max()),float(capacity.max()),live_hits
