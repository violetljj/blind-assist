"""Metric two-view range anchors from past/current RGB and externally supplied camera poses."""
import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class View:
    gray: np.ndarray
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    corners: np.ndarray | None = None


def camera_rays(pixels, K):
    return np.column_stack(((pixels[:,0]-K[0,2])/K[0,0],
                            (pixels[:,1]-K[1,2])/K[1,1],np.ones(len(pixels))))


def triangulate(uv0, uv1, ref, cur):
    """Closest points on world rays, with positive depths, parallax and reprojection tests."""
    ray0 = camera_rays(uv0,ref.K) @ ref.R.T
    ray1 = camera_rays(uv1,cur.K) @ cur.R.T
    mag0 = np.linalg.norm(ray0,axis=1)
    mag1 = np.linalg.norm(ray1,axis=1)
    v0,v1 = ray0/mag0[:,None],ray1/mag1[:,None]
    baseline = cur.t-ref.t
    a,b = v0@baseline,v1@baseline
    cosine = np.clip((v0*v1).sum(axis=1),-1,1)
    sine2 = 1-cosine**2
    inv = 1/np.maximum(sine2,1.e-10)
    d0,d1 = (a-cosine*b)*inv,(cosine*a-b)*inv
    gap = np.linalg.norm(d0[:,None]*v0-d1[:,None]*v1-baseline,axis=1)
    # Reprojection error of the 3-D midpoint in roughly half-resolution pixels.
    reproj = .5*gap*np.maximum(ref.K[0,0]/np.maximum(d0,.1),cur.K[0,0]/np.maximum(d1,.1))
    good = ((d0>0) & (d1>0) & (cosine>0) & (sine2>math.sin(math.radians(1.4))**2)
            & (reproj<=.70))
    return d1/mag1,good,np.sqrt(sine2)


def weighted_median(x,w):
    order = np.argsort(x)
    index = np.searchsorted(w[order].cumsum(),.5*w.sum())
    return float(x[order[index]])


class MetricScale:
    """Both readouts use identical accepted correspondences and a fresh state per clip."""

    def __init__(self):
        self.past = deque(maxlen=10)
        self.log_scale = 0.
        self.attempts = self.updates = self.n_tracks = 0

    def _matches(self, ref, cur):
        if ref.corners is None:
            ref.corners = cv2.goodFeaturesToTrack(ref.gray,maxCorners=180,qualityLevel=.012,minDistance=5,blockSize=5)
        if ref.corners is None or len(ref.corners)<8:
            return None
        uv0 = ref.corners[:,0,:]
        rotated = camera_rays(uv0,ref.K) @ ref.R.T @ cur.R
        z = np.maximum(.1,rotated[:,2])
        guess = np.column_stack((cur.K[0,0]*rotated[:,0]/z+cur.K[0,2],
                                 cur.K[1,1]*rotated[:,1]/z+cur.K[1,2])).reshape(-1,1,2).copy(order="C").astype(np.float32)
        lk = {"winSize": (15,15),"maxLevel": 3,"criteria": (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,20,.03),
                  "flags": cv2.OPTFLOW_USE_INITIAL_FLOW}
        end,ok,_ = cv2.calcOpticalFlowPyrLK(ref.gray,cur.gray,ref.corners,guess,**lk)
        if end is None:
            return None
        back,back_ok,_ = cv2.calcOpticalFlowPyrLK(cur.gray,ref.gray,end,ref.corners.copy(),**lk)
        if back is None:
            return None
        keep = (ok[:,0]>0) & (back_ok[:,0]>0) & (np.linalg.norm(back[:,0]-uv0,axis=1)<.65)
        uv0,uv1 = uv0[keep],end[keep,0]
        if len(uv1)<8:
            return None
        zmetric,ok,parallax = triangulate(uv0,uv1,ref,cur)
        ok &= (zmetric>=.45) & (zmetric<=9.)
        return uv1[ok],zmetric[ok],parallax[ok]

    def step(self, gray, depth, ls, K_full, R, t):
        # OpenCV resize to one half maps full pixel i to (i+.5)/2-.5.
        k = K_full.astype(np.float64).copy()
        k[:2,:] *= .5
        k[:2,2] -= .25
        cur = View(cv2.equalizeHist(gray), k, R.astype(np.float64), t.astype(np.float64))
        pool = [(abs(delta-.32),v) for v in reversed(self.past)
                if .12 <= (delta:=np.linalg.norm(v.t-cur.t)) <= 1.15 and np.sum(v.R*cur.R)>2.45]
        self.past.append(cur)
        global_s = math.exp(self.log_scale)
        if not pool:
            return global_s,global_s,0
        self.attempts += 1
        ref = min(pool,key=lambda item:item[0])[1]
        matches = self._matches(ref,cur)
        if matches is None or len(matches[0])<8:
            return global_s,global_s,0
        uv,zmetric,parallax = matches
        # Recover full-resolution image coordinates, then indices of [::4,::4] samples.
        xy = (uv*2+.5)/4
        h,w = depth.shape
        keep = (xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<w-1)&(xy[:,1]<h-1)
        xy,zmetric,parallax = xy[keep],zmetric[keep],parallax[keep]
        u,v = xy[:,0].astype(int),xy[:,1].astype(int)
        fu,fv = xy[:,0]-u,xy[:,1]-v
        neighbors = np.stack((depth[v,u],depth[v,u+1],depth[v+1,u],depth[v+1,u+1]),axis=1)
        pred = ((1-fv)*((1-fu)*neighbors[:,0]+fu*neighbors[:,1])
                +fv*((1-fu)*neighbors[:,2]+fu*neighbors[:,3]))
        keep = (np.isfinite(pred)&(pred>.3)&(pred<15)&(neighbors.max(1)-neighbors.min(1)<.15*pred))
        keep &= (zmetric>.25*pred)&(zmetric<4.*pred)
        u,v,pred,zmetric,parallax = (x[keep] for x in (u,v,pred,zmetric,parallax))
        if len(pred)<8 or len(np.unique((v//12)*16+u//12))<3:
            return global_s,global_s,0
        values = np.log(zmetric/pred)
        weights = np.clip(parallax/.07,.3,2)*np.exp(-np.clip(ls[v,u],-3,-.5)*.25)
        center = weighted_median(values,weights)
        mad = weighted_median(abs(values-center),weights)
        if mad>.38:
            return global_s,global_s,0
        # Bounded robust update to prevent a single moving surface setting the entire clip.
        gain = min(.55, .18+len(values)/160) / (1+3*mad)
        self.log_scale += float(np.clip(gain*(center-self.log_scale),-.24,.24))
        self.log_scale = float(np.clip(self.log_scale,math.log(.4),math.log(2.5)))
        self.n_tracks += len(values)
        self.updates += 1
        global_s = math.exp(self.log_scale)
        local = self._local_scale(depth,u,v,pred,values,weights)
        return global_s,local,len(values)

    def _local_scale(self,depth,u,v,pred,values,weights):
        # A spatial Gaussian and tent interpolation in log-depth avoid spreading
        # corrections across unrelated ranges. Missing supports revert to global.
        bins = 12
        def index(z):
            return np.clip(np.log(np.clip(z,.3,20.)/.3)/.4,0,bins-1.001)
        def gather(volume, coord):
            a = coord.astype(int)
            f = coord-a
            lo = np.take_along_axis(volume,a[...,None],axis=2)[...,0]
            hi = np.take_along_axis(volume,(a+1)[...,None],axis=2)[...,0]
            return (1-f)*lo+f*hi
        coord = index(pred)
        base = coord.astype(int)
        f = coord-base
        mass = np.zeros((*depth.shape,bins),np.float32)
        sums = np.zeros_like(mass)
        delta = np.clip(values-self.log_scale,-.55,.55)
        for layer,portion in [(base,1-f),(base+1,f)]:
            np.add.at(mass,(v,u,layer),weights*portion)
            np.add.at(sums,(v,u,layer),weights*portion*delta)
        kernel = {"ksize": (25,25),"sigmaX": 4.,"borderType": cv2.BORDER_CONSTANT}
        coord = index(np.nan_to_num(depth,nan=20.))
        support = gather(cv2.GaussianBlur(mass,**kernel),coord)
        correction = gather(cv2.GaussianBlur(sums,**kernel),coord)/(support+.012)
        return np.exp(self.log_scale+correction).astype(np.float32)
