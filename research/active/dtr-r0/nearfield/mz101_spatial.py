"""Fixed observable-only stereo/ToF points and shared current-corridor readout.

Coordinates are forward/right/up, in metres. Native depth and scene actors are
not inputs. Known rig and camera/body pose are a declared controlled assumption.
"""
from __future__ import annotations

import math
import time
import cv2
import numpy as np

PARTS = ('BODY', 'HEAD')
BOXES = np.array([[[.18, -.28, .65], [3.18, .28, 1.4]],
                  [[.13, -.18, 1.4], [3.13, .18, 1.85]]])
MIN_DEPTH, MAX_DEPTH = .5, 4.
RIG = dict(width=640, height=360, hfov_deg=70., baseline_m=.10, tof_hfov_deg=45.)
SGBM = dict(minDisparity=0, numDisparities=96, blockSize=5,
            P1=8*25, P2=32*25, disp12MaxDiff=1, preFilterCap=31,
            uniquenessRatio=10, speckleWindowSize=0, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)


def rotation(yaw, pitch, roll=0.):
    """UE Rotator basis: positive pitch looks up; positive yaw looks right."""
    y,p,r = np.radians([yaw,pitch,roll])
    cy,sy,cp,sp,cr,sr = np.cos(y),np.sin(y),np.cos(p),np.sin(p),np.cos(r),np.sin(r)
    return np.array([[cp*cy, sr*sp*cy-cr*sy, -(cr*sp*cy+sr*sy)],
                     [cp*sy, sr*sp*sy+cr*cy, cy*sr-cr*sp*sy],
                     [sp, -sr*cp, cr*cp]])


def rays():
    a = np.arange(8)*45/8+45/16
    el,az = np.meshgrid(22.5-a, a-22.5, indexing='ij')
    xyz = np.stack([np.ones_like(az),np.tan(np.radians(az)),np.tan(np.radians(el))],-1).reshape(-1,3)
    return xyz/np.linalg.norm(xyz,axis=1,keepdims=True)


def tof_points(ranges, valid):
    r = np.asarray(ranges).reshape(64)
    v = np.asarray(valid).reshape(64).astype(bool)
    ok = v & np.isfinite(r) & (r >= MIN_DEPTH) & (r <= MAX_DEPTH)
    return rays()[ok]*r[ok,None]


def stereo_depth(left, right, rig=RIG):
    start=time.perf_counter()
    if left.shape != right.shape or left.shape[:2] != (rig['height'],rig['width']):
        raise ValueError('Stereo image/rig shape mismatch')
    l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    dl=cv2.StereoSGBM_create(**SGBM).compute(l,r).astype(np.float32)/16.
    right_cfg={**SGBM,'minDisparity':-SGBM['numDisparities']}
    dr=cv2.StereoSGBM_create(**right_cfg).compute(r,l).astype(np.float32)/16.
    yy,xx=np.indices(dl.shape)
    xr=np.rint(xx-dl).astype(int)
    xr_safe=np.clip(xr,0,rig['width']-1)
    # Preserve thin objects: consistency and small connected support, no large
    # speckle removal, dilation, GT-guided filling or mono-depth fallback.
    ok=(dl>0)&(xr>=0)&(xr<rig['width'])&(dr[yy,xr_safe]>-96)&(np.abs(dl+dr[yy,xr_safe])<=1.)
    f=rig['width']/(2*math.tan(math.radians(rig['hfov_deg']/2)))
    depth=f*rig['baseline_m']/np.maximum(dl,.01)
    ok &= (depth>=MIN_DEPTH)&(depth<=MAX_DEPTH)
    n,labels,stats,_=cv2.connectedComponentsWithStats(ok.astype('uint8'),8)
    keep=np.zeros(n,bool)
    keep[1:]=stats[1:,cv2.CC_STAT_AREA]>=8
    ok &= keep[labels]
    return np.where(ok,depth,np.nan).astype('float32'), dict(seconds=time.perf_counter()-start,
        valid_pixels=int(ok.sum()),total_pixels=ok.size)


def depth_points(depth, rig=RIG):
    yy,xx=np.nonzero(np.isfinite(depth))
    x=depth[yy,xx]
    f=rig['width']/(2*math.tan(math.radians(rig['hfov_deg']/2)))
    # UE SceneCapture intrinsics use half-size principal point.
    return np.stack([x,(xx-rig['width']/2)*x/f,-(yy-rig['height']/2)*x/f],axis=1)


def readout(points, pose, *, common_fov=True):
    """Return measured spatial support only; zero is UNKNOWN/no alert, not clear."""
    points=np.asarray(points).reshape(-1,3)
    if common_fov and len(points):
        ok=(np.abs(np.degrees(np.arctan2(points[:,1],points[:,0])))<=22.5)&(np.abs(np.degrees(np.arctan2(points[:,2],points[:,0])))<=20.)
        points=points[ok]
    body=points@rotation(pose['yaw'],pose['pitch'],pose.get('roll',0.)).T+np.asarray(pose['camera_in_body_m'])
    counts=[]
    for low,high in BOXES:
        # Classify original points; voxelisation only deduplicates support.
        q=body[((body>=low)&(body<=high)).all(1)]
        counts.append(len(np.unique(np.floor(q/.05).astype('int32'),axis=0)))
    return np.asarray(counts,dtype='int32'), body


def hysteresis(support, episodes):
    support=np.asarray(support)>0
    out=np.zeros_like(support)
    on=np.zeros(support.shape[1],int);off=on.copy();active=on.astype(bool)
    previous=None
    for i,ep in enumerate(episodes):
        if ep!=previous:
            on[:]=0;off[:]=0;active[:]=False
        on=np.where(support[i],on+1,0);off=np.where(support[i],0,off+1)
        active=(active|(on>=2))&(off<2)
        out[i]=active;previous=ep
    return out
