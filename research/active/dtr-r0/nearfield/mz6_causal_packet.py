"""Bounded raw-observation range hypotheses. No pose, truth or native-depth input."""
from collections import deque
import math

import cv2
import numpy as np

HISTORY = 2
HZ = 12


def zone_ids(points):
    """320x180 resized RGB, original HFoV100 and generic45x45 angular zones."""
    points = np.asarray(points)
    f = 160 / math.tan(math.radians(50))
    az = np.degrees(np.arctan((points[:,0]-159.5)/f))
    el = np.degrees(np.arctan(-(points[:,1]-89.5)/f))
    valid = (np.abs(az)<=22.5) & (np.abs(el)<=22.5)
    col = np.clip(np.floor((az+22.5)/5.625),0,7).astype(int)
    row = np.clip(np.floor((22.5-el)/5.625),0,7).astype(int)
    return np.where(valid,row*8+col,-1)


def correspondences(before, current, zero_motion=False):
    points = cv2.goodFeaturesToTrack(before, maxCorners=1000, qualityLevel=.01,
                                    minDistance=3, blockSize=3)
    if points is None:
        return np.zeros((64,64),dtype=int)
    if zero_motion:
        target = points.copy()
        keep = np.ones(len(points),dtype=bool)
    else:
        target, status, err = cv2.calcOpticalFlowPyrLK(before,current,points,None,
            winSize=(15,15),maxLevel=2)
        back, status_back, _ = cv2.calcOpticalFlowPyrLK(current,before,target,None,
            winSize=(15,15),maxLevel=2)
        keep = (status[:,0]>0) & (status_back[:,0]>0)
        keep &= np.linalg.norm(back[:,0]-points[:,0],axis=1)<=1.5
    p, q = points[:,0], target[:,0]
    keep &= np.isfinite(q).all(1) & (q[:,0]>=0) & (q[:,0]<320) & (q[:,1]>=0) & (q[:,1]<180)
    # Same local photometric check for measured-flow and zero-motion controls.
    for i in np.flatnonzero(keep):
        old_patch=cv2.getRectSubPix(before,(7,7),tuple(p[i])).astype(float)
        new_patch=cv2.getRectSubPix(current,(7,7),tuple(q[i])).astype(float)
        keep[i]=np.mean(np.abs(old_patch-new_patch))<=20
    a,b = zone_ids(p),zone_ids(np.nan_to_num(q))
    keep &= (a>=0)&(b>=0)
    counts=np.zeros((64,64),dtype=int)
    np.add.at(counts,(a[keep],b[keep]),1)
    return counts


class CausalPacket:
    def __init__(self, zero_motion=False):
        self.history=deque(maxlen=HISTORY)
        self.clip=None
        self.zero_motion=zero_motion

    def step(self, clip, index, gray, ranges, valid):
        if clip != self.clip:
            self.history.clear(); self.clip=clip
        if self.history and index<=self.history[-1][0]:
            raise ValueError('Strictly increasing causal frame indices required')
        assert gray.shape==(180,320) and gray.dtype==np.uint8
        assert ranges.shape==valid.shape==(64,2)
        assert np.isfinite(ranges[valid]).all()
        out, flags = ranges.copy(), valid.copy()
        completions=[]
        for old_index, old_gray, old_ranges, old_valid in reversed(self.history):
            age=index-old_index
            if age>HISTORY:
                continue
            counts=correspondences(old_gray,gray,self.zero_motion)
            tolerance=.05+1.5*age/HZ
            source=(old_valid.all(1)&((old_ranges[:,1]-old_ranges[:,0])>=.30))
            for dest in np.flatnonzero(flags.sum(1)==1):
                current_range=out[dest,flags[dest]][0]
                compatible=(source & (counts[:,dest]>=3)
                    & (np.abs(old_ranges[:,1]-current_range)<=tolerance)
                    & (old_ranges[:,0]<current_range-.15))
                choices=np.flatnonzero(compatible)
                if not len(choices):
                    continue
                src=int(choices[np.argmax(counts[choices,dest])])
                out[dest]=[old_ranges[src,0],current_range]; flags[dest]=True
                completions.append(dict(source_zone=src,destination_zone=int(dest),age_frames=age,
                    matches=int(counts[src,dest]),range_m=float(old_ranges[src,0]),
                    background_residual_m=float(abs(old_ranges[src,1]-current_range)),
                    interval_halfwidth_m=tolerance))
        # Never recycle completed ranges, and never persist truth or simulator pose.
        self.history.append((index,gray.copy(),ranges.copy(),valid.copy()))
        return out,flags,completions


def mean3(logits, clips):
    out=np.empty_like(logits)
    history=deque(maxlen=3); previous=None
    for i,clip in enumerate(clips):
        if clip!=previous:
            history.clear(); previous=clip
        history.append(logits[i])
        out[i]=np.mean(history,axis=0)
    return out
