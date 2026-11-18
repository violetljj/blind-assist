"""Causal, calibrated RGB matching for challenging overly near monocular depth.

No active depth, body labels, model scores or scene annotations are inputs.
Timestamps use 100 ns. Camera matrices are OpenCV pinholes and camera-to-world poses.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np


@dataclass
class View:
    ts: int
    path: Path
    K: np.ndarray
    c2w: np.ndarray


def holoset_views(cap, extra_dir):
    paths = {int(p.stem): p for directory in (cap.directory / "rgb", Path(extra_dir))
             for p in directory.glob("*.png") if p.stem.isdigit()}
    views = []
    for ts, path in sorted(paths.items()):
        if ts not in cap.pv:
            continue
        fx, fy, pv = cap.pv[ts]
        pose = pv.copy()
        pose[:3, :3] = pv[:3, :3] @ np.diag([1., -1., -1.])
        K = np.array([[fx, 0., cap.cx], [0., fy, cap.cy], [0., 0., 1.]])
        views.append(View(ts, path, K, pose))
    return views


def half_camera(K):
    # OpenCV's 2:1 area reduction puts output centres at (2*u+.5,2*v+.5).
    return np.array([[.5,0.,-.25],[0.,.5,-.25],[0.,0.,1.]]) @ K


def average(im):
    return cv2.blur(im, (5, 5))


def match_pair(src, dst, src_K, dst_K, relative, diagnostics=False, return_hypotheses=False):
    h, w = src.shape
    K, D = half_camera(src_K), half_camera(dst_K)
    rot, t = relative[:3,:3], relative[:3,3]
    base = D @ rot @ np.linalg.inv(K)
    shift = D @ t
    v,u = np.indices((h,w), dtype=np.float32)
    q = np.stack((u,v,np.ones_like(u)), axis=-1) @ base.T
    mu = average(src)
    var = np.maximum(average(src*src)-mu*mu, 0)
    usable = var >= 16.
    usable[:3] = usable[-3:] = False
    usable[:,:3] = usable[:,-3:] = False
    # Keep disparity steps small at long baselines. Depth hypotheses span 0.30–12 m.
    n = int(np.clip(np.ceil((1/.30-1/12)*np.linalg.norm(t)*max(D[0,0],D[1,1])/.85), 80, 224))
    rho = np.linspace(1/12, 1/.30, n, dtype=np.float32)
    curve = np.empty((n,h,w), np.int16)
    for i, inverse in enumerate(rho):
        H = base.copy()
        H[:,2] += shift*inverse
        warped = cv2.warpPerspective(dst, H, (w,h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=-1024.)
        inside = cv2.erode(warped, np.ones((5,5), np.uint8)) >= 0
        avg = average(warped)
        vv = np.maximum(average(warped*warped)-avg*avg, 0)
        corr = (average(src*warped)-mu*avg) / np.sqrt(np.maximum(var*vv, 1.))
        ok = (usable & inside & (vv>=16.) & ((q[:,:,2]+shift[2]*inverse)>.05)
              & (abs((warped-avg)-(src-mu))<=15.))
        curve[i] = np.where(ok, np.rint(1000*np.clip(corr,-1,1)), -2000).astype(np.int16)
    best = curve.max(axis=0)
    peak = rho[curve.argmax(axis=0)]
    # Include every nearly equivalent hypothesis, including disconnected peaks.
    plausible = curve >= (best-55)
    lo = rho[plausible.argmax(axis=0)]
    hi = rho[(n-1)-plausible[::-1].argmax(axis=0)]
    denom = np.maximum(q[:,:,2] + shift[2]*peak, .05)
    dx = (shift[0]*q[:,:,2] - shift[2]*q[:,:,0]) / denom**2
    dy = (shift[1]*q[:,:,2] - shift[2]*q[:,:,1]) / denom**2
    slope = np.hypot(dx,dy)
    error = (rho[1]-rho[0])/2 + 1.25/np.maximum(slope,1e-4)
    valid = ((best >= 900) & (peak*slope >= 3.) & ((hi-lo) <= .45*peak) & (error < .30*peak))
    bounds = np.stack((peak, np.maximum(0,lo-error), hi+error)).astype(np.float32)
    bounds[:,~valid] = 0
    if return_hypotheses:
        # Register plausible samples on one common inverse-depth grid for pair-to-pair comparison.
        common_indices = np.rint(np.linspace(0,n-1,96)).astype(int)
        acceptable = (best>=900) & usable
        packed = np.packbits(plausible[common_indices] & acceptable[None],axis=0)
        # Use the smaller derivative over the peak and the nearest plausible hypothesis.
        hi_denom = np.maximum(q[:,:,2]+shift[2]*hi,.05)
        near_slope = slope * denom**2/hi_denom**2
        margin = (rho[1]-rho[0])/2 + 1.25/np.maximum(np.minimum(slope,near_slope),1e-4)
        near_inv = np.where(acceptable,hi+margin,0).astype(np.float32)
        return bounds, packed, near_inv
    if diagnostics:
        return bounds, dict(texture=usable, peak=best>=900, parallax=peak*slope>=3.,
                            distinct=(hi-lo)<=.45*peak, localized=error<.30*peak)
    return bounds


class RGBVerifier:
    def __init__(self, views):
        self.views = {v.ts:v for v in views}
        self.clock = np.asarray(sorted(self.views), np.int64)
        self.image = lru_cache(maxsize=28)(self._image)
        self.pair = lru_cache(maxsize=56)(self._pair)

    def _image(self, ts):
        im = cv2.imread(str(self.views[ts].path), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(self.views[ts].path)
        return cv2.resize(im, (im.shape[1]//2, im.shape[0]//2), interpolation=cv2.INTER_AREA).astype(np.float32)

    def _pair(self, src, dst):
        a,b = self.views[src], self.views[dst]
        return match_pair(self.image(src), self.image(dst), a.K, b.K, np.linalg.inv(b.c2w) @ a.c2w)

    def witnesses(self, source, cutoff):
        a = self.views[source]
        left,right = np.searchsorted(self.clock, [source-9_000_000, min(cutoff,source+9_000_001)])
        candidates = []
        for ts in self.clock[left:right]:
            offset = (int(ts)-source)/1e7
            b = self.views[ts]
            baseline = np.linalg.norm(b.c2w[:3,3]-a.c2w[:3,3])
            if abs(offset)>=.05 and .05<=baseline<=.70 and a.c2w[:3,2]@b.c2w[:3,2]>.82:
                candidates.append((int(ts), offset))
        selected = []
        for target in (.16, -.20, .40, -.42):
            possible = [c for c in candidates if c[1]*target>0 and all(
                np.linalg.norm(self.views[c[0]].c2w[:3,3]-self.views[t].c2w[:3,3]) >= .05 for t in selected)]
            if possible:
                selected.append(min(possible,key=lambda p:abs(p[1]-target))[0])
        return sorted(selected)

    def classify(self, source_rgb_ts, cutoff_ts, maps, mono):
        source, cutoff = int(source_rgb_ts), int(cutoff_ts)
        if source>=cutoff:
            raise ValueError("A history exposure must precede the decision")
        dates = self.witnesses(source,cutoff)
        u,v = [np.rint((a-.5)/2).astype(np.int32) for a in maps]
        median = np.full(mono.shape,np.nan,np.float32)
        lower = np.full_like(median,np.nan)
        info = dict(source=source,cutoff=cutoff,witnesses=dates,scale_tiles=0,rgb_scale=None,
                    tri_m=median,lower_m=lower,mono_rgb_m=np.full_like(median,np.nan),valid=np.zeros_like(u,bool))
        if len(dates)<2 or max(dates)<=source:
            return np.zeros_like(u,bool), info
        pairs = []
        for ts in dates:
            match = self.pair(source,ts)
            valid = (u>=0)&(u<match.shape[2])&(v>=0)&(v<match.shape[1])
            sampled = match[:,v.clip(0,match.shape[1]-1),u.clip(0,match.shape[2]-1)]
            sampled[:,~valid] = 0
            pairs.append(sampled)
        p,lo,hi = np.stack(pairs,axis=1)
        seen = np.zeros_like(u,bool)
        for i,j in combinations(range(len(dates)),2):
            if max(dates[i],dates[j])<=source:
                continue
            good = ((p[i]>0)&(p[j]>0)&(abs(p[i]-p[j]) <= .15*np.maximum(p[i],p[j]))
                    &(lo[i]<=hi[j])&(lo[j]<=hi[i]))
            seen |= good
        # Any additional individually unambiguous match that disagrees vetoes this track.
        highest = np.max(p,axis=0)
        lowest = np.min(np.where(p>0,p,np.inf),axis=0)
        seen &= highest-lowest <= .15*highest
        seen &= np.isfinite(mono)&(mono>0)
        median[seen] = 1/(p.sum(axis=0)[seen]/(p>0).sum(axis=0)[seen])
        lower[seen] = 1/hi.max(axis=0)[seen]
        info["valid"] = seen
        medians = []
        for yy in np.array_split(np.arange(u.shape[0]),4):
            for xx in np.array_split(np.arange(u.shape[1]),4):
                ix = np.ix_(yy,xx)
                mask = seen[ix]
                if mask.sum()>=12:
                    medians.append(float(np.median(median[ix][mask]/mono[ix][mask])))
        info["scale_tiles"] = len(medians)
        if len(medians)<4:
            return np.zeros_like(u,bool),info
        scale = float(np.median(medians))
        pred = mono*scale
        info.update(rgb_scale=scale,mono_rgb_m=pred)
        return seen & (lower > pred + np.maximum(.20,.20*pred)),info
