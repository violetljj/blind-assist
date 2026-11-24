"""Causal harmonic bearing and de-rotated, signed visual translation estimates."""
import numpy as np
import cv2
from scipy.stats import vonmises

FREQUENCIES = np.linspace(.7, 1.3, 13)


def unit(v):
    n = np.linalg.norm(v)
    return v/n if n>1e-10 else np.zeros_like(v)


def angle(v):
    return np.rad2deg(np.arctan2(v[1], v[0]))


def corridor_risk(bearing, angular_half_width, mu, kappa):
    """Bayes occupancy of a single known bearing/range obstacle, in radians."""
    if not 0 <= angular_half_width <= np.pi or kappa < 0:
        raise ValueError("Invalid aperture or concentration")
    offset = (bearing-mu+np.pi)%(2*np.pi)-np.pi
    return float(vonmises.cdf(offset+angular_half_width,kappa)
                 -vonmises.cdf(offset-angular_half_width,kappa))


def harmonic(t, acc):
    """Four seconds ending now, gravity-aligned forward/left/up coordinates.

    z selects cadence. A principal-axis estimate combines the AP second harmonic
    and the orthogonal ML fundamental. Braking half-plane selects its sign.
    """
    if len(t)<40 or t[-1]-t[0]<3.8 or np.diff(t).max()>.15:
        return None
    window = np.hanning(len(t))
    centered = acc - np.average(acc, weights=window, axis=0)
    a = centered * (2*window/window.sum())[:,None]
    e = np.exp(-2j*np.pi*FREQUENCIES[:,None]*(t-t[-1]))
    first, second = e @ a, (e*e) @ a
    k = np.argmax(abs(second[:,2]))
    c1, c2 = first[k,:2], second[k,:2] * np.exp(-1j*np.angle(second[k,2]))
    cross = np.array([-c1[1], c1[0]])
    mat = (np.outer(c2,c2.conj())+np.outer(cross,cross.conj())).real
    vals, vecs = np.linalg.eigh(mat)
    axis = vecs[:,-1]
    signed = -np.imag(c2 @ axis)
    bearing = axis * (1. if signed>=0 else -1.)
    contrast = (vals[-1]-vals[0])/max(vals.sum(),1e-10)
    phase_margin = abs(signed)/max(np.linalg.norm(c2),1e-10)
    vertical = abs(second[k,2])
    quality = contrast*phase_margin
    available = vertical>=.30 and np.linalg.norm(c2)>=.10 and quality>=.35
    return {"direction":bearing,"valid":bool(available),"quality":float(quality),
            "cross_real":c2.real, "cross_imag":c2.imag,
            "vertical":float(vertical),"phase_margin":float(phase_margin),
            "signed_phase_rad":float(np.angle(c2 @ bearing)),
            "frequency":float(FREQUENCIES[k])}


def visual_foe(img0, img1, k, from_old_to_new):
    """KLT rays, known local rotation, then incidence + cheirality for translation."""
    corners = cv2.goodFeaturesToTrack(img0, maxCorners=120, qualityLevel=.009,
                                      minDistance=8, blockSize=5)
    if corners is None or len(corners)<16:
        return None
    opts = dict(winSize=(17,17),maxLevel=3)
    moved, valid, _ = cv2.calcOpticalFlowPyrLK(img0,img1,corners,None,**opts)
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(img1,img0,moved,None,**opts)
    ok = valid.ravel().astype(bool) & reverse.ravel().astype(bool)
    ok &= np.linalg.norm((corners-back)[:,0],axis=1)<1.4
    if ok.sum()<16:
        return None
    def rays(x):
        return np.c_[x[ok,0],np.ones(ok.sum())] @ np.linalg.inv(k).T
    old = rays(corners) @ from_old_to_new.T
    good = old[:,2]>.2
    old = old[good]/old[good,2,None]
    now = rays(moved)[good]
    flow = (now-old)[:,:2]
    normals = np.cross(now,old)
    lengths = np.linalg.norm(normals,axis=1)
    good = lengths>.002
    if good.sum()<12:
        return None
    normals = normals[good]/lengths[good,None]
    now, flow = now[good], flow[good]
    wt = np.ones(len(normals))
    for _ in range(3):
        _,s,v = np.linalg.svd(normals*np.sqrt(wt[:,None]),full_matrices=False)
        direction = v[-1]
        residual = np.abs(normals@direction)
        wt = np.minimum(1., .08/np.maximum(residual,1e-9))
    expected = now[:,:2]*direction[2]-direction[None,:2]
    sign = (expected*flow).sum(1)
    orientation = np.mean(sign>0)
    if orientation < .5:
        direction = -direction
        orientation = 1.-orientation
    if s[1]<.1*s[0] or np.median(residual)>.15 or orientation<.75:
        return None
    return direction


class CausalBodyHeadingFilter:
    """Fuse valid directions in the same current horizontal frame; define fallback.

    kappa is the length of the directional resultant for explicitly assumed
    von-Mises factors. It is a working confidence, not calibrated to real heads.
    """
    def __init__(self):
        self.last_world = None

    def update(self, gait, optical, world_from_plane=None, stopped=False):
        has_gait = bool(gait and gait["valid"])
        if (stopped or (not has_gait and optical is None)) and self.last_world is not None and world_from_plane is not None:
            return unit(world_from_plane.T@self.last_world), 0.
        summed = np.array([.5,0.])
        if has_gait:
            summed += 3*gait["quality"]*gait["direction"]
        if optical is not None:
            summed += 5*unit(optical)
        norm = np.linalg.norm(summed)
        v = unit(summed)
        if world_from_plane is not None and (optical is not None or (gait and gait["valid"])):
            self.last_world = world_from_plane@v
        return v,float(norm)
