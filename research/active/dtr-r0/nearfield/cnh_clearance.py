"""Certified clearance distance (Development prototype, fast lane; not a frozen method).

Per frame and query box: D_q = the largest forward depth D in [0.3, 3.0] such that the
box slab Z in [0.3, D) is certified free of a reference-class target. A cell (zone,
bin) is certified free when (i) it lies before the first significant return of its
zone (not occluded), (ii) no significant residual is observed in it, and (iii) the
noise model says a reference target there would have been detected (z_cap >= kappa).
Free and occupied evidence is kept in a gravity-free world voxel memory over the last
T frames (static-world assumption; occupied overrides free). Inputs: H3, public
ambient, calib bias, calibrated gain, poses. Geometry truth is used only to score.
"""
from dataclasses import replace
import numpy as np
from cnh_route_sensor import SensorParameters, synthesize_response, angular_rays
from cnh_track_a_readout import cell_points, _transform, BOXES, START, WIDTH

KAPPA, VOXEL, T_MEM, EPS = 3., .1, 8, .1
SLAB = .1


def reference_counts(gain, rho, fill):
    """Expected reference-target counts per H3 bin centre: best of 1-bin or 2-bin window."""
    _, w = angular_rays(16)
    p = replace(SensorParameters(), signal_counts=gain, noise_scale=0.)
    empty = synthesize_response(np.full((8, 8, 256), np.inf), rho, 1., w, params=p, seed=0)['histogram'][3, 3]
    counts, width = np.zeros(16), np.ones(16)
    for b in range(16):
        r = START+(b+.5)*WIDTH
        d = np.full((8, 8, 256), np.inf)
        d[3, 3, :int(round(256*fill))] = r
        h = (synthesize_response(d, rho, 1., w, params=p, seed=0)['histogram'][3, 3]-empty).reshape(16, 8).sum(-1)
        one, two = h[b], (h[b]+h[b+1] if b < 15 else -np.inf)
        counts[b], width[b] = (one, 1.) if one >= two/np.sqrt(2) else (two, 2.)
    return counts, width


def cell_states(r, v, ref):
    """[64,16] states for one frame: +1 certified free, -1 occupied/occluded, 0 unknown."""
    counts, width = ref
    z = (r/np.sqrt(np.maximum(v, 1e-9))).reshape(64, 16)
    z2 = np.concatenate([(r[..., :-1]+r[..., 1:]).reshape(64, 15)/np.sqrt(np.maximum(v[..., :-1]+v[..., 1:], 1e-9).reshape(64, 15)),
                         np.full((64, 1), -np.inf)], 1)
    detect = (z >= KAPPA) | (z2 >= KAPPA) | (np.roll(z2, 1, 1) >= KAPPA) & (np.arange(16) > 0)
    first = np.where(detect.any(1), detect.argmax(1), 16)
    zcap = counts[None]/np.sqrt(np.maximum(width[None]*v.reshape(64, 16), 1e-9))
    bins = np.arange(16)[None]
    free = (bins < first[:, None]) & ~detect & (zcap >= KAPPA)
    state = np.zeros((64, 16), int)
    state[free] = 1
    state[detect] = -1
    return state


def world_voxels(state, world_from_tof):
    """Voxel keys of free (+1) and occupied (-1) cells from 4x4 sub-rays at bin centres."""
    pts = _transform(cell_points(), world_from_tof).reshape(64, 16, 16, 3)
    out = {}
    for s in (1, -1):
        sel = state == s
        if sel.any():
            out[s] = np.unique(np.floor(pts[sel].reshape(-1, 3)/VOXEL).astype(np.int64), axis=0)
    return out


def box_samples(step=.05):
    out = []
    for lo, hi in BOXES:
        xs, ys, zs = (np.arange(lo[k]+step/2, hi[k], step) for k in range(3))
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        out.append(np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1))
    return out


BOX_POINTS = box_samples()


def clearance_sequence(hist, ambient, bias, poses_tof, poses_q, ref):
    """[N,6] certified clearance distances (m) for one sequence, causal."""
    r = hist-bias
    v = 16*ambient[..., None]+np.maximum(bias, 0)
    memory = []
    out = np.full((len(r), 6), .3)
    for i in range(len(r)):
        memory.append(world_voxels(cell_states(r[i], v[i], ref), poses_tof[i]))
        memory = memory[-T_MEM:]
        free = {tuple(k) for m in memory for k in m.get(1, [])}
        occ = {tuple(k) for m in memory for k in m.get(-1, [])}
        free -= occ
        for q, pts in enumerate(BOX_POINTS):
            world = _transform(pts, poses_q[i])
            keys = np.floor(world/VOXEL).astype(np.int64)
            ok = np.fromiter((tuple(k) in free for k in keys), bool, len(keys))
            d = .3
            for z0 in np.arange(.3, 3.0, SLAB):
                sl = (pts[:, 2] >= z0) & (pts[:, 2] < z0+SLAB)
                if not sl.any() or ok[sl].mean() < 1-EPS:
                    break
                d = z0+SLAB
            out[i, q] = d
    return out


def score(clear, witness, labels, main, strata=None):
    """False-clear rate and utility. witness: nearest in-box contributor Z (NaN if none)."""
    m = main[:, None].repeat(6, 1)
    pos = (labels == 1) & m & np.isfinite(witness)
    neg = (labels == 0) & m
    false_clear = pos & (witness < clear)
    res = dict(positives=int(pos.sum()), negatives=int(neg.sum()), false_clear=int(false_clear.sum()),
               FCR=float(false_clear.sum()/max(1, pos.sum())),
               median_clear_neg=float(np.median(clear[neg])) if neg.any() else None,
               unknown_neg=float(np.mean(clear[neg] <= .3)) if neg.any() else None,
               median_gap_pos=float(np.median((witness-clear)[pos])) if pos.any() else None)
    if strata is not None:
        res['FCR_by_stratum'] = {s: float((false_clear & (strata == s)).sum()/max(1, (pos & (strata == s)).sum()))
                                 for s in ('tiny', 'realistic', 'wide')}
    return res
