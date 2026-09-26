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

KAPPA, KAPPA_OCC, VOXEL, T_MEM, EPS = 3., 3., .1, 8, .1
SLAB = .1
POOL = False            # also detect with 2x2-zone windows (marks all four zones)


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


def _marking_z(h, pool, b):
    """Strongest unit-noise statistic that marks zone (3,3) occupied at a bin <= b."""
    wins = [h[3:4, 3:4]]+([h[i:i+2, j:j+2] for i in (2, 3) for j in (2, 3)] if pool else [])
    best = -np.inf
    for w in wins:
        s, n = w.sum((0, 1)), w.shape[0]*w.shape[1]
        best = max(best, (s[:b+1]/np.sqrt(n)).max(), ((s[:b+1]+s[1:b+2])/np.sqrt(2*n)).max())
    return best


def reference_worst(gain, rho, side_m=None, fill=.25, pool=True, n_offsets=6,
                    range_offsets=(.1, .35, .6, .85), max_bin=9):
    """Worst-placement reference counts [16] (use width 1: z_cap = counts/sqrt(v_zone)).

    A square reference target (side side_m, or sqrt(fill) of a zone when side_m is None)
    overlaps zone (3,3) anywhere in angle and lies anywhere inside range bin b. For each
    placement the strongest statistic that marks that zone at a bin <= b is taken (1x1
    zone and, with pool, the four 2x2 windows containing it; 1 or 2 bins), assuming
    locally uniform zone noise; counts[b] is the minimum over placements. Bins beyond
    max_bin get 0 (never certified).
    """
    dirs, w = angular_rays(16)
    tx, ty = dirs[..., 0]/dirs[..., 2], dirs[..., 1]/dirs[..., 2]
    edge = np.tan(np.deg2rad(45/2))
    wz = 2*edge/8
    c0 = -edge+3.5*wz
    base = SensorParameters()
    p = replace(base, signal_counts=1., noise_scale=0., neighbour_leak=0.)
    empty = synthesize_response(np.full((8, 8, 256), np.inf), rho, 1., w, params=p, seed=0)['histogram']
    counts = np.zeros(16)
    for b in range(1, max_bin+1):
        worst = np.inf
        for f in range_offsets:
            r = START+(b+f)*WIDTH
            # All target rays share one range, so the response is linear in per-zone
            # solid angle: one leak-free full-zone profile, then the model's zone leak.
            d = np.full((8, 8, 256), np.inf)
            d[3, 3] = r
            prof = (synthesize_response(d, rho, 1., w, params=p, seed=0)['histogram']-empty)[3, 3]
            prof = prof.reshape(16, 8).sum(-1)/w[3, 3].sum()
            t = np.sqrt(fill) if side_m is None else side_m/(r*wz)
            for dx in np.linspace(0, .5+t/2-1/32, n_offsets):
                for dy in np.linspace(0, .5+t/2-1/32, n_offsets):
                    mask = (np.abs(tx-c0-dx*wz) <= t*wz/2) & (np.abs(ty-c0-dy*wz) <= t*wz/2)
                    if not mask[3, 3].any():
                        continue
                    e = (w*mask).sum(-1)
                    old, leak = e.copy(), base.neighbour_leak/4
                    e[1:] += leak*(old[:-1]-old[1:])
                    e[:-1] += leak*(old[1:]-old[:-1])
                    e[:, 1:] += leak*(old[:, :-1]-old[:, 1:])
                    e[:, :-1] += leak*(old[:, 1:]-old[:, :-1])
                    worst = min(worst, _marking_z(e[..., None]*prof, pool, b))
        counts[b] = gain*worst
    return counts, np.ones(16)


def _pooled_detect(r, v):
    """[64,16] zones marked by a 2x2-window detection (1 or 2 bins)."""
    R = r[:-1, :-1]+r[1:, :-1]+r[:-1, 1:]+r[1:, 1:]
    V = v[:-1, :-1]+v[1:, :-1]+v[:-1, 1:]+v[1:, 1:]
    z = R/np.sqrt(np.maximum(V, 1e-9))
    z2 = np.concatenate([(R[..., :-1]+R[..., 1:])/np.sqrt(np.maximum(V[..., :-1]+V[..., 1:], 1e-9)),
                         np.full(R.shape[:2]+(1,), -np.inf)], -1)
    win = (z >= KAPPA_OCC) | (z2 >= KAPPA_OCC) | (np.roll(z2, 1, -1) >= KAPPA_OCC) & (np.arange(16) > 0)
    out = np.zeros((8, 8, 16), bool)
    for i in (0, 1):
        for j in (0, 1):
            out[i:i+7, j:j+7] |= win
    return out.reshape(64, 16)


def cell_states(r, v, ref):
    """[64,16] states for one frame: +1 certified free, -1 occupied/occluded, 0 unknown."""
    counts, width = ref
    z = (r/np.sqrt(np.maximum(v, 1e-9))).reshape(64, 16)
    z2 = np.concatenate([(r[..., :-1]+r[..., 1:]).reshape(64, 15)/np.sqrt(np.maximum(v[..., :-1]+v[..., 1:], 1e-9).reshape(64, 15)),
                         np.full((64, 1), -np.inf)], 1)
    detect = (z >= KAPPA_OCC) | (z2 >= KAPPA_OCC) | (np.roll(z2, 1, 1) >= KAPPA_OCC) & (np.arange(16) > 0)
    if POOL:
        detect = detect | _pooled_detect(r, v)
    first = np.where(detect.any(1), detect.argmax(1), 16)
    zcap = counts[None]/np.sqrt(np.maximum(width[None]*v.reshape(64, 16), 1e-9))
    bins = np.arange(16)[None]
    free = (bins < first[:, None]) & ~detect & (zcap >= KAPPA)
    state = np.zeros((64, 16), int)
    state[free] = 1
    state[detect] = -1
    return state


_DIRS = angular_rays(4)[0].reshape(64, 16, 3)
_DIRS = _DIRS/np.linalg.norm(_DIRS, axis=-1, keepdims=True)
_RADII = START+(np.arange(16)[:, None]+(np.arange(6)+.5)/6)*WIDTH           # [bin, 6 depth samples]
_CELL_SAMPLES = (_DIRS[:, None, :, None, :]*_RADII[None, :, None, :, None]).reshape(64, 16, 96, 3)


def world_voxels(state, world_from_tof):
    """Voxel keys of free (+1) and occupied (-1) cells: 4x4 sub-rays x 6 depths per cell."""
    pts = _transform(_CELL_SAMPLES, world_from_tof)
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
