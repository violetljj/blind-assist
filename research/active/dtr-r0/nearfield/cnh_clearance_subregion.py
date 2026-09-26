"""Certifiability map and per-subregion certified clearance (Development, fast lane).

Extends cnh_clearance: every query box is split into three vertical bands (upper,
middle, lower). For each frame the readout keeps, per band and 0.1 m forward slab,
the fraction of the band volume certified free by the voxel memory; the band's
clearance D is the largest depth whose slabs are all >= 1-EPS certified. A band
that certifies nothing (D = 0.3) is UNKNOWN, not clear. Geometry truth (object
triangles clipped to each band) is used only for scoring. Not a frozen method.

Rule-fix outputs (2026-09-26 v2): per band slab also whether any occupied voxel lies
in it (occ) and whether a k x k block of 5 cm (x, y) columns that are not fully
certified could hide a square reference target (hole{k}, k = 1, 2, 4; box outer
edges padded as uncertified). Optional worst-placement reference and 2x2 pooling.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import cnh_clearance as cc
from cnh_track_a_readout import BOXES, noisy_poses, _transform
import cnh_track_a_scale_evaluate as se
import cnh_track_a_v13_sensor as sensor_module

SLABS = np.round(np.arange(.3, 3.0, cc.SLAB), 3)
BANDS = ('upper', 'middle', 'lower')          # y points down: upper band = smallest y


def band_bounds():
    out = []
    for lo, hi in BOXES:
        edges = np.linspace(lo[1], hi[1], 4)
        out.append([(np.array([lo[0], edges[b], lo[2]]), np.array([hi[0], edges[b+1], hi[2]])) for b in range(3)])
    return out


BAND_BOUNDS = band_bounds()


HOLE_K = (1, 2, 4)


def box_grids(step=.05):
    """Per box: sample points in meshgrid order, grid shape, row indices of each band."""
    out = []
    for lo, hi in BOXES:
        xs, ys, zs = (np.arange(lo[k]+step/2, hi[k], step) for k in range(3))
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        slab = np.clip(((zs-.3)/cc.SLAB+1e-9).astype(int), 0, len(SLABS)-1)
        assert len(zs) == 2*len(SLABS) and (slab == np.repeat(np.arange(len(SLABS)), 2)).all()
        edges = np.linspace(lo[1], hi[1], 4)
        rows = [np.flatnonzero((ys >= edges[b]) & (ys < edges[b+1])) for b in range(3)]
        out.append((np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1), (len(xs), len(ys), len(SLABS), 2), rows))
    return out


BOX_GRIDS = box_grids()


def hole_flags(nc, rows, k):
    """[3, S] per band: some k x k block of uncertified columns overlaps the band."""
    if k > 1:
        nc = np.pad(nc, ((k-1, k-1), (k-1, k-1), (0, 0)), constant_values=True)
    c = np.pad(nc.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0)))
    full = (c[k:, k:]-c[:-k, k:]-c[k:, :-k]+c[:-k, :-k]) == k*k      # window (i, j) covers rows j-k+1..j
    per_row = full.any(0)
    out = np.zeros((3, nc.shape[-1]), bool)
    for b, r in enumerate(rows):
        out[b] = per_row[r.min():r.max()+k].any(0)
    return out


def encode(keys):
    k = keys.astype(np.int64)+(1 << 20)
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def certified_fractions(hist, ambient, bias, poses_tof, poses_q, ref):
    """Causal voxel memory readout. frac [N,6,3,S]: certified fraction of each band slab;
    occ and hole{k} (same shape, bool): occupied voxel / hideable uncertified block."""
    r = hist-bias
    v = 16*ambient[..., None]+np.maximum(bias, 0)
    memory, n, shape4 = [], len(r), (len(r), 6, 3, len(SLABS))
    out = {'frac': np.zeros(shape4), 'occ': np.zeros(shape4, bool)} | {f'hole{k}': np.zeros(shape4, bool) for k in HOLE_K}
    for i in range(n):
        vox = cc.world_voxels(cc.cell_states(r[i], v[i], ref), poses_tof[i])
        memory.append({s: encode(k) for s, k in vox.items()})
        memory = memory[-cc.T_MEM:]
        free = np.unique(np.concatenate([m[1] for m in memory if 1 in m] or [np.zeros(0, np.int64)]))
        occ = np.unique(np.concatenate([m[-1] for m in memory if -1 in m] or [np.zeros(0, np.int64)]))
        free = np.setdiff1d(free, occ, assume_unique=True)
        for q, (pts, shape, rows) in enumerate(BOX_GRIDS):
            keys = encode(np.floor(_transform(pts, poses_q[i])/cc.VOXEL))
            ok = np.isin(keys, free).reshape(shape)
            oc = np.isin(keys, occ).reshape(shape)
            for b, rw in enumerate(rows):
                out['frac'][i, q, b] = ok[:, rw].sum((0, 1, 3))/(shape[0]*len(rw)*2)
                out['occ'][i, q, b] = oc[:, rw].any((0, 1, 3))
            nc = ~ok.all(-1)
            for k in HOLE_K:
                out[f'hole{k}'][i, q] = hole_flags(nc, rows, k)
    return out


def clearance_from_fractions(frac, eps=cc.EPS):
    """Largest D with all slabs in [0.3, D) certified >= 1-eps; 0.3 means UNKNOWN."""
    good = frac >= 1-eps
    first_bad = np.where(~good.all(-1), (~good).argmax(-1), good.shape[-1])
    return .3+first_bad*cc.SLAB


def _clip(poly, axis, bound, keep_greater):
    out = []
    n = len(poly)
    for k in range(n):
        a, b = poly[k], poly[(k+1) % n]
        ina = a[axis] >= bound if keep_greater else a[axis] <= bound
        inb = b[axis] >= bound if keep_greater else b[axis] <= bound
        if ina:
            out.append(a)
        if ina != inb:
            t = (bound-a[axis])/(b[axis]-a[axis])
            out.append(a+t*(b-a))
    return out


def min_z_in_box(tris, lo, hi):
    """Minimum forward z of triangle surfaces inside the AABB [lo, hi] (NaN if none)."""
    tmin, tmax = tris.min(1), tris.max(1)
    cand = np.flatnonzero(((tmax >= lo) & (tmin <= hi)).all(1))
    best = np.inf
    for t in cand:
        poly = list(tris[t])
        for axis in range(3):
            poly = _clip(poly, axis, lo[axis], True)
            if poly:
                poly = _clip(poly, axis, hi[axis], False)
            if not poly:
                break
        if poly:
            best = min(best, min(p[2] for p in poly))
    return np.nan if not np.isfinite(best) else best


def band_truth(config, frames_q):
    """Per frame, box, band: nearest in-band surface z and its object (reference test applied later)."""
    objs = [o for o in config['objects'] if o.get('category') not in ('LOW',) and o['id'] != 100]
    z = np.full((len(frames_q), 6, 3), np.nan)
    who = np.full((len(frames_q), 6, 3), -1)
    for i, wq in enumerate(frames_q):
        q_from_world = np.linalg.inv(wq)
        for o in objs:
            tris = np.asarray(o['triangles_world'], float)
            tq = _transform(tris.reshape(-1, 3), q_from_world).reshape(-1, 3, 3)
            for q in range(6):
                for b, (lo, hi) in enumerate(BAND_BOUNDS[q]):
                    w = min_z_in_box(tq, lo, hi)
                    if np.isfinite(w) and not (w >= z[i, q, b]):
                        z[i, q, b], who[i, q, b] = w, o['id']
    return z, who


def reference_mask(config, who, z, rho_min=.5, fill=.25):
    """Reference class: rho >= rho_min and cross-section >= fill of one zone at that distance."""
    zone = np.tan(np.deg2rad(45/8))
    info = {o['id']: o for o in config['objects']}
    out = np.zeros(who.shape, bool)
    for idx in zip(*np.nonzero(who >= 0)):
        o = info[int(who[idx])]
        ext = o.get('extent')
        area = np.inf if ext is None else np.prod(sorted(ext)[1:])
        out[idx] = o.get('rho', .5) >= rho_min and area >= fill*(z[idx]*zone)**2
    return out


REF_SIDES = (.05, .10, .20)


def reference_size_mask(config, who, side):
    """Size reference class: rho >= 0.5 and a side x side square fits in the largest face."""
    info = {o['id']: o for o in config['objects']}
    out = np.zeros(who.shape, bool)
    for idx in zip(*np.nonzero(who >= 0)):
        o = info[int(who[idx])]
        ext = o.get('extent')
        out[idx] = o.get('rho', .5) >= .5 and (ext is None or sorted(ext)[1] >= side)
    return out


def unit_truth(geometry, sensor, unit, records, cache=None):
    """Band truth and scoring labels for one unit (independent of motion and readout; cached)."""
    target = None if cache is None else cache/f'unit{unit}.npz'
    if target is not None and target.exists():
        return dict(np.load(target))
    data = json.loads((geometry/f'unit{unit:02d}'/f'unit{unit:02d}.json').read_text(encoding='utf-8-sig'))
    configs = {c['config']: c for c in data['configs']}
    with np.load(sensor/f'unit{unit:02d}-mount-10-observations.npz') as f:
        obs_config = f['config']
    rows = {}
    for rec in records:
        c = configs[rec['config']]
        wq = np.asarray(c['world_from_Q_10hz'])[::2] if 'world_from_Q_10hz' in c else np.asarray(c['world_from_Q'])
        z, who = band_truth(c, wq)
        sizes = np.array([[[c['size_classes'].get(str(k), 'background') if k >= 0 else 'none' for k in row] for row in f]
                          for f in who])
        item = dict(truth=z, who=who, ref=reference_mask(c, who, z), witness=rec['witness'], labels=rec['labels'],
                    main=rec['main'], size=sizes, config=np.full(len(z), rec['config']),
                    obs_index=np.flatnonzero(obs_config == rec['config']))
        item |= {f'ref_s{int(round(100*s)):02d}': reference_size_mask(c, who, s) for s in REF_SIDES}
        assert len(item['obs_index']) == len(z)
        for k, v in item.items():
            rows.setdefault(k, []).append(v)
    out = {k: np.concatenate(v) for k, v in rows.items()}
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **out)
    return out


def run_unit(geometry, sensor, unit, bias, ref, motion, snr_index=1, truth_cache=None):
    split, records, _ = se.unit_records(geometry, sensor, unit, -10, snr_index)
    rows = {}
    for rec in records:
        poses = rec['poses'] if motion == 'GT' else noisy_poses(rec['poses'], rec['ego_seed'], dt=.2)
        pq = np.array([p @ np.linalg.inv(t) for p, t in zip(poses, rec['tq'])])
        for k, v in certified_fractions(rec['hist'], rec['ambient'], bias, poses, pq, ref).items():
            rows.setdefault(k, []).append(v)
    return {k: np.concatenate(v) for k, v in rows.items()} | unit_truth(geometry, sensor, unit, records, truth_cache)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--sensor', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--calib', type=int, nargs='+', required=True)
    p.add_argument('--units', type=int, nargs='+', required=True)
    p.add_argument('--family', required=True)
    p.add_argument('--kappa-occ', type=float, default=cc.KAPPA_OCC)
    p.add_argument('--kappa-cap', type=float, default=cc.KAPPA, help='certify only if expected reference z >= this')
    p.add_argument('--motion', choices=('GT', 'noisy'), default='noisy')
    p.add_argument('--snr', type=int, choices=(3, 6, 12), default=6)
    p.add_argument('--placement', choices=('centred', 'worst'), default='centred',
                   help='reference response: centred in one zone (original) or worst placement')
    p.add_argument('--ref-size', type=float, default=None, help='square reference side (m); default 25%% of a zone')
    p.add_argument('--pool', action='store_true', help='also detect with 2x2-zone windows')
    p.add_argument('--truth-cache', type=Path, default=None)
    a = p.parse_args()
    sensor_module.FAMILY = a.family
    cc.KAPPA_OCC, cc.KAPPA, cc.POOL = a.kappa_occ, a.kappa_cap, a.pool
    a.out.mkdir(parents=True, exist_ok=True)
    params = json.loads((a.sensor/f'unit{a.units[0]:02d}-mount-10.json').read_text(encoding='utf-8'))['parameters'][str(a.snr)]
    si = (3, 6, 12).index(a.snr)
    if a.placement == 'worst':
        ref = cc.reference_worst(params['signal_counts'], .5, side_m=a.ref_size, pool=a.pool)
    else:
        assert a.ref_size is None, 'centred placement only models the 25%-zone class'
        ref = cc.reference_counts(params['signal_counts'], .5, .25)
    bias = np.median(np.concatenate([r['hist'] for u in a.calib for r in se.unit_records(a.geometry, a.sensor, u, -10, si)[1]]), 0)
    for u in a.units:
        target = a.out/f'unit{u}.npz'
        if not target.exists():
            np.savez_compressed(target, **run_unit(a.geometry, a.sensor, u, bias, ref, a.motion, si, a.truth_cache))
        print(u, flush=True)


if __name__ == '__main__':
    main()
