"""Certifiability map and per-subregion certified clearance (Development, fast lane).

Extends cnh_clearance: every query box is split into three vertical bands (upper,
middle, lower). For each frame the readout keeps, per band and 0.1 m forward slab,
the fraction of the band volume certified free by the voxel memory; the band's
clearance D is the largest depth whose slabs are all >= 1-EPS certified. A band
that certifies nothing (D = 0.3) is UNKNOWN, not clear. Geometry truth (object
triangles clipped to each band) is used only for scoring. Not a frozen method.
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


def band_points(step=.05):
    """Sample points per (box, band) and their slab index."""
    pts, slab = [], []
    for q, (lo, hi) in enumerate(BOXES):
        xs, ys, zs = (np.arange(lo[k]+step/2, hi[k], step) for k in range(3))
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        p = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        edges = np.linspace(lo[1], hi[1], 4)
        for b in range(3):
            sel = (p[:, 1] >= edges[b]) & (p[:, 1] < edges[b+1])
            pts.append(p[sel])
            slab.append(np.clip(((p[sel, 2]-.3)/cc.SLAB+1e-9).astype(int), 0, len(SLABS)-1))
    return pts, slab


BAND_POINTS, BAND_SLAB = band_points()


def encode(keys):
    k = keys.astype(np.int64)+(1 << 20)
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def certified_fractions(hist, ambient, bias, poses_tof, poses_q, ref):
    """[N, 6, 3, S] fraction of each band slab certified free (causal voxel memory)."""
    r = hist-bias
    v = 16*ambient[..., None]+np.maximum(bias, 0)
    memory, out = [], np.zeros((len(r), 6, 3, len(SLABS)))
    for i in range(len(r)):
        vox = cc.world_voxels(cc.cell_states(r[i], v[i], ref), poses_tof[i])
        memory.append({s: encode(k) for s, k in vox.items()})
        memory = memory[-cc.T_MEM:]
        free = np.unique(np.concatenate([m[1] for m in memory if 1 in m] or [np.zeros(0, np.int64)]))
        occ = np.unique(np.concatenate([m[-1] for m in memory if -1 in m] or [np.zeros(0, np.int64)]))
        free = np.setdiff1d(free, occ, assume_unique=True)
        for j, (pts, slab) in enumerate(zip(BAND_POINTS, BAND_SLAB)):
            ok = np.isin(encode(np.floor(_transform(pts, poses_q[i])/cc.VOXEL)), free)
            out[i, j//3, j % 3] = np.bincount(slab, ok, len(SLABS))/np.maximum(np.bincount(slab, minlength=len(SLABS)), 1)
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


def run_unit(geometry, sensor, unit, bias, ref, motion, snr_index=1):
    split, records, _ = se.unit_records(geometry, sensor, unit, -10, snr_index)
    data = json.loads((geometry/f'unit{unit:02d}'/f'unit{unit:02d}.json').read_text(encoding='utf-8-sig'))
    configs = {c['config']: c for c in data['configs']}
    rows = dict(frac=[], truth=[], ref=[], witness=[], labels=[], main=[], size=[], config=[])
    for rec in records:
        c = configs[rec['config']]
        poses = rec['poses'] if motion == 'GT' else noisy_poses(rec['poses'], rec['ego_seed'], dt=.2)
        pq = np.array([p @ np.linalg.inv(t) for p, t in zip(poses, rec['tq'])])
        rows['frac'].append(certified_fractions(rec['hist'], rec['ambient'], bias, poses, pq, ref))
        wq = np.asarray(c['world_from_Q_10hz'])[::2] if 'world_from_Q_10hz' in c else np.asarray(c['world_from_Q'])
        z, who = band_truth(c, wq)
        rows['truth'].append(z)
        rows['ref'].append(reference_mask(c, who, z))
        rows['witness'].append(rec['witness'])
        rows['labels'].append(rec['labels'])
        rows['main'].append(rec['main'])
        sizes = np.array([[[c['size_classes'].get(str(k), 'background') if k >= 0 else 'none' for k in row] for row in f]
                          for f in who])
        rows['size'].append(sizes)
        rows['config'].append(np.full(len(z), rec['config']))
    return {k: np.concatenate(v) for k, v in rows.items()}


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
    a = p.parse_args()
    sensor_module.FAMILY = a.family
    cc.KAPPA_OCC, cc.KAPPA = a.kappa_occ, a.kappa_cap
    a.out.mkdir(parents=True, exist_ok=True)
    params = json.loads((a.sensor/f'unit{a.units[0]:02d}-mount-10.json').read_text(encoding='utf-8'))['parameters'][str(a.snr)]
    si = (3, 6, 12).index(a.snr)
    ref = cc.reference_counts(params['signal_counts'], .5, .25)
    bias = np.median(np.concatenate([r['hist'] for u in a.calib for r in se.unit_records(a.geometry, a.sensor, u, -10, si)[1]]), 0)
    for u in a.units:
        target = a.out/f'unit{u}.npz'
        if not target.exists():
            np.savez_compressed(target, **run_unit(a.geometry, a.sensor, u, bias, ref, a.motion, si))
        print(u, flush=True)


if __name__ == '__main__':
    main()
