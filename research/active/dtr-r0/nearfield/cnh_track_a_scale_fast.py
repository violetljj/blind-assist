"""Faster, bitwise-identical scoring for the frozen scale readouts (engineering only).

Same arithmetic as cnh_track_a_scale_evaluate.score_unit, reorganized:
- query-support windows are built once per frame and tau and shared by S2 (GT,
  noisy) and S2r4 instead of being rebuilt inside every scan call;
- S2r4 transport matrices are built once per frame pair instead of once per tau;
- the memory scan accumulates per-voxel sums with np.add.at in the same source
  order as the original dictionary loop (0.0 + t_j1 + t_j2 ...), so results are
  bitwise equal;
- all (condition, unit) jobs share one process pool (load balancing).
Methods, parameters, selections and statistics are unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from cnh_track_a_readout import (query_weights, noisy_poses, cell_points, _transform, _poses, EDGE, BOXES)
from cnh_track_a_v12_readout import accumulate, s1, supported_windows
from cnh_track_a_v13_head_diag import transport
from cnh_track_a_v13_evaluate import observability, TAUS
import cnh_track_a_scale_evaluate as se
import cnh_track_a_v13_sensor as sensor_module


def scan_windows(r, v, sources, windows):
    """cnh_track_a_v12_readout.scan_transported with prebuilt window selectors."""
    output = np.full(6, -50.)
    for q, u in enumerate(windows):
        if not u.shape[0]:
            continue
        total, var = np.zeros(u.shape[0]), np.zeros(u.shape[0])
        for j, a in sources:
            c = u @ a
            c.sum_duplicates()
            total += c @ r[j].ravel()
            var += c.multiply(c) @ v[j].ravel()
        output[q] = max(-50., float(np.max(total/np.sqrt(np.maximum(var, 1e-9)))))
    return output


def windows_all(weights):
    """Selectors for tau .25 plus row indices of the (nested) .5/.75 windows.

    Support grows as tau falls, so every window admissible at .5/.75 is admissible
    at .25; rows keep supported_windows' order (window shape, then C-order start).
    """
    from cnh_scan_development import WINDOWS
    support = {t: np.asarray(weights).reshape(6, 8, 8, 16) >= t for t in TAUS}
    ids = np.arange(1024).reshape(8, 8, 16)
    selectors, rows = [], {t: [] for t in TAUS}
    for q in range(6):
        rr, cc, count, sel = [], [], 0, {t: [] for t in TAUS}
        for h, w, d in WINDOWS:
            shape = (h, w, d)
            adm = {t: np.lib.stride_tricks.sliding_window_view(support[t][q], shape).all((-3, -2, -1)) for t in TAUS}
            base = adm[TAUS[0]]
            cells = np.lib.stride_tricks.sliding_window_view(ids, shape)[base].reshape(-1, h*w*d)
            rr.append(np.repeat(np.arange(count, count+len(cells)), h*w*d))
            cc.append(cells.ravel())
            for t in TAUS:
                sel[t].append(count+np.flatnonzero(adm[t][base]))
            count += len(cells)
        row, col = np.concatenate(rr), np.concatenate(cc)
        selectors.append(sparse.csr_matrix((np.ones(len(col)), (row, col)), shape=(count, 1024)))
        for t in TAUS:
            rows[t].append(np.concatenate(sel[t]))
    return selectors, rows


def scan_multi(r, v, sources, selectors, rows):
    """scan_transported for all three taus from one set of sparse products."""
    out = {t: np.full(6, -50.) for t in TAUS}
    for q, u in enumerate(selectors):
        if not u.shape[0]:
            continue
        coeffs = []
        for j, a in sources:
            c = u @ a
            c.sum_duplicates()
            coeffs.append((j, c))
        for t in TAUS:
            idx = rows[t][q]
            if not len(idx):
                continue
            total, var = np.zeros(len(idx)), np.zeros(len(idx))
            for j, c in coeffs:
                ct = c[idx]
                total += ct @ r[j].ravel()
                var += ct.multiply(ct) @ v[j].ravel()
            out[t][q] = max(-50., float(np.max(total/np.sqrt(np.maximum(var, 1e-9)))))
    return out


def memory_scan_fast(r, v, p, i, tq):
    """cnh_track_a_v12_readout.memory_scan (k=8) with vectorized voxel accumulation."""
    tq = _poses(np.asarray(tq)[None], 1)[0]
    original = cell_points()
    old_radius = np.linalg.norm(original, axis=-1)
    source = np.broadcast_to(np.arange(1024)[:, None], (1024, 16))
    parts = [([], [], []) for _ in range(6)]
    for j in range(max(0, i-7), i+1):
        current = _transform(original, np.linalg.inv(p[i]) @ p[j])
        radius = np.linalg.norm(current, axis=-1)
        front = current[..., 2] > 0
        tangent = np.divide(current[..., :2], current[..., 2, None],
                            out=np.full((1024, 16, 2), np.inf), where=front[..., None])
        outside = ~(front & (np.abs(tangent) < EDGE).all(-1))
        point_q = _transform(current, tq)
        gain = np.ones_like(radius)
        if j != i:
            np.divide(old_radius, radius, out=gain, where=radius > 0)
            gain *= gain
        for q, (lo, hi) in enumerate(BOXES):
            keep = outside & (radius > 0) & (point_q >= lo).all(-1) & (point_q <= hi).all(-1)
            if not keep.any():
                continue
            voxels = np.floor(point_q[keep]/.2).astype(np.int64)
            unique, inverse = np.unique(voxels, axis=0, return_inverse=True)
            a = sparse.coo_matrix((gain[keep]/16, (inverse.ravel(), source[keep])), shape=(len(unique), 1024)).tocsr()
            a.sum_duplicates()
            parts[q][0].append(unique)
            parts[q][1].append(a @ r[j].ravel())
            parts[q][2].append(a.multiply(a) @ v[j].ravel())
    result = np.full(6, -50.)
    for q, (keys, totals, variances) in enumerate(parts):
        if not keys:
            continue
        uniq, inv = np.unique(np.concatenate(keys), axis=0, return_inverse=True)
        acc_t, acc_v = np.zeros(len(uniq)), np.zeros(len(uniq))
        np.add.at(acc_t, inv.ravel(), np.concatenate(totals))
        np.add.at(acc_v, inv.ravel(), np.concatenate(variances))
        result[q] = max(-50., float(np.max(acc_t/np.sqrt(np.maximum(acc_v, 1e-9)))))
    return result


def score_unit_fast(job):
    geometry, sensor, output, unit, mount, snr_index, family, with_r4 = job
    sensor_module.FAMILY = family
    target = Path(output)/f'unit{unit:02d}.npz'
    if target.exists():
        return unit
    bias = np.load(Path(output)/'bias.npy')
    split, records, step = se.unit_records(Path(geometry), Path(sensor), unit, mount, snr_index)
    scores = {}
    identity = sparse.eye(1024, format='csr')
    for rec in records:
        r = rec['hist']-bias
        v = 16*rec['ambient'][..., None]+np.maximum(bias, 0)
        weights = [query_weights(t) for t in rec['tq']]
        packs = [windows_all(w) for w in weights]
        out = {'B0': np.array([np.einsum('zb,qzb->q', y.reshape(64, 16), w) for y, w in zip(rec['hist'], weights)])}
        for tau in TAUS:
            out[f'S1|{tau}'] = np.array([s1(r[i:i+1], v[i:i+1], w, tau)[0] for i, w in enumerate(weights)])
            out[f'S1cell|{tau}'] = np.array([se.s1_cell(r[i], v[i], w, tau) for i, w in enumerate(weights)])
        for motion, p in (('GT', rec['poses']), ('noisy', noisy_poses(rec['poses'], rec['ego_seed'], dt=.2))):
            acc = accumulate(r, v, p, 4)
            out[f'B1-R/{motion}'] = np.array([np.einsum('zb,qzb->q', m.reshape(64, 16), w) for m, w in zip(acc['mean'], weights)])
            multi = [None]+[scan_multi(r, v, acc['sources'][i], *packs[i]) for i in range(1, len(r))]
            for tau in TAUS:
                out[f'S2/{motion}|{tau}'] = np.array([out[f'S1|{tau}'][0] if i == 0 else multi[i][tau]
                                                      for i in range(len(r))])
            pp = _poses(p, len(r))
            out[f'memory/{motion}'] = np.array([memory_scan_fast(r, v, pp, i, rec['tq'][i]) for i in range(len(r))])
            if with_r4 and motion == 'noisy':
                src = [None]+[[(i, identity)]+[(j, transport(np.linalg.inv(pp[i]) @ pp[j], -2)) for j in range(max(0, i-3), i)]
                              for i in range(1, len(r))]
                multi4 = [None]+[scan_multi(r, v, src[i], *packs[i]) for i in range(1, len(r))]
                for tau in TAUS:
                    out[f'S2r4/noisy|{tau}'] = np.array([out[f'S1|{tau}'][0] if i == 0 else multi4[i][tau]
                                                         for i in range(len(r))])
        for key, value in out.items():
            scores.setdefault(key, []).append(value)
    arrays = {k.replace('/', '__').replace('|', '@'): np.concatenate(v) for k, v in scores.items()}
    extra = dict(labels=np.concatenate([r['labels'] for r in records]), main=np.concatenate([r['main'] for r in records]),
                 boundary=np.concatenate([r['boundary'] for r in records]),
                 witness=np.concatenate([r['witness'] for r in records]), strata=se.strata_of(records),
                 config=np.repeat([r['config'] for r in records], 12), frame=np.tile(np.arange(12), len(records)),
                 split=np.array(split))
    if split == 'audit':
        extra['visibility'] = observability(Path(sensor), records, mount, 5, step=step)
    np.savez_compressed(target, **arrays, **extra)
    return unit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--sensor', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--family', required=True)
    p.add_argument('--split-counts', type=int, nargs=3, required=True)
    p.add_argument('--conditions', nargs='+', required=True, help='name:mount:snr, all with primary statistics')
    p.add_argument('--workers', type=int, default=17)
    a = p.parse_args()
    t0 = time.monotonic()
    ntr, nca, nau = a.split_counts
    n = ntr+nca+nau
    conds = []
    for spec in a.conditions:
        name, mount, snr = spec.split(':')
        out = a.root/name
        out.mkdir(parents=True, exist_ok=True)
        snr_index = (3, 6, 12).index(int(snr))
        if not (out/'bias.npy').exists():
            calib = []
            for u in range(ntr, ntr+nca):
                if (a.geometry/f'unit{u:02d}'/f'unit{u:02d}.json').exists():
                    calib.extend(r['hist'] for r in se.unit_records(a.geometry, a.sensor, u, int(mount), snr_index)[1])
            np.save(out/'bias.npy', np.median(np.concatenate(calib), axis=0))
        conds.append((name, int(mount), snr_index, out))
    jobs = [(str(a.geometry), str(a.sensor), str(out), u, mount, si, a.family, True)
            for _, mount, si, out in conds for u in range(ntr, n)
            if (a.geometry/f'unit{u:02d}'/f'unit{u:02d}.json').exists()]
    # Interleave conditions so every condition progresses together.
    jobs.sort(key=lambda j: (j[3], j[2]))
    with ProcessPoolExecutor(a.workers) as pool:
        for _ in pool.map(score_unit_fast, jobs):
            pass
    for name, mount, si, out in conds:
        report = se.analyze(out, n, dict(train=ntr, calib=nca, audit=nau), True)
        report['wall_s_pipeline'] = time.monotonic()-t0
        report['implementation'] = 'cnh_track_a_scale_fast (bitwise-identical scoring, shared pool)'
        (out/'result.json').write_text(json.dumps(report, indent=1, allow_nan=False)+'\n', encoding='utf-8')
        print(name, json.dumps({arm: {g: round(v[g]['macro_AP'], 4) for g in ('HEAD', 'BODY')} for arm, v in report['arms'].items()}))


if __name__ == '__main__':
    main()
