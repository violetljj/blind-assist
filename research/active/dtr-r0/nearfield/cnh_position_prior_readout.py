"""Consumed-v3 position-prior mechanism diagnostic; never a camera result.

All arms reuse S2's four-frame transported numerator, full window covariance,
tau=.75 query support and seven window shapes. Only allowed windows change.
G1/G3 masks are query-independent first-hit non-BACKGROUND oracle footprints;
G2 additionally intersects the per-ray range bin +/-1 (privileged reference,
not a physical upper bound). G4 uses no oracle: all query-admissible S2 windows
with z>=2.5 in frames t-1,t-2, transported by the same noisy poses. It keeps
transported angular cells, discards their depth, then uses the G1 intersection
rule. This tests causal search restriction, not independent extra energy.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np

FAMILY = 'cnh-track-a-scale-v3-20260926'
SEED = 20260927
ARMS = ('G0', 'G1', 'G2', 'G3a', 'G3b', 'G3c', 'G3d', 'G4local')


def footprint_pool(geometry, sensor, units):
    """Even-unit angular templates only, no target labels or evaluation outcomes."""
    pool = []
    for unit in units:
        if unit % 2:
            continue
        data = json.loads((geometry/f'unit{unit:02d}'/f'unit{unit:02d}.json').read_text(encoding='utf-8-sig'))
        with np.load(sensor/f'unit{unit:02d}-mount-10-observations.npz') as f:
            configs = f['config']
        with np.load(sensor/f'unit{unit:02d}-mount-10-oracle.npz') as f:
            oid = f['object_id']
        for c in data['configs']:
            for obj in c['objects']:
                if obj['category'] == 'BACKGROUND':
                    continue
                for row in np.flatnonzero(configs == c['config']):
                    xy = np.argwhere((oid[row] == obj['id']).any(-1))
                    if len(xy):
                        pool.append(xy-xy.min(0))
    if not pool:
        raise ValueError('No angular templates')
    return pool


def oracle_masks(oid, distance, objects, unit, config, pool):
    from cnh_route_sensor import angular_rays
    from cnh_track_a_readout import START, WIDTH, EDGE
    rays, _ = angular_rays(16)
    xy = (rays[..., :2]/rays[..., 2:]+EDGE)/(2*EDGE)*8
    n = len(oid)
    masks = {a: np.zeros((n, 8, 8, 16), bool) for a in ARMS[1:-1]}
    ids = [o['id'] for o in objects if o['category'] != 'BACKGROUND']
    counts = []
    for t in range(n):
        visible = 0
        for ident in ids:
            hit = oid[t] == ident
            if not hit.any():
                continue
            visible += 1
            zone = hit.any(-1)
            masks['G1'][t] |= zone[..., None]
            rng = np.random.default_rng(np.random.SeedSequence([SEED, unit, config, t, int(ident)]))
            angle = rng.uniform(0, 2*np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])
            for arm, magnitude in (('G3a', .5), ('G3b', 1.)):
                shifted = np.floor(xy[hit]+magnitude*direction).astype(int)
                valid = ((shifted >= 0) & (shifted < 8)).all(1)
                col, row = shifted[valid].T
                masks[arm][t, row, col, :] = True
            if rng.random() >= .2:
                masks['G3c'][t] |= zone[..., None]
            row, col, ray = np.where(hit)
            bins = np.floor((distance[t][hit]-START)/WIDTH).astype(int)
            for offset in (-1, 0, 1):
                b = bins+offset
                good = (b >= 0) & (b < 16)
                masks['G2'][t, row[good], col[good], b[good]] = True
        masks['G3d'][t] = masks['G1'][t]
        rng = np.random.default_rng(np.random.SeedSequence([SEED, unit, config, t, 99999]))
        template = pool[int(rng.integers(len(pool)))]
        # Translate a sampled real footprint without clipping its size.
        origin = np.array([rng.integers(8-template[:, k].max()) for k in range(2)])
        rc = template+origin
        masks['G3d'][t, rc[:, 0], rc[:, 1], :] = True
        counts.append(visible)
    return masks, counts


def sequence_scores(hist, ambient, bias, tq, noisy, masks):
    import torch
    import torch.nn.functional as F
    import cnh_track_a_gpu_readout as g
    from cnh_scan_development import WINDOWS
    n = len(hist)
    r = g.T(hist)-g.T(bias)
    v = 16*g.T(ambient)[..., None]+g.T(bias).clamp_min(0)
    rf, vf = r.reshape(n, 1024), v.reshape(n, 1024)
    vv = v.reshape(n, 8, 8, 16)
    support = (g.query_weights(torch.as_tensor(tq, dtype=g.D64, device=g.DEV)) >= .75).reshape(n, 6, 8, 8, 16)
    pairs = [(i, j) for i in range(1, n) for j in range(max(0, i-3), i)]
    I = torch.tensor([i for i, j in pairs], device=g.DEV)
    J = torch.tensor([j for i, j in pairs], device=g.DEV)
    p = torch.as_tensor(noisy, dtype=g.D64, device=g.DEV)
    rel = torch.linalg.inv(p[I]) @ p[J]
    A = g.transport(rel, 1).to(g.DT)
    total = rf.clone().index_add_(0, I, torch.bmm(A, rf[J].unsqueeze(-1)).squeeze(-1)).reshape(n, 8, 8, 16)
    At = A.transpose(1, 2).reshape(len(pairs)*1024, 1, 8, 8, 16)
    vj = vf[J]
    out = {a: torch.full((n, 6), -float('inf'), device=g.DEV, dtype=g.DT) for a in ARMS}
    out['G0'].fill_(-50.)
    zs, admissible = {}, {}
    candidate_cells = torch.zeros((n, 8, 8, 16), dtype=torch.bool, device=g.DEV)
    for shape in WINDOWS:
        nn = int(np.prod(shape))
        c = F.avg_pool3d(At, shape, stride=1)*nn
        c = c.reshape(len(pairs), 1024, *c.shape[-3:])
        var = g.boxsum(vv, shape).clone().index_add_(0, I, torch.einsum('ps,psxyz->pxyz', vj, c*c))
        z = g.boxsum(total, shape)/var.clamp_min(1e-9).sqrt()
        adm = g.boxsum(support.to(g.DT), shape) > nn-.5
        zs[shape], admissible[shape] = z, adm
        good = (z >= 2.5) & adm.any(1)
        occupied = F.conv_transpose3d(good[:, None].to(g.DT), torch.ones((1, 1, *shape), device=g.DEV))[:, 0] > 0
        candidate_cells |= occupied
        out['G0'] = torch.maximum(out['G0'], torch.where(adm, z[:, None], -float('inf')).flatten(2).max(-1).values)
    del A, At, c
    # Temporal candidates are computed solely from past sensor scores. Reused
    # S2 history makes evidence correlated; calib measures the resulting null.
    local = torch.zeros((n, 8, 8, 16), dtype=torch.bool, device=g.DEV)
    centers = g._orig.mean(1)
    for i in range(1, n):
        for j in range(max(0, i-2), i):
            pts = centers[candidate_cells[j].flatten()]
            if not len(pts):
                continue
            trans = torch.linalg.inv(p[i]) @ p[j]
            pts = pts @ trans[:3, :3].T + trans[:3, 3]
            z = pts[:, 2]
            xy = pts[:, :2]/z[:, None].clamp_min(1e-12)
            valid = (z > 0) & (xy.abs() < g.EDGE).all(1)
            rc = torch.floor((xy[valid]+g.EDGE)/(2*g.EDGE)*8).long().clamp(0, 7)
            local[i, rc[:, 1], rc[:, 0], :] = True
    all_masks = {a: torch.as_tensor(m, dtype=g.DT, device=g.DEV) for a, m in masks.items()}
    all_masks['G4local'] = local.to(g.DT)
    for shape in WINDOWS:
        for arm, mask in all_masks.items():
            allowed = g.boxsum(mask, shape) > 0
            adm = admissible[shape] & allowed[:, None]
            val = torch.where(adm, zs[shape][:, None], -float('inf')).flatten(2).max(-1).values
            out[arm] = torch.maximum(out[arm], val)
    return {a: value.double().cpu().numpy() for a, value in out.items()}


def score_unit(job):
    evidence, output, unit, pool_file = job
    import torch
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    import cnh_track_a_scale_evaluate as se
    from cnh_track_a_readout import noisy_poses
    se.sensor_module.FAMILY = FAMILY
    evidence, output = Path(evidence), Path(output)
    geometry, sensor = evidence/'geometry', evidence/'sensor'
    readouts = evidence/'readouts-gpu/primary-mount-10-snr6'
    start = time.monotonic()
    pool = [np.asarray(x, int) for x in json.loads(Path(pool_file).read_text())]
    split, records, step = se.unit_records(geometry, sensor, unit, -10, 1)
    assert split == 'audit'
    data = json.loads((geometry/f'unit{unit:02d}'/f'unit{unit:02d}.json').read_text(encoding='utf-8-sig'))
    obj = {c['config']: c['objects'] for c in data['configs']}
    with np.load(sensor/f'unit{unit:02d}-mount-10-oracle.npz') as f:
        oid, dist = f['object_id'], f['raydistance']
    with np.load(sensor/f'unit{unit:02d}-mount-10-observations.npz') as f:
        config = f['config']
    with np.load(readouts/f'unit{unit:02d}.npz') as f:
        saved = {k: f[k] for k in ('config', 'frame', 'labels', 'witness', 'strata')}
        baseline = f['S2__noisy@0.75']
    bias = np.load(readouts/'bias.npy')
    scores = {a: [] for a in ARMS}
    candidates = []
    for rec in records:
        c = rec['config']
        rows = np.flatnonzero(config == c)[::step]
        masks, counts = oracle_masks(oid[rows], dist[rows], obj[c], unit, c, pool)
        result = sequence_scores(rec['hist'], rec['ambient'], bias, rec['tq'],
                                 noisy_poses(rec['poses'], rec['ego_seed'], dt=.2), masks)
        for a in ARMS:
            scores[a].append(result[a])
        candidates.extend(counts)
    arrays = {a: np.concatenate(value) for a, value in scores.items()}
    assert np.array_equal(saved['config'], np.repeat([r['config'] for r in records], 12))
    error = float(np.max(np.abs(arrays['G0']-baseline)))
    rel = float(np.max(np.abs(arrays['G0']-baseline)/np.maximum(1., np.abs(baseline))))
    if rel > 1e-5:
        raise ValueError(f'G0 mismatch unit {unit}: {error}, {rel}')
    for a in ARMS[1:]:
        if np.any(arrays[a] > arrays['G0']+1e-5):
            raise ValueError(f'Masked score exceeds G0: {a}')
    target = output/f'unit{unit:02d}.npz'
    tmp = target.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **saved, **arrays, G0_consistency_max_abs=error,
                        G0_consistency_max_rel=rel, oracle_candidate_counts=np.asarray(candidates))
    tmp.replace(target)
    return dict(unit=unit, seconds=time.monotonic()-start, G0_max_abs=error, G0_max_rel=rel,
                backend='cuda' if torch.cuda.is_available() else 'cpu',
                device=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu',
                peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--units', type=int, nargs='+')
    a = p.parse_args()
    if a.evidence.name != FAMILY+'-v1':
        raise ValueError('Only consumed v3 evidence is authorized')
    readouts = a.evidence/'readouts-gpu/primary-mount-10-snr6'
    units = []
    for f in sorted(readouts.glob('unit*.npz')):
        with np.load(f) as d:
            if str(d['split']) == 'audit':
                units.append(int(f.stem[4:]))
    if len(units) != 63:
        raise ValueError(f'Expected 63 consumed v3 units, got {len(units)}')
    a.output.mkdir(parents=True, exist_ok=True)
    pool_file = a.output.parent/'angular_templates.json'
    if not pool_file.exists():
        pool = footprint_pool(a.evidence/'geometry', a.evidence/'sensor', units)
        pool_file.write_text(json.dumps([x.tolist() for x in pool]))
    chosen = a.units or units
    if not set(chosen) <= set(units):
        raise ValueError('Units outside consumed audit')
    jobs = [(str(a.evidence), str(a.output), u, str(pool_file)) for u in chosen if not (a.output/f'unit{u:02d}.npz').exists()]
    start = time.monotonic()
    progress = dict(total=len(chosen), complete=len(chosen)-len(jobs), status='running', workers=a.workers,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    receipt = a.output/'progress.json'
    receipt.write_text(json.dumps(progress))
    try:
        with ProcessPoolExecutor(a.workers) as executor:
            for future in as_completed([executor.submit(score_unit, j) for j in jobs]):
                result = future.result()
                progress.update(complete=progress['complete']+1, last=result, elapsed_s=time.monotonic()-start)
                receipt.write_text(json.dumps(progress))
                print(json.dumps(result), flush=True)
        progress['status'] = 'complete'
    except BaseException as e:
        progress.update(status='failed', error=str(e))
        raise
    finally:
        receipt.write_text(json.dumps(progress, indent=2))


if __name__ == '__main__':
    main()
