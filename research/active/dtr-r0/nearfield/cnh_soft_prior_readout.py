"""Consumed repaired-v2 soft angular priors: synthetic confidence, not RGB.

Keep existing candidate geometry/RNG exactly. Confidence labels identify real
non-background objects, never whether a query is positive. No oracle range.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import ndtr, ndtri
import cnh_candidate_quality_readout as q

CONDITIONS = ('G1', 'FA1', 'FA3', 'BGFA1', 'BGFA3', 'DROP40', 'COMBO_MODERATE', 'COMBO_SEVERE')
AUCS = (.8, .9, .95, .99)
DELTAS = (0., .25, .5, .75, 1., 1.5, 2., 3.)
CONF_SEED = 20260929


def key(arm, auc, d):
    return f'C__{arm}__A{round(auc*100)}__D{d}'


def candidate_fields(oid, objects, unit, config, pool):
    """Union masks and per-zone maximum confidence, preserving old geometry.

    Equal-variance latent normal separation sqrt(2)*Phi^-1(AUC). Confidence
    is Phi(latent), not a posterior probability. Per-candidate draws shared
    across severity/AUC. False/true type is privileged, independent of ToF.
    """
    from cnh_route_sensor import angular_rays
    from cnh_track_a_readout import EDGE
    rays, _ = angular_rays(16)
    xy = (rays[..., :2]/rays[..., 2:]+EDGE)/(2*EDGE)*8
    ids = [o['id'] for o in objects if o['category'] != 'BACKGROUND']
    bg_ids = [o['id'] for o in objects if o['category'] == 'BACKGROUND']
    masks = {a: np.zeros((len(oid), 8, 8, 16), bool) for a in CONDITIONS}
    fields = np.zeros((len(CONDITIONS), len(AUCS), len(oid), 8, 8), np.float32)
    samples = {a: [] for a in CONDITIONS}
    offsets = np.sqrt(2)*ndtri(np.array(AUCS))
    for t, frame in enumerate(oid):
        rng = np.random.default_rng(np.random.SeedSequence([q.SEED, unit, config, t, 99999]))
        uniform, _ = q.fake_footprints(pool, rng, 10)
        bg, edge = q.background_support(frame, bg_ids)
        rng = np.random.default_rng(np.random.SeedSequence([q.SEED, unit, config, t, 99998]))
        structured, _ = q.fake_footprints(pool, rng, 10, bg, edge)
        real = []
        for ident in ids:
            hit = frame == ident
            if not hit.any():
                continue
            rng = np.random.default_rng(np.random.SeedSequence([q.SEED, unit, config, t, int(ident)]))
            angle = rng.uniform(0, 2*np.pi)
            direction, drop = np.array([np.cos(angle), np.sin(angle)]), rng.random()
            latent = np.random.default_rng(np.random.SeedSequence([CONF_SEED, unit, config, t, 0, int(ident)])).normal()
            pix = {shift: q.to_pixels(xy[hit], direction, shift) for shift in (0., .5, 1.)}
            real.append((drop, pix, latent))
        fake_latents = [np.random.default_rng(np.random.SeedSequence([CONF_SEED, unit, config, t, 1+kind])).normal(size=10)
                        for kind in (0, 1)]
        for ai, arm in enumerate(CONDITIONS):
            shift, dilation, drop_rate, nf, structured_flag = q.SPECS[arm]
            candidates = [(q.to_zones(pix[shift], dilation), latent, True)
                          for drop, pix, latent in real if drop >= drop_rate]
            candidates += [(mask, latent, False) for mask, latent in zip(
                (structured if structured_flag else uniform)[:nf], fake_latents[int(structured_flag)][:nf])]
            for mask, latent, is_real in candidates:
                if not mask.any():
                    continue  # no candidate after clipping / no background placement
                confidence = ndtr(latent+offsets*is_real).astype(np.float32)
                fields[ai, :, t] = np.maximum(fields[ai, :, t], confidence[:, None, None]*mask[None])
                masks[arm][t] |= mask[..., None]
                # Evaluated frames only; candidate denominator is frame-level.
                if t >= 3:
                    samples[arm].append((int(is_real), float(latent)))
    return masks, fields.reshape(-1, len(oid), 8, 8), samples


def sequence_scores(hist, ambient, bias, tq, noisy, masks, fields):
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
    A = g.transport(torch.linalg.inv(p[I]) @ p[J], 1).to(g.DT)
    total = rf.clone().index_add_(0, I, torch.bmm(A, rf[J].unsqueeze(-1)).squeeze(-1)).reshape(n, 8, 8, 16)
    At = A.transpose(1, 2).reshape(len(pairs)*1024, 1, 8, 8, 16)
    out = {a: torch.full((n, 6), -float('inf'), device=g.DEV, dtype=g.DT) for a in CONDITIONS}
    out['G0'] = torch.full((n, 6), -50., device=g.DEV, dtype=g.DT)
    stack = torch.as_tensor(np.stack([masks[a] for a in CONDITIONS]), device=g.DEV, dtype=g.DT)
    conf = torch.as_tensor(fields, device=g.DEV, dtype=g.DT)
    weighted = torch.full((len(DELTAS), len(fields), n, 6), -50., device=g.DEV, dtype=g.DT)
    for shape in WINDOWS:
        size = int(np.prod(shape))
        c = F.avg_pool3d(At, shape, stride=1)*size
        c = c.reshape(len(pairs), 1024, *c.shape[-3:])
        var = g.boxsum(vv, shape).clone().index_add_(0, I, torch.einsum('ps,psxyz->pxyz', vf[J], c*c))
        z = g.boxsum(total, shape)/var.clamp_min(1e-9).sqrt()
        adm = g.boxsum(support.to(g.DT), shape) > size-.5
        score = torch.where(adm, z[:, None], -float('inf')).flatten(2).max(-1).values
        out['G0'] = torch.maximum(out['G0'], score)
        allowed = g.boxsum(stack, shape) > 0
        hard = torch.where(adm[None] & allowed[:, :, None], z[None, :, None], -float('inf')).flatten(3).amax(-1)
        for ai, arm in enumerate(CONDITIONS):
            out[arm] = torch.maximum(out[arm], hard[ai])
        # Angular max across the window footprint; depth remains unrestricted.
        pooled = F.max_pool2d(conf.reshape(-1, 1, 8, 8), shape[:2], stride=1)
        pooled = pooled.reshape(len(fields), n, *pooled.shape[-2:])[..., None]
        for di, delta in enumerate(DELTAS[1:], 1):
            adjusted = z[None]+delta*pooled
            score = torch.where(adm[None], adjusted[:, :, None], -float('inf')).flatten(3).amax(-1)
            weighted[di] = torch.maximum(weighted[di], score)
    weighted[0] = out['G0'][None]
    result = {a: s.double().cpu().numpy() for a, s in out.items()}
    arr = weighted.double().cpu().numpy()
    for ai, arm in enumerate(CONDITIONS):
        for ui, auc in enumerate(AUCS):
            for di in range(len(DELTAS)):
                result[key(arm, auc, di)] = arr[di, ai*len(AUCS)+ui]
    return result


def score_unit(job):
    source, target, unit = Path(job[0]), Path(job[1]), job[2]
    import torch
    import cnh_track_a_scale_evaluate as se
    from cnh_track_a_readout import noisy_poses
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no silent CPU fallback')
    torch.cuda.reset_peak_memory_stats()
    se.sensor_module.FAMILY = q.FAMILY
    start = time.monotonic()
    split, records, step = se.unit_records(source/'geometry', source/'sensor', unit, -10, 1)
    bias = np.load(source/'bias.npy')
    pool = [np.asarray(x, int) for x in json.loads((source/'angular_templates.json').read_text())]
    data = json.loads((source/'geometry'/f'unit{unit:02d}'/f'unit{unit:02d}.json').read_text(encoding='utf-8-sig'))
    objects = {c['config']: c['objects'] for c in data['configs']}
    with np.load(source/'sensor'/f'unit{unit:02d}-mount-10-oracle.npz') as f:
        oid = f['object_id']
    with np.load(source/'sensor'/f'unit{unit:02d}-mount-10-observations.npz') as f:
        assert int(f['rate']) == 5
        config = f['config']
    scores, samples = {}, {a: [] for a in CONDITIONS}
    for rec in records:
        rows = np.flatnonzero(config == rec['config'])[::step]
        masks, fields, stats = candidate_fields(oid[rows], objects[rec['config']], unit, rec['config'], pool)
        noisy = noisy_poses(rec['poses'], rec['ego_seed'], dt=.2)
        result = sequence_scores(rec['hist'], rec['ambient'], bias, rec['tq'], noisy, masks, fields)
        for arm, score in result.items():
            scores.setdefault(arm, []).append(score)
        for arm in CONDITIONS:
            samples[arm].extend(stats[arm])
    arrays = {a: np.concatenate(v) for a, v in scores.items()}
    parity = {}
    with np.load(source/'scores'/f'unit{unit:02d}.npz') as old:
        for arm in ('G0', *CONDITIONS):
            x, y = arrays[arm], old[arm]
            if not np.array_equal(np.isneginf(x), np.isneginf(y)):
                raise AssertionError(f'{unit} {arm}: mask parity')
            ix = np.isfinite(y)
            parity[arm] = float(np.max(np.abs(x[ix]-y[ix])/np.maximum(1., np.abs(y[ix])))) if ix.any() else 0.
            if parity[arm] > 1e-5:
                raise AssertionError(f'{unit} {arm}: score parity {parity[arm]}')
        for arm in CONDITIONS:
            for auc in AUCS:
                np.testing.assert_array_equal(arrays[key(arm, auc, 0)], arrays['G0'])
        extra = {k: old[k] for k in ('config', 'frame', 'labels', 'witness', 'strata', 'split', 'family',
                                    'G0_consistency_max_abs', 'G0_consistency_max_rel')}
    path = target/'scores'/f'unit{unit:02d}.npz'
    tmp = path.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **arrays, **extra)
    tmp.replace(path)
    # Latents suffice to independently recompute empirical AUC for every arm.
    np.savez_compressed(target/'scores'/f'confidence{unit:02d}.npz', **{a: np.asarray(v, float).reshape(-1, 2) for a, v in samples.items()})
    receipt = dict(unit=unit, split=split, seconds=time.monotonic()-start, baseline_relative_parity=parity,
                   zero_delta_exact=True, backend='cuda', device=torch.cuda.get_device_name(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
    (target/'scores'/f'unit{unit:02d}.json').write_text(json.dumps(receipt))
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--units', nargs='+', type=int)
    a = p.parse_args()
    if a.source.name != q.FAMILY+'-dev-repaired' or a.output.resolve() == a.source.resolve():
        raise ValueError('Use the existing repaired v2 source and a separate output')
    (a.output/'scores').mkdir(parents=True, exist_ok=True)
    allowed = [u for u in range(96, 192) if u != 143]
    units = a.units or allowed
    if not set(units) <= set(allowed):
        raise ValueError('Invalid cohort')
    jobs = [(str(a.source), str(a.output), u) for u in units if not (a.output/'scores'/f'unit{u:02d}.json').exists()]
    progress = dict(status='running', complete=len(units)-len(jobs), total=len(units), workers=a.workers,
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    receipt = a.output/'scoring_progress.json'
    start = time.monotonic()
    try:
        receipt.write_text(json.dumps(progress))
        with ProcessPoolExecutor(a.workers) as workers:
            for future in as_completed([workers.submit(score_unit, job) for job in jobs]):
                result = future.result()
                progress.update(complete=progress['complete']+1, last=result, elapsed_s=time.monotonic()-start)
                receipt.write_text(json.dumps(progress))
                print(json.dumps(result), flush=True)
        progress['status'] = 'complete'
    except BaseException as exc:
        progress.update(status='failed', error=str(exc))
        raise
    finally:
        receipt.write_text(json.dumps(progress, indent=2))


if __name__ == '__main__':
    main()
