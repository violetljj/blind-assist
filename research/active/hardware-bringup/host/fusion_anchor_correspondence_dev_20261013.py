"""Two fixed zone-correspondence hypotheses on already consumed real data.

No ground truth enters fitting. Selection residual is same-source consistency,
not accuracy. All unsuccessful candidates retain the exact baseline frame.
"""
import copy
import hashlib
import json
import time
from collections import Counter
import numpy as np
from atom_camera import AtomCamera
import atom_tof_extrinsic_dev_20261012 as X
import real_rgb_small_model_dev_20261012 as S
import real_rgb_query_feasibility_dev_20261012 as F
from fusion_refinement_cache_dev_20261013 import OUT
from fusion_refinement_readout_dev_20261013 import project
from fusion_anchor_candidates_dev_20261013 import robust_affine, fit_quality
from fusion_refinement_score_dev_20261013 import metrics
from fusion_temporal_candidates_dev_20261013 import run_candidates

DEST = OUT.parent / 'fusion-refinement-v2-dev-20261013'


def zone_samples(disp, K, Rct, rays):
    rc = rays @ Rct.T
    u = K[0, 0] * rc[..., 0] / rc[..., 2] + K[0, 2]
    v = K[1, 1] * rc[..., 1] / rc[..., 2] + K[1, 2]
    h, w = disp.shape
    inside = (rc[..., 2] > .05) & (u >= 0) & (u < w-1) & (v >= 0) & (v < h-1)
    d = disp[np.clip(v.round().astype(int), 0, h-1), np.clip(u.round().astype(int), 0, w-1)]
    return [row[mask] if mask.sum() >= 13 else np.array([]) for row, mask in zip(d, inside)]


def two_modes(values):
    """Deterministic two-means; require supported, separated components."""
    if len(values) < 13:
        return np.array([])
    centers = np.percentile(values, [25, 75])
    for _ in range(6):
        group = np.abs(values[:, None] - centers).argmin(axis=1)
        if min(np.bincount(group, minlength=2)) < .2 * len(values):
            return np.array([np.median(values)])
        new = np.array([np.mean(values[group == k]) for k in range(2)])
        if np.allclose(new, centers, rtol=0, atol=1e-10):
            break
        centers = new
    spread = np.array([np.std(values[group == k]) for k in range(2)])
    if abs(centers[1] - centers[0]) <= 2 * max(spread.max(), 1e-9):
        return np.array([np.median(values)])
    return np.sort(centers)


def propose(samples, dz, inv, seed, method):
    diag = dict(method=method, accepted=False, same_source_consistency=True)
    if seed is None or seed[0] <= 0:
        return None, dict(diag, reason='no_positive_seed')
    if method == 'homogeneous_replace':
        # Relative metric IQR uses the baseline gauge; never an outcome label.
        spread = np.array([np.subtract(*np.percentile(v, [75, 25])) if len(v) else np.nan for v in samples])
        good = np.isfinite(inv) & (inv > 0) & (spread / seed[0] / inv <= .2)
        selected = np.where(good, dz, np.nan)
        proposed = robust_affine(selected, inv)
        fit = proposed['fit']
        diag.update(selected_zones=int(good.sum()), selected_quality=proposed['diagnostics'])
    else:
        modes = [two_modes(v) for v in samples]
        fit = seed
        for iteration in range(3):
            selected = np.array([m[np.argmin(abs(m - (fit[0]*x + fit[1])))] if len(m) and np.isfinite(x) else np.nan for m, x in zip(modes, inv)])
            proposed = robust_affine(selected, inv)
            if proposed['fit'] is None:
                fit = None
                break
            fit = proposed['fit']
        diag.update(iterations=iteration+1, multimodal_zones=sum(len(m) == 2 for m in modes), selected_quality=proposed['diagnostics'])
    # Same original all-zone quality check as robust_replace; selecting a
    # subset/mode cannot by itself make a previously inconsistent frame pass.
    original_quality = fit_quality(dz, inv, fit)
    diag.update(original_quality=original_quality, accepted=bool(original_quality['accepted']))
    return (fit if diag['accepted'] else None), diag


def main():
    started = time.perf_counter()
    DEST.mkdir(parents=True, exist_ok=True)
    source = OUT/'robust_readout.json'
    data = json.loads(source.read_text())
    cache = json.loads((OUT/'cache_manifest.json').read_text())
    cam = AtomCamera(); K = cam.K_upright; w, h = cam.upright_size
    vv, uu = np.mgrid[:h, :w]
    dirs = np.stack(((uu-K[0, 2])/K[0, 0], (vv-K[1, 2])/K[1, 1], np.ones_like(uu)), -1).reshape(-1, 3)
    Rct = np.array(json.loads((X.OUT/'camera_tof_extrinsic.json').read_text())['rotation'])
    rays = X.zone_rays()
    results = dict(sessions={}); comparison = dict(sessions={}); counters = Counter()
    for key, s in data['sessions'].items():
        arms = {name: s['arms'][name] for name in ('baseline', 'robust_replace')}
        arms.update(homogeneous_replace=[], mode_replace=[])
        for original, c in zip(arms['baseline'], cache['sessions'][key]['frames']):
            if time.perf_counter()-started > 550:
                raise TimeoutError('600 command-wall budget reserve')
            rows = {name: copy.deepcopy(original) for name in ('homogeneous_replace', 'mode_replace')}
            if 'cache' in c and not original['quality'].get('accepted', False):
                counters['attempted_frames'] += 1
                with np.load(OUT/c['cache']) as payload:
                    disp, valid = payload['disp'], payload['valid']
                dz, zfac = S.zone_disparity(disp, K, Rct, rays)
                samples = zone_samples(disp, K, Rct, rays)
                dist = np.asarray(c['dist']); keep = np.isfinite(dist) & (dist > .3) & (dist < 4.)
                inv = np.full(64, np.nan); inv[keep] = 1/(dist[keep]*zfac[keep])
                corridor = None
                for method, row in rows.items():
                    fit, diag = propose(samples, dz, inv, original['fit'], method)
                    row['correspondence_diagnostic'] = diag
                    if fit is not None:
                        counters[method+'_accepted'] += 1
                        if corridor is None:
                            corridor = dirs @ (F.rot(c['pitch'], c['roll']) @ Rct.T).T
                        row.update(project(disp, valid, fit, corridor), fit=fit, anchor=method)
            for name, row in rows.items():
                arms[name].append(row)
        results['sessions'][key] = dict(label=s['label'], session=s['session'], arms=arms)
        pred = {name: run_candidates(fr)['baseline_2of3'] for name, fr in arms.items()}
        report = {}
        for name, fr in arms.items():
            out = pred[name]; base = pred['baseline']
            report[name] = dict(metrics(fr, out),
                added_query_frames=int((out & ~base).sum()), removed_query_frames=int((base & ~out).sum()),
                added_any_frames=int((out.any(1) & ~base.any(1)).sum()), removed_any_frames=int((base.any(1) & ~out.any(1)).sum()),
                fallback_frames=sum('correspondence_diagnostic' in r and not r['correspondence_diagnostic']['accepted'] for r in fr),
                sources=dict(Counter(r['anchor'] for r in fr)))
        comparison['sessions'][key] = dict(label=s['label'], arms=report)
        print(key, {n: {f: m[f] for f in ('any','H','B','added_query_frames','removed_query_frames','rgb_available')} for n,m in report.items()}, flush=True)
    (DEST/'correspondence_readout.json').write_text(json.dumps(results), encoding='utf8')
    (DEST/'correspondence_comparison.json').write_text(json.dumps(comparison, indent=1), encoding='utf8')
    receipt = dict(wall_s=time.perf_counter()-started, gpu_seconds=0, download_bytes=0,
        counters=dict(counters), input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
        warning='Consumed Development, no independent labels; mode selection and quality use same sensor/disparity evidence.')
    (DEST/'correspondence_receipt.json').write_text(json.dumps(receipt, indent=1), encoding='utf8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
