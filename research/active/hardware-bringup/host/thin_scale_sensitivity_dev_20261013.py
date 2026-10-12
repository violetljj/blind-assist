"""Label-blind affine support diagnostic; never refits or changes geometry.

Anchor min/max is an observed support envelope, not calibrated confidence.
Floor/reused fits are not applicable to this fresh ToF support hypothesis.
"""
import hashlib
import json
import time
from collections import Counter

import numpy as np
from atom_camera import AtomCamera
import atom_tof_extrinsic_dev_20261012 as X
import real_rgb_small_model_dev_20261012 as S
import real_rgb_query_feasibility_dev_20261012 as F
from fusion_anchor_candidates_dev_20261013 import fit_quality

CACHE = F.ROOT / 'artifacts.local/work/fusion-refinement-dev-20261013'
SOURCE = F.ROOT / 'artifacts.local/work/thin-local-geometry-dev-20261013/geometry_frames.jsonl'
OUT = F.ROOT / 'artifacts.local/work/thin-confidence-dev-20261013'


def quantiles(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return np.percentile(values, [0, 10, 50, 90, 100]).tolist() if len(values) else None


def support(values, valid, lo, hi):
    inside = valid & np.isfinite(values) & (values >= lo) & (values <= hi)
    return int(inside.sum()), bool(inside.sum() >= 20)


def main():
    start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    manifest = json.loads((CACHE / 'cache_manifest.json').read_text())
    data = json.loads((CACHE / 'robust_readout.json').read_text())
    frames = {(k, r['i']): r for k, s in manifest['sessions'].items() for r in s['frames']}
    fits = {(k, r['i']): r for k, s in data['sessions'].items() for r in s['arms']['robust_replace']}
    K = AtomCamera().K_upright
    Rct = np.array(json.loads((X.OUT / 'camera_tof_extrinsic.json').read_text())['rotation'])
    rays = X.zone_rays()
    assert support(np.arange(40), np.ones(40, bool), 0, 19) == (20, True)
    assert support(np.arange(40), np.ones(40, bool), 0, 18) == (19, False)
    counts = Counter(); failures = []
    with SOURCE.open(encoding='utf8') as source, (OUT / 'scale_sensitivity.jsonl').open('w', encoding='utf8') as out:
        for line in source:
            if time.perf_counter() - start > 140:
                raise TimeoutError('scale support 150 command-wall-second budget reserve')
            row = json.loads(line); key = row['key']; i = row['i']
            meta, old = frames[key, i], fits[key, i]
            fit = old['fit']; anchor = old['anchor']; diag = None
            disp = valid = None; dq = iq = None
            if meta.get('cache'):
                with np.load(CACHE / meta['cache']) as p:
                    disp, valid = p['disp'], p['valid']
                dz, zfac = S.zone_disparity(disp, K, Rct, rays)
                dist = np.asarray(meta['dist'])
                keep = np.isfinite(dist) & (dist > .3) & (dist < 4) & np.isfinite(zfac) & (zfac > 0)
                inv = np.full(64, np.nan); inv[keep] = 1 / (dist[keep] * zfac[keep])
                good = np.isfinite(dz) & np.isfinite(inv) & (inv > 0)
                dq, iq = quantiles(dz[good]), quantiles(inv[good])
                diag = fit_quality(dz, inv, fit)
            applicable = anchor in ('tof', 'robust_tof') and dq is not None and fit is not None and fit[0] > 0
            frame = dict(key=key, i=i, rgb_index=row['rgb_index'], anchor=anchor, fit=fit,
                         applicable=applicable, quality=diag, anchor_disparity_quantiles=dq,
                         anchor_inverse_z_quantiles=iq, candidates=[])
            if fit and dq:
                frame['disparity_span_over_slope'] = (dq[3] - dq[1]) / fit[0]
                frame['slope_over_anchor_disparity_p90_p10_span'] = fit[0] / (dq[3] - dq[1]) if dq[3] > dq[1] else None
            for j, c in enumerate(row['candidates']):
                d = dict(candidate_index=j, **{n:c[n] for n in ('x1', 'y1', 'x2', 'y2')},
                         applicable=applicable, gate=None, status='NOT_APPLICABLE', inside_count=None,
                         outside_fraction_valid=None, candidate_disparity_quantiles=None,
                         legal_inverse_points=0, projected_z_median_m=None)
                if disp is not None:
                    p = np.array([[c['x1'], c['y1']], [c['x2'], c['y2']]])
                    xy = np.rint(p[0] + np.linspace(0, 1, 40)[:, None] * (p[1] - p[0])).astype(int)
                    assert ((xy >= 0).all() and (xy[:, 0] < disp.shape[1]).all() and (xy[:, 1] < disp.shape[0]).all())
                    values = disp[xy[:, 1], xy[:, 0]]; mask = valid[xy[:, 1], xy[:, 0]] & np.isfinite(values)
                    d['candidate_disparity_quantiles'] = quantiles(values[mask])
                    if fit and fit[0] > 0:
                        projected_inverse = (values - fit[1]) / fit[0]
                        legal = mask & np.isfinite(projected_inverse) & (projected_inverse > 1e-3)
                        d['legal_inverse_points'] = int(legal.sum())
                        if legal.any(): d['projected_z_median_m'] = float(np.median(1 / projected_inverse[legal]))
                        previous = c['geometry']['robust_replace']
                        if previous['camera_z_median_m'] is not None:
                            assert np.isclose(previous['camera_z_median_m'], d['projected_z_median_m'])
                    if applicable:
                        n, passed = support(values, mask, dq[0], dq[4])
                        d.update(gate=passed, status='WITHIN_SUPPORT' if passed else 'EXTRAPOLATED', inside_count=n,
                                 outside_fraction_valid=float(1 - n / mask.sum()) if mask.any() else None)
                counts[d['status']] += 1
                frame['candidates'].append(dict(c, scale_support=d))
            counts['frames'] += 1
            counts['applicable_frames'] += applicable
            if (key, i) in [('073554', 59), ('073635', 39), ('073821', 81)]: failures.append(frame)
            out.write(json.dumps(frame, allow_nan=False) + '\n')
    assert counts['frames'] == 1677
    assert source_hash == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    result = dict(counts=counts, wall_s=time.perf_counter() - start, geometry_source_sha256=source_hash,
                  check='1677 frames; unchanged source SHA; legal inverse projection equals original camera z median; exact 20/40 support boundary',
                  quantile_order=['min','p10','median','p90','max'], gate='At least 20/40 sampled pixels within fresh ToF anchor disparity min/max; floor/reused NOT_APPLICABLE',
                  note='Support is not calibrated confidence; rejected or nonapplicable candidates remain UNKNOWN, never FREE.')
    (OUT / 'scale_sensitivity_summary.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    (OUT / 'scale_sensitivity_failures.json').write_text(json.dumps(failures, indent=2), encoding='utf8')
    print(json.dumps(result))
    for r in failures:
        print(json.dumps({k:v for k,v in r.items() if k != 'candidates'}))


if __name__ == '__main__': main()
