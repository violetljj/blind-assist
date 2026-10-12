"""Frame-local anchor candidates; consumed Development only, no frozen changes.

The numerical quality cutoffs below are engineering hypotheses, not calibrated
confidence or obstacle truth. A rejected RGB anchor is UNKNOWN, never FREE.
Relative disparity has no common affine gauge across frames: only ``baseline``
may reuse a previous fit, to reproduce the existing script's behavior.
"""
from __future__ import annotations

import numpy as np

QUALITY = dict(min_zones=6, min_relative_inverse_span=.10,
               max_median_relative_inverse_error=.10,
               relative_inverse_inlier_error=.20, min_inlier_fraction=.60)


def fit_quality(dz, inv_z, fit):
    """Assess fit in metric inverse depth, invariant to positive affine disparity."""
    m = np.isfinite(dz) & np.isfinite(inv_z) & (inv_z > 0)
    x, y = np.asarray(inv_z)[m], np.asarray(dz)[m]
    d = dict(zones=int(m.sum()), accepted=False, reasons=[])
    if len(x) < QUALITY['min_zones']:
        d['reasons'].append('insufficient_zones')
    if fit is None or not np.all(np.isfinite(fit)) or fit[0] <= 0:
        d['reasons'].append('no_positive_fit')
        return d
    if not len(x):
        return d
    span = float((np.percentile(x, 90) - np.percentile(x, 10)) / np.median(x))
    a, b = fit
    error = np.abs((y - b) / a - x) / x
    d.update(relative_inverse_span=span,
             median_relative_inverse_error=float(np.median(error)),
             inlier_fraction=float(np.mean(error <= QUALITY['relative_inverse_inlier_error'])))
    if span < QUALITY['min_relative_inverse_span']:
        d['reasons'].append('insufficient_depth_span')
    if d['median_relative_inverse_error'] > QUALITY['max_median_relative_inverse_error']:
        d['reasons'].append('large_metric_residual')
    if d['inlier_fraction'] < QUALITY['min_inlier_fraction']:
        d['reasons'].append('insufficient_consensus')
    d['accepted'] = not d['reasons']
    return d


def select_anchor(disp, dirs, K, Rct, rays, dist, pitch, roll, height, rng,
                  policy='quality_floor', previous_fit=None):
    """Return fit/source/fresh/diagnostics for one frame.

    baseline: existing ToF -> floor -> previous fit, without quality rejection.
    fresh_only: same ordering, but no stale fit (isolates the stale-fit effect).
    quality_floor: quality-approved ToF -> existing per-frame floor -> unknown.
    quality_tof: quality-approved ToF -> unknown (floor dependence ablation).
    ``dist`` is 64 camera-aligned ToF-zone ranges in metres, NaN if invalid.
    Caller owns RNG and must give comparisons identical per-frame seeds.
    """
    import real_rgb_small_model_dev_20261012 as S
    import real_rgb_query_feasibility_dev_20261012 as F

    if policy not in ('baseline', 'fresh_only', 'quality_floor', 'quality_tof'):
        raise ValueError(policy)
    dz, zfac = S.zone_disparity(disp, K, Rct, rays)
    dist = np.asarray(dist)
    valid = np.isfinite(dist) & (dist > .3) & (dist < 4.)
    inv_z = np.full_like(dist, np.nan, dtype=float)
    inv_z[valid] = 1 / (dist[valid] * zfac[valid])
    fit = S.fit_affine(dz, inv_z)
    quality = fit_quality(dz, inv_z, fit)
    diagnostics = dict(tof=quality, thresholds=dict(QUALITY),
                       previous_fit_reused=False)
    if policy.startswith('quality') and not quality['accepted']:
        fit = None
    source = 'tof' if fit is not None else None
    if fit is None and policy != 'quality_tof':
        fit = S.floor_anchor(disp, dirs, F.rot(pitch, roll), Rct, height, rng)
        source = 'floor' if fit is not None else None
        # Existing floor RANSAC requires >=30% support; this is not an
        # independently validated floor detector or a guarantee of height.
        diagnostics['floor_attempted'] = True
        diagnostics['floor_assumed_height_m'] = float(height)
    fresh = fit is not None
    if fit is None and policy == 'baseline' and previous_fit is not None:
        fit, source = previous_fit, 'reused'
        diagnostics['previous_fit_reused'] = True
    return dict(fit=fit, source=source, fresh=fresh, diagnostics=diagnostics)


def robust_affine(dz, inv_z):
    """Deterministic exhaustive two-point consensus, then OLS consensus refit.

    At most 2016 pairs for 64 zones. Consensus uses the same 20% relative
    inverse-depth threshold as QUALITY. Final acceptance applies the unchanged
    QUALITY contract to ALL valid zones, never just selected inliers. Additional
    support/coverage checks apply to the final consensus itself.
    """
    dz, inv_z = np.asarray(dz, float), np.asarray(inv_z, float)
    valid = np.isfinite(dz) & np.isfinite(inv_z) & (inv_z > 0)
    x, y = inv_z[valid], dz[valid]
    diag = dict(method='exhaustive_pair_consensus', zones=len(x), pairs=0)
    if len(x) < QUALITY['min_zones']:
        diag.update(fit_quality(dz, inv_z, None))
        return dict(fit=None, diagnostics=diag)
    ii, jj = np.triu_indices(len(x), 1)
    good = np.abs(x[ii] - x[jj]) > 1e-9 * max(float(np.median(x)), 1.)
    ii, jj = ii[good], jj[good]
    a = (y[ii] - y[jj]) / (x[ii] - x[jj])
    b = y[ii] - a * x[ii]
    good = np.isfinite(a) & np.isfinite(b) & (a > 0)
    a, b = a[good], b[good]
    diag['pairs'] = int(len(a))
    if not len(a):
        diag.update(fit_quality(dz, inv_z, None))
        return dict(fit=None, diagnostics=diag)
    error = np.abs((y[None, :] - b[:, None]) / a[:, None] - x) / x
    support = error <= QUALITY['relative_inverse_inlier_error']
    counts = support.sum(1)
    # Deterministic ranking: largest consensus, then smallest all-zone median.
    order = np.lexsort((np.median(error, axis=1), -counts))
    best = int(order[0]); inlier = support[best]
    fit = (float(a[best]), float(b[best]))
    for _ in range(3):
        if inlier.sum() < QUALITY['min_zones']:
            break
        fitted, *_ = np.linalg.lstsq(np.stack([x[inlier], np.ones(inlier.sum())], 1), y[inlier], rcond=None)
        if fitted[0] <= 0 or not np.all(np.isfinite(fitted)):
            fit = None
            break
        fit = tuple(float(v) for v in fitted)
        new = np.abs((y - fit[1]) / fit[0] - x) / x <= QUALITY['relative_inverse_inlier_error']
        if np.array_equal(new, inlier):
            break
        inlier = new
    quality = fit_quality(dz, inv_z, fit)
    consensus_quality = fit_quality(y[inlier], x[inlier], fit)
    if inlier.sum() < QUALITY['min_zones'] or (
            consensus_quality.get('relative_inverse_span', 0.) < QUALITY['min_relative_inverse_span']):
        quality['accepted'] = False
        quality['reasons'].append('insufficient_consensus_coverage')
    diag.update(quality)
    diag.update(consensus_zones=int(inlier.sum()), consensus_quality=consensus_quality,
                proposed_fit=fit)
    return dict(fit=fit if quality['accepted'] else None, diagnostics=diag)


def robust_numerical_checks():
    x = np.linspace(.25, 2., 64)
    y = 3. * x + 2.
    # 20 adversarial zones destroy OLS while 44 preserve the metric relation.
    contaminated = y.copy(); contaminated[::3] += 20.
    r = robust_affine(contaminated, x)
    assert r['fit'] is not None and np.allclose(r['fit'], (3., 2.))
    t = robust_affine(7. * contaminated + 11., x)
    assert t['fit'] is not None and np.allclose(t['fit'], (21., 25.))
    assert t['diagnostics']['consensus_zones'] == r['diagnostics']['consensus_zones']
    assert robust_affine(y[:5], x[:5])['fit'] is None
    assert robust_affine(np.ones(64), np.ones(64))['fit'] is None
    assert robust_affine(-y, x)['fit'] is None
    return dict(checks=5, passed=True)


def numerical_checks():
    """Focused checks: affine gauge, degeneracy, and residual rejection."""
    x = np.linspace(.25, 2., 64)
    y = 3. * x + 2.
    q = fit_quality(y, x, (3., 2.))
    assert q['accepted']
    transformed = 7. * y + 11.
    q2 = fit_quality(transformed, x, (21., 25.))
    assert q2['accepted']
    assert abs(q['relative_inverse_span'] - q2['relative_inverse_span']) < 1e-12
    assert q2['median_relative_inverse_error'] < 1e-12
    flat = np.ones(64)
    assert not fit_quality(3. * flat + 2., flat, (3., 2.))['accepted']
    assert not fit_quality(y, x, (3., 20.))['accepted']
    assert not fit_quality(y[:5], x[:5], (3., 2.))['accepted']
    return dict(checks=5, passed=True)


if __name__ == '__main__':
    print(numerical_checks())
