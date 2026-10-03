"""Known-scene signed-CNH information diagnostic; no model fitting.

Standalone likelihood/AUC routines never open scene or observation data.
Nominal signed H3 counts are Skellam: sum(positive Poisson draws) minus
sum(independent ambient-estimate Poisson draws). The zero-noise signed mean,
including residual crosstalk, and declared ambient determine both rates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import gammaln, ive, logsumexp
from scipy.stats import rankdata, skellam

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-displacement-ceiling-20261003'
RUN = 'CNH_DISPLACEMENT_CEILING_20261003'
INTRUSION_CM = [1, 2, 5, -5, -10, -15, -20]
PRIMARY_POS = [0, 1]
PRIMARY_NEG = [4, 5, 6]
TEMPLATE_IDS = [0, 1, 2, 4, 5, 6]
FRAMES = np.arange(3, 16)
BINS = [[.6, .9], [.9, 1.2], [1.2, 1.6], [1.6, 2.1], [2.1, 2.6]]
THRESHOLD = .8557642486787612
N_BOOT = 1000
BOOT_SEED = 2026100305


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf8')


def _log_bessel_series(order, x, relative_tolerance=1e-13):
    """Stable positive I_n series for scaled-Bessel underflow cases.

    Stop only when the remaining series has a geometric relative-error bound.
    This is numerical evaluation of the exact law, not a Gaussian fallback.
    """
    order, x = np.broadcast_arrays(np.asarray(order, float), np.asarray(x, float))
    if np.any(x <= 0) or np.any(order < 0):
        raise ValueError('Positive Bessel argument and nonnegative order required')
    log_term = np.zeros_like(x)
    log_total = np.zeros_like(x)
    active = np.ones(x.shape, bool)
    for j in range(1, 100001):
        if not np.any(active):
            return order*np.log(x/2)-gammaln(order+1)+log_total
        log_term[active] += 2*np.log(x[active]/2)-np.log(j)-np.log(order[active]+j)
        log_total[active] = np.logaddexp(log_total[active], log_term[active])
        ratio = x*x/(4*(j+1)*(order+j+1))
        bounded = active & (ratio < 1)
        tail_log = np.full(x.shape, np.inf)
        tail_log[bounded] = (log_term[bounded]+np.log(ratio[bounded])
                             -np.log1p(-ratio[bounded])-log_total[bounded])
        active &= tail_log > np.log(relative_tolerance)
    raise FloatingPointError('Exact Bessel series did not meet its tail bound')


def skellam_logpmf_signed(observed, expected, background, *, noise_scale=1., output_gain=1.):
    """Elementwise signed-count log likelihood from mean and summed ambient.

    expected is a noise_scale=0 H3 coarse template, including residual xtalk.
    background is the per-H3-bin sum of ambient-estimate Poisson rates, normally
    eight times the coarse native-zone ambient metadata. No latent count draw
    or true displacement identity is an input.
    """
    if noise_scale != 1. or output_gain != 1.:
        raise ValueError('UNSUPPORTED exact law: requires noise_scale=output_gain=1')
    y, e, a = np.broadcast_arrays(np.asarray(observed, float), np.asarray(expected, float), np.asarray(background, float))
    if not np.isfinite(y).all() or not np.isfinite(e).all() or not np.isfinite(a).all():
        raise ValueError('Nonfinite likelihood inputs')
    if np.any(y != np.rint(y)) or np.any(e < 0) or np.any(a < 0):
        raise ValueError('Integer signed observation and nonnegative Poisson means required')
    positive, negative = e+a, a
    out = np.full(y.shape, -np.inf)
    both = (positive > 0) & (negative > 0)
    if np.any(both):
        k, p, n = y[both], positive[both], negative[both]
        x = 2*np.sqrt(p*n)
        scaled = ive(np.abs(k), x)
        log_i = np.empty(x.shape)
        good = np.isfinite(scaled) & (scaled > 0)
        log_i[good] = np.log(scaled[good])+x[good]
        if np.any(~good):
            log_i[~good] = _log_bessel_series(np.abs(k[~good]), x[~good])
        out[both] = -p-n+.5*k*(np.log(p)-np.log(n))+log_i
    positive_only = (positive > 0) & (negative == 0) & (y >= 0)
    out[positive_only] = (-positive[positive_only]+y[positive_only]*np.log(positive[positive_only])
                          -gammaln(y[positive_only]+1))
    negative_only = (positive == 0) & (negative > 0) & (y <= 0)
    out[negative_only] = (-negative[negative_only]-y[negative_only]*np.log(negative[negative_only])
                          -gammaln(-y[negative_only]+1))
    out[(positive == 0) & (negative == 0) & (y == 0)] = 0.
    return out


def rolling_sum(values, window=8):
    """Causal frame-axis=last sums; no repeated likelihood calculations."""
    values = np.asarray(values, float)
    if values.ndim < 1 or window < 1 or int(window) != window or not np.isfinite(values).all():
        raise ValueError('Finite per-frame log likelihood and positive window required')
    prefix = np.concatenate((np.zeros((*values.shape[:-1], 1)), np.cumsum(values, axis=-1)), axis=-1)
    ends = np.arange(1, values.shape[-1]+1)
    begins = np.maximum(0, ends-int(window))
    return prefix[..., ends]-prefix[..., begins]


def mixture_llr(frame_loglike, positive_templates, negative_templates, window=8):
    """Class-balanced template mixture; inputs [...,template,frame].

    Every noisy sample is scored against every candidate template. Class
    mixtures have equal prior mass regardless of their template counts.
    """
    values = np.asarray(frame_loglike, float)
    pos, neg = np.asarray(positive_templates, int), np.asarray(negative_templates, int)
    if values.ndim < 2 or not len(pos) or not len(neg):
        raise ValueError('Both nonempty template classes are required')
    if len(set(pos.tolist())) != len(pos) or len(set(neg.tolist())) != len(neg) or set(pos)&set(neg):
        raise ValueError('Template classes must be disjoint without duplicates')
    if np.any(pos < 0) or np.any(neg < 0) or np.any(pos >= values.shape[-2]) or np.any(neg >= values.shape[-2]):
        raise ValueError('Template indices outside declared candidates')
    totals = rolling_sum(values, window)
    return (logsumexp(totals[..., pos, :], axis=-2)-np.log(len(pos))
            -logsumexp(totals[..., neg, :], axis=-2)+np.log(len(neg)))


def binary_auc(positive_scores, negative_scores):
    """Rank AUC with half-credit ties; no threshold fitting or filtering."""
    pos, neg = np.asarray(positive_scores, float).ravel(), np.asarray(negative_scores, float).ravel()
    if not len(pos) or not len(neg):
        return None
    if not np.isfinite(pos).all() or not np.isfinite(neg).all():
        raise ValueError('Nonfinite AUC scores; retain an explicit unsupported denominator')
    rank = rankdata(np.concatenate((pos, neg)), method='average')
    return float((rank[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))


def m3_smooth(raw):
    """Inherited five-logit causal smoothing, frame3 first retained decision."""
    raw = np.asarray(raw, float)
    if raw.shape != (7, 4, 13, 2) or not np.isfinite(raw).all():
        raise ValueError('Finite M3 raw logits [delta,replica,13,2] required')
    output = np.empty_like(raw)
    weights = np.array([1., 2., 4., 8., 16.])
    for t in range(13):
        start = max(0, t-4)
        weight = weights[-(t-start+1):]
        output[:, :, t] = (raw[:, :, start:t+1]*weight[None, None, :, None]).sum(2)/weight.sum()
    return output


def frame_likelihood(hist, expected, ambient):
    """One scene, independent bin likelihoods once per sample/template/frame.

    Only hist/expected/ambient are accepted. Privileged counts and
    ambient_estimate arrays are deliberately outside the function interface.
    """
    if hist.shape != (7, 4, 16, 8, 8, 16) or expected.shape != (7, 16, 8, 8, 16) or ambient.shape != (16, 8, 8):
        raise ValueError('CNH observation/template/ambient axes differ')
    if hist.dtype != np.int32:
        raise ValueError('Raw signed int32 observations required')
    samples = hist.reshape(28, 16, 8, 8, 16)
    likelihood = np.empty((28, len(TEMPLATE_IDS), 16), float)
    for ti, template in enumerate(TEMPLATE_IDS):
        values = skellam_logpmf_signed(samples, expected[template][None], 8*ambient[None, ..., None])
        likelihood[:, ti] = values.sum(axis=(-3, -2, -1))
    if not np.isfinite(likelihood).all():
        raise ValueError('Unsupported nonfinite scene likelihood; do not silently discard a scene')
    return likelihood


def auc_cell(score, frame_keep, positive=PRIMARY_POS, negative=PRIMARY_NEG):
    """Within-scene AUC, pooled branch/replica/frame only inside a fixed bin."""
    score = np.asarray(score, float)
    if score.shape != (7, 4, 13) or np.asarray(frame_keep).shape != (13,):
        raise ValueError('Within-scene AUC axes differ')
    pos = score[np.asarray(positive), :, :][..., frame_keep].ravel()
    neg = score[np.asarray(negative), :, :][..., frame_keep].ravel()
    return dict(auc=binary_auc(pos, neg), positive_n=int(len(pos)), negative_n=int(len(neg)), frames=int(np.sum(frame_keep)))


def analyze_scene(unit, hist, expected, ambient, truth, raw, object_id=None, expected32=None):
    if truth['unit'] != unit or truth['intrusion_cm'] != INTRUSION_CM or truth['mode'] not in (0, 1) or truth['group'] not in (0, 1):
        raise ValueError('Declared scene identity/primary straight-motion selection differs')
    front = np.asarray(truth['front_range_m'], float)
    categories = np.asarray(truth['categories'])
    labels = np.asarray(truth['labels'])
    if front.shape != (16,) or not np.isfinite(front).all() or categories.shape != (7, 16, 2) or labels.shape != (7, 16, 6):
        raise ValueError('Evaluation truth axes/values differ')
    frame_ll = frame_likelihood(hist, expected, ambient)
    # Static displacement hypotheses are mixed after complete temporal likelihoods,
    # rather than resampling the displacement hypothesis independently each frame.
    oracle = mixture_llr(frame_ll, [0, 1], [3, 4, 5], 8).reshape(7, 4, 16)[..., FRAMES]
    oracle12 = mixture_llr(frame_ll, [0, 1], [3, 4, 5], 12).reshape(7, 4, 16)[..., FRAMES]
    q = int(truth['group'])
    scores = dict(oracle8=oracle, oracle12=oracle12, M3_raw=np.asarray(raw, float)[..., q], M3_smooth=m3_smooth(raw)[..., q])
    ranges = front[FRAMES]
    masks = {f'{low:g}-{high:g}m': (ranges >= low) & (ranges < high) for low, high in BINS}
    masks['primary1.2-2.1m'] = (ranges >= 1.2) & (ranges < 2.1)
    cells, threshold_curves, truth_counts, fits, visibility = {}, {}, {}, {}, {}
    for name, keep in masks.items():
        cells[name] = {arm: auc_cell(score, keep) for arm, score in scores.items()}
        threshold_curves[name] = {}
        for delta in range(7):
            value = scores['M3_smooth'][delta, :, keep]
            threshold_curves[name][str(INTRUSION_CM[delta])] = dict(stops=int((value >= THRESHOLD).sum()), n=int(value.size))
        truth_counts[name] = {}
        for role, deltas in (('nominal_shallow', PRIMARY_POS), ('nominal_outside10_15_20', PRIMARY_NEG)):
            selected = categories[deltas, :, q][:, FRAMES][:, keep].ravel()
            names, counts = np.unique(selected, return_counts=True)
            # Truth categories are deterministic branch/frame counts; each is
            # repeated K4 times in the score denominator, never four new scenes.
            truth_counts[name][role] = dict(branch_frames=int(len(selected)), noisy_score_samples=int(len(selected)*4),
                categories={str(k): int(v) for k, v in zip(names, counts)},
                unknown_branch_frames=int(sum(int(v) for k, v in zip(names, counts) if str(k).upper() in ('UNKNOWN', 'NOT_EVALUABLE'))))
        if keep.any():
            means = np.array([scores['M3_smooth'][d, :, keep].mean() for d in range(7)])
            design = np.column_stack((INTRUSION_CM, np.ones(7)))
            slope, intercept = np.linalg.lstsq(design, means, rcond=None)[0]
            residual = means-design@np.array([slope, intercept])
            fits[name] = dict(slope_logit_per_cm=float(slope), intercept_logit=float(intercept),
                branch_mean_logit=means.tolist(), branch_mean_fit_RMSE=float(np.sqrt(np.mean(residual**2))),
                fitted_threshold_crossing_cm=float((THRESHOLD-intercept)/slope) if abs(slope) > 1e-12 else None,
                crossing_within_sampled_margin=bool(abs(slope)>1e-12 and min(INTRUSION_CM)<=(THRESHOLD-intercept)/slope<=max(INTRUSION_CM)))
        else:
            fits[name] = None
        if object_id is not None:
            if object_id.shape != (7, 16, 8, 8, 256):
                raise ValueError('First-visible object-id axes differ')
            visible = object_id[:, FRAMES][:, keep]
            visibility[name] = {str(INTRUSION_CM[d]): dict(target_rays=int((visible[d] == 0).sum()), rays=int(visible[d].size),
                frames=int(keep.sum()), frames_with_target_return=int(np.any(visible[d] == 0, axis=(-3, -2, -1)).sum())) for d in range(7)}
    total_ll = rolling_sum(frame_ll, 8)
    total_ll12 = rolling_sum(frame_ll, 12)
    pairwise = {}
    for inside in (0, 1, 2):
        for outside in PRIMARY_NEG:
            pi, ni = TEMPLATE_IDS.index(inside), TEMPLATE_IDS.index(outside)
            pair_oracle = (total_ll[:, pi]-total_ll[:, ni]).reshape(7, 4, 16)[..., FRAMES]
            pair_oracle12 = (total_ll12[:, pi]-total_ll12[:, ni]).reshape(7, 4, 16)[..., FRAMES]
            arms = dict(oracle8=pair_oracle, oracle12=pair_oracle12, M3_raw=scores['M3_raw'], M3_smooth=scores['M3_smooth'])
            key = f'in{INTRUSION_CM[inside]}_out{-INTRUSION_CM[outside]}cm'
            pairwise[key] = {name: {arm: auc_cell(score, keep, [inside], [outside]) for arm, score in arms.items()}
                             for name, keep in masks.items()}
    sensitivity = None
    if expected32 is not None:
        if expected32.shape != expected.shape or not np.isfinite(expected32).all() or np.any(expected32 < 0):
            raise ValueError('32-ray expected-template sensitivity axes/means differ')
        sensitivity = {}
        for title, templates in (('Gaussian_template16', expected), ('Gaussian_template32', expected32)):
            distances = []
            for inside in PRIMARY_POS:
                for outside in PRIMARY_NEG:
                    positive, negative = templates[inside], templates[outside]
                    variance = .5*(positive+negative)+16*ambient[..., None]
                    if np.any(variance <= 0):
                        raise ValueError('Positive Gaussian sensitivity variance required')
                    separation = ((positive-negative)**2/variance).sum(axis=(-3,-2,-1))
                    distances.append(np.sqrt(rolling_sum(separation,8)[FRAMES]))
            distances = np.stack(distances)
            sensitivity[title] = {name: dict(mean_noise_normalized_expected_distance=float(distances[:,keep].mean())
                if keep.any() else None, candidate_pairs=6, pair_frames=int(6*keep.sum())) for name,keep in masks.items()}
    sequence={}
    for delta in range(7):
        alarm=scores['M3_smooth'][delta]>=THRESHOLD
        stopped=alarm.any(1)
        first=alarm.argmax(1)
        timely=stopped&(ranges[first]>=.9)
        category=categories[delta,FRAMES,q]
        sequence[str(INTRUSION_CM[delta])]=dict(n=4,first_stops=int(stopped.sum()),timely_stops=int(timely.sum()),
            right_censored=int(4 if ranges.min()>.9 else 0),
            geometry_all_clear=bool(np.all(category=='clear')),
            geometry_all_pass=bool(np.all(category=='pass0-10cm')),
            geometry_categories=sorted(set(category.tolist())),
            lead_to0p5m_s=((ranges[first[stopped]]-.5)/.8).tolist())
    return dict(unit=unit, group=q, mode=int(truth['mode']), context=truth['context'], front_range_m=ranges.tolist(),
        cells=cells, M3_frozen_threshold_curves=threshold_curves, truth_category_counts=truth_counts, pairwise=pairwise,
        descriptive_logit_fits=fits, target_visibility=visibility, template_subdivision_sensitivity=sensitivity,
        M3_nominal_sequence=sequence)


def macro_cell(scenes, getter, keep, boot):
    values = np.array([getter(scene)['auc'] if getter(scene)['auc'] is not None else np.nan for scene in scenes])
    eligible = np.asarray(keep, bool) & np.isfinite(values)
    n = int(eligible.sum())
    if not n:
        return dict(value=None, ci95=[None, None], scenes=0, unknown_or_empty_scenes=int(np.sum(keep)), positive_n=0, negative_n=0)
    numerator = boot@np.where(eligible, np.nan_to_num(values), 0)
    denominator = boot@eligible.astype(float)
    sampled = np.divide(numerator, denominator, out=np.full(N_BOOT, np.nan), where=denominator > 0)
    interval = np.nanpercentile(sampled, [2.5, 97.5]).tolist()
    return dict(value=float(values[eligible].mean()), ci95=interval, scenes=n,
        unknown_or_empty_scenes=int(np.sum(keep)-n), positive_n=sum(getter(scenes[i])['positive_n'] for i in np.flatnonzero(eligible)),
        negative_n=sum(getter(scenes[i])['negative_n'] for i in np.flatnonzero(eligible)),
        per_scene_min=float(np.min(values[eligible])), per_scene_median=float(np.median(values[eligible])),
        per_scene_max=float(np.max(values[eligible])))


def aggregate(scenes):
    if len(scenes) != 48 or len({s['unit'] for s in scenes}) != 48:
        raise ValueError('Exactly 48 distinct controlled scenes required')
    rng = np.random.default_rng(BOOT_SEED)
    boot = np.asarray([np.bincount(rng.integers(48, size=48), minlength=48) for _ in range(N_BOOT)])
    strata = {'all': np.ones(48, bool)}
    for q, title in ((0, 'HEAD'), (1, 'BODY')):
        strata[title] = np.asarray([s['group'] == q for s in scenes])
    for context in ('none', 'panel'):
        strata[context] = np.asarray([s['context'] == context for s in scenes])
        for q, title in ((0, 'HEAD'), (1, 'BODY')):
            strata[title+'/'+context] = np.asarray([s['group'] == q and s['context'] == context for s in scenes])
    for mode in (0, 1):
        strata[f'mode{mode}'] = np.asarray([s['mode'] == mode for s in scenes])
    bins = list(scenes[0]['cells'])
    arms = ('oracle8', 'oracle12', 'M3_raw', 'M3_smooth')
    cells = {name: {bin_name: {arm: macro_cell(scenes, lambda s, b=bin_name, a=arm: s['cells'][b][a], keep, boot)
        for arm in arms} for bin_name in bins} for name, keep in strata.items()}
    pairwise = {pair: {bin_name: {arm: macro_cell(scenes, lambda s, p=pair, b=bin_name, a=arm: s['pairwise'][p][b][a], strata['all'], boot)
        for arm in arms} for bin_name in bins} for pair in scenes[0]['pairwise']}
    curves = {}
    for name, keep in strata.items():
        curves[name] = {}
        for bin_name in bins:
            curves[name][bin_name] = {}
            for delta in map(str, INTRUSION_CM):
                picked = [s['M3_frozen_threshold_curves'][bin_name][delta] for s, flag in zip(scenes, keep) if flag]
                n = sum(s['n'] for s in picked)
                stops = sum(s['stops'] for s in picked)
                per_scene = [s['stops']/s['n'] for s in picked if s['n']]
                curves[name][bin_name][delta] = dict(stops=stops, n=n, rate=stops/n if n else None,
                    scene_macro_rate=float(np.mean(per_scene)) if per_scene else None)
    crossing = {}
    distance_names = list(scenes[0]['cells'])[:-1]
    sampling = {}
    for (low,high),name in zip(BINS,distance_names):
        observed=np.concatenate([np.asarray(s['front_range_m'])[(np.asarray(s['front_range_m'])>=low)&(np.asarray(s['front_range_m'])<high)] for s in scenes])
        actual_low,actual_high = (float(observed.min()),float(observed.max())) if len(observed) else (None,None)
        sampling[name]=dict(nominal_bin_m=[low,high],actual_front_minmax_m=[actual_low,actual_high],scene_frames=len(observed),
            per_branch_noise_samples=int(len(observed)*4),
            lead_relative_to0p5m_at0p8mps_s=[(actual_low-.5)/.8,(actual_high-.5)/.8] if len(observed) else [None,None],
            all_sampled_frames_at_or_before_timely0p9m=bool(len(observed) and actual_low>=.9),
            any_sampled_frame_at_or_before_timely0p9m=bool(len(observed) and actual_high>=.9))
    for pair in pairwise:
        crossing[pair] = {}
        for arm in arms:
            counts = {name: 0 for name in reversed(distance_names)}
            counts['not_reached'] = 0
            for scene in scenes:
                reached = next((name for name in reversed(distance_names)
                    if scene['pairwise'][pair][name][arm]['auc'] is not None and scene['pairwise'][pair][name][arm]['auc'] >= .9), None)
                counts[reached or 'not_reached'] += 1
            macro_reached = next((name for name in reversed(distance_names)
                if pairwise[pair][name][arm]['value'] is not None and pairwise[pair][name][arm]['value'] >= .9), None)
            crossing[pair][arm] = dict(farthest_sampled_bin_macro_AUC_ge_0p9=macro_reached, per_scene_first_bin_counts=counts,
                farthest_macro_bin_sampling=sampling[macro_reached] if macro_reached else None,
                interpretation='Point AUC crossing on sampled distance bins; four replicas per branch; no interpolated resolution or monotonicity guarantee')
    main = cells['all']['primary1.2-2.1m']
    oracle, m3 = main['oracle8']['value'], main['M3_smooth']['value']
    def branch(oracle):
        return ('LOW_KNOWN_SCENE_DISCRIMINABILITY' if oracle is not None and oracle <= .75 else
                'READOUT_GAP_CANDIDATE' if oracle is not None and m3 is not None and oracle >= .9 and m3 <= .75 else
                'INTERMEDIATE_OR_BOTH_DISCRIMINABLE')
    decision, decision12 = branch(oracle), branch(main['oracle12']['value'])
    primary_fits=[s['descriptive_logit_fits']['primary1.2-2.1m'] for s in scenes]
    def distribution(values):
        finite=[v for v in values if v is not None and np.isfinite(v)]
        return dict(n=len(finite),median=float(np.median(finite)) if finite else None,
            IQR=np.percentile(finite,[25,75]).tolist() if finite else [None,None],
            min=float(np.min(finite)) if finite else None,max=float(np.max(finite)) if finite else None)
    slope_summary=dict(slope_logit_per_cm=distribution([f['slope_logit_per_cm'] for f in primary_fits]),
        intercept_logit=distribution([f['intercept_logit'] for f in primary_fits]),
        fitted_threshold_crossing_cm=distribution([f['fitted_threshold_crossing_cm'] for f in primary_fits]),
        branch_mean_fit_RMSE=distribution([f['branch_mean_fit_RMSE'] for f in primary_fits]),
        sampled_range_cm=[min(INTRUSION_CM),max(INTRUSION_CM)],
        crossing_in_sampled_range=sum(f['crossing_within_sampled_margin'] for f in primary_fits),
        crossing_off_sampled_range=sum(f['fitted_threshold_crossing_cm'] is not None and not f['crossing_within_sampled_margin'] for f in primary_fits),
        slope_negative=sum(f['slope_logit_per_cm']<0 for f in primary_fits),
        slope_near_zero=sum(abs(f['slope_logit_per_cm'])<=1e-12 for f in primary_fits),
        interpretation='Seven branch-mean points per scene; OLS descriptions, no causal resolution/bias decomposition')
    sensitivity_keep = np.array([s['template_subdivision_sensitivity'] is not None for s in scenes])
    sensitivity = None
    if sensitivity_keep.any():
        if int(sensitivity_keep.sum()) != 8:
            raise ValueError('Exactly first eight balanced template-sensitivity scenes required')
        sensitivity = {}
        for title in ('Gaussian_template16','Gaussian_template32'):
            sensitivity[title] = {}
            for name in bins:
                values = [s['template_subdivision_sensitivity'][title][name]['mean_noise_normalized_expected_distance']
                    for s in scenes if s['template_subdivision_sensitivity'] is not None]
                finite = [v for v in values if v is not None]
                sensitivity[title][name] = dict(scene_macro_expected_distance=float(np.mean(finite)) if finite else None,
                    per_scene=values, scenes=len(finite), interpretation='Expected-template noise-normalized Euclidean separation; not sampled AUC or a32-ray sensor result')
    sequence_summary={}
    for stratum in ('all','HEAD','BODY'):
        selected=[s for s,keep in zip(scenes,strata[stratum]) if keep]
        sequence_summary[stratum]={}
        for title,deltas,geometry_filter in (('nominal_inside1_2cm',[1,2],None),
            ('actual_clear_outside15_20cm',[-15,-20],'geometry_all_clear'),('actual_pass_outside5_10cm',[-5,-10],'geometry_all_pass')):
            allcells=[s['M3_nominal_sequence'][str(d)] for s in selected for d in deltas]
            matched=[c for c in allcells if geometry_filter is None or c[geometry_filter]]
            n=sum(c['n'] for c in matched);stops=sum(c['first_stops'] for c in matched);minutes=n*13*.2/60
            lead=[v for c in matched for v in c['lead_to0p5m_s']]
            sequence_summary[stratum][title]=dict(n=n,expected_nominal_n=sum(c['n'] for c in allcells),
                category_mismatch_or_unknown_n=sum(c['n'] for c in allcells if geometry_filter is not None and not c[geometry_filter]),
                first_stops=stops,timely_stops=sum(c['timely_stops'] for c in matched),
                right_censored=sum(c['right_censored'] for c in matched),proxy_minutes=minutes,
                first_stops_per_proxy_min=stops/minutes if minutes else None,
                median_lead_given_first_stop_s=float(np.median(lead)) if lead else None)
    return dict(status='COMPLETE', verdict=decision, units=[s['unit'] for s in scenes], cells=cells, pairwise=pairwise,
        M3_frozen_threshold_curves=curves, sampled_AUC_crossings=crossing, per_scene=scenes,
        distance_bin_sampling=sampling,primary_logit_fit_summary=slope_summary,template_subdivision_sensitivity=sensitivity,
        M3_sequence_description=sequence_summary,
        n=dict(scenes=48, intrusion_branches=7, noise_replicas=4, frame3_to15_query_samples=48*7*4*13,
            primary_nominal_query_samples=48*5*4*13),
        bootstrap=dict(replicates=N_BOOT, seed=BOOT_SEED, cluster='whole base scenes with all displacements/replicas/frames',
            pairing='one common bootstrap matrix for every arm/bin/subgroup',
            conditional_on='fixed known-scene templates, proxy noise law and frozen M3 models'),
        decision=dict(range_m=[1.2, 2.1], oracle_AUC=oracle, M3_smoothed_AUC=m3,
            oracle12_AUC=main['oracle12']['value'], original8_branch=decision, matched12_branch=decision12,
            eight_frame_alone_cannot_close_readout=bool(oracle is not None and oracle<=.75 and main['oracle12']['value'] is not None and main['oracle12']['value']>.75),
            interpretation='Descriptive next-question branches only;12frame matches maximal physical exposure history of five smoothed eight-frame M3 logits; no automatic training, baseline promotion or physical sensor ceiling claim'))


def validate_plan(plan):
    expected = dict(run=RUN, intrusion_cm=INTRUSION_CM, K=4, frames=FRAMES.tolist(), bins=BINS,
                    threshold=THRESHOLD, template_window=8, M3_smoothing_weights=[1,2,4,8,16])
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError('PLAN/evaluator mismatch: '+key)
    units = plan['units']
    if len(units) != 48 or len(set(units)) != 48 or any(u%3 not in (0,1) for u in units):
        raise ValueError('Expected48 distinct straight-motion base scenes')
    params = plan['nominal_sensor_params']
    if params['noise_scale'] != 1 or params['output_gain'] != 1:
        raise ValueError('UNSUPPORTED exact signed law: nonnominal noise_scale/output_gain')


def bind_receipt(path, unit, plan_digest, hashes, expected_file):
    receipt = read(path)
    if receipt.get('status') != 'COMPLETE' or receipt.get('unit') != unit or receipt.get('plan_sha256') != plan_digest:
        raise ValueError('Incomplete or wrong unit receipt: '+str(path))
    key = str(expected_file.relative_to(OUT)).replace('\\','/')
    if set(receipt['output_sha256']) != {key}:
        raise ValueError('Receipt output identity differs: '+str(path))
    digest = sha(expected_file)
    if receipt['output_sha256'][key] != digest:
        raise ValueError('Unit payload changed: '+str(expected_file))
    hashes[str(expected_file)] = digest
    hashes[str(path)] = sha(path)
    return receipt


def evaluation_inputs():
    plan_path, scores_path, amendment_path = OUT/'PLAN.json', OUT/'scores_receipt.json', OUT/'PLAN_AMENDMENT.json'
    plan, scores, amendment = read(plan_path), read(scores_path), read(amendment_path)
    validate_plan(plan)
    plan_digest = sha(plan_path)
    if amendment.get('run') != RUN or amendment.get('original_plan_sha256') != plan_digest or amendment.get('oracle_matched_history_window') != 12:
        raise ValueError('Matching pre-evaluation twelve-frame amendment required')
    if scores.get('status') != 'COMPLETE' or scores.get('plan_sha256') != plan_digest or scores.get('units') != plan['units']:
        raise ValueError('All48 complete sealed M3 scores are required before truth access')
    hashes = {str(p): sha(p) for p in (plan_path,scores_path,amendment_path)}
    for unit in plan['units']:
        obs = OUT/'observations'/f'unit{unit}.npz'
        template = OUT/'templates'/f'unit{unit}.npz'
        truth = OUT/'truth'/f'unit{unit}.json'
        score = OUT/'scores'/f'unit{unit}.npz'
        observation_receipt = bind_receipt(obs.with_suffix('.json'),unit,plan_digest,hashes,obs)
        if observation_receipt.get('noise_scale') != 1 or observation_receipt.get('output_gain') != 1:
            raise ValueError('UNSUPPORTED per-unit signed noise law')
        bind_receipt(template.with_suffix('.json'),unit,plan_digest,hashes,template)
        bind_receipt(truth.with_name(f'unit{unit}.receipt.json'),unit,plan_digest,hashes,truth)
        bind_receipt(score.with_suffix('.json'),unit,plan_digest,hashes,score)
        key = f'scores/unit{unit}.npz'
        if scores['outputs'].get(key) != hashes[str(score)]:
            raise ValueError('Complete score aggregate does not bind this unit')
    expected_outputs = {f'scores/unit{u}.npz' for u in plan['units']}
    if set(scores['outputs']) != expected_outputs:
        raise ValueError('Complete score aggregate unit outputs differ')
    # Bind evaluator identity separately without rewriting the generation PLAN.
    request = dict(run=RUN,plan_sha256=plan_digest,amendment_sha256=sha(amendment_path),
        scores_receipt_sha256=sha(scores_path),evaluator_sha256=sha(__file__),input_sha256=hashes,
        operations=dict(training=False,model_inference=False,privileged_counts=False,ambient_estimate=False,
            known_scene_expected_templates=True,truth_read_after_complete_scores=True),
        exact_law='Skellam(expected+8ambient,8ambient)',oracle_windows=[8,12],
        primary_template_ids=[0,1,4,5,6],bootstrap=dict(replicates=N_BOOT,seed=BOOT_SEED))
    request_path = OUT/'evaluation_request.json'
    if request_path.exists():
        if read(request_path) != request:
            raise ValueError('Existing evaluation request differs; preserve prior evidence')
    else:
        save(request_path,request)
    return plan, amendment, request


def verify_hashes(hashes):
    for path,digest in hashes.items():
        if sha(path) != digest:
            raise ValueError('Evaluation input changed: '+path)


def metric_text(metric):
    if metric['value'] is None:
        return '-- (0 scenes)'
    return f"{metric['value']:.3f} [{metric['ci95'][0]:.3f},{metric['ci95'][1]:.3f}] (s={metric['scenes']}; n+={metric['positive_n']}, n−={metric['negative_n']})"


def write_report(result):
    main = result['cells']['all']['primary1.2-2.1m']
    lines = [result['verdict'],'','# 已知场景受控位移：signed CNH信息测量','',
        '固定48个直行场景，HEAD/BODY×none/panel×mode0/1各6；每场景7个横向位移分支，每分支4个独立光子噪声序列。仅移动同一目标的横向位置，尺寸、反射率、背景与真实姿态固定；每噪声复本的姿态估计跨位移共享。无新训练。','',
        '主比较为名义伸入1/2cm对身体外10/15/20cm；精确外10cm仍属保留三级定义的pass0–10cm，不把全部外侧分支称为clear。先在每场景1.2–2.1m距离范围计算AUC，再等权宏平均；噪声复本、帧和位移不作为独立场景。已知场景oracle采用零噪声候选模板和精确Skellam类内混合似然比；每样本评分全部候选模板，不选择真实位移模板。oracle只读取signed CNH、期望模板与公共ambient，不读取潜在counts/ambient_estimate。','',
        '原PLAN的8帧oracle保留；评价前修正补充12帧oracle，匹配M3五个8帧读出的最大物理历史。M3报告原始logit和固定五logit因果平滑，共用旧阈值，无本批校准。12帧oracle仍知道完整场景和真实姿态，是条件信息诊断，不是部署方法。','',
        '| 1.2–2.1m主比较 | 场景宏AUC及95%区间；场景与样本分母 |','| --- | --- |']
    for arm in ('oracle8','oracle12','M3_raw','M3_smooth'):
        lines.append('| '+arm+' | '+metric_text(main[arm])+' |')
    lines += ['', '描述性分支：'+json.dumps(result['decision'],ensure_ascii=False),'',
        '8帧低可辨性而12帧更高时，不能用原8帧结果关闭读出探索。以上分支不自动启动训练，不晋升基线，也不证明所有方法不可能。','',
        '| 距离档 | oracle8 | oracle12 | M3原始 | M3平滑 |','| --- | --- | --- | --- | --- |']
    for name,metrics in result['cells']['all'].items():
        lines.append('| '+name+' | '+' | '.join(metric_text(metrics[a]) for a in ('oracle8','oracle12','M3_raw','M3_smooth'))+' |')
    lines += ['', '![Conditional displacement diagnostic](displacement_diagnostic.png)', '',
        '| 距离档 | 实际采样front范围，m | 相对0.5m提前量范围，s | 全档满足及时≥0.9m |',
        '| --- | --- | --- | --- |']
    for name,sampling in result['distance_bin_sampling'].items():
        lines.append('| '+name+' | '+str(sampling['actual_front_minmax_m'])+' | '+str(sampling['lead_relative_to0p5m_at0p8mps_s'])+' | '+str(sampling['all_sampled_frames_at_or_before_timely0p9m'])+' |')
    lines += ['', '| 分层，主1.2–2.1m | oracle8 | oracle12 | M3平滑 |','| --- | --- | --- | --- |']
    for name,bins in result['cells'].items():
        if name != 'all':
            lines.append('| '+name+' | '+' | '.join(metric_text(bins['primary1.2-2.1m'][a]) for a in ('oracle8','oracle12','M3_smooth'))+' |')
    lines += ['', '| 独立位移对，主1.2–2.1m | oracle8 | oracle12 | M3平滑 |','| --- | --- | --- | --- |']
    for pair,bins in result['pairwise'].items():
        lines.append('| '+pair+' | '+' | '.join(metric_text(bins['primary1.2-2.1m'][a]) for a in ('oracle8','oracle12','M3_smooth'))+' |')
    lines += ['', '| 位移对 | oracle8最远采样档AUC≥.9 | oracle12 | M3平滑 |','| --- | --- | --- | --- |']
    for pair,arms in result['sampled_AUC_crossings'].items():
        lines.append('| '+pair+' | '+' | '.join(str(arms[a]['farthest_sampled_bin_macro_AUC_ge_0p9']) for a in ('oracle8','oracle12','M3_smooth'))+' |')
    lines += ['', '首次可分仅为采样距离档的AUC点值 crossing；每场景的首次档/not-reached分布见result.json，不插值为厘米分辨率，不保证随距离单调，也不是及时报警通过。','',
        '| 距离档，M3冻结阈值 | 内1cm | 内2cm | 内5cm | 外5cm | 外10cm | 外15cm | 外20cm |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for name,curve in result['M3_frozen_threshold_curves']['all'].items():
        texts=[]
        for delta in map(str,INTRUSION_CM):
            cell=curve[delta]
            texts.append(f"{cell['stops']}/{cell['n']} ({cell['rate']:.3f})" if cell['rate'] is not None else '--')
        lines.append('| '+name+' | '+' | '.join(texts)+' |')
    lines += ['', f'原冻结阈值={THRESHOLD!r}。上述逐查询/帧触发比例含噪声复本，不是首停率或现场误停负担；内5cm和外5cm不参与主模板混合或主分支。','',
        '| 距离档 | 名义浅分支真实类别（分支×帧） | 名义外10/15/20分支真实类别（分支×帧） |', '| --- | --- | --- |']
    for name in result['cells']['all']:
        columns=[]
        for role in ('nominal_shallow','nominal_outside10_15_20'):
            counts={};unknown=0;denominator=0
            for scene in result['per_scene']:
                cell=scene['truth_category_counts'][name][role];unknown+=cell['unknown_branch_frames'];denominator+=cell['branch_frames']
                for category,n in cell['categories'].items():
                    counts[category]=counts.get(category,0)+n
            columns.append(json.dumps(counts,ensure_ascii=False)+f'; n={denominator}, UNKNOWN/NOT_EVALUABLE={unknown}')
        lines.append('| '+name+' | '+' | '.join(columns)+' |')
    lines += ['', '未知类别保留分母；低信号或候选均值相同的样本仍参与AUC，平局计0.5，不能当畅通。名义分支判别与真实三级碰撞语义分别报告。','',
        '| 距离档，M3 logit对位移OLS | 场景斜率中位数，logit/cm | 截距中位数 | 分支均值拟合RMSE中位数 | 样本范围内阈值交点场景 |',
        '| --- | --- | --- | --- | --- |']
    for name in result['cells']['all']:
        fits=[s['descriptive_logit_fits'][name] for s in result['per_scene'] if s['descriptive_logit_fits'][name] is not None]
        if fits:
            lines.append(f"| {name} | {np.median([f['slope_logit_per_cm'] for f in fits]):.4f} | {np.median([f['intercept_logit'] for f in fits]):.3f} | {np.median([f['branch_mean_fit_RMSE'] for f in fits]):.3f} | {sum(f['crossing_within_sampled_margin'] for f in fits)}/{len(fits)} |")
    lines += ['', '斜率、截距和拟合阈值交点是logit响应描述。flat或截距偏移不能武断拆成传感器分辨率与模型偏置的因果归因；非线性残差和已知场景先验同样可能影响它们。','',
        '主1.2–2.1m场景斜率/截距/交点分布：','', '```json',json.dumps(result['primary_logit_fit_summary'],ensure_ascii=False,indent=2),'```','',
        '| M3序列描述，分层 | 名义内1/2cm及时/分母 | 实际clear外15/20首停/代理分钟 | 实际pass外5/10首停/分母 |',
        '| --- | --- | --- | --- |']
    for name,cells in result['M3_sequence_description'].items():
        inside,clear,passing=(cells[k] for k in ('nominal_inside1_2cm','actual_clear_outside15_20cm','actual_pass_outside5_10cm'))
        lines.append(f"| {name} | {inside['timely_stops']}/{inside['n']} | {clear['first_stops']}/{clear['proxy_minutes']:.2f} | {passing['first_stops']}/{passing['n']} |")
    lines += ['', 'M3及时按首次报警对应front≥0.9m；未覆盖0.9m的轨迹右删失单列，不作漏停。本测量分布的名义内1/2cm轨迹分母为384，不能直接与旧26/31比较；没有新模型胜利。clear成本只计全13帧实际clear的外15/20，外5/10实际pass仅描述，不计误停；类别不一致/UNKNOWN保留在结果的原名义分母与单列计数。清晰代理分钟含首停后暴露，不是步行分钟。','',
        '| 主1.2–2.1m目标可见性 | target first-visible rays/全部采样rays | 至少一条target ray的分支帧 |','| --- | --- | --- |']
    for delta in map(str,INTRUSION_CM):
        cells=[s['target_visibility']['primary1.2-2.1m'][delta] for s in result['per_scene']]
        lines.append(f"| {delta}cm | {sum(c['target_rays'] for c in cells)}/{sum(c['rays'] for c in cells)} | {sum(c['frames_with_target_return'] for c in cells)}/{sum(c['frames'] for c in cells)} |")
    lines += ['', '可见ray为未加角权的数值采样数量，不是物理面积、信号可检出性或UNKNOWN语义。','']
    if result['template_subdivision_sensitivity'] is not None:
        lines += ['首8个平衡场景的16/32数值采样敏感性，仅比较候选期望间噪声归一化距离，不生成32采样观测或32传感器AUC：','',
            '```json',json.dumps(result['template_subdivision_sensitivity'],ensure_ascii=False,indent=2),'```','']
    lines += ['边界：已消费受控仿真Development；光子、角响应、脉冲和安装都是代理假设。panel gap .34–.40m比先前near-panel更宽，避免目标/背景相交，不能直接外推旧场景。已知场景template oracle知道尺寸、反射率、背景、真实姿态及候选位移集合，优于部署可用信息。','',
        '1000次整场景共同bootstrap，仅条件于本场景设计、K4噪声、固定模板和冻结模型；不含重训练、模拟器标定和硬件不确定性。名义位移识别不是完整碰撞报警能力，8/12帧条件判别也不是任意长历史或真实设备物理上限。','',
        '分母：'+json.dumps(result['n'],ensure_ascii=False),'',
        '输入封存、评价前12帧修正和评价源码SHA见result.json provenance；原8帧PLAN与失败证据保留。','']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')


def write_figure(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12.5,4.6),layout='constrained')
    colors={'oracle8':'#314e8b','oracle12':'#19a290','M3_raw':'#cf9e42','M3_smooth':'#bd5379'}
    names={'oracle8':'Known scene, 8 frames','oracle12':'Known scene, 12 frames','M3_raw':'M3 raw','M3_smooth':'M3 causal five'}
    distance_names=list(result['distance_bin_sampling'])
    x=np.array([np.mean(result['distance_bin_sampling'][n]['actual_front_minmax_m']) for n in distance_names])
    for arm in ('oracle8','oracle12','M3_raw','M3_smooth'):
        metrics=[result['cells']['all'][n][arm] for n in distance_names]
        y=np.array([m['value'] for m in metrics]);ci=np.array([m['ci95'] for m in metrics])
        axes[0].plot(x,y,'o-',color=colors[arm],label=names[arm],linewidth=1.7,markersize=4)
        axes[0].fill_between(x,ci[:,0],ci[:,1],color=colors[arm],alpha=.1)
    axes[0].axhline(.5,color='#999999',linestyle=':',linewidth=1)
    axes[0].set(xlabel='Target front range (m)',ylabel='Equal-scene macro AUC',ylim=(0,1.03),title='Nominal in1/2 vs out10/15/20 cm')
    axes[0].legend(loc='lower left',frameon=False,fontsize=8)
    axes[0].grid(alpha=.16)
    xs=np.asarray(INTRUSION_CM,float);order=np.argsort(xs)
    rng=np.random.default_rng(2026100305)
    for q,name,color in ((0,'HEAD','#314e8b'),(1,'BODY','#d18a4d')):
        values=np.array([s['descriptive_logit_fits']['primary1.2-2.1m']['branch_mean_logit'] for s in result['per_scene'] if s['group']==q])
        for i,margin in enumerate(xs):
            axes[1].scatter(np.full(len(values),margin)+rng.uniform(-.22,.22,len(values)),values[:,i],color=color,alpha=.18,s=10,linewidths=0)
        axes[1].plot(xs[order],values.mean(0)[order],'o-',color=color,label=name+' scene mean',linewidth=1.8,markersize=4)
    axes[1].axhline(THRESHOLD,color='#555555',linestyle='--',linewidth=1.3,label='Frozen threshold')
    axes[1].set(xlabel='Signed intrusion (cm; outside is negative)',ylabel='M3 smoothed logit',title='1.2–2.1 m: scene means and distribution',xticks=[-20,-15,-10,-5,0,5])
    axes[1].legend(loc='best',frameon=False,fontsize=8)
    axes[1].grid(alpha=.16)
    fig.suptitle('Controlled displacement:48 scenes, four noise replicas; conditional simulation only',fontsize=12)
    for suffix in ('png','svg'):
        fig.savefig(OUT/f'displacement_diagnostic.{suffix}',dpi=180)
    plt.close(fig)


def evaluate():
    result_path = OUT/'result.json'
    if result_path.exists():
        result=read(result_path)
        if result.get('status') != 'COMPLETE':
            raise ValueError('Existing incomplete result requires inspection')
        verify_hashes(result['provenance']['input_sha256'])
        print('Existing COMPLETE displacement result verified; no repeat',flush=True)
        return result
    plan,amendment,request=evaluation_inputs()
    scenes=[];started=time.monotonic()
    cache_dir=OUT/'evaluation_units';cache_dir.mkdir(exist_ok=True)
    for unit in plan['units']:
        cache=cache_dir/f'unit{unit}.json'
        if cache.exists():
            stored=read(cache)
            if stored.get('request_sha256')!=sha(OUT/'evaluation_request.json'):
                raise ValueError('Existing likelihood cache differs from sealed request')
            scenes.append(stored['scene'])
            print('reused likelihood scene',unit,flush=True)
            continue
        with np.load(OUT/'observations'/f'unit{unit}.npz',allow_pickle=False) as obs:
            hist,ambient=obs['hist'],obs['ambient']
        with np.load(OUT/'templates'/f'unit{unit}.npz',allow_pickle=False) as templates:
            expected,object_id=templates['expected'],templates['object_id']
            expected32=templates['expected32'] if 'expected32' in templates.files else None
        with np.load(OUT/'scores'/f'unit{unit}.npz',allow_pickle=False) as scores:
            raw=scores['raw']
        truth=read(OUT/'truth'/f'unit{unit}.json')
        scene=analyze_scene(unit,hist,expected,ambient,truth,raw,object_id,expected32)
        save(cache,dict(status='COMPLETE',request_sha256=sha(OUT/'evaluation_request.json'),scene=scene))
        scenes.append(scene)
        print('likelihood completed',len(scenes),'/48',unit,'elapsed_s',round(time.monotonic()-started,1),flush=True)
    result=aggregate(scenes)
    verify_hashes(request['input_sha256'])
    result['provenance']=dict(**request,plan=plan,amendment=amendment,evaluation_elapsed_s=time.monotonic()-started,
        evaluation_request_sha256=sha(OUT/'evaluation_request.json'))
    write_figure(result)
    write_report(result)
    save(result_path,result)
    print(result['verdict'],json.dumps(result['decision']),flush=True)
    return result


def check():
    signed = np.arange(-12, 31)
    for expectation, background in ((0., 4.), (3., 4.), (20., 8.), (100., 32.)):
        np.testing.assert_allclose(skellam_logpmf_signed(signed, expectation, background),
            skellam.logpmf(signed, expectation+background, background), rtol=1e-11, atol=1e-11)
    # Far tails can underflow scipy's PMF; the exact positive series stays finite.
    tail = skellam_logpmf_signed(np.array([2000.]), np.array([0.]), np.array([32.]))
    assert np.isfinite(tail).all() and tail[0] < -1000
    order, x = np.array([0., 3., 100.]), np.array([1., 5., 40.])
    np.testing.assert_allclose(_log_bessel_series(order, x), np.log(ive(order, x))+x, atol=1e-11)
    assert binary_auc([0., 1.], [0., -1.]) == .875
    assert binary_auc([0., 0.], [0., 0.]) == .5
    rng = np.random.default_rng(2026100305)
    values = rng.normal(size=(3, 5, 16))
    actual = rolling_sum(values, 8)
    for frame in range(16):
        np.testing.assert_allclose(actual[..., frame], values[..., max(0, frame-7):frame+1].sum(-1))
    llr = mixture_llr(values, [0, 1], [2, 3, 4])
    changed = values.copy(); changed[..., 10:] += 100
    np.testing.assert_array_equal(mixture_llr(changed, [0, 1], [2, 3, 4])[..., :10], llr[..., :10])
    zero = mixture_llr(np.zeros((2, 5, 16)), [0, 1], [2, 3, 4])
    assert np.allclose(zero, 0)
    try:
        skellam_logpmf_signed(0, 1, 1, noise_scale=.5)
    except ValueError:
        pass
    else:
        raise AssertionError('Non-nominal noise law accepted')
    print('PASS exact signed-CNH law, finite far-tail series, class-balanced mixtures, causal eight-frame sums and tied AUC; synthetic only')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('check','evaluate','finalize'))
    args = parser.parse_args()
    {'check':check,'evaluate':evaluate,'finalize':evaluate}[args.stage]()
