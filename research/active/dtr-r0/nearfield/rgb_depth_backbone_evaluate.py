"""Frozen RGB backbone comparison, inference caches only, consumed Development.

Run calibrate first (original train56 + pooled cal304), then final (eval136).
Prediction rows contain scan/frame/rgb_sha256/depth_K/depth_shape/path/sha256;
path is a native-grid metric .npy, or .npz with key depth. No model is loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import time
import numpy as np

from rgb_body_query_interval_distribution import load, utc
from rgb_body_query_input_diagnostic import sha
from rgb_body_query_calibrated_geometry import fit_models, correct
from rgb_body_query_bounded_evaluate import manifest_rows
from rgb_body_query_negative_frozen_score import paths, EVAL, numeric
from rgb_body_query_query_calibration_probe import FREE, COHORTS, nth_score, select_cutoff, write
from rgb_body_query_reference_eval import rays, ray_interval
from rgb_body_query_residual_readout import summary, evaluate, paired
from rgb_body_query_scene_diagnostic import write_csv

BANDS = ('0.3-0.8m', '0.8-1.5m', '1.5-3m')
ARMS = ('raw', 'affine', 'depthpro_affine')
QUANTILES = (0., .01, .05, .5, .95, .99, 1.)


def key(row):
    return row['scan'], row['frame']


def train_manifest(repo):
    _, root, _, _ = paths(repo)
    path = root/'train-cal-sensor/dataset_manifest.json'
    source = load(path)
    rows = [r for r in source['rows'] if r['split'] == 'train']
    assert len(rows) == 56 and len({r['environment'] for r in rows}) == 7
    return dict(rows=rows, queries=source['queries']), path


def prediction_index(path, registered):
    data = load(registered(path))
    assert data['status'] == 'COMPLETE', 'Partial model is NOT_RUN; do not silently subset frames'
    lookup = {key(r): r for r in data['rows']}
    assert len(lookup) == len(data['rows']), 'Duplicate prediction identities'
    return lookup


def native_prediction(ref, predictions, registered):
    p = predictions[key(ref)]
    for field in ('rgb_sha256', 'depth_K', 'depth_shape'):
        assert p[field] == ref[field], f'Public input identity mismatch: {field}'
    path = registered(p['path'], p['sha256'])
    if path.suffix == '.npy':
        z = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as f:
            z = f['depth']
    assert list(z.shape) == ref['depth_shape']
    return np.asarray(z, np.float64)


def fit_train(repo, predictions, out, registered, check):
    manifest, path = train_manifest(repo)
    registered(path)
    xs, ys, identities = [], [], []
    for ref in manifest['rows']:
        check()
        z = native_prediction(ref, predictions, registered)
        registered(ref['reference_path'], ref['reference_sha256'])
        with np.load(ref['reference_path']) as f:
            gt, observed = f['depth'], f['observed']
        mask = observed & np.isfinite(gt) & (gt > 0) & np.isfinite(z) & (z > 0)
        xs.append(np.log(z[mask])); ys.append(np.log(gt[mask].astype(np.float64)))
        identities.append(dict(scan=ref['scan'], frame=ref['frame'], environment=ref['environment'], paired_pixels=int(mask.sum())))
    x, y = np.concatenate(xs), np.concatenate(ys)
    models, history = fit_models(x, y)
    fit = dict(status='FROZEN_TRAIN_ONLY', frozen_utc=utc(), affine=models['log_affine'], history=history,
               train_frames=identities, paired_pixels=len(x), recipe='Original OLS +8 Huber IRLS; .2 log residual; slope clipped [.25,4]; pooled valid observed pixels',
               eval_reference_read=False, cal_reference_read=False, neural_training_calls=0,
               recipe_source_sha256=sha(fit_models.__code__.co_filename))
    write(out/'fit.json', fit)
    return fit


def baseline_index(repo, cohort, registered):
    root = repo/'artifacts.local/work/rgb-body-query-bounded-residual-dev-20261010'
    return prediction_index(root/f'predictions/{cohort}/predictions.json', registered)


def metrics(truth, prediction):
    if not len(truth):
        return dict(status='NOT_EVALUABLE', paired_pixels=0, meanAbsLog=None, medianAbsRel=None,
                    absLog_quantiles=None, absRel_quantiles=None, signedLog_quantiles=None, prediction_quantiles_m=None)
    log = np.log(prediction/truth)
    rel = np.abs(prediction-truth)/truth
    return dict(status='EVALUABLE', paired_pixels=len(truth), meanAbsLog=float(np.abs(log).mean()), medianAbsRel=float(np.median(rel)),
                absLog_quantiles=np.quantile(np.abs(log), QUANTILES).tolist(), absRel_quantiles=np.quantile(rel, QUANTILES).tolist(),
                signedLog_quantiles=np.quantile(log, QUANTILES).tolist(), prediction_quantiles_m=np.quantile(prediction, QUANTILES).tolist())


def score(repo, cohort, manifest, predictions, affine, out, registered, check, checks):
    baseline = baseline_index(repo, 'cal' if cohort == 'cal_fit_description' else cohort, registered)
    info = load(registered(repo/'artifacts.local/work/rgb-body-query-interval-distribution-dev-20261010/candidate/training_inputs.json'))
    data = {a: [] for a in ARMS}
    pixel_parts = defaultdict(list); frame_metrics = []
    out.mkdir(parents=True, exist_ok=True)
    for ref in manifest['rows']:
        check()
        z = native_prediction(ref, predictions, registered)
        bp = baseline[key(ref)]
        registered(bp['sampled_depth_path'], bp['sampled_depth_sha256'])
        registered(ref['reference_path'], ref['reference_sha256'])
        with np.load(bp['sampled_depth_path']) as f:
            dp = f['depth']
        with np.load(ref['reference_path']) as f:
            gt, observed, labels = f['depth'], f['observed'], f['labels']
        assert dp.shape == z.shape == gt.shape and labels.shape == (27, *gt.shape)
        public_valid = np.isfinite(dp) & (dp > 0)
        candidate_valid = np.isfinite(z) & (z > 0)
        # The frozen public domain supplies all denominators. Invalid candidates
        # cannot supply support; they never remove a reference negative query.
        depths = dict(raw=z, affine=correct(z, affine), depthpro_affine=correct(dp, info['affine']))
        rx, ry = rays(ref['depth_K'], ref['depth_shape'])
        for j, query in enumerate(manifest['queries']):
            check()
            entry, exit, domain = ray_interval(rx, ry, query)
            lab = labels[j]
            pos, free, unk = [int((lab == n).sum()) for n in (1, 0, 2)]
            state = 'POSITIVE' if pos >= 16 else FREE if pos == 0 and unk == 0 and free >= 16 else 'UNKNOWN'
            assert state == ref['queries'][j]['state']
            checks['unchanged_reference_queries'] += 1
            for arm, estimate in depths.items():
                mask = public_valid & domain
                if arm != 'depthpro_affine': mask &= candidate_valid
                s = np.full(z.shape, -np.inf)
                s[mask] = np.minimum(estimate[mask]-entry[mask], exit[mask]-estimate[mask])
                data[arm].append(dict(cohort=cohort, environment=ref['environment'], scan=ref['scan'], frame=ref['frame'],
                    query=query['name'], distance_band=f'{query["low"][2]:g}-{query["high"][2]:g}m', reference_state=state,
                    valid_ray_count=int(mask.sum()), frozen_public_domain_rays=int((public_valid & domain).sum()),
                    missing_candidate_public_rays=int((public_valid & domain & ~candidate_valid).sum()) if arm != 'depthpro_affine' else 0,
                    query_score=nth_score(s), known_positive_score=nth_score(s[lab == 1])))
        # Diagnostic bands use GT Z only, never query scores or threshold choice.
        observed_valid = observed & np.isfinite(gt) & (gt > 0)
        for band, lo, hi in zip(BANDS, (.3, .8, 1.5), (.8, 1.5, 3.)):
            gtband = observed_valid & (gt >= lo) & (gt < hi if hi < 3. else gt <= hi)
            for arm, estimate in depths.items():
                valid = np.isfinite(estimate) & (estimate > 0)
                mask = gtband & valid
                gv, pv = gt[mask].astype(np.float64), estimate[mask]
                pixel_parts[ref['scan'], band, arm].append((gv, pv))
                frame_metrics.append(dict(cohort=cohort, scan=ref['scan'], frame=ref['frame'], band=band, arm=arm,
                    observed_band_pixels=int(gtband.sum()), missing_prediction_pixels=int((gtband & ~valid).sum()), **metrics(gv, pv)))
        checks['frames_scored'] += 1
    pixel_metrics = {}
    for (scan, band, arm), parts in pixel_parts.items():
        check()
        gv = np.concatenate([p[0] for p in parts]); pv = np.concatenate([p[1] for p in parts])
        pixel_metrics[f'{scan}/{band}/{arm}'] = metrics(gv, pv)
    write(out/'pixel_metrics.json', dict(quantiles=QUANTILES, groups=pixel_metrics,
        definition='Native observed sensor pixels; GT Z bands; missing excluded and counted; no clipping/trimming; consumed Development',
        tails='Original full native prediction and native reference caches preserve every tail value'))
    write_csv(out/'pixel_frame_metrics.csv', frame_metrics)
    write(out/'scores.json', dict(arms=data))
    # Check old baseline scores rather than rerunning DepthPro inference.
    old = repo/'artifacts.local/work/rgb-body-query-bounded-residual-dev-20261010/evaluation'
    priorpath = old/'calibration/scores.json' if cohort == 'cal_fit_description' else old/f'final/{cohort}_scores.json'
    previous = { (r['scan'], r['frame'], r['query']): r for r in numeric(load(registered(priorpath))['arms']['affine']) }
    for row in data['depthpro_affine']:
        prior = previous[row['scan'], row['frame'], row['query']]
        assert all(row[k] == prior[k] for k in ('query_score', 'known_positive_score', 'reference_state', 'valid_ray_count'))
        checks['exact_cached_depthpro_query_reproduction'] += 1
    return data


def select(data):
    return {a: select_cutoff([r['query_score'] for r in rows if r['reference_state'] == FREE]) for a, rows in data.items()}


def expanded_summary(rows, pairs=False):
    result = summary(rows, pairs)
    captures = defaultdict(list)
    for row in rows: captures[row['scan']].append(row)
    for scan, rr in captures.items():
        current = summary(rr, pairs)
        result['capture/'+scan] = current['all']
        for band in BANDS:
            if 'band/'+band in current: result['capture_band/'+scan+'/'+band] = current['band/'+band]
    return result


def save(out, protocol, evaluation, cuts):
    out.mkdir(parents=True, exist_ok=True)
    rows, pp, summaries, pair_summaries = [], [], {}, {}
    for cohort, arms in evaluation.items():
        write(out/f'{cohort}_evaluation.json', arms)
        summaries[cohort] = {a: expanded_summary(rr) for a, rr in arms.items()}
        for arm, groups in summaries[cohort].items():
            rows.extend(dict(protocol=protocol, cohort=cohort, arm=arm, group=g, **s) for g, s in groups.items())
        for arm in ARMS[:2]:
            pairs = paired(arms[arm], arms['depthpro_affine'], cohort, arm, 'depthpro_affine', protocol)
            pp.extend(pairs); pair_summaries[cohort+'/'+arm] = expanded_summary(pairs, True)
    write(out/'results.json', dict(status='COMPLETE', protocol=protocol, thresholds=cuts, cohorts=summaries, pairs=pair_summaries,
        limits='Consumed Development; sampled FREE not volume clearance; native sensor truth; correlated frames; no real-world safety claim'))
    write_csv(out/'summary.csv', rows); write_csv(out/'query_pairs.csv', pp)


def run(args):
    started = time.perf_counter()
    out = args.output.resolve(); repo = args.repo.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'plan.json').exists(): raise FileExistsError('Preserve earlier stage; use a new attempt and account prior wall')
    receipt = dict(status='STARTING', stage=args.stage, model=args.model, GPU_s=0, training_calls=0, download_bytes=0)
    inputs, checks = [], Counter(); registered_hashes = {}
    def check():
        if time.perf_counter()-started >= args.budget_s: raise TimeoutError('Evaluation command-wall allocation reached')
    def registered(path, expected=None):
        check(); path = Path(path)
        digest = registered_hashes.get(str(path))
        if digest is None:
            digest = sha(path); registered_hashes[str(path)] = digest; inputs.append(dict(path=str(path), sha256=digest))
        if expected is not None: assert digest == expected, f'Changed input: {path}'
        return path
    write(out/'plan.json', dict(stage=args.stage, model=args.model, frozen_utc=utc(), source_sha256=sha(__file__),
        predictions_sha256=sha(args.predictions), budget_cpu_command_wall_s=args.budget_s,
        query_score='16th largest absolute min(z-entry,exit-z); reference query states unchanged; frozen DP public valid domain',
        affine_recipe='Same56train only, original 8-Huber-IRLS recipe',
        cutoff='Other two captures strict sampledFREE5% LOCO main; pooled304cal supplemental; preserve ties; no eval threshold tuning',
        strong_signal='Literal both additional captures near witness increase with no FREE increase and <=5% mid/far net loss; 40777065 nearPOS0 therefore literal signal NOT_EVALUABLE',
        data_role='Consumed Development', GPU=0, neural_training=0))
    shutil.copyfile(__file__, out/'executed_evaluate.py')
    try:
        predictions = prediction_index(args.predictions, registered)
        if args.stage == 'calibrate':
            fit = fit_train(repo, predictions, out, registered, check)
            manifest, sourcepaths = manifest_rows(repo)
            for path in sourcepaths: registered(path)
            assert {key(r) for r in manifest['rows']}.isdisjoint({(r['scan'], r['frame']) for r in fit['train_frames']})
            data = score(repo, 'cal_fit_description', manifest, predictions, fit['affine'], out/'cal_scores', registered, check, checks)
            assert all(len(rr) == 8208 for rr in data.values())
            cuts = select(data)
            assert all(c['negative_queries'] == 588 for c in cuts.values())
            write(out/'thresholds.json', dict(status='FROZEN_BEFORE_EVALUATION', frozen_utc=utc(), arms=cuts,
                evaluation_reference_read=False, fit_sha256=sha(out/'fit.json'), prediction_manifest_sha256=sha(args.predictions)))
            ev = {'cal_fit_description': {a: evaluate(rr, cuts[a], False) for a, rr in data.items()}}
            save(out/'cal_fit_description', 'pooled-cal-description', ev, cuts)
        else:
            assert args.calibration is not None
            cal = args.calibration.resolve()
            assert load(registered(cal/'terminal.json'))['status'] == 'COMPLETE'
            frozen = load(registered(cal/'thresholds.json'))
            fit = load(registered(cal/'fit.json'))
            assert frozen['status'] == 'FROZEN_BEFORE_EVALUATION'
            assert frozen['fit_sha256'] == sha(cal/'fit.json') and frozen['prediction_manifest_sha256'] == sha(args.predictions)
            caldata = load(registered(cal/'cal_scores/scores.json'))
            calids = {key(r) for r in caldata['arms']['raw']}
            trainids = {key(r) for r in fit['train_frames']}
            _, _, _, folders = paths(repo); data = {}; evalids = set()
            for cohort in EVAL:
                manifest = load(registered(folders[cohort]/'dataset_manifest.json'))
                ids = {key(r) for r in manifest['rows']}
                assert ids.isdisjoint(calids | trainids | evalids); evalids.update(ids)
                data[cohort] = score(repo, cohort, manifest, predictions, fit['affine'], out/'scores'/cohort, registered, check, checks)
            assert len(evalids) == 136 and sum(len(v['raw']) for v in data.values()) == 3672
            cuts = frozen['arms']
            pooled = {c: {a: evaluate(rr, cuts[a], False) for a, rr in ss.items()} for c, ss in data.items()}
            save(out/'pooled-supplemental', 'pooled-supplemental', pooled, cuts)
            main, maincuts = {}, {}
            for held in COHORTS:
                choices = select({a: [r for c in COHORTS if c != held for r in data[c][a]] for a in ARMS})
                maincuts[held] = choices
                write(out/f'{held}_LOCO_thresholds.json', dict(status='FROZEN_BEFORE_HELD_DECISIONS', held_capture=held,
                    other_captures=[c for c in COHORTS if c != held], arms=choices, selection='Only other capture strictFREE'))
                main[held] = {a: evaluate(rr, choices[a], False) for a, rr in data[held].items()}
            save(out/'main-loco', 'main-loco', main, maincuts)
            write(out/'results.json', dict(status='COMPLETE', model=args.model, main='main-loco', supplement='pooled-supplemental',
                eval_frames=136, pooled_cal_frames=304, train_frames=56, posthoc_cutoffs=False, strong_signal='NOT_EVALUABLE_40777065_NEAR_POS_ZERO'))
        receipt['status'] = 'COMPLETE'
    except Exception as exc:
        receipt.update(status='FAILED_PARTIAL', error=repr(exc)); raise
    finally:
        receipt.update(wall_s=time.perf_counter()-started, completed_utc=utc(), checks=dict(checks))
        write(out/'inputs.json', inputs); write(out/'terminal.json', receipt); print(receipt, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True); p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--model', required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=('calibrate', 'final'), required=True); p.add_argument('--calibration', type=Path)
    p.add_argument('--budget-s', type=float, required=True)
    run(p.parse_args())
