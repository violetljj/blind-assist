"""Frozen optical-Z point migration diagnostic on consumed Development.

No model inference, training, downloads or decision-threshold selection. exp(mu)
is the log-normal Z median, not its arithmetic mean. Pixels and frames are related.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import time

import numpy as np

from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_margin_baseline import sampled
from rgb_body_query_reference_eval import rays
from rgb_body_query_scene_diagnostic import write_csv


ARMS = ('raw_depthpro', 'log_affine', 'depth_ray_gaussian_point', 'geometry_gaussian_point')
BANDS = (('all', 0., np.inf), ('below_0.3', 0., .3), ('0.3_0.8', .3, .8),
         ('0.8_1.5', .8, 1.5), ('1.5_3', 1.5, 3.), ('ge_3', 3., np.inf))


def load(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def checked_npz(path, digest):
    if sha(path) != digest:
        raise ValueError(f'Input changed: {path}')
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def metrics(true, pred):
    n = len(true)
    if not n:
        return dict(points=0)
    log_error = np.log(pred)-np.log(true)
    ratio = pred/true
    return dict(points=n, mean_signed_log_error=float(log_error.mean()),
                median_signed_log_error=float(np.median(log_error)),
                mean_absolute_log_error=float(np.abs(log_error).mean()),
                median_absolute_log_error=float(np.median(np.abs(log_error))),
                ratio_q10=float(np.quantile(ratio, .1)), ratio_q50=float(np.median(ratio)),
                ratio_q90=float(np.quantile(ratio, .9)), true_z_median_m=float(np.median(true)),
                predicted_z_median_m=float(np.median(pred)),
                far_gt3_total=int((true > 3).sum()),
                far_gt3_predicted_le3=int(((true > 3)&(pred <= 3)).sum()))


def support(inputs, bounds):
    outside = (inputs < bounds['q01']) | (inputs > bounds['q99'])
    return dict(points=len(inputs), outside_joint_count=int(outside.any(axis=1).sum()),
                outside_joint_fraction=float(outside.any(axis=1).mean()),
                outside_log_depth_fraction=float(outside[:, 0].mean()),
                outside_ray_x_fraction=float(outside[:, 1].mean()),
                outside_ray_y_fraction=float(outside[:, 2].mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    root = out.parent; plan = load(root/'plan.json'); plan_sha = sha(root/'plan.json')
    previous = load(out/'failed_attempt.json') if (out/'failed_attempt.json').exists() else {}
    booked_previous = previous.get('cpu_wall_booked_upper_bound_s', 0)
    budget = plan['budget']['point_branch_s'] - booked_previous
    if plan['budget']['gpu_s'] != 0 or plan['budget']['download_bytes'] != 0:
        raise ValueError('This diagnostic requires frozen CPU-only/no-download plan')
    def check():
        if time.perf_counter()-start >= budget-3:
            raise TimeoutError('Point diagnostic budget reached')

    work = args.repo/'artifacts.local/work'
    prior = work/'rgb-body-query-interval-distribution-dev-20261010'
    query = work/'rgb-body-query-query-level-dev-20261009'
    training_path = prior/'candidate/training_inputs.json'; info = load(training_path)
    a, b = info['affine']['a'], info['affine']['b']
    train_sensor = prior/'train-cal-sensor'
    train_manifest = load(train_sensor/'dataset_manifest.json')
    train_dp = load(train_sensor/'depthpro/predictions.json')
    train_lookup = {(r['scan'], r['frame']): r for r in train_dp['rows']}
    train_inputs = []; train_ids = []
    masks = out/'masks'; masks.mkdir(exist_ok=True)
    for r in train_manifest['rows']:
        if r['split'] != 'train':
            continue
        check(); p = train_lookup[r['scan'], r['frame']]
        depth, _ = sampled(r, p)
        ref = checked_npz(r['reference_path'], r['reference_sha256']); z = ref['depth']
        matched = ref['observed'] & np.isfinite(z) & (z > 0) & np.isfinite(depth) & (depth > 0)
        rx, ry = rays(r['depth_K'], r['depth_shape'])
        train_inputs.append(np.stack((np.log(depth[matched].astype(np.float64)), rx[matched], ry[matched]), -1))
        mask_path = masks/f'train_{r["scan"]}_{r["frame"]:06d}.npz'
        np.savez_compressed(mask_path, matched_observed=matched)
        train_ids.append(dict(scan=r['scan'], frame=r['frame'], points=int(matched.sum()),
                              reference_path=r['reference_path'], reference_sha256=r['reference_sha256'],
                              prediction_path=p['path'], prediction_sha256=p['sha256'],
                              mask_path=str(mask_path), mask_sha256=sha(mask_path)))
    tx = np.concatenate(train_inputs); del train_inputs
    if len(tx) != info['points'] or len(tx) != 1591791:
        raise ValueError('Training point denominator changed')
    bounds = dict(q01=np.quantile(tx, .01, axis=0), q99=np.quantile(tx, .99, axis=0))
    support_rows = [dict(cohort='train', scope='pooled', **support(tx, bounds))]
    train_bounds = dict(columns=['raw_DepthPro_log_Z', 'public_ray_x', 'public_ray_y'],
                        q01=bounds['q01'].tolist(), q99=bounds['q99'].tolist(),
                        training_points=len(tx), training_frames=len(train_ids),
                        rule='Outside any marginal train q01/q99; descriptive support, not causal OOD test')
    del tx
    write(out/'train_support.json', dict(**train_bounds, identities=train_ids))
    sources = {'cal': train_sensor,
               'original_validation24': query/'fixed-grid-sensor/original_validation24',
               'new_3rscan64': query/'fixed-grid-sensor/new_3rscan64',
               'arkit16': query/'fixed-grid-sensor/arkit16',
               'arkit_40777060': query/'additional-arkit-sensor/40777060',
               'arkit_40777065': query/'additional-arkit-sensor/40777065'}
    frames = []; pooled = []; frame_averages = []; identities = []; focused = []
    for cohort, sensor in sources.items():
        check(); manifest_path = sensor/'dataset_manifest.json'
        prediction_path = prior/'candidate/predictions'/cohort/'predictions.json'
        refs = {(r['scan'], r['frame']): r for r in load(manifest_path)['rows']}
        predictions = load(prediction_path)['rows']; groups = defaultdict(list); cohort_inputs = []
        for row in predictions:
            check(); r = refs[row['scan'], row['frame']]
            ref = checked_npz(r['reference_path'], r['reference_sha256']); true = ref['depth'].astype(np.float64)
            depth = checked_npz(row['sampled_depth_path'], row['sampled_depth_sha256'])['depth'].astype(np.float64)
            if sha(row['depthpro_source_path']) != row['depthpro_source_sha256']:
                raise ValueError('Original DepthPro source changed')
            dist = {name: checked_npz(d['path'], d['sha256']) for name, d in row['distributions'].items()}
            pred = dict(raw_depthpro=depth, log_affine=np.exp(a*np.log(np.where(depth > 0, depth, 1))+b),
                        depth_ray_gaussian_point=np.exp(dist['depth_ray_gaussian']['mu']),
                        geometry_gaussian_point=np.exp(dist['geometry_gaussian']['mu']))
            observed = ref['observed'] & np.isfinite(true) & (true > 0)
            matched = observed & np.isfinite(depth) & (depth > 0)
            for name, values in pred.items():
                matched &= np.isfinite(values) & (values > 0)
            matched &= dist['depth_ray_gaussian']['valid'] & dist['geometry_gaussian']['valid']
            rx, ry = rays(r['depth_K'], r['depth_shape'])
            inputs = np.stack((np.log(depth[matched]), rx[matched], ry[matched]), -1)
            cohort_inputs.append(inputs)
            identity = dict(cohort=cohort, environment=r['environment'], scan=r['scan'], frame=r['frame'])
            support_rows.append(dict(**identity, scope='frame', **support(inputs, bounds)))
            mask_path = masks/f'{cohort}_{r["scan"]}_{r["frame"]:06d}.npz'
            np.savez_compressed(mask_path, observed_reference=observed, matched_all_arms=matched)
            identities.append(dict(**identity, observed_reference_points=int(observed.sum()), matched_points=int(matched.sum()),
                unmatched_observed_points=int((observed & ~matched).sum()), mask_path=str(mask_path), mask_sha256=sha(mask_path),
                reference_path=r['reference_path'], reference_sha256=r['reference_sha256'],
                sampled_depth_path=row['sampled_depth_path'], sampled_depth_sha256=row['sampled_depth_sha256'],
                depthpro_source_path=row['depthpro_source_path'], depthpro_source_sha256=row['depthpro_source_sha256'],
                distributions=row['distributions']))
            z = true[matched]; values = {name: p[matched] for name, p in pred.items()}
            groups[r['environment']].append((z, values)); groups['__all__'].append((z, values))
            for band, lo, hi in BANDS:
                bm = (z >= lo)&(z < hi)
                for arm in ARMS:
                    frames.append(dict(**identity, band=band, arm=arm, **metrics(z[bm], values[arm][bm])))
            # Independent scalar recomputation for the first real frame of every cohort.
            if len([x for x in focused if x['cohort'] == cohort]) == 0:
                coords = list(zip(*np.where(matched)))[:257]
                for arm in ARMS:
                    tt = np.array([float(ref['depth'][i, j]) for i, j in coords])
                    pp = np.array([float(pred[arm][i, j]) for i, j in coords])
                    scalar_errors = [math.log(float(p))-math.log(float(t)) for p, t in zip(pp, tt)]
                    got = metrics(tt, pp)
                    assert math.isclose(got['mean_signed_log_error'], sum(scalar_errors)/len(scalar_errors), abs_tol=2e-14)
                    assert math.isclose(got['mean_absolute_log_error'], sum(abs(v) for v in scalar_errors)/len(scalar_errors), abs_tol=2e-14)
                    far = sum(t > 3 for t in tt); crossing = sum(t > 3 and p <= 3 for p, t in zip(pp, tt))
                    assert got['far_gt3_total'] == far and got['far_gt3_predicted_le3'] == crossing
                # Load the original full prediction through an independent saved mapping.
                dp_row = dict(path=row['depthpro_source_path'], sha256=row['depthpro_source_sha256'])
                again, _ = sampled(r, dp_row)
                assert np.allclose(depth[matched], again[matched], atol=0, rtol=0)
                independent_count = sum(bool(ref['observed'][i,j]) and math.isfinite(float(ref['depth'][i,j])) and float(ref['depth'][i,j]) > 0
                    and all(math.isfinite(float(p[i,j])) and float(p[i,j]) > 0 for p in pred.values())
                    and bool(dist['depth_ray_gaussian']['valid'][i,j]) and bool(dist['geometry_gaussian']['valid'][i,j])
                    for i in range(true.shape[0]) for j in range(true.shape[1]))
                assert independent_count == int(matched.sum())
                focused.append(dict(**identity, scalar_points=len(coords), arms=list(ARMS), matched_points=independent_count,
                                    original_depthpro_mapping_exact=True, status='PASS'))
        support_rows.append(dict(cohort=cohort, scope='pooled', **support(np.concatenate(cohort_inputs), bounds)))
        for env, items in groups.items():
            z = np.concatenate([x[0] for x in items])
            for arm in ARMS:
                pp = np.concatenate([x[1][arm] for x in items])
                for band, lo, hi in BANDS:
                    check(); bm=(z >= lo)&(z < hi)
                    pooled.append(dict(cohort=cohort, scope='pooled' if env=='__all__' else 'environment',
                                       environment=env, band=band, arm=arm, **metrics(z[bm], pp[bm])))
        print('COHORT_COMPLETE', cohort, len(predictions), flush=True)
    for cohort in sources:
        for arm in ARMS:
            for band, _, _ in BANDS:
                ff = [r for r in frames if r['cohort']==cohort and r['arm']==arm and r['band']==band and r['points']]
                numeric = ['mean_signed_log_error','mean_absolute_log_error','median_absolute_log_error','ratio_q50']
                frame_averages.append(dict(cohort=cohort, arm=arm, band=band, contributing_frames=len(ff),
                    **{'frame_average_'+k: float(np.mean([r[k] for r in ff])) if ff else None for k in numeric}))
    for name, data in [('pooled', pooled),('frames', frames),('frame_averages', frame_averages),('input_support', support_rows)]:
        fields = list(dict.fromkeys(k for r in data for k in r))
        write_csv(out/f'{name}.csv', [{k: r.get(k) for k in fields} for r in data])
        write(out/f'{name}.json', data)
    write(out/'point_identities.json', dict(mask_rule='Observed finite positive true Z and common finite positive predictions with distribution validity',
          rows=identities, frames=len(identities), matched_points=sum(r['matched_points'] for r in identities)))
    write(out/'focused_check.json', dict(status='PASS', frames=focused, scalar_points=sum(r['scalar_points'] for r in focused),
                                        independent_original_mapping_and_full_frame_denominator=True))
    check()
    if sha(root/'plan.json') != plan_sha:
        raise ValueError('Parent plan changed during diagnostic')
    write(out/'completion_receipt.json', dict(status='COMPLETE', cpu_wall_s=time.perf_counter()-start, budget_cpu_wall_s=budget,
          prior_failed_measured_cpu_wall_s=previous.get('measured_cpu_wall_s'),
          prior_failed_booked_s=booked_previous, cumulative_cpu_wall_booked_s=booked_previous+time.perf_counter()-start,
          branch_budget_cpu_wall_s=plan['budget']['point_branch_s'],
          gpu_s=0, download_bytes=0, new_training=False, inference_calls=0, thresholds_changed=False,
          source_sha256=sha(__file__), plan_sha256=plan_sha, training_inputs_sha256=sha(training_path),
          frames=len(identities), point_masks=len(train_ids)+len(identities),
          matched_points=sum(r['matched_points'] for r in identities), limitations='Correlated observed optical-Z points; descriptive diagnostics do not identify causes or full-volume safety'))
    print(json.dumps(load(out/'completion_receipt.json')), flush=True)


if __name__ == '__main__':
    main()
