"""Frozen Depth Pro scale diagnostics on native measured 3RScan rays.

The calibrated scale uses only cal reference; environment-specific scales use
scoring reference and are GT-assisted diagnostics, not performance bounds.
No query threshold, network, saved prediction or reference is modified.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from rgb_body_query_3rscan import sample_prediction
from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_reference_eval import confusion, ratios, rays, ray_interval

COUNTS = ('tp', 'fn', 'fp', 'tn')
LAYERS = ('raw', 'cal_global', 'environment_gt_assisted')


def paired_mask(reference, prediction, observed):
    return observed & np.isfinite(reference) & (reference > 0) & np.isfinite(prediction) & (prediction > 0)


def fit_scale(log_ratios):
    values = np.asarray(log_ratios, np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError('Need nonempty finite paired log ratios')
    return float(np.exp(np.median(values)))


def depth_error(reference, prediction, mask, scale):
    truth = reference[mask].astype(np.float64)
    estimate = prediction[mask].astype(np.float64)*scale
    residual = np.log(estimate/truth)
    return dict(paired_pixels=int(mask.sum()), absrel_sum=float((np.abs(estimate-truth)/truth).sum()),
                absolute_log_sum=float(np.abs(residual).sum()), signed_log_sum=float(residual.sum()),
                absrel_mean=float((np.abs(estimate-truth)/truth).mean()),
                absolute_log_mean=float(np.abs(residual).mean()), signed_log_mean=float(residual.mean()),
                median_signed_log=float(np.median(residual)))


def summarize(rows):
    counts = {k: sum(r[k] for r in rows) for k in COUNTS}
    positives = [r for r in rows if r['reference_state'] == 'POSITIVE']
    free = [r for r in rows if r['reference_state'] == 'FREE_ON_SAMPLED_RAYS']
    return dict(**counts, **ratios(counts), frames=len({(r['scan'], r['frame']) for r in rows}),
                query_positive_total=len(positives), query_positive_hits=sum(r['predicted_positive'] for r in positives),
                query_positive_known_witness_hits=sum(r['tp'] >= 16 for r in positives),
                sampled_free_total=len(free), sampled_free_false_support=sum(r['predicted_positive'] for r in free),
                unknown_query_total=sum(r['reference_state'] == 'UNKNOWN' for r in rows))


def grouped_summary(records):
    result = {}
    for layer in LAYERS:
        current = [r for r in records if r['layer'] == layer]
        group = defaultdict(list)
        for r in current:
            group['all'].append(r)
            group['split/'+r['split']].append(r)
            group['environment/'+r['environment']].append(r)
            group['query/'+r['query']].append(r)
            group['family/'+r['query_family']].append(r)
            group['distance/'+r['distance_band']].append(r)
            group['split_family/'+r['split']+'/'+r['query_family']].append(r)
            group['split_distance/'+r['split']+'/'+r['distance_band']].append(r)
            group['environment_distance/'+r['environment']+'/'+r['distance_band']].append(r)
            group['environment_query/'+r['environment']+'/'+r['query']].append(r)
        result[layer] = {name: summarize(rs) for name, rs in group.items()}
    return result


def run(source, output, budget_s=240):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'plan.json').exists():
        raise FileExistsError('Use a new output; preserve previous diagnostic')
    paths = dict(reference=source/'dataset_manifest.json', prediction=source/'depthpro/predictions.json',
                 previous_evaluation=source/'evaluation.json')
    manifest = json.loads(paths['reference'].read_text('utf-8-sig'))
    preds = json.loads(paths['prediction'].read_text('utf-8-sig'))
    plan = dict(layers=list(LAYERS), source_manifests={k:dict(path=str(p), sha256=sha(p)) for k,p in paths.items()},
                source_code_sha256=sha(__file__), cpu_budget_s=budget_s, gpu_budget_s=0, download_budget_bytes=0,
                scale_definition='exp(median(log(measured_z/predicted_z))) across all finite positive paired observed sensor rays; pixels pooled',
                fitting_scope='cal_global: cal only, one scalar applied unchanged to all splits; environment_gt_assisted: each environment all 8 scoring frames',
                frozen='15 existing queries, closed intervals, >=16 support, saved predictions/reference; no query-score optimization',
                interpretation='scale adjustment may explain a portion of disparity; cannot alone identify cause; GT-assisted is not an upper bound',
                data_role='consumed official-train Development; sensor reference is not metrology truth')
    write(output/'plan.json', plan)
    source_by_key = {(r['scan'], r['frame']):r for r in manifest['rows']}
    cached, log_by_environment, cal_logs, identities = [], defaultdict(list), [], []
    for p in preds['rows']:
        if time.perf_counter()-start >= budget_s:
            raise TimeoutError('Scale diagnostic CPU budget reached')
        r = source_by_key[p['scan'],p['frame']]
        if any(r[k] != p[k] for k in ('environment','split','rgb_sha256')):
            raise ValueError('Prediction/source identity mismatch')
        if sha(p['path']) != p['sha256'] or sha(r['reference_path']) != r['reference_sha256']:
            raise ValueError('Changed saved payload')
        with np.load(p['path']) as npz:
            depth = npz['depth']
        with np.load(r['reference_path']) as ref:
            sampled = sample_prediction(depth, ref['map_x'], ref['map_y'])
            truth, labels, observed, k = ref['depth'], ref['labels'], ref['observed'], ref['depth_K']
        valid = paired_mask(truth,sampled,observed)
        logs = np.log(truth[valid].astype(np.float64)/sampled[valid])
        log_by_environment[r['environment']].append(logs)
        if r['split'] == 'cal':
            cal_logs.append(logs)
        cached.append((r,truth,sampled,labels,valid,k))
        identities.append(dict(scan=r['scan'], frame=r['frame'], prediction_sha256=p['sha256'],
                               reference_sha256=r['reference_sha256'], rgb_sha256=r['rgb_sha256']))
    cal_values = np.concatenate(cal_logs)
    cal_scale = fit_scale(cal_values)
    environments = {}
    for env, logs in log_by_environment.items():
        values = np.concatenate(logs)
        erows = [r for r in manifest['rows'] if r['environment'] == env]
        environments[env] = dict(scale=fit_scale(values), fitting_pixels=len(values), split=erows[0]['split'],
                                 frames=len(erows), log_ratio_q05_q50_q95=np.quantile(values,[.05,.5,.95]).tolist())
    write(output/'fitted_scales.json', dict(cal_global=dict(scale=cal_scale, fitting_pixels=len(cal_values),
          fitting_frames=sum(r['split']=='cal' for r in manifest['rows']),
          environments=[env for env,e in environments.items() if e['split']=='cal']),
          environments=environments, identities=identities, plan_sha256=sha(output/'plan.json')))
    records, depth_records, checked = [], [], 0
    for r,truth,sampled,labels,valid,k in cached:
        if time.perf_counter()-start >= budget_s:
            raise TimeoutError('Scale diagnostic CPU budget reached')
        rx,ry = rays(k, sampled.shape)
        for layer, scale in zip(LAYERS, (1.,cal_scale,environments[r['environment']]['scale'])):
            scaled = sampled*scale
            depth_records.append(dict(environment=r['environment'], scan=r['scan'], frame=r['frame'], split=r['split'],
                                      layer=layer, scale=scale, **depth_error(truth,sampled,valid,scale)))
            for i,q in enumerate(manifest['queries']):
                entry,exit,reachable = ray_interval(rx,ry,q)
                predicted = np.isfinite(scaled) & reachable & (scaled >= entry) & (scaled <= exit)
                c = confusion(labels[i],predicted)
                if layer == 'cal_global':
                    # Independent direct XYZ membership, rather than ray-box interval arithmetic.
                    xyz = (rx*scaled,ry*scaled,scaled)
                    direct = np.isfinite(scaled)
                    for coord,lo,hi in zip(xyz,q['low'],q['high']):
                        direct &= (coord >= lo) & (coord <= hi)
                    if confusion(labels[i], direct) != c or int(direct.sum()) != int(predicted.sum()):
                        raise AssertionError('Direct XYZ / interval mismatch')
                    checked += 1
                records.append(dict(environment=r['environment'], scan=r['scan'], frame=r['frame'], split=r['split'],
                                    layer=layer, scale=scale, query=q['name'],
                                    query_family='legacy6' if i<6 else 'new_distance_bands9',
                                    distance_band=f'{q["low"][2]:g}-{q["high"][2]:g}m',
                                    reference_state=r['queries'][i]['state'], predicted_support=int(predicted.sum()),
                                    predicted_positive=int(predicted.sum())>=16, **c))
    previous = json.loads(paths['previous_evaluation'].read_text('utf-8-sig'))
    old = {(r['scan'],r['frame'],r['query']):r for r in previous['records']}
    raw = [r for r in records if r['layer']=='raw']
    for r in raw:
        p = old[r['scan'],r['frame'],r['query']]
        if any(r[key] != p[key] for key in (*COUNTS,'predicted_support','predicted_positive','reference_state')):
            raise AssertionError('Raw diagnostic failed frozen evaluation reproduction')
    depth_summary = {}
    for layer in LAYERS:
        group = defaultdict(list)
        for r in depth_records:
            if r['layer'] == layer:
                group['all'].append(r); group['split/'+r['split']].append(r)
                group['environment/'+r['environment']].append(r)
        depth_summary[layer] = {}
        for name,rs in group.items():
            n = sum(r['paired_pixels'] for r in rs)
            depth_summary[layer][name] = dict(paired_pixels=n,
                 **{k:sum(r[s] for r in rs)/n for k,s in [('absrel_mean','absrel_sum'),
                   ('absolute_log_mean','absolute_log_sum'),('signed_log_mean','signed_log_sum')]})
    result = dict(status='COMPLETE', records=records, summary=grouped_summary(records),
                  depth_records=depth_records, depth_summary=depth_summary,
                  fitted_scales_sha256=sha(output/'fitted_scales.json'),
                  verification=dict(raw_frozen_records_equal=len(raw), cal_global_direct_xyz_records_equal=checked),
                  wall_s=time.perf_counter()-start, limits=plan['interpretation'])
    write(output/'evaluation.json', result)
    print(json.dumps(dict(scales=dict(cal_global=cal_scale, environments=environments),
          validation={layer:result['summary'][layer]['split/validation'] for layer in LAYERS},
          depth_validation={layer:depth_summary[layer]['split/validation'] for layer in LAYERS},
          wall_s=result['wall_s'], verification=result['verification'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-s', type=float, default=240)
    args = parser.parse_args()
    run(args.source.resolve(),args.output.resolve(),args.budget_s)
