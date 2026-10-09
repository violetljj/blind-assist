"""Cal-only signed metre margin for frozen Depth Pro ray-box readout.

Original ray reachability and all saved heads/cutoffs stay fixed. This is a new
calibrated baseline, not a correction of the previous working-point comparison.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import time

import numpy as np

from rgb_body_query_3rscan import sample_prediction
from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_reference_eval import rays, ray_interval, confusion, ratios
from rgb_body_query_scale_diagnostic import summarize
from rgb_body_query_scene_diagnostic import paired_counts, write_csv
from rgb_body_query_frozen_transfer import aggregate

TARGET = .14190356447903094
SCALE = 1.012320716490867
ARMS = ('depthpro_margin', 'depthpro_raw', 'depthpro_cal_scale', 'depth_only', 'geometry')


def margin_support(depth, entry, exit, reachable, delta):
    margin = np.minimum(depth-entry, exit-depth)
    return reachable & np.isfinite(depth) & (depth > 0) & (margin >= -delta)


def focused_checks():
    z = np.array([1., 2., 3., 1.5, 2., np.nan, 0., 2.])
    a = np.ones(8); b = np.full(8, 3.); domain = np.ones(8, bool); domain[-1] = False
    checked = 0
    for delta in (-1., -.5, 0., .5, 1.):
        direct = domain & np.isfinite(z) & (z > 0) & (a-delta <= b+delta) & (z >= a-delta) & (z <= b+delta)
        assert np.array_equal(direct, margin_support(z, a, b, domain, delta)); checked += z.size
    assert not margin_support(z, a, b, domain, -1.01).any()
    rng = np.random.default_rng(7)
    z = rng.uniform(.1, 8., 10000); a = rng.uniform(.3, 3., 10000); b = a+rng.uniform(0., 3., 10000)
    domain = rng.random(10000) > .2
    for delta in (-1., -.39, 0., .47, 1.):
        direct = domain & (a-delta <= b+delta) & (z >= a-delta) & (z <= b+delta)
        assert np.array_equal(direct, margin_support(z, a, b, domain, delta)); checked += z.size
    return dict(status='PASS', direct_vs_margin_points=checked, closed_boundary_negative_and_inverted_interval=True)


def load_json(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def sampled(row, prediction):
    if sha(prediction['path']) != prediction['sha256']:
        raise ValueError('Prediction changed')
    with np.load(prediction['path']) as a:
        full = a['depth']
    if sha(row['reference_path']) != row['reference_sha256']:
        raise ValueError('Reference changed')
    with np.load(row['reference_path']) as a:
        depth = sample_prediction(full, a['map_x'], a['map_y'])
        labels = a['labels']
    return depth, labels


def calibration(sensor, output, check_budget):
    manifest_path = sensor/'dataset_manifest.json'; prediction_path = sensor/'depthpro/predictions.json'
    manifest = load_json(manifest_path); predictions = load_json(prediction_path)
    pred_by_key = {(p['scan'], p['frame']): p for p in predictions['rows']}
    selected = [r for r in manifest['rows'] if r['split'] == 'cal']
    if len(selected) != 16 or len({r['environment'] for r in selected}) != 2:
        raise ValueError('Original 16-frame/two-environment cal required')
    margins = {0: [], 1: []}; denominators = {0: 0, 1: 0}; identities = []
    for row in selected:
        check_budget()
        p = pred_by_key[row['scan'], row['frame']]
        depth, labels = sampled(row, p)
        rx, ry = rays(row['depth_K'], row['depth_shape'])
        for j, q in enumerate(manifest['queries']):
            entry, exit, domain = ray_interval(rx, ry, q)
            m = np.minimum(depth-entry, exit-depth)
            valid = domain & np.isfinite(depth) & (depth > 0)
            for label in (0, 1):
                mask = labels[j] == label
                denominators[label] += int(mask.sum())
                margins[label].append(m[mask & valid])
        identities.append(dict(environment=row['environment'],scan=row['scan'],frame=row['frame'],
                               reference_sha256=row['reference_sha256'],prediction_sha256=p['sha256']))
    sorted_margins = {label:np.sort(np.concatenate(v)) for label,v in margins.items()}
    rows = []
    for integer in range(-100, 101):
        delta = integer/100.
        count = {label: len(v)-int(np.searchsorted(v, -delta, side='left')) for label,v in sorted_margins.items()}
        fpr = count[0]/denominators[0]
        rows.append(dict(delta_m=delta, fp=count[0], tn=denominators[0]-count[0], tp=count[1], fn=denominators[1]-count[1],
                         fpr=fpr, recall=count[1]/denominators[1], distance_to_target=abs(fpr-TARGET)))
    best = min(rows, key=lambda r:(r['distance_to_target'], abs(r['delta_m']), r['delta_m']))
    result = dict(status='FROZEN_AFTER_CAL', selected=best, exact_target_matched=best['fpr']==TARGET,
                  target_cal_fpr=TARGET, candidates=rows, fitting_frames=identities,
                  selection_used_recall=False, transfer_reference_reads_before_freeze=False,
                  source_manifests={str(p):sha(p) for p in (manifest_path,prediction_path)},
                  plan_sha256=sha(output/'plan.json'), source_sha256=sha(Path(__file__)))
    write(output/'calibration.json', result)
    return result


def grouped(records):
    buckets = defaultdict(list)
    for r in records:
        buckets['all'].append(r)
        buckets['environment/'+r['environment']].append(r)
        buckets['band/'+r['distance_band']].append(r)
        buckets['environment_band/'+r['environment']+'/'+r['distance_band']].append(r)
    return {name:summarize(rows) for name,rows in buckets.items()}


def evaluate_cohort(name, sensor, head_root, output, delta, check_budget, original=False, features=None):
    manifest_path = sensor/'dataset_manifest.json'; manifest = load_json(manifest_path)
    head_ev_path = head_root/'evaluation.json'; head_ev = load_json(head_ev_path)
    if original:
        source = load_json(sensor/'depthpro/predictions.json')
        depths = {(r['scan'],r['frame']):r for r in source['rows']}
        rows = [r for r in manifest['rows'] if r['split']=='validation']
        scores = {arm:{(manifest['rows'][p['row']]['scan'],manifest['rows'][p['row']]['frame']):p
                        for p in head_ev['arms'][arm]['predictions']} for arm in ('depth_only','geometry')}
        cutoffs = {arm:head_ev['arms'][arm]['cutoff'] for arm in scores}
    else:
        source = load_json(head_root/'predictions.json'); depths={(r['scan'],r['frame']):r for r in source['rows']}
        rows = manifest['rows']; scores=None; cutoffs=source['cutoffs']
    expected_cutoffs = dict(depth_only=.5618626475334167, geometry=.6106688380241394)
    if cutoffs != expected_cutoffs:
        raise ValueError('Head cutoff changed')
    records = {arm:[] for arm in ARMS}; pairs=[]; raw_reproduced=0; head_reproduced=0
    old_records = {(r['scan'],r['frame'],r['query']):r for r in load_json(sensor/'evaluation.json')['records']} if original else None
    transfer_records = {arm:{(r['scan'],r['frame'],r['query']):r for r in head_ev['arms'][arm]['records']} for arm in ARMS[1:]} if not original else None
    for row in rows:
        check_budget(); key=(row['scan'],row['frame']); item=depths[key]
        if original:
            depth, labels = sampled(row,item)
            entries={a:scores[a][key] for a in scores}
        else:
            if sha(item['predicted_depth_path']) != item['predicted_depth_sha256']:
                raise ValueError('Sampled predicted depth changed')
            with np.load(item['predicted_depth_path']) as a: depth=a['depth']
            if sha(row['reference_path'])!=row['reference_sha256']:raise ValueError('Reference changed')
            with np.load(row['reference_path']) as a:labels=a['labels']
            entries=item['predictions']
        probs={}
        for arm,p in entries.items():
            if sha(p['path'])!=p['sha256']:raise ValueError('Head score changed')
            probs[arm]=np.load(p['path'],allow_pickle=False)
        rx,ry=rays(row['depth_K'],row['depth_shape'])
        for j,q in enumerate(manifest['queries']):
            entry,exit,domain=ray_interval(rx,ry,q)
            masks={a:(s[j]>=cutoffs[a])&domain for a,s in probs.items()}
            masks['depthpro_margin']=margin_support(depth,entry,exit,domain,delta)
            masks['depthpro_raw']=margin_support(depth,entry,exit,domain,0.)
            masks['depthpro_cal_scale']=margin_support(depth*SCALE,entry,exit,domain,0.)
            for arm,pred in masks.items():
                rec=dict(cohort=name,environment=row['environment'],scan=row['scan'],frame=row['frame'],split=row['split'],
                         query=q['name'],distance_band=f'{q["low"][2]:g}-{q["high"][2]:g}m',
                         reference_state=row['queries'][j]['state'],predicted_support=int(pred.sum()),
                         predicted_positive=int(pred.sum())>=16,**confusion(labels[j],pred))
                records[arm].append(rec)
                previous=None
                if original and arm=='depthpro_raw':previous=old_records[key+(q['name'],)]
                if original and arm in ('depth_only','geometry'):
                    previous=next(r for r in head_ev['arms'][arm]['validation']['records'] if (r['scan'],r['frame'],r['query'])==key+(q['name'],))
                if not original and arm!='depthpro_margin':previous=transfer_records[arm][key+(q['name'],)]
                if previous is not None:
                    for field in ('tp','fn','fp','tn','predicted_support','predicted_positive'):
                        if rec[field]!=previous[field]:raise AssertionError(f'Previous score differs {name} {arm} {field}')
                    if arm.startswith('depthpro'):raw_reproduced+=1
                    else:head_reproduced+=1
            for candidate, baseline in [('depthpro_margin','depthpro_raw'),('depthpro_margin','geometry'),
                                         ('depth_only','depthpro_margin'),('depth_only','depthpro_raw'),('depth_only','geometry')]:
                pairs.append(dict(cohort=name,arm=candidate,baseline=baseline,environment=row['environment'],scan=row['scan'],
                                  frame=row['frame'],query=q['name'],band=f'{q["low"][2]:g}-{q["high"][2]:g}m',
                                  **paired_counts(labels[j],masks[candidate],masks[baseline])))
    increments={}
    for label,keys in [('all',['cohort','arm','baseline']),('environment',['cohort','arm','baseline','environment']),
                        ('band',['cohort','arm','baseline','band']),('environment_band',['cohort','arm','baseline','environment','band'])]:
        increments[label]=aggregate(pairs,keys);write_csv(output/f'{name}_increments_{label}.csv',increments[label])
    write_csv(output/f'{name}_frame_query_increments.csv',pairs)
    result=dict(status='COMPLETE',arms={a:dict(summary=grouped(rs),records=rs) for a,rs in records.items()},increments=increments,
                source_manifests={str(p):sha(p) for p in (manifest_path,head_ev_path)},
                checks=dict(raw_records_equal=raw_reproduced,head_records_equal=head_reproduced,all_paired_identities=True),cutoffs=cutoffs)
    write(output/f'{name}_evaluation.json',result)
    return result


def run(repo,output,budget_s):
    start=time.perf_counter(); output.mkdir(parents=True,exist_ok=True)
    if (output/'calibration.json').exists():raise FileExistsError('Preserve previous calibrated run')
    plan=load_json(output/'plan.json')
    if plan['target_cal_fpr']!=TARGET:raise ValueError('Unexpected target')
    def check_budget():
        if time.perf_counter()-start>=budget_s:raise TimeoutError('CPU budget reached')
    focused=focused_checks();write(output/'focused_check.json',focused)
    base=repo/'artifacts.local/work'
    old=base/'rgb-body-query-cross-session-dev-20261009/sensor'
    learned=base/'rgb-body-query-metric-diagnostic-dev-20261009/readout'
    transfer=base/'rgb-body-query-transfer-dev-20261009'
    cal=calibration(old,output,check_budget); frozen_sha=sha(output/'calibration.json'); delta=cal['selected']['delta_m']
    print('CAL_FROZEN',json.dumps(cal['selected']),flush=True)
    results={}
    for name,sensor,heads,original in [('original_validation',old,learned,True),
                                     ('new_3rscan',transfer/'sensor',transfer/'readout',False),
                                     ('arkit',transfer/'camera-sensor',transfer/'camera-readout',False)]:
        results[name]=evaluate_cohort(name,sensor,heads,output,delta,check_budget,original)
        print(name,json.dumps({a:v['summary']['all'] for a,v in results[name]['arms'].items()}),flush=True)
    if sha(output/'calibration.json')!=frozen_sha:raise AssertionError('Calibration changed during transfer evaluation')
    write(output/'evaluation.json',dict(status='COMPLETE',selected_delta_m=delta,actual_cal_fpr=cal['selected']['fpr'],
          target_cal_fpr=TARGET,calibration_sha256=frozen_sha,cohorts={n:{a:v['summary']['all'] for a,v in r['arms'].items()} for n,r in results.items()},
          interpretation='Specified consumed Development only; a matched calibrated geometric result does not disprove RGB distance value or decide independent contribution',
          limits='Correlated query-ray units; UNKNOWN excluded; sampled-free query support is not whole-box/body FPR; no walking/thin/mobile evidence',
          wall_s=time.perf_counter()-start,cpu_budget_s=budget_s,gpu_s=0,download_bytes=0,checks=focused))
    shutil.copyfile(__file__,output/'executed_margin_baseline.py')
    write(output/'completion_receipt.json',dict(status='COMPLETE',wall_s=time.perf_counter()-start,cpu_budget_s=budget_s,
         gpu_s=0,download_bytes=0,calibration_sha256=frozen_sha,source_sha256=sha(Path(__file__)),
         result_sha256=sha(output/'evaluation.json'),resource_release='CPU-only subprocess exits; no workers or services'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--budget-s',type=float,default=600);a=p.parse_args();run(a.repo.resolve(),a.output.resolve(),a.budget_s)
