"""Fixed camera-local subvolumes on cached real sensor rays, Development only.

All 27 fragments are declared before labels are read. References cannot certify
whole-volume clearance. Frozen old query heads are extrapolated to new queries;
this adapter neither trains them nor reruns image-only Depth Pro inference.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
from rgb_body_query_3rscan import sensor_labels, sample_prediction
from rgb_body_query_reference_eval import rays, ray_interval, confusion
from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_scene_diagnostic import write_csv

EDGES = ((-.9, -.3, .3, .9), (-.55, -.18, .18, .55), (.3, .8, 1.5, 3.))
PUBLIC = ('environment', 'scan', 'split', 'frame', 'rgb_path', 'rgb_sha256',
          'color_K', 'color_shape', 'depth_K', 'depth_shape')
SOURCES = {'original_validation24': ('rgb-body-query-cross-session-dev-20261009/sensor', 24),
           'new_3rscan64': ('rgb-body-query-transfer-dev-20261009/sensor', 64),
           'arkit16': ('rgb-body-query-transfer-dev-20261009/camera-sensor', 16)}

def load(p): return json.loads(Path(p).read_text('utf-8-sig'))

def queries():
    return [dict(name=f'fragment_x{x}_y{y}_z{z}',
                 low=[EDGES[0][x], EDGES[1][y], EDGES[2][z]],
                 high=[EDGES[0][x+1], EDGES[1][y+1], EDGES[2][z+1]])
            for x in range(3) for y in range(3) for z in range(3)]

def independent_xyz_labels(depth, k, query, observed):
    """Independent 3D point inclusion and three-axis line-box clipping."""
    h, w = depth.shape; yy, xx = np.indices((h, w), dtype=np.float64)
    directions = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(k).T
    enter = np.full((h, w), -np.inf); leave = np.full((h, w), np.inf)
    for axis in range(3):
        d = directions[..., axis]; low, high = query['low'][axis], query['high'][axis]
        nonzero = np.abs(d) > 1e-12
        t0 = np.divide(low, d, out=np.zeros_like(d), where=nonzero)
        t1 = np.divide(high, d, out=np.zeros_like(d), where=nonzero)
        enter = np.maximum(enter, np.where(nonzero, np.minimum(t0, t1), -np.inf))
        leave = np.minimum(leave, np.where(nonzero, np.maximum(t0, t1), np.inf))
        leave[(~nonzero) & ((low > 0) | (high < 0))] = -np.inf
    domain = enter <= leave + 1e-12
    valid = observed & np.isfinite(depth) & (depth > 0)
    xyz = directions * depth[..., None]
    inside = np.ones((h, w), bool)
    for axis in range(3):
        # Match the optical-Z closed-interval tolerance in coordinate units.
        tolerance = np.abs(directions[..., axis]) * 1e-12
        inside &= (xyz[..., axis] >= query['low'][axis]-tolerance)
        inside &= (xyz[..., axis] <= query['high'][axis]+tolerance)
    label = np.full((h, w), 255, np.uint8); label[domain] = 2
    label[domain & valid & inside] = 1
    label[domain & valid & (depth > leave + 1e-12)] = 0
    return label, domain

def prepare(repo, output, budget_s=300):
    start = time.perf_counter(); output.mkdir(parents=True, exist_ok=True)
    if (output/'plan.json').exists(): raise FileExistsError('Preserve prior fixed grid run')
    fixed = queries()
    plan = dict(status='PLAN_BEFORE_NEW_LABELS', created_utc=datetime.now(timezone.utc).isoformat(),
                queries=fixed, axis_edges=EDGES, naming='fragment_x{0..2}_y{0..2}_z{0..2}; x then y then z ascending',
                selection='Every Cartesian fragment in every prespecified existing validation frame; no label/prediction-based selection',
                sources=SOURCES, budget_cpu_s=budget_s, gpu_s=0, downloads=0,
                coordinates='camera right x, camera down y, optical forward Z metres',
                boundaries='Closed boxes, so shared faces can occur in adjacent fragments; correlated query-ray units',
                frozen='Old checkpoints/normalization/cutoffs fixed, no training. New bounds are Development query extension',
                reference_contract='Native first-return rays: in box positive, after box free, before box/missing unknown, ray misses255; min16',
                limitations='Subvolumes are not whole body queries, sampled free does not certify whole-volume clearance or walking events',
                source_sha256=sha(__file__))
    write(output/'plan.json', plan)
    results = {}; checks = []; complete = True
    def budget():
        if time.perf_counter()-start >= budget_s: raise TimeoutError('CPU prepare budget reached')
    try:
        for name, (relative, expected) in SOURCES.items():
            budget(); source = repo/'artifacts.local/work'/relative
            mp, pp = source/'dataset_manifest.json', source/'depthpro/predictions.json'
            manifest, predictions = load(mp), load(pp)
            rows = [r for r in manifest['rows'] if r['split']=='validation']
            if len(rows)!=expected: raise ValueError(f'{name} frame count changed')
            if predictions['status']!='COMPLETE': raise ValueError('Source predictions incomplete')
            lookup = {(p['scan'], p['frame']):p for p in predictions['rows']}
            dest = output/name; dest.mkdir(parents=True, exist_ok=True)
            new_rows, pred_rows, coverage = [], [], []
            for r in rows:
                budget(); p = lookup[r['scan'], r['frame']]
                if sha(r['rgb_path'])!=r['rgb_sha256'] or sha(r['reference_path'])!=r['reference_sha256']:
                    raise ValueError('RGB/reference identity changed')
                if sha(p['path'])!=p['sha256'] or p['rgb_sha256']!=r['rgb_sha256']:
                    raise ValueError('Cached prediction identity changed')
                with np.load(r['reference_path']) as ref:
                    payload = {k:ref[k].copy() for k in ref.files if k!='labels'}
                for key in ('depth_K','color_K'):
                    if not np.array_equal(payload[key],np.asarray(r[key])): raise ValueError('K changed')
                if list(payload['depth'].shape)!=r['depth_shape']: raise ValueError('Reference shape changed')
                labels, states = [], []
                for q in fixed:
                    label, state = sensor_labels(payload['depth'],payload['depth_K'],q,payload['observed'])
                    independent, domain = independent_xyz_labels(payload['depth'],payload['depth_K'],q,payload['observed'])
                    if not np.array_equal(label,independent): raise ValueError(f'XYZ label mismatch {name}/{r["frame"]}/{q["name"]}')
                    reach = state['domain_pixels']; known = state['positive_pixels']+state['free_ray_pixels']
                    state.update(observed_reachable_rays=int((domain & payload['observed']).sum()),
                                 reachable_ray_fraction=reach/label.size, known_reference_fraction=known/reach if reach else None)
                    labels.append(label); states.append(state)
                    coverage.append(dict(cohort=name,environment=r['environment'],scan=r['scan'],frame=r['frame'],**state))
                ref_path = dest/f'{r["scan"]}_{r["frame"]:06d}.npz'
                np.savez_compressed(ref_path,labels=np.stack(labels),**payload)
                # Reopen all generated reference arrays: native ray identity is unchanged.
                with np.load(ref_path) as checked:
                    for key,val in payload.items():
                        if not np.array_equal(checked[key],val,equal_nan=True): raise ValueError('Native reference array changed')
                new = dict(r,source_reference_path=r['reference_path'],source_reference_sha256=r['reference_sha256'],
                           reference_path=str(ref_path),reference_sha256=sha(ref_path),queries=states)
                new_rows.append(new)
                pred_rows.append({k:v for k,v in p.items() if k!='inference_s'})
                checks.append(dict(cohort=name,scan=r['scan'],frame=r['frame'],rgb_sha256=r['rgb_sha256'],
                                   source_reference_sha256=r['reference_sha256'],new_reference_sha256=new['reference_sha256'],
                                   prediction_sha256=p['sha256'],xyz_queries_equal=27,native_arrays_unchanged=True))
            counts = dict(Counter(q['state'] for r in new_rows for q in r['queries']))
            m = dict(manifest,rows=new_rows,queries=fixed,frames=len(new_rows),state_counts_by_split={'validation':counts},
                     status='REAL_SENSOR_FIXED_FRAGMENT_READY',plan_sha256=sha(output/'plan.json'),
                     source_manifest_path=str(mp),source_manifest_sha256=sha(mp),
                     query_contract='Fixed 27 camera-local fragments, sampled first-return reference only; no whole-body clearance',
                     official_data_role='Consumed Development, query bounds extension; no new independent frame selection')
            write(dest/'dataset_manifest.json',m)
            obs = dict(rows=[{k:r[k] for k in PUBLIC} for r in new_rows],queries=fixed,dataset_manifest_sha256=sha(dest/'dataset_manifest.json'))
            if any(set(r)!=set(PUBLIC) for r in obs['rows']): raise ValueError('Evaluator field leakage')
            write(dest/'observations.json',obs)
            reused = dict(status='COMPLETE',execution_status='REUSED',rows=pred_rows,source_manifest_path=str(pp),source_manifest_sha256=sha(pp),
                          original_weight_sha256=predictions['weight_sha256'],weight_sha256=predictions['weight_sha256'],
                          observations_sha256=sha(dest/'observations.json'),inference_s=0,allocation_wall_s=0,completed_calls=0,
                          reused_frames=len(pred_rows),gpu_s_this_run=0,evaluator_inputs=False,scale_fit=False,
                          reason='RGB/full-image K/native frame unchanged. Depth Pro is image-only and independent of query list; only compatible manifest is regenerated.')
            (dest/'depthpro').mkdir(exist_ok=True)
            write(dest/'depthpro/predictions.json',reused)
            write_csv(dest/'reference_coverage.csv',coverage)
            results[name]=dict(path=str(dest),frames=len(new_rows),environments=len({r['environment'] for r in new_rows}),
                              query_records=len(new_rows)*27,state_counts=counts,
                              positive_query_ray_units=sum(c['positive_pixels'] for c in coverage),
                              free_query_ray_units=sum(c['free_ray_pixels'] for c in coverage),
                              unknown_query_ray_units=sum(c['unknown_pixels'] for c in coverage),
                              reachable_query_ray_units=sum(c['domain_pixels'] for c in coverage),
                              zero_reachable_queries=sum(c['domain_pixels']==0 for c in coverage),
                              manifest_sha256=sha(dest/'dataset_manifest.json'),observations_sha256=sha(dest/'observations.json'),
                              reused_prediction_manifest_sha256=sha(dest/'depthpro/predictions.json'))
            write(output/'progress.json',dict(status='RUNNING',cohorts=results,elapsed_s=time.perf_counter()-start))
    except Exception as e:
        complete=False; write(output/'failure.json',dict(status='NOT_RUN_REMAINDER' if isinstance(e,TimeoutError) else 'FAILED',error=repr(e),completed_cohorts=results)); raise
    finally:
        shutil.copyfile(__file__,output/'executed_fixed_grid.py')
        write(output/'completion_receipt.json',dict(status='COMPLETE' if complete else 'INCOMPLETE',cohorts=results,
              elapsed_cpu_wall_s=time.perf_counter()-start,budget_cpu_s=budget_s,gpu_s=0,download_bytes=0,
              placement='TASK_NOT_GPU_SUITABLE: scalar sensor label adapter and SHA audit; no model inference',
              checks=dict(frame_rows=checks,independent_xyz_queries=len(checks)*27,public_field_isolation=True),
              plan_sha256=sha(output/'plan.json'),source_sha256=sha(__file__),resources='No workers or services retained'))
    print(json.dumps(results),flush=True)

def geometry_predictions(depth, k, shape, fixed_queries, fit, calibration):
    """Evaluator helper: reuse frozen original-train fit and original-cal margins."""
    from rgb_body_query_calibrated_geometry import correct
    from rgb_body_query_margin_baseline import margin_support, SCALE
    rx,ry=rays(k,shape)
    corrected={name:correct(depth,model) for name,model in fit['models'].items()}
    for q in fixed_queries:
        entry,exit,domain=ray_interval(rx,ry,q)
        result={'depthpro_raw':margin_support(depth,entry,exit,domain,0),
                'depthpro_cal_scale':margin_support(depth*SCALE,entry,exit,domain,0)}
        for name,z in corrected.items():
            result[name+'_direct']=margin_support(z,entry,exit,domain,0)
            result[name+'_margin']=margin_support(z,entry,exit,domain,calibration['arms'][name]['selected']['delta_m'])
        yield result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--budget-s',type=float,default=300)
    a=p.parse_args();prepare(a.repo.resolve(),a.output.resolve(),a.budget_s)
