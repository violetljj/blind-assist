"""Frozen RGB-predicted depth query heads: transfer without fitting.

Observation-only prediction precedes reference scoring. The original trained
normalization, checkpoints and actual cal cutoffs are reused unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_3rscan import color_coordinates, sample_prediction
from rgb_body_query_reference_eval import confusion, rays, ray_interval
from rgb_body_query_sensor_pilot import summarize
from rgb_body_query_scene_diagnostic import paired_counts, write_csv


def band(q):
    return f'{q["low"][2]:g}-{q["high"][2]:g}m'


def aggregate(rows, keys):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    result = []
    for key, items in groups.items():
        counts = {k: sum(r[k] for r in items) for k in
                  ('positive', 'free', 'rgb_tp', 'geometry_tp', 'rgb_fp', 'geometry_fp',
                   'positive_rescue', 'positive_loss', 'free_rescue', 'free_added')}
        assert counts['positive_rescue']-counts['positive_loss'] == counts['rgb_tp']-counts['geometry_tp']
        assert counts['free_added']-counts['free_rescue'] == counts['rgb_fp']-counts['geometry_fp']
        result.append(dict(zip(keys, key), **counts,
            candidate_recall=counts['rgb_tp']/counts['positive'] if counts['positive'] else None,
            baseline_recall=counts['geometry_tp']/counts['positive'] if counts['positive'] else None,
            candidate_fpr=counts['rgb_fp']/counts['free'] if counts['free'] else None,
            baseline_fpr=counts['geometry_fp']/counts['free'] if counts['free'] else None,
            frame_query_records=len(items), frames=len({(r['scan'], r['frame']) for r in items})))
    return result


def predict(source, sensor, output, budget_s):
    start = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'predictions.json').exists():
        raise FileExistsError('Preserve prior predictions; no implicit restart')
    obs = json.loads((sensor/'observations.json').read_text('utf-8-sig'))
    dp = json.loads((sensor/'depthpro/predictions.json').read_text('utf-8-sig'))
    old = json.loads((source/'receipt.json').read_text('utf-8-sig'))
    ev = json.loads((source/'evaluation.json').read_text('utf-8-sig'))
    if dp['status'] != 'COMPLETE':
        raise ValueError('Complete RGB predictions required')
    rows, queries = obs['rows'], obs['queries']
    priors = {(r['scan'], r['frame']): r for r in dp['rows']}
    result = dict(status='STARTING', rows=[], budget_s=budget_s, source_sha256=sha(Path(__file__)),
        observations_sha256=sha(sensor/'observations.json'), depthpro_manifest_sha256=sha(sensor/'depthpro/predictions.json'),
        frozen_evaluation_sha256=sha(source/'evaluation.json'), frozen_receipt_sha256=sha(source/'receipt.json'),
        cutoffs={a:ev['arms'][a]['cutoff'] for a in ('depth_only','geometry')},
        target_cal_fpr=old['target_cal_fpr'], fit=False, evaluator_reads=False)
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError('Bounded batched head inference requires CUDA')
    torch.cuda.reset_peak_memory_stats()
    result.update(device=torch.cuda.get_device_name(0), framework=torch.__version__)
    heads={}
    try:
        for arm in ('depth_only','geometry'):
            path=source/f'{arm}_final.pt'
            if sha(path) != old['stages']['train']['arms'][arm]['sha256']:
                raise ValueError('Checkpoint identity changed')
            cp=torch.load(path,map_location='cuda',weights_only=True)
            head=torch.nn.Sequential(torch.nn.Linear(41,64),torch.nn.ReLU(),torch.nn.Linear(64,64),
                                    torch.nn.ReLU(),torch.nn.Linear(64,1)).cuda()
            head.load_state_dict(cp['state_dict']);head.eval().requires_grad_(False);heads[arm]=head
        norm=source/'observation_normalization.npz'
        if sha(norm) != old['normalization']['sha256']:
            raise ValueError('Normalization changed')
        with np.load(norm) as a:
            mean,std=a['mean'],a['std']
        bounds=torch.tensor([q['low']+q['high'] for q in queries],device='cuda',dtype=torch.float32)
        bounds/=torch.tensor([.9,.55,6.,.9,.55,6.],device='cuda')
        for i,r in enumerate(rows):
            if time.perf_counter()-start>=budget_s-2:
                raise TimeoutError('Frozen head inference budget reached')
            p=priors[r['scan'],r['frame']]
            if p['rgb_sha256'] != r['rgb_sha256'] or sha(p['path']) != p['sha256']:
                raise ValueError('RGB prediction identity changed')
            with np.load(p['path']) as a:
                full=a['depth']
            mx,my=color_coordinates(np.array(r['depth_K']),np.array(r['color_K']),r['depth_shape'])
            depth=sample_prediction(full,mx,my)
            rx,ry=rays(r['depth_K'],r['depth_shape']); n=depth.size
            finite=np.isfinite(depth)&(depth>0)
            observed=torch.zeros((n,33),device='cuda')
            safe=np.where(finite,depth,1.).reshape(-1)
            observed[:,32]=torch.as_tensor(np.clip((np.log(safe)-mean[32])/std[32],-20,20),device='cuda')
            ray=torch.as_tensor(np.stack([rx,ry],-1).reshape(-1,2),device='cuda',dtype=torch.float32)
            saved={}
            # Preserve the training-run evaluator's flattened query-point batch
            # layout: changing GEMM batch size can move cutoff-adjacent values.
            all_q=torch.arange(len(queries),device='cuda').repeat_interleave(n)
            all_p=torch.arange(n,device='cuda').repeat(len(queries))
            for arm in heads:
                visual=observed if arm=='depth_only' else torch.zeros_like(observed)
                parts=[]
                with torch.inference_mode():
                    for k in range(0,len(all_p),65536):
                        pi,qi=all_p[k:k+65536],all_q[k:k+65536]
                        x=torch.cat([visual[pi],ray[pi],bounds[qi]],-1)
                        parts.append(heads[arm](x).sigmoid()[:,0].cpu().numpy())
                arr=np.concatenate(parts).reshape((len(queries),)+depth.shape)
                if arm=='depth_only': arr[:,~finite]=0.
                path=output/f'{arm}_{r["scan"]}_{r["frame"]:06d}.npy'
                np.save(path,arr)
                saved[arm]=dict(path=str(path),sha256=sha(path))
            sample_path=output/f'predicted_depth_{r["scan"]}_{r["frame"]:06d}.npz'
            np.savez_compressed(sample_path,depth=depth)
            result['rows'].append(dict(row=i,environment=r['environment'],scan=r['scan'],frame=r['frame'],
                predictions=saved,predicted_depth_path=str(sample_path),predicted_depth_sha256=sha(sample_path),
                finite_predicted_pixels=int(finite.sum()),total_pixels=n))
            result.update(status='RUNNING',elapsed_s=time.perf_counter()-start)
            write(output/'predictions.json',result)
            if (i+1)%8==0:print(f'heads {i+1}/{len(rows)}',flush=True)
        result['status']='COMPLETE'
    except Exception as e:
        result.update(status='PARTIAL_BUDGET' if isinstance(e,TimeoutError) else 'FAILED',error=repr(e));raise
    finally:
        result.update(allocation_wall_s=time.perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      resource_release='Owned subprocess exits; no workers/services retained')
        write(output/'predictions.json',result)


def evaluate(sensor, output, budget_s):
    start=time.perf_counter()
    if (output/'evaluation.json').exists():raise FileExistsError('Preserve scored result')
    pm=json.loads((output/'predictions.json').read_text('utf-8-sig'))
    manifest=json.loads((sensor/'dataset_manifest.json').read_text('utf-8-sig'))
    if pm['status']!='COMPLETE':raise ValueError('Complete frozen observations required')
    if sha(sensor/'observations.json') != pm['observations_sha256']:raise ValueError('Observations changed')
    qs=manifest['queries']; records=defaultdict(list); pairs=[]
    for item,r in zip(pm['rows'],manifest['rows'],strict=True):
        if time.perf_counter()-start>=budget_s:raise TimeoutError('Score budget reached')
        if (r['scan'],r['frame'])!=(item['scan'],item['frame']):raise ValueError('Frame order changed')
        if sha(r['reference_path'])!=r['reference_sha256']:raise ValueError('Reference changed')
        with np.load(r['reference_path']) as a: labels=a['labels']
        p=item['predicted_depth_path']
        if sha(p)!=item['predicted_depth_sha256']:raise ValueError('Prediction changed')
        with np.load(p) as a: depth=a['depth']
        scores={}
        for arm,s in item['predictions'].items():
            if sha(s['path'])!=s['sha256']:raise ValueError('Score changed')
            scores[arm]=np.load(s['path'],allow_pickle=False)
        rx,ry=rays(r['depth_K'],r['depth_shape'])
        for j,q in enumerate(qs):
            entry,exit,domain=ray_interval(rx,ry,q)
            predictions={a:(p[j]>=pm['cutoffs'][a])&domain for a,p in scores.items()}
            for name,scale in [('depthpro_raw',1.),('depthpro_cal_scale',1.012320716490867)]:
                z=depth*scale;predictions[name]=domain&np.isfinite(z)&(z>=entry-1e-12)&(z<=exit+1e-12)
            for arm,pred in predictions.items():
                records[arm].append(dict(environment=r['environment'],scan=r['scan'],frame=r['frame'],split=r['split'],
                    query=q['name'],band=band(q),reference_state=r['queries'][j]['state'],
                    predicted_support=int(pred.sum()),predicted_positive=int(pred.sum())>=16,**confusion(labels[j],pred)))
            for base in ('geometry','depthpro_raw','depthpro_cal_scale'):
                pairs.append(dict(arm='depth_only',baseline=base,environment=r['environment'],scan=r['scan'],frame=r['frame'],
                                  query=q['name'],band=band(q),**paired_counts(labels[j],predictions['depth_only'],predictions[base])))
    arms={}
    for arm,rs in records.items():
        s=summarize(rs);s['known_witness_hits']=sum(r['reference_state']=='POSITIVE' and r['tp']>=16 for r in rs)
        arms[arm]=dict(summary=s,records=rs)
    increments={}
    for name,keys in [('all',['arm','baseline']),('environment',['arm','baseline','environment']),
                      ('band',['arm','baseline','band']),('environment_band',['arm','baseline','environment','band'])]:
        increments[name]=aggregate(pairs,keys);write_csv(output/f'increments_{name}.csv',increments[name])
    write_csv(output/'frame_query_increments.csv',pairs)
    result=dict(status='COMPLETE',arms=arms,increments=increments,cutoffs=pm['cutoffs'],target_cal_fpr=pm['target_cal_fpr'],
        frozen=True,thresholds_refitted=False,checkpoint_updated=False,reference_manifest_sha256=sha(sensor/'dataset_manifest.json'),
        predictions_manifest_sha256=sha(output/'predictions.json'),source_sha256=sha(Path(__file__)),wall_s=time.perf_counter()-start,
        limitations='Specified new sensor-ray Development environments only; related rays; no body clearance, walking events or broad camera generalization')
    write(output/'evaluation.json',result)
    print(json.dumps({a:v['summary'] for a,v in arms.items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['predict','evaluate'])
    p.add_argument('--source',type=Path);p.add_argument('--sensor',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--budget-s',type=float,default=240.)
    a=p.parse_args()
    if a.action=='predict':predict(a.source.resolve(),a.sensor.resolve(),a.output.resolve(),a.budget_s)
    else:evaluate(a.sensor.resolve(),a.output.resolve(),a.budget_s)
