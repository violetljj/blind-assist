"""CPU-only reproduction and independent saved-output transfer scoring audit."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from rgb_body_query_3rscan import color_coordinates, sample_prediction
from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_reference_eval import rays, ray_interval


def load(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def counts(label, prediction):
    return dict(tp=int(((label == 1) & prediction).sum()),
                fn=int(((label == 1) & ~prediction).sum()),
                fp=int(((label == 0) & prediction).sum()),
                tn=int(((label == 0) & ~prediction).sum()))


def summarize(rows):
    c = {k:sum(r[k] for r in rows) for k in ('tp','fn','fp','tn')}
    return dict(**c, recall=c['tp']/(c['tp']+c['fn']) if c['tp']+c['fn'] else None,
                fpr=c['fp']/(c['fp']+c['tn']) if c['fp']+c['tn'] else None,
                iou=c['tp']/(c['tp']+c['fn']+c['fp']) if c['tp']+c['fn']+c['fp'] else None,
                frames=len({(r['scan'],r['frame']) for r in rows}),
                query_positive_hits=sum(r['reference_state']=='POSITIVE' and r['predicted_positive'] for r in rows),
                query_positive_total=sum(r['reference_state']=='POSITIVE' for r in rows),
                sampled_free_false_support=sum(r['reference_state']=='FREE_ON_SAMPLED_RAYS' and r['predicted_positive'] for r in rows),
                sampled_free_total=sum(r['reference_state']=='FREE_ON_SAMPLED_RAYS' for r in rows),
                unknown_query_total=sum(r['reference_state']=='UNKNOWN' for r in rows),
                known_witness_hits=sum(r['reference_state']=='POSITIVE' and r['tp']>=16 for r in rows))


def adapter(repo, output, budget_s):
    start=time.perf_counter()
    output.mkdir(parents=True,exist_ok=True)
    if (output/'adapter.json').exists(): raise FileExistsError('Preserve adapter result')
    old=repo/'artifacts.local/work/rgb-body-query-metric-diagnostic-dev-20261009'
    source=old/'readout'; ev=load(source/'evaluation.json'); receipt=load(source/'receipt.json')
    obs=load(repo/'artifacts.local/work/rgb-body-query-cross-session-dev-20261009/sensor/observations.json')
    fm=load(old/'features/feature_manifest.json')
    dp=load(repo/'artifacts.local/work/rgb-body-query-cross-session-dev-20261009/sensor/depthpro/predictions.json')
    i=next(i for i,r in enumerate(obs['rows']) if r['split']=='validation')
    row=obs['rows'][i]; f=fm['rows'][i]
    p=next(p for p in dp['rows'] if (p['scan'],p['frame'])==(row['scan'],row['frame']))
    for path,digest in [(f['feature_path'],f['feature_sha256']),(p['path'],p['sha256']),
                        (source/'observation_normalization.npz',receipt['normalization']['sha256'])]:
        assert sha(path)==digest
    with np.load(f['feature_path']) as a: depth=a['predicted_depth']
    with np.load(p['path']) as a: full=a['depth']
    mx,my=color_coordinates(row['depth_K'],row['color_K'],row['depth_shape'])
    assert np.array_equal(sample_prediction(full,mx,my),depth)
    with np.load(source/'observation_normalization.npz') as a: mean,std=a['mean'],a['std']
    import torch
    torch.set_num_threads(4)
    rx,ry=rays(row['depth_K'],row['depth_shape']); n=depth.size
    ray=torch.as_tensor(np.stack([rx,ry],-1).reshape(-1,2),dtype=torch.float32)
    visual=torch.zeros((n,33));visual[:,32]=torch.as_tensor(np.clip((np.log(depth.reshape(-1))-mean[32])/std[32],-20,20))
    bounds=torch.tensor([q['low']+q['high'] for q in obs['queries']],dtype=torch.float32)
    bounds/=torch.tensor([.9,.55,6.,.9,.55,6.])
    results={}
    for arm in ('depth_only','geometry'):
        cp_path=source/f'{arm}_final.pt'
        assert sha(cp_path)==receipt['stages']['train']['arms'][arm]['sha256']
        cp=torch.load(cp_path,map_location='cpu',weights_only=True)
        assert cp['steps']==200 and cp['seed']==7
        head=torch.nn.Sequential(torch.nn.Linear(41,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
        head.load_state_dict(cp['state_dict']);head.eval().requires_grad_(False)
        vi=visual if arm=='depth_only' else torch.zeros_like(visual)
        predictions=[]
        with torch.inference_mode():
            for q in bounds:
                x=torch.cat([vi,ray,q.expand(n,6)],-1)
                predictions.append(head(x).sigmoid()[:,0].numpy().reshape(depth.shape))
        actual=np.stack(predictions)
        saved=next(p for p in ev['arms'][arm]['predictions'] if p['row']==i)
        assert sha(saved['path'])==saved['sha256']
        expected=np.load(saved['path'],allow_pickle=False)
        error=float(np.max(np.abs(actual-expected)))
        assert error<=2e-6,(arm,error)
        cutoff=ev['arms'][arm]['cutoff']
        differences=[]
        for j,q in enumerate(obs['queries']):
            domain=ray_interval(rx,ry,q)[2]
            differences.append(int((((actual[j]>=cutoff) ^ (expected[j]>=cutoff)) & domain).sum()))
        all_q=torch.arange(len(bounds)).repeat_interleave(n)
        all_p=torch.arange(n).repeat(len(bounds))
        original_cpu=[]
        with torch.inference_mode():
            for j in range(0,len(all_q),65536):
                pi,qi=all_p[j:j+65536],all_q[j:j+65536]
                original_cpu.append(head(torch.cat([vi[pi],ray[pi],bounds[qi]],-1)).sigmoid()[:,0].numpy())
        original_cpu=np.concatenate(original_cpu).reshape(actual.shape)
        boundary=[]
        for j,q in enumerate(obs['queries']):
            domain=ray_interval(rx,ry,q)[2]
            for y,x in np.argwhere(((actual[j]>=cutoff) ^ (expected[j]>=cutoff)) & domain):
                boundary.append(dict(query=q['name'],y=int(y),x=int(x),cpu_main=float(actual[j,y,x]),
                                     cpu_original=float(original_cpu[j,y,x]),saved_cuda=float(expected[j,y,x]),cutoff=cutoff))
        results[arm]=dict(max_absolute_probability_error=error, thresholded_domain_differences=sum(differences),cutoff=cutoff,
                          checkpoint_sha256=sha(cp_path), saved_prediction_sha256=saved['sha256'],
                          actual_thresholded_domain_differences=sum(differences),boundary=boundary,
                          cpu_original_vs_cpu_main_max_error=float(np.max(np.abs(original_cpu-actual))))
        if time.perf_counter()-start>budget_s: raise TimeoutError('Adapter CPU budget reached')
    exact=all(a['actual_thresholded_domain_differences']==0 for a in results.values())
    result=dict(status='PASS' if exact else 'NOT_EXACT_FLOAT_BOUNDARY', arms=results, row=i, scan=row['scan'],frame=row['frame'],
                evaluator_inputs_read=False, cpu_wall_s=time.perf_counter()-start,gpu_s=0,source_sha256=sha(Path(__file__)))
    write(output/'adapter.json',result);print(json.dumps(result),flush=True)


def score(source,sensor,readout,output,budget_s):
    start=time.perf_counter();output.mkdir(parents=True,exist_ok=True)
    if (output/'score.json').exists(): raise FileExistsError('Preserve score audit')
    pm=load(readout/'predictions.json'); ev=load(readout/'evaluation.json')
    m=load(sensor/'dataset_manifest.json');obs=load(sensor/'observations.json')
    old=load(source/'receipt.json');old_ev=load(source/'evaluation.json')
    assert pm['status']==ev['status']=='COMPLETE'
    assert len(pm['rows'])==len(m['rows'])==len(obs['rows'])
    assert m['queries']==obs['queries']
    assert sha(sensor/'dataset_manifest.json')==ev['reference_manifest_sha256']==obs['dataset_manifest_sha256']
    assert sha(sensor/'observations.json')==pm['observations_sha256']
    assert sha(readout/'predictions.json')==ev['predictions_manifest_sha256']
    assert sha(source/'receipt.json')==pm['frozen_receipt_sha256'] and sha(source/'evaluation.json')==pm['frozen_evaluation_sha256']
    assert pm['cutoffs']==ev['cutoffs']=={a:old_ev['arms'][a]['cutoff'] for a in ('depth_only','geometry')}
    assert pm['fit'] is False and pm['evaluator_reads'] is False
    assert ev['frozen'] and not ev['thresholds_refitted'] and not ev['checkpoint_updated']
    assert ev['target_cal_fpr']==pm['target_cal_fpr']==old['target_cal_fpr']
    assert sha(sensor/'depthpro/predictions.json')==pm['depthpro_manifest_sha256']
    assert pm['source_sha256']==ev['source_sha256']==sha(Path(__file__).with_name('rgb_body_query_frozen_transfer.py'))
    assert sha(source/'observation_normalization.npz')==old['normalization']['sha256']
    import torch
    for arm in ('depth_only','geometry'):
        checkpoint=source/f'{arm}_final.pt'
        assert sha(checkpoint)==old['stages']['train']['arms'][arm]['sha256']
        cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert cp['steps']==200 and cp['seed']==7
    records=defaultdict(list); pairs=[]; file_count=0
    for item,row,observation in zip(pm['rows'],m['rows'],obs['rows'],strict=True):
        if time.perf_counter()-start>=budget_s: raise TimeoutError('Score CPU budget reached')
        identity=(row['scan'],row['frame'])
        assert identity==(item['scan'],item['frame'])==(observation['scan'],observation['frame'])
        assert row['rgb_sha256']==observation['rgb_sha256'] and sha(row['rgb_path'])==row['rgb_sha256']
        assert sha(row['reference_path'])==row['reference_sha256']
        with np.load(row['reference_path']) as a: labels=a['labels']
        assert sha(item['predicted_depth_path'])==item['predicted_depth_sha256']
        with np.load(item['predicted_depth_path']) as a: depth=a['depth']
        scores={}
        for arm,p in item['predictions'].items():
            assert sha(p['path'])==p['sha256'];file_count+=1
            scores[arm]=np.load(p['path'],allow_pickle=False)
            assert scores[arm].shape==labels.shape and np.isfinite(scores[arm]).all()
        rx,ry=rays(row['depth_K'],row['depth_shape'])
        for j,q in enumerate(m['queries']):
            entry,exit,domain=ray_interval(rx,ry,q)
            preds={a:(p[j]>=pm['cutoffs'][a])&domain for a,p in scores.items()}
            for arm,scale in [('depthpro_raw',1.),('depthpro_cal_scale',1.012320716490867)]:
                z=depth*scale;preds[arm]=domain&np.isfinite(z)&(z>=entry-1e-12)&(z<=exit+1e-12)
            for arm,pred in preds.items():
                records[arm].append(dict(environment=row['environment'],scan=row['scan'],frame=row['frame'],split=row['split'],
                    query=q['name'],band=f'{q["low"][2]:g}-{q["high"][2]:g}m',reference_state=row['queries'][j]['state'],
                    predicted_support=int(pred.sum()),predicted_positive=int(pred.sum())>=16,**counts(labels[j],pred)))
            p=preds['depth_only'];lab=labels[j]
            for baseline in ('geometry','depthpro_raw','depthpro_cal_scale'):
                b=preds[baseline]
                pairs.append(dict(arm='depth_only',baseline=baseline,environment=row['environment'],scan=row['scan'],frame=row['frame'],
                    band=f'{q["low"][2]:g}-{q["high"][2]:g}m',positive=int((lab==1).sum()),free=int((lab==0).sum()),
                    rgb_tp=int(((lab==1)&p).sum()),geometry_tp=int(((lab==1)&b).sum()),rgb_fp=int(((lab==0)&p).sum()),geometry_fp=int(((lab==0)&b).sum()),
                    positive_rescue=int(((lab==1)&p&~b).sum()),positive_loss=int(((lab==1)&~p&b).sum()),
                    free_rescue=int(((lab==0)&~p&b).sum()),free_added=int(((lab==0)&p&~b).sum())))
    summaries={}
    for arm,rs in records.items():
        assert rs==ev['arms'][arm]['records'],arm
        summaries[arm]=summarize(rs)
        assert summaries[arm]==ev['arms'][arm]['summary'],(arm,summaries[arm],ev['arms'][arm]['summary'])
    key_fields={'all':['arm','baseline'],'environment':['arm','baseline','environment'],
                'band':['arm','baseline','band'],'environment_band':['arm','baseline','environment','band']}
    count_fields=('positive','free','rgb_tp','geometry_tp','rgb_fp','geometry_fp','positive_rescue','positive_loss','free_rescue','free_added')
    slice_counts={}
    for name,keys in key_fields.items():
        grouped=defaultdict(list)
        for row in pairs:grouped[tuple(row[k] for k in keys)].append(row)
        actual=[]
        for key,rs in grouped.items():
            c={k:sum(r[k] for r in rs) for k in count_fields}
            assert c['positive_rescue']-c['positive_loss']==c['rgb_tp']-c['geometry_tp']
            assert c['free_added']-c['free_rescue']==c['rgb_fp']-c['geometry_fp']
            actual.append(dict(zip(keys,key),**c,
                candidate_recall=c['rgb_tp']/c['positive'] if c['positive'] else None,
                baseline_recall=c['geometry_tp']/c['positive'] if c['positive'] else None,
                candidate_fpr=c['rgb_fp']/c['free'] if c['free'] else None,
                baseline_fpr=c['geometry_fp']/c['free'] if c['free'] else None,
                frame_query_records=len(rs),frames=len({(r['scan'],r['frame']) for r in rs})))
        expected=ev['increments'][name]
        assert actual==expected,name
        slice_counts[name]=len(actual)
    result=dict(status='PASS',summaries=summaries,increment_slice_counts=slice_counts,score_files=file_count,
                all_frame_query_records_exact=True,all_summaries_exact=True,paired_rescue_loss_exact=True,
                cutoffs=pm['cutoffs'],target_cal_fpr=pm['target_cal_fpr'],checkpoints_normalization_frozen=True,
                predictions_manifest_sha256=sha(readout/'predictions.json'),evaluation_sha256=sha(readout/'evaluation.json'),
                reference_manifest_sha256=sha(sensor/'dataset_manifest.json'),cpu_wall_s=time.perf_counter()-start,gpu_s=0,
                source_sha256=sha(Path(__file__)))
    write(output/'score.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['adapter','score']);p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[4])
    p.add_argument('--source',type=Path);p.add_argument('--sensor',type=Path);p.add_argument('--readout',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--budget-s',type=float,default=240.)
    a=p.parse_args()
    if a.action=='adapter':adapter(a.repo.resolve(),a.output.resolve(),a.budget_s)
    else:score(a.source.resolve(),a.sensor.resolve(),a.readout.resolve(),a.output.resolve(),a.budget_s)
