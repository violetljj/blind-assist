"""Posthoc point-distance readout of frozen log-distance means.

exp(mu) is the Gaussian log-Z model's Z median, not mean Z. No model changes.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import time
from collections import defaultdict
import numpy as np
from rgb_body_query_interval_distribution import load, utc, choose_cutoff, summaries, interval_probability
from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_reference_eval import rays, ray_interval, confusion
from rgb_body_query_scene_diagnostic import paired_counts, write_csv
from rgb_body_query_frozen_transfer import aggregate
from rgb_body_query_margin_baseline import TARGET

ARMS={'depth_ray_mean_margin_gridcal':'depth_ray_gaussian', 'geometry_mean_margin_gridcal':'geometry_gaussian'}

def scores(row,queries,info):
    rx,ry=rays(row['depth_K'],row['depth_shape']);dist={}
    for name,base in ARMS.items():
        p=row['distributions'][base]
        if sha(p['path'])!=p['sha256']:raise ValueError('Frozen distribution changed')
        with np.load(p['path']) as f:dist[name]=(f['mu'],f['sigma'],f['valid'])
    with np.load(row['sampled_depth_path']) as f:dp=f['depth']
    valid=np.isfinite(dp)&(dp>0);corrected=np.full(dp.shape,np.nan)
    corrected[valid]=np.exp(info['affine']['a']*np.log(dp[valid].astype(np.float64))+info['affine']['b'])
    for q in queries:
        entry,exit,domain=ray_interval(rx,ry,q);point={};cdf={}
        for name,(mu,sigma,v) in dist.items():
            margin=np.full(mu.shape,-np.inf);mask=v&domain;z=np.exp(mu)
            margin[mask]=np.minimum(z[mask]-entry[mask],exit[mask]-z[mask]);point[name]=margin
            cdf[ARMS[name]]=interval_probability(mu,sigma,entry,exit,domain,v)
        margin=np.full(dp.shape,-np.inf);mask=valid&domain
        margin[mask]=np.minimum(corrected[mask]-entry[mask],exit[mask]-corrected[mask]);cdf['affine_margin_gridcal']=margin
        yield point,cdf,domain

def run(repo,root,budget):
    start=time.perf_counter();candidate=root/'candidate';out=root/'mean-probe';out.mkdir(parents=True,exist_ok=True)
    if (out/'execution_plan.json').exists():raise FileExistsError('Preserve prior probe')
    plan=dict(status='POSTHOC_PLAN_BEFORE_CAL',created_utc=utc(),arms=ARMS,amendment_sha256=sha(root/'mean_probe_amendment.json'),
              primary_receipt_sha256=sha(candidate/'completion_receipt.json'),executed_primary_source_sha256=sha(candidate/'executed_interval_distribution.py'),
              source_sha256=sha(__file__),definition='z_point=exp(mu), Z median under log-Gaussian, not arithmetic expected Z; signed opticalZ margin min(z-entry,exit-z)',
              rule='Original cal16/fixed27 free-score conservative closest FPR <= target; scores>=cutoff; no validation selection',budget_cpu_s=budget,
              new_training=False,new_gpu=False,primary4_unchanged=True)
    write(out/'execution_plan.json',plan);info=load(candidate/'training_inputs.json');primarycal=load(candidate/'calibration.json')
    complete=False;failure=None
    def check():
        if time.perf_counter()-start>=budget:raise TimeoutError('Mean probe CPU budget reached')
    try:
        sensor=root/'train-cal-sensor';manifest=load(sensor/'dataset_manifest.json');pm=load(candidate/'predictions/cal/predictions.json')
        refs={(r['scan'],r['frame']):r for r in manifest['rows'] if r['split']=='cal'}
        negative={a:[] for a in ARMS};positive={a:[] for a in ARMS}
        for row in pm['rows']:
            check();ref=refs[row['scan'],row['frame']]
            if sha(ref['reference_path'])!=ref['reference_sha256']:raise ValueError('Cal reference changed')
            with np.load(ref['reference_path']) as f:labels=f['labels']
            for j,(point,_,_) in enumerate(scores(row,manifest['queries'],info)):
                for a,s in point.items():negative[a].append(s[labels[j]==0]);positive[a].append(s[labels[j]==1])
        cal=dict(status='FROZEN_AFTER_CAL',frozen_utc=utc(),frames=primarycal['frames'],target_fpr=TARGET,selection_used_recall=False,arms={},
                 prediction_manifest_path=str(candidate/'predictions/cal/predictions.json'),prediction_manifest_sha256=sha(candidate/'predictions/cal/predictions.json'),
                 primary_training_receipt_sha256=sha(candidate/'training_receipt.json'),amendment_sha256=sha(root/'mean_probe_amendment.json'),posthoc=True)
        for a in ARMS:
            neg=np.concatenate(negative[a]);pos=np.concatenate(positive[a]);choice=choose_cutoff(neg,TARGET,cdf=False)
            choice.update(tp=int((pos>=choice['cutoff']).sum()),fn=int((pos<choice['cutoff']).sum()),delta_m=-choice['cutoff'],cutoff_unit='metres signed interval margin')
            p=out/f'{a}_cal_free_sorted.npy';np.save(p,np.sort(neg));choice.update(sorted_free_path=str(p),sorted_free_sha256=sha(p));cal['arms'][a]=choice
        write(out/'calibration.json',cal);print('PROBE_CAL_FROZEN',cal['arms'],flush=True)
        old=repo/'artifacts.local/work/rgb-body-query-query-level-dev-20261009'
        cohorts={n:old/'fixed-grid-sensor'/n for n in ('original_validation24','new_3rscan64','arkit16')}
        cohorts.update({f'arkit_{n}':old/'additional-arkit-sensor'/n for n in ('40777060','40777065')})
        allsummary=[]
        for name,path in cohorts.items():
            check();manifest=load(path/'dataset_manifest.json');refs={(r['scan'],r['frame']):r for r in manifest['rows']}
            pmpath=candidate/'predictions'/name/'predictions.json';pm=load(pmpath);records=defaultdict(list);pairs=[];witness=[]
            for row in pm['rows']:
                check();r=refs[row['scan'],row['frame']]
                if sha(r['reference_path'])!=r['reference_sha256']:raise ValueError('Evaluation reference changed')
                with np.load(r['reference_path']) as f:labels=f['labels']
                for j,(point,cdf,domain) in enumerate(scores(row,manifest['queries'],info)):
                    q=manifest['queries'][j];state=r['queries'][j]['state'];band=f'{q["low"][2]:g}-{q["high"][2]:g}m'
                    masks={a:(s>=cal['arms'][a]['cutoff'])&domain for a,s in point.items()}
                    basemasks={a:(s>=primarycal['arms'][a]['cutoff'])&domain for a,s in cdf.items()}
                    basemasks.update(masks)
                    for a,mask in masks.items():
                        rec=dict(cohort=name,environment=r['environment'],scan=r['scan'],frame=r['frame'],query=q['name'],distance_band=band,reference_state=state,
                                 predicted_support=int(mask.sum()),predicted_positive=int(mask.sum())>=16,**confusion(labels[j],mask))
                        rec['positive_known_witness']=rec['tp']>=16;records[a].append(rec)
                        for b,bmask in basemasks.items():
                            if a==b:continue
                            pairs.append(dict(cohort=name,arm=a,baseline=b,environment=r['environment'],scan=r['scan'],frame=r['frame'],query=q['name'],band=band,**paired_counts(labels[j],mask,bmask)))
                            ca=rec['tp']>=16;cb=int(((labels[j]==1)&bmask).sum())>=16
                            witness.append(dict(cohort=name,arm=a,baseline=b,environment=r['environment'],scan=r['scan'],frame=r['frame'],query=q['name'],band=band,reference_state=state,
                                                candidate_known_witness=ca,baseline_known_witness=cb,positive_rescue=state=='POSITIVE' and ca and not cb,positive_loss=state=='POSITIVE' and cb and not ca))
            increments={}
            for title,keys in [('all',['cohort','arm','baseline']),('environment',['cohort','arm','baseline','environment']),('band',['cohort','arm','baseline','band']),('environment_band',['cohort','arm','baseline','environment','band'])]:
                increments[title]=aggregate(pairs,keys);write_csv(out/f'{name}_increments_{title}.csv',increments[title])
            write_csv(out/f'{name}_frame_query_pairs.csv',pairs);write_csv(out/f'{name}_query_witness_pairs.csv',witness)
            result=dict(status='COMPLETE',arms={a:dict(records=rr,summary=summaries(rr)) for a,rr in records.items()},increments=increments,
                        calibration_sha256=sha(out/'calibration.json'),primary_prediction_manifest_path=str(pmpath),primary_prediction_manifest_sha256=sha(pmpath),
                        manifest_path=str(path/'dataset_manifest.json'),manifest_sha256=sha(path/'dataset_manifest.json'),posthoc=True)
            write(out/f'{name}_evaluation.json',result)
            allsummary.extend(dict(cohort=name,arm=a,**v['summary']['all']) for a,v in result['arms'].items())
            print('PROBE_EVAL_COMPLETE',name,flush=True)
        write_csv(out/'summary.csv',allsummary);complete=True
    except Exception as exc:failure=repr(exc);raise
    finally:write(out/'completion_receipt.json',dict(status='COMPLETE' if complete else 'PARTIAL',failure=failure,cpu_wall_s=time.perf_counter()-start,gpu_s=0,download_bytes=0,new_training=False,
                 source_sha256=sha(__file__),execution_plan_sha256=sha(out/'execution_plan.json'),resources_released=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--cpu-budget-s',type=float,default=150)
    a=p.parse_args();run(a.repo.resolve(),a.root.resolve(),a.cpu_budget_s)
