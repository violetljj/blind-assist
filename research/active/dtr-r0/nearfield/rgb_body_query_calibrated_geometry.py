"""Train-only low-parameter Depth Pro correction, then original-cal ray margin.

Consumed Development. Every predeclared correction is reported; transfer labels
never fit correction parameters or select a margin. No learned head retraining.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import numpy as np
from rgb_body_query_margin_baseline import TARGET, SCALE, load_json, sampled, grouped, margin_support, focused_checks as interval_checks
from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_reference_eval import rays, ray_interval, confusion
from rgb_body_query_scene_diagnostic import paired_counts, write_csv
from rgb_body_query_frozen_transfer import aggregate

CORRECTIONS = ('log_shift', 'log_affine')
CUTOFFS = dict(depth_only=.5618626475334167, geometry=.6106688380241394)

def utc(): return datetime.now(timezone.utc).isoformat()

def correct(depth, model):
    valid = np.isfinite(depth) & (depth > 0)
    out = np.full(depth.shape, np.nan, dtype=np.float64)
    out[valid] = np.exp(model['a'] * np.log(depth[valid].astype(np.float64)) + model['b'])
    return out

def fit_models(x, y):
    """Fixed 8-step Huber IRLS; all train sensor pixels pooled equally."""
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Need nonempty finite log pairs')
    shift = dict(a=1., b=float(np.median(y-x)))
    weights = np.ones(len(x)); history = []
    for step in range(9):  # initial weighted OLS plus eight robust updates
        sw = weights.sum(); mx = float(np.dot(weights,x)/sw); my = float(np.dot(weights,y)/sw)
        dx = x-mx; variance = float(np.dot(weights,dx*dx))
        if variance <= 0: raise ValueError('Train log-depth has no variance')
        a = float(np.clip(np.dot(weights,dx*(y-my))/variance, .25, 4.))
        b = my-a*mx
        residual = y-(a*x+b)
        history.append(dict(step=step,a=a,b=b,huber_abs_residual_mean=float(np.mean(np.abs(residual)))))
        weights = np.minimum(1., .2/np.maximum(np.abs(residual), np.finfo(np.float64).tiny))
    return dict(log_shift=shift, log_affine=dict(a=a,b=b)), history

def split_rows(manifest, split, expected_frames, expected_envs):
    rows = [r for r in manifest['rows'] if r['split']==split]
    if len(rows)!=expected_frames or len({r['environment'] for r in rows})!=expected_envs:
        raise ValueError('Original split identity mismatch')
    for other in {r['split'] for r in manifest['rows']} - {split}:
        if {r['environment'] for r in rows} & {r['environment'] for r in manifest['rows'] if r['split']==other}:
            raise ValueError('Environment leakage across splits')
    return rows

def checks():
    result=interval_checks()
    z=np.array([.25,1.,4.,np.nan,0.,-1.])
    for model in (dict(a=1.,b=np.log(2.)),dict(a=.5,b=np.log(3.))):
        c=correct(z,model)
        assert np.allclose(c[:3],np.exp(model['b'])*z[:3]**model['a'])
        assert np.isnan(c[3:]).all()
    x=np.linspace(-1,2,101); y=1.2*x+.3
    models,_=fit_models(x,y)
    assert np.allclose([models['log_affine']['a'],models['log_affine']['b']],[1.2,.3],atol=1e-10)
    assert np.isclose(models['log_shift']['b'],np.median(y-x))
    fake=dict(rows=[dict(split='train',environment='a'),dict(split='cal',environment='b')])
    assert split_rows(fake,'train',1,1)[0]['environment']=='a'
    fake['rows'][1]['environment']='a'
    try: split_rows(fake,'train',1,1)
    except ValueError: pass
    else: raise AssertionError('Failed split-isolation check')
    return dict(result, correction_formula=True, synthetic_affine_fit=True, split_isolation=True)

def fit(sensor,out,check_budget):
    manifest=load_json(sensor/'dataset_manifest.json'); source=load_json(sensor/'depthpro/predictions.json')
    lookup={(p['scan'],p['frame']):p for p in source['rows']}
    xs=[]; ys=[]; identities=[]
    for r in split_rows(manifest,'train',56,7):
        check_budget(); p=lookup[r['scan'],r['frame']]; pred,_=sampled(r,p)
        with np.load(r['reference_path']) as ref:
            truth=ref['depth']; observed=ref['observed']
        valid=observed & np.isfinite(truth) & (truth>0) & np.isfinite(pred) & (pred>0)
        xs.append(np.log(pred[valid].astype(np.float64))); ys.append(np.log(truth[valid].astype(np.float64)))
        identities.append(dict(environment=r['environment'],scan=r['scan'],frame=r['frame'],split=r['split'],
                               paired_pixels=int(valid.sum()),reference_sha256=r['reference_sha256'],prediction_sha256=p['sha256']))
    x=np.concatenate(xs); y=np.concatenate(ys); models,history=fit_models(x,y)
    result=dict(status='FROZEN_TRAIN_FIT',frozen_at_utc=utc(),models=models,affine_history=history,
                fitting_pixels=len(x),fitting_frames=identities,train_reference_only=True,
                plan_sha256=sha(out/'plan.json'),source_sha256=sha(__file__),
                fit_errors={name:dict(mean_abs_log=float(np.mean(np.abs(y-(m['a']*x+m['b'])))),
                                     median_signed_log=float(np.median(m['a']*x+m['b']-y))) for name,m in models.items()})
    write(out/'fit.json',result); return result

def calibrate(sensor,out,models,check_budget):
    manifest=load_json(sensor/'dataset_manifest.json'); source=load_json(sensor/'depthpro/predictions.json')
    lookup={(p['scan'],p['frame']):p for p in source['rows']}; store={a:{0:[],1:[]} for a in CORRECTIONS}
    denominator={0:0,1:0}; identities=[]
    for r in split_rows(manifest,'cal',16,2):
        check_budget(); p=lookup[r['scan'],r['frame']]; depth,labels=sampled(r,p)
        corrected={a:correct(depth,m) for a,m in models.items()}; rx,ry=rays(r['depth_K'],r['depth_shape'])
        for j,q in enumerate(manifest['queries']):
            entry,exit,domain=ray_interval(rx,ry,q)
            for label in (0,1): denominator[label]+=int((labels[j]==label).sum())
            for arm,z in corrected.items():
                margin=np.minimum(z-entry,exit-z); valid=domain & np.isfinite(z) & (z>0)
                for label in (0,1): store[arm][label].append(margin[(labels[j]==label)&valid])
        identities.append(dict(environment=r['environment'],scan=r['scan'],frame=r['frame'],split=r['split'],
                               reference_sha256=r['reference_sha256'],prediction_sha256=p['sha256']))
    calibrations={}
    for arm in CORRECTIONS:
        sorted_m={k:np.sort(np.concatenate(v)) for k,v in store[arm].items()}; curve=[]
        for i in range(-100,101):
            delta=i/100.; count={k:len(v)-int(np.searchsorted(v,-delta,'left')) for k,v in sorted_m.items()}
            fpr=count[0]/denominator[0]
            curve.append(dict(delta_m=delta,tp=count[1],fn=denominator[1]-count[1],fp=count[0],tn=denominator[0]-count[0],
                              recall=count[1]/denominator[1],fpr=fpr,distance_to_target=abs(fpr-TARGET)))
        chosen=min(curve,key=lambda v:(v['distance_to_target'],abs(v['delta_m']),v['delta_m']))
        calibrations[arm]=dict(selected=chosen,exact_target_matched=chosen['fpr']==TARGET,curve=curve)
        write_csv(out/f'{arm}_calibration_curve.csv',curve)
    result=dict(status='FROZEN_AFTER_CAL',frozen_at_utc=utc(),arms=calibrations,target_cal_fpr=TARGET,
                calibration_frames=identities,selection_used_recall=False,transfer_reference_reads_before_freeze=False,
                fit_sha256=sha(out/'fit.json'),plan_sha256=sha(out/'plan.json'))
    write(out/'calibration.json',result); return result

def evaluate(name,sensor,heads,out,models,cal,check_budget,original=False):
    manifest=load_json(sensor/'dataset_manifest.json'); previous=load_json(heads/'evaluation.json')
    if original:
        source=load_json(sensor/'depthpro/predictions.json'); rows=split_rows(manifest,'validation',24,3)
        lookup={(p['scan'],p['frame']):p for p in source['rows']}
        scores={a:{(manifest['rows'][p['row']]['scan'],manifest['rows'][p['row']]['frame']):p for p in previous['arms'][a]['predictions']} for a in CUTOFFS}
        cutoffs={a:previous['arms'][a]['cutoff'] for a in CUTOFFS}
    else:
        source=load_json(heads/'predictions.json'); rows=manifest['rows']; lookup={(p['scan'],p['frame']):p for p in source['rows']}; cutoffs=source['cutoffs']
    if cutoffs!=CUTOFFS: raise ValueError('Saved head cutoffs changed')
    records=defaultdict(list); pairs=[]; reproduction=0
    old_raw={(r['scan'],r['frame'],r['query']):r for r in load_json(sensor/'evaluation.json')['records']} if original else None
    for r in rows:
        check_budget(); key=r['scan'],r['frame']; p=lookup[key]
        if original: depth,labels=sampled(r,p); entries={a:scores[a][key] for a in CUTOFFS}
        else:
            if sha(p['predicted_depth_path'])!=p['predicted_depth_sha256'] or sha(r['reference_path'])!=r['reference_sha256']: raise ValueError('Changed payload')
            with np.load(p['predicted_depth_path']) as data: depth=data['depth']
            with np.load(r['reference_path']) as ref: labels=ref['labels']
            entries=p['predictions']
        probs={}
        for a,item in entries.items():
            if sha(item['path'])!=item['sha256']: raise ValueError('Changed saved head score')
            probs[a]=np.load(item['path'],allow_pickle=False)
        corrected={a:correct(depth,m) for a,m in models.items()}; rx,ry=rays(r['depth_K'],r['depth_shape'])
        previous_by_arm={a:{(v['scan'],v['frame'],v['query']):v for v in (previous['arms'][a]['validation']['records'] if original else previous['arms'][a]['records'])} for a in CUTOFFS}
        for j,q in enumerate(manifest['queries']):
            entry,exit,domain=ray_interval(rx,ry,q); band=f'{q["low"][2]:g}-{q["high"][2]:g}m'
            masks={a:(v[j]>=CUTOFFS[a])&domain for a,v in probs.items()}
            masks['depthpro_raw']=margin_support(depth,entry,exit,domain,0.)
            masks['depthpro_cal_scale']=margin_support(depth*SCALE,entry,exit,domain,0.)
            for a,z in corrected.items():
                masks[a+'_direct']=margin_support(z,entry,exit,domain,0.)
                masks[a+'_margin']=margin_support(z,entry,exit,domain,cal['arms'][a]['selected']['delta_m'])
            for a,mask in masks.items():
                rec=dict(cohort=name,environment=r['environment'],scan=r['scan'],frame=r['frame'],split=r['split'],query=q['name'],
                         distance_band=band,reference_state=r['queries'][j]['state'],predicted_support=int(mask.sum()),predicted_positive=int(mask.sum())>=16,
                         **confusion(labels[j],mask)); records[a].append(rec)
                old=previous_by_arm[a][key+(q['name'],)] if a in CUTOFFS else (old_raw[key+(q['name'],)] if original and a=='depthpro_raw' else None)
                if old is not None:
                    assert all(rec[k]==old[k] for k in ('tp','fn','fp','tn','predicted_support','predicted_positive')); reproduction+=1
            comparisons=[(a,b) for a in [c+'_margin' for c in CORRECTIONS] for b in ('depthpro_raw','geometry','depth_only')]
            comparisons += [('depth_only',c+'_margin') for c in CORRECTIONS]+[(c+'_margin',c+'_direct') for c in CORRECTIONS]
            for a,b in comparisons:
                pairs.append(dict(cohort=name,arm=a,baseline=b,environment=r['environment'],scan=r['scan'],frame=r['frame'],query=q['name'],band=band,
                                  **paired_counts(labels[j],masks[a],masks[b])))
    increments={}
    for label,keys in [('all',['cohort','arm','baseline']),('environment',['cohort','arm','baseline','environment']),('band',['cohort','arm','baseline','band']),('environment_band',['cohort','arm','baseline','environment','band'])]:
        increments[label]=aggregate(pairs,keys); write_csv(out/f'{name}_increments_{label}.csv',increments[label])
    write_csv(out/f'{name}_frame_query_increments.csv',pairs)
    result=dict(status='COMPLETE',evaluation_started_after_cal=True,arms={a:dict(summary=grouped(rs),records=rs) for a,rs in records.items()},
                increments=increments,checks=dict(frozen_records_reproduced=reproduction),source_manifests={str(p):sha(p) for p in (sensor/'dataset_manifest.json',heads/'evaluation.json')})
    write(out/f'{name}_evaluation.json',result); return result

def run(repo,out,budget):
    start=time.perf_counter(); out.mkdir(parents=True,exist_ok=True)
    if (out/'fit.json').exists(): raise FileExistsError('Preserve prior run')
    def check_budget():
        if time.perf_counter()-start>=budget: raise TimeoutError('CPU research budget reached')
    plan=dict(status='FROZEN_BEFORE_FIT',created_at_utc=utc(),cpu_budget_s=budget,gpu_s=0,download_bytes=0,compute_status='TASK_NOT_GPU_SUITABLE',
              corrections=list(CORRECTIONS),train_fit='Only original train 56 frames/7 environments, all finite positive observed paired sensor points pooled; no environment-specific fit',
              log_shift='b=median(log measured_z-log predicted_z), a=1',log_affine='OLS initial plus 8 Huber IRLS iterations; threshold 0.2 log units; each slope clipped [0.25,4] and intercept weighted mean residual; no model selection',
              correction='exp(a*log(z)+b), finite positive only, invalid retained as nan',delta_definition='Signed optical-Z interval margin metres; original reachable mask fixed; negative contracts, inverted invalid',
              delta_candidates_m=[i/100 for i in range(-100,101)],target_cal_fpr=TARGET,selection='Nearest absolute cal FPR error, ties smallest abs(delta) then ascending delta; recall not used',
              cal='Original cal 16 frames/2 environments; save fit then cal freeze before transfer reference access',head_cutoffs=CUTOFFS,
              cohorts=['original_validation24','new_3rscan64','arkit16','native_vga8','derived_256_8'],
              decision_check='Report all predetermined arms, actual FPR and paired rescue/loss; no superiority gate or transfer FPR matching claim; correlated query-ray N, UNKNOWN not negative',
              source_sha256=sha(__file__))
    write(out/'plan.json',plan); write(out/'focused_check.json',checks())
    base=repo/'artifacts.local/work'; old=base/'rgb-body-query-cross-session-dev-20261009/sensor'
    fitting=fit(old,out,check_budget); print('FIT_FROZEN',json.dumps(fitting['models']),flush=True)
    cal=calibrate(old,out,fitting['models'],check_budget); frozen={p:sha(out/p) for p in ('fit.json','calibration.json','plan.json')}
    print('CAL_FROZEN',json.dumps({a:v['selected'] for a,v in cal['arms'].items()}),flush=True)
    transfer=base/'rgb-body-query-transfer-dev-20261009'; prior=base/'rgb-body-query-input-baseline-dev-20261009'
    cohorts=[('original_validation',old,base/'rgb-body-query-metric-diagnostic-dev-20261009/readout',True),('new_3rscan',transfer/'sensor',transfer/'readout',False),('arkit',transfer/'camera-sensor',transfer/'camera-readout',False)]
    cohorts += [(name,prior/'paired-sensor'/name,prior/'paired-readout'/name,False) for name in ('native_vga','derived_256')]
    results={}
    for name,sensor,heads,original in cohorts:
        check_budget(); r=evaluate(name,sensor,heads,out,fitting['models'],cal,check_budget,original)
        results[name]={a:v['summary']['all'] for a,v in r['arms'].items()}; print(name,json.dumps(results[name]),flush=True)
    assert all(sha(out/p)==s for p,s in frozen.items())
    result=dict(status='COMPLETE',models=fitting['models'],calibration={a:v['selected'] for a,v in cal['arms'].items()},cohorts=results,frozen_sha256=frozen,
                wall_s=time.perf_counter()-start,cpu_budget_s=budget,gpu_s=0,download_bytes=0,limits='Consumed correlated Development; sensor reference not metrology truth; no whole-box/body FPR, walking lead time, thin obstacle or deployment evidence; no transfer recalibration')
    write(out/'evaluation.json',result); shutil.copyfile(__file__,out/'executed_calibrated_geometry.py')
    write(out/'completion_receipt.json',dict(status='COMPLETE',wall_s=time.perf_counter()-start,cpu_budget_s=budget,gpu_s=0,download_bytes=0,source_sha256=sha(__file__),
          result_sha256=sha(out/'evaluation.json'),frozen_sha256=frozen,resource_release='CPU subprocess exits, no workers or services'))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--budget-s',type=float,default=600)
    args=p.parse_args(); run(args.repo.resolve(),args.output.resolve(),args.budget_s)
