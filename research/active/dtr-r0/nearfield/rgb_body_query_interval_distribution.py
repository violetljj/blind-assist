"""Query-independent distance distributions and analytic interval readout.

Consumed Development: train observed metric distances, cal thresholds, then
freeze before transfer scoring. A visible first surface is not volume truth.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import ndtr
from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_margin_baseline import sampled, TARGET
from rgb_body_query_3rscan import color_coordinates, sample_prediction
from rgb_body_query_reference_eval import rays, ray_interval, confusion, ratios
from rgb_body_query_scene_diagnostic import paired_counts, write_csv
from rgb_body_query_frozen_transfer import aggregate
from rgb_body_query_fixed_grid import PUBLIC, geometry_predictions

DISTRIBUTIONS = ('affine_gaussian', 'depth_ray_gaussian', 'geometry_gaussian')
ARMS = DISTRIBUTIONS + ('affine_margin_gridcal',)

def load(p): return json.loads(Path(p).read_text('utf-8-sig'))
def utc(): return datetime.now(timezone.utc).isoformat()

def interval_probability(mu, sigma, entry, exit, domain, valid):
    if not np.isfinite(mu).all() or not np.isfinite(sigma).all() or (sigma <= 0).any():
        raise ValueError('Finite mean and positive sigma required')
    mask = domain & valid & (entry > 0) & (exit >= entry)
    out = np.zeros(mu.shape, np.float64)
    out[mask] = ndtr((np.log(exit[mask])-mu[mask])/sigma[mask])-ndtr((np.log(entry[mask])-mu[mask])/sigma[mask])
    return np.clip(out, 0., 1.)

def focused_checks():
    rng=np.random.default_rng(7); mu=rng.normal(size=10000); sigma=rng.uniform(.03,2.,10000)
    one=np.ones(10000,bool); a=np.full(10000,.3); b=np.full(10000,.8); c=np.full(10000,1.5)
    ab=interval_probability(mu,sigma,a,b,one,one); bc=interval_probability(mu,sigma,b,c,one,one)
    ac=interval_probability(mu,sigma,a,c,one,one)
    assert np.allclose(ac,ab+bc,atol=3e-16,rtol=1e-14)
    assert np.all(ac+1e-15>=ab)
    assert not interval_probability(mu,sigma,c,a,one,one).any()
    assert not interval_probability(mu,sigma,a,c,~one,one).any()
    assert not interval_probability(mu,sigma,a,c,one,~one).any()
    return dict(status='PASS',partition_additivity=True,containment_monotonicity=True,
                invalid_query_zero=True,invalid_prediction_zero=True,points=10000)

def summaries(records):
    result={}
    for title,keys in [('all',[]),('environment',['environment']),('band',['distance_band']),('environment_band',['environment','distance_band'])]:
        groups=defaultdict(list)
        for r in records: groups[tuple(r[k] for k in keys)].append(r)
        for key,rr in groups.items():
            counts={k:sum(r[k] for r in rr) for k in ('tp','fn','fp','tn')}
            pos=[r for r in rr if r['reference_state']=='POSITIVE']
            free=[r for r in rr if r['reference_state']=='FREE_ON_SAMPLED_RAYS']
            unk=[r for r in rr if r['reference_state']=='UNKNOWN']
            result['/'.join([title]+list(key))]=dict(**counts,**ratios(counts),frames=len({(r['scan'],r['frame']) for r in rr}),
                query_positive_total=len(pos),query_positive_hits=sum(r['predicted_positive'] for r in pos),
                query_positive_known_witness_hits=sum(r['tp']>=16 for r in pos),sampled_free_total=len(free),
                sampled_free_false_support=sum(r['predicted_positive'] for r in free),unknown_query_total=len(unk),
                unknown_query_full_support=sum(r['predicted_positive'] for r in unk))
    return result

def public_depth(r,p):
    if sha(p['path'])!=p['sha256'] or p['rgb_sha256']!=r['rgb_sha256']:
        raise ValueError('Public cached prediction changed')
    with np.load(p['path']) as f: full=f['depth']
    mx,my=color_coordinates(np.asarray(r['depth_K']),np.asarray(r['color_K']),r['depth_shape'])
    return sample_prediction(full,mx,my)

def train_points(sensor,fit,cpu_check):
    manifest=load(sensor/'dataset_manifest.json'); dp=load(sensor/'depthpro/predictions.json')
    lookup={(p['scan'],p['frame']):p for p in dp['rows']}
    rows=[r for r in manifest['rows'] if r['split']=='train']
    assert len(rows)==56 and len({r['environment'] for r in rows})==7
    xs=[]; ys=[]; ids=[]
    for r in rows:
        cpu_check(); p=lookup[r['scan'],r['frame']]; depth,_=sampled(r,p)
        with np.load(r['reference_path']) as f: z=f['depth']; observed=f['observed']
        valid=observed & np.isfinite(z)&(z>0)&np.isfinite(depth)&(depth>0)
        rx,ry=rays(r['depth_K'],r['depth_shape'])
        xs.append(np.stack([np.log(depth[valid].astype(np.float64)),rx[valid],ry[valid]],-1));ys.append(np.log(z[valid].astype(np.float64)))
        ids.append(dict(environment=r['environment'],scan=r['scan'],frame=r['frame'],split='train',paired_pixels=int(valid.sum()),
                        reference_path=r['reference_path'],reference_sha256=r['reference_sha256'],prediction_path=p['path'],prediction_sha256=p['sha256']))
    x64=np.concatenate(xs);x=x64.astype(np.float32); y=np.concatenate(ys)
    if len(y)!=1591791: raise ValueError('Original paired training point identity mismatch')
    a,b=fit['models']['log_affine']['a'],fit['models']['log_affine']['b']
    sigma=float(np.sqrt(np.mean((y-(a*x64[:,0]+b))**2)))
    mean=float(x64[:,0].mean());std=float(x64[:,0].std())
    return x,y.astype(np.float32),dict(points=len(y),frames=ids,normalization=dict(log_dp_mean=mean,log_dp_std=std,ray_normalization='public ray x/y unchanged'),
         affine=dict(a=a,b=b),affine_residual_rms=sigma,sigma_initial=float(np.clip(sigma,.030001,1.999999)),geometry_mean=float(y.mean()))

def gpu_train_predict(repo,out,sensor,cohorts,x,y,info,budget):
    start=time.perf_counter(); receipt=dict(status='STARTING',budget_gpu_allocation_wall_s=budget,arms={},predictions={})
    import torch
    torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for predeclared training stage')
    torch.cuda.reset_peak_memory_stats(); models={}; data=None; target=None
    def check():
        if time.perf_counter()-start>=budget-3: raise TimeoutError('Candidate GPU allocation budget reached')
    try:
        data=torch.tensor(x,device='cuda');target=torch.tensor(y,device='cuda')
        norm=info['normalization']; mean,std=norm['log_dp_mean'],norm['log_dp_std']
        normalized=data.clone();normalized[:,0]=(normalized[:,0]-mean)/std
        a,b=info['affine']['a'],info['affine']['b'];initial=info['sigma_initial']
        sigma_logit=float(np.log((initial-.03)/(2.-initial)))
        for arm in ('depth_ray_gaussian','geometry_gaussian'):
            check(); torch.manual_seed(7);torch.cuda.manual_seed_all(7)
            model=torch.nn.Sequential(torch.nn.Linear(3,64),torch.nn.ReLU(),torch.nn.Linear(64,64),torch.nn.ReLU(),torch.nn.Linear(64,2)).cuda()
            torch.nn.init.zeros_(model[-1].weight);torch.nn.init.zeros_(model[-1].bias)
            optimizer=torch.optim.Adam(model.parameters(),lr=.002)
            generator=torch.Generator(device='cuda').manual_seed(7);history=[];steps=0
            for step in range(600):
                check();ix=torch.randint(len(target),(8192,),generator=generator,device='cuda');inputs=normalized[ix].clone()
                if arm=='geometry_gaussian': inputs[:,0]=0
                output=model(inputs)
                base=info['geometry_mean']
                mu=base+output[:,0];sig=.03+1.97*torch.sigmoid(output[:,1]+sigma_logit)
                loss=(torch.log(sig)+.5*((target[ix]-mu)/sig)**2+.5*np.log(2*np.pi)).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite train loss')
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();steps=step+1
                if step in (0,99,199,299,399,499,599):history.append(dict(step=steps,nll=float(loss.item())))
            model.eval().requires_grad_(False);models[arm]=model
            path=out/f'{arm}_final.pt';torch.save(dict(state_dict=model.state_dict(),arm=arm,steps=steps,normalization=norm,initial=info),path)
            receipt['arms'][arm]=dict(steps=steps,checkpoint_path=str(path),sha256=sha(path),history=history,final_only=True,seed=7,sampling_seed=7)
            del optimizer;torch.cuda.synchronize()
            print('TRAIN_COMPLETE',arm,steps,flush=True)
        del data,target,normalized;data=target=None
        # Inference reads only public observations and cached RGB prediction manifests.
        for name,path in [('cal',sensor)]+list(cohorts.items()):
            check();obs=load(path/'observations.json');dp=load(path/'depthpro/predictions.json')
            if any(set(r)-set(PUBLIC) for r in obs['rows']):raise ValueError('Evaluator truth in public observations')
            lookup={(p['scan'],p['frame']):p for p in dp['rows']};dest=out/'predictions'/name;dest.mkdir(parents=True,exist_ok=True);saved=[]
            for r in obs['rows']:
                if name=='cal' and r['split']!='cal':continue
                check();depth=public_depth(r,lookup[r['scan'],r['frame']]);valid=np.isfinite(depth)&(depth>0)
                safe=np.where(valid,depth,1.);log=np.log(safe.astype(np.float64));rx,ry=rays(r['depth_K'],r['depth_shape'])
                inp=np.stack([(log-mean)/std,rx,ry],-1).reshape(-1,3).astype(np.float32);t=torch.tensor(inp,device='cuda');distribution={}
                distribution['affine_gaussian']=(a*log+b,np.full(depth.shape,initial),valid)
                with torch.inference_mode():
                    for arm,model in models.items():
                        current=t.clone()
                        if arm=='geometry_gaussian':current[:,0]=0.
                        raw=model(current).cpu().numpy().astype(np.float64)
                        base=np.full(depth.shape,info['geometry_mean'])
                        mu=base+raw[:,0].reshape(depth.shape)
                        sig=.03+1.97/(1.+np.exp(-(raw[:,1].reshape(depth.shape)+sigma_logit)))
                        distribution[arm]=(mu,sig,valid if arm=='depth_ray_gaussian' else np.ones(depth.shape,bool))
                row=dict(r,distributions={})
                dp_path=dest/f'depth_{r["scan"]}_{r["frame"]:06d}.npz';np.savez(dp_path,depth=depth)
                row.update(sampled_depth_path=str(dp_path),sampled_depth_sha256=sha(dp_path),depthpro_source_path=lookup[r['scan'],r['frame']]['path'],depthpro_source_sha256=lookup[r['scan'],r['frame']]['sha256'])
                for arm,(mu,sig,mask) in distribution.items():
                    p=dest/f'{arm}_{r["scan"]}_{r["frame"]:06d}.npz';np.savez(p,mu=mu,sigma=sig,valid=mask)
                    row['distributions'][arm]=dict(path=str(p),sha256=sha(p))
                saved.append(row);del t
            pm=dict(status='COMPLETE',rows=saved,observations_sha256=sha(path/'observations.json'),depthpro_manifest_sha256=sha(path/'depthpro/predictions.json'),
                    evaluator_reads=False,queries_not_network_inputs=True)
            write(dest/'predictions.json',pm);receipt['predictions'][name]=dict(path=str(dest/'predictions.json'),sha256=sha(dest/'predictions.json'),frames=len(saved))
            print('PREDICT_COMPLETE',name,len(saved),flush=True)
        torch.cuda.synchronize();receipt['status']='COMPLETE'
        return receipt
    except Exception as exc:
        receipt.update(status='PARTIAL',error=repr(exc));raise
    finally:
        receipt.update(gpu_allocation_wall_s=time.perf_counter()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),device=torch.cuda.get_device_name(0),framework=torch.__version__,resources_released=True)
        models.clear();data=target=None;torch.cuda.empty_cache()
        write(out/'gpu_receipt.json',receipt)

def frame_scores(row,queries,info):
    rx,ry=rays(row['depth_K'],row['depth_shape'])
    dist={}
    for arm,item in row['distributions'].items():
        if sha(item['path'])!=item['sha256']:raise ValueError('Distribution changed')
        with np.load(item['path']) as f:dist[arm]=(f['mu'],f['sigma'],f['valid'])
    if sha(row['sampled_depth_path'])!=row['sampled_depth_sha256']:raise ValueError('Sampled depth changed')
    with np.load(row['sampled_depth_path']) as f:depth=f['depth']
    valid=np.isfinite(depth)&(depth>0); corrected=np.full(depth.shape,np.nan)
    corrected[valid]=np.exp(info['affine']['a']*np.log(depth[valid].astype(np.float64))+info['affine']['b'])
    for q in queries:
        entry,exit,domain=ray_interval(rx,ry,q)
        result={a:interval_probability(mu,sig,entry,exit,domain,v) for a,(mu,sig,v) in dist.items()}
        margin=np.full(depth.shape,-np.inf);m=domain&valid
        margin[m]=np.minimum(corrected[m]-entry[m],exit[m]-corrected[m]);result['affine_margin_gridcal']=margin
        yield result,domain

def choose_cutoff(scores,target,cdf=True):
    values=np.sort(scores.astype(np.float64));n=len(values);allowed=int(np.floor(target*n))
    # Threshold is selected from free scores only; include score ties as a group.
    if not n:raise ValueError('Cal free reference required')
    unique,counts=np.unique(values,return_counts=True);tail=np.cumsum(counts[::-1])[::-1]
    eligible=(tail<=allowed)&np.isfinite(unique)
    if cdf:eligible &= unique>0
    if eligible.any(): cutoff=float(unique[np.flatnonzero(eligible)[0]])
    else:cutoff=float(np.nextafter(values[-1],np.inf))
    fp=int(n-np.searchsorted(values,cutoff,side='left'))
    if fp/n>target:raise AssertionError('Nonconservative cal threshold')
    return dict(cutoff=cutoff,fp=fp,tn=n-fp,fpr=fp/n,free=n,allowed_fp=allowed,
                ties_at_cutoff=int((values==cutoff).sum()),exact_target_matched=fp/n==target)

def calibrate(out,sensor,info,cpu_check):
    started=utc();manifest=load(sensor/'dataset_manifest.json');pm=load(out/'predictions/cal/predictions.json')
    refs={(r['scan'],r['frame']):r for r in manifest['rows'] if r['split']=='cal'}
    assert len(refs)==16 and len({r['environment'] for r in refs.values()})==2
    free={a:[] for a in ARMS};positive={a:[] for a in ARMS};ids=[]
    for row in pm['rows']:
        cpu_check();ref=refs[row['scan'],row['frame']]
        if sha(ref['reference_path'])!=ref['reference_sha256']:raise ValueError('Cal reference changed')
        with np.load(ref['reference_path']) as f:labels=f['labels']
        for j,(scores,_) in enumerate(frame_scores(row,manifest['queries'],info)):
            for arm,p in scores.items():free[arm].append(p[labels[j]==0]);positive[arm].append(p[labels[j]==1])
        ids.append(dict(scan=row['scan'],frame=row['frame'],environment=row['environment'],reference_path=ref['reference_path'],reference_sha256=ref['reference_sha256']))
    result=dict(status='FROZEN_AFTER_CAL',started_utc=started,frozen_utc=utc(),target_fpr=TARGET,selection_used_recall=False,
                rule='Closest achievable free FPR <= target, all score ties included; CDF zero scores excluded; scores >= cutoff',frames=ids,arms={},
                training_receipt_sha256=sha(out/'training_receipt.json'),predictions_sha256=sha(out/'predictions/cal/predictions.json'))
    for arm in ARMS:
        neg=np.concatenate(free[arm]);pos=np.concatenate(positive[arm]);choice=choose_cutoff(neg,TARGET,cdf=arm!='affine_margin_gridcal')
        choice.update(tp=int((pos>=choice['cutoff']).sum()),fn=int((pos<choice['cutoff']).sum()),cutoff_unit='metres signed interval margin' if arm=='affine_margin_gridcal' else 'interval probability')
        if arm=='affine_margin_gridcal':choice['delta_m']=-choice['cutoff']
        p=out/f'{arm}_cal_free_sorted.npy';np.save(p,np.sort(neg));choice.update(sorted_free_path=str(p),sorted_free_sha256=sha(p))
        result['arms'][arm]=choice
    write(out/'calibration.json',result);print('CAL_FROZEN',json.dumps(result['arms']),flush=True);return result

def evaluate(name,sensor,heads,out,info,cal,oldroot,cpu_check):
    manifest=load(sensor/'dataset_manifest.json');pm=load(out/'predictions'/name/'predictions.json');hm=load(heads/'predictions.json')
    hlookup={(r['scan'],r['frame']):r for r in hm['rows']};refs={(r['scan'],r['frame']):r for r in manifest['rows']}
    fit=load(oldroot.parent/'rgb-body-query-calibrated-transfer-dev-20261009/geometry/fit.json')
    oldcal=load(oldroot.parent/'rgb-body-query-calibrated-transfer-dev-20261009/geometry/calibration.json')
    records=defaultdict(list);pairs=[];witness=[]
    for row in pm['rows']:
        cpu_check();key=row['scan'],row['frame'];r=refs[key];h=hlookup[key]
        if sha(r['reference_path'])!=r['reference_sha256']:raise ValueError('Evaluation reference changed')
        with np.load(r['reference_path']) as f:labels=f['labels']
        hs=h['predictions']['depth_only']
        if sha(hs['path'])!=hs['sha256']:raise ValueError('Saved old head changed')
        oldhead=np.load(hs['path']);depth=np.load(row['sampled_depth_path'])['depth']
        geo=geometry_predictions(depth,row['depth_K'],row['depth_shape'],manifest['queries'],fit,oldcal)
        for j,((scores,domain),baselinegeo) in enumerate(zip(frame_scores(row,manifest['queries'],info),geo)):
            q=manifest['queries'][j];base=dict(old_depth_only=(oldhead[j]>=hm['cutoffs']['depth_only'])&domain,old_log_affine_margin=baselinegeo['log_affine_margin'])
            masks={a:(scores[a]>=cal['arms'][a]['cutoff'])&domain for a in ARMS}
            band=f'{q["low"][2]:g}-{q["high"][2]:g}m';state=r['queries'][j]['state']
            for arm,mask in masks.items():
                rec=dict(cohort=name,environment=r['environment'],scan=r['scan'],frame=r['frame'],query=q['name'],distance_band=band,
                         reference_state=state,predicted_support=int(mask.sum()),predicted_positive=int(mask.sum())>=16,**confusion(labels[j],mask))
                rec['positive_known_witness']=rec['tp']>=16;records[arm].append(rec)
                for baseline,bmask in {**base,'affine_margin_gridcal':masks['affine_margin_gridcal'],'geometry_gaussian':masks['geometry_gaussian']}.items():
                    if arm==baseline:continue
                    pair=dict(cohort=name,arm=arm,baseline=baseline,environment=r['environment'],scan=r['scan'],frame=r['frame'],query=q['name'],band=band,**paired_counts(labels[j],mask,bmask))
                    pairs.append(pair)
                    ca=int(((labels[j]==1)&mask).sum())>=16;cb=int(((labels[j]==1)&bmask).sum())>=16
                    witness.append(dict(cohort=name,arm=arm,baseline=baseline,environment=r['environment'],scan=r['scan'],frame=r['frame'],query=q['name'],band=band,
                         reference_state=state,candidate_known_witness=ca,baseline_known_witness=cb,positive_rescue=state=='POSITIVE' and ca and not cb,positive_loss=state=='POSITIVE' and cb and not ca,
                         candidate_full_support=int(mask.sum())>=16,baseline_full_support=int(bmask.sum())>=16))
    increments={}
    for title,keys in [('all',['cohort','arm','baseline']),('environment',['cohort','arm','baseline','environment']),('band',['cohort','arm','baseline','band']),('environment_band',['cohort','arm','baseline','environment','band'])]:
        increments[title]=aggregate(pairs,keys);write_csv(out/'evaluation'/f'{name}_increments_{title}.csv',increments[title])
    write_csv(out/'evaluation'/f'{name}_query_witness_pairs.csv',witness)
    write_csv(out/'evaluation'/f'{name}_frame_query_ray_pairs.csv',pairs)
    result=dict(status='COMPLETE',arms={a:dict(records=rr,summary=summaries(rr)) for a,rr in records.items()},increments=increments,
        calibration_sha256=sha(out/'calibration.json'),manifest_sha256=sha(sensor/'dataset_manifest.json'),prediction_manifest_sha256=sha(out/'predictions'/name/'predictions.json'),
        old_head_manifest_path=str(heads/'predictions.json'),old_head_manifest_sha256=sha(heads/'predictions.json'),old_geometry_fit_path=str(oldroot.parent/'rgb-body-query-calibrated-transfer-dev-20261009/geometry/fit.json'),
        old_geometry_cal_path=str(oldroot.parent/'rgb-body-query-calibrated-transfer-dev-20261009/geometry/calibration.json'),old_calibration_difference='Old saved head/affine-margin calibrated on original15queries; new4arms calibrated on fixed27queries. Practical comparison, not isolated architecture causal evidence.')
    write(out/'evaluation'/f'{name}_evaluation.json',result)
    print('EVALUATION_COMPLETE',name,json.dumps({a:v['summary']['all'] for a,v in result['arms'].items()}),flush=True)
    return result

def run(repo,root,cpu_budget,gpu_budget):
    wall=time.perf_counter();out=root/'candidate';out.mkdir(parents=True,exist_ok=True)
    if (out/'execution_plan.json').exists():raise FileExistsError('Preserve prior execution')
    sensor=root/'train-cal-sensor';old=repo/'artifacts.local/work/rgb-body-query-query-level-dev-20261009'
    cohorts={n:old/'fixed-grid-sensor'/n for n in ('original_validation24','new_3rscan64','arkit16')}
    cohorts.update({f'arkit_{n}':old/'additional-arkit-sensor'/n for n in ('40777060','40777065')})
    heads={n:old/'fixed-grid-readout'/n for n in ('original_validation24','new_3rscan64','arkit16')}
    heads.update({f'arkit_{n}':old/'additional-arkit-readout'/n for n in ('40777060','40777065')})
    fitpath=repo/'artifacts.local/work/rgb-body-query-calibrated-transfer-dev-20261009/geometry/fit.json'
    plan=dict(status='FROZEN_BEFORE_TRAIN',created_utc=utc(),root_plan_sha256=sha(root/'plan.json'),arms=list(ARMS),cpu_auxiliary_budget_s=cpu_budget,gpu_budget_s=gpu_budget,
        network='3->64 ReLU ->64 ReLU ->2; query boundaries never input',mean='Both learned arms base same train logtrue mean; same seeded weights and zero output; affine Gaussian independently uses reused train affine',
        sigma='.03+1.97*sigmoid(raw_sigma + initial_logit); initial affine training residual RMS clipped (.030001,1.999999)',
        normalization='Train paired logDP mean/std; public ray x/y unchanged; geometry normalized depth channel replaced zero',
        training='Both arms same seed7 initialized hidden weights and sample seed7, 600 final Adam .002 steps, batch8192 GaussianNLL, no val selection',
        calibration='All4arms original cal16 fixed27 free-score conservative closest FPR <= target; affine margin cutoff is metres delta=-cutoff',
        source_sha256=sha(__file__),fit_path=str(fitpath),fit_sha256=sha(fitpath),queries=load(sensor/'observations.json')['queries'])
    write(out/'execution_plan.json',plan);write(out/'focused_check.json',focused_checks())
    cpu=0.;cpu_phase=time.perf_counter();gpu=0.;complete=False;failure=None
    def check():
        if cpu+time.perf_counter()-cpu_phase>=cpu_budget:raise TimeoutError('Candidate CPU auxiliary budget reached')
    try:
        x,y,info=train_points(sensor,load(fitpath),check)
        write(out/'training_inputs.json',dict(info,manifest_sha256=sha(sensor/'dataset_manifest.json'),prediction_manifest_sha256=sha(sensor/'depthpro/predictions.json')))
        cpu+=time.perf_counter()-cpu_phase
        receipt=gpu_train_predict(repo,out,sensor,cohorts,x,y,info,gpu_budget);gpu=receipt['gpu_allocation_wall_s'];cpu_phase=time.perf_counter()
        write(out/'training_receipt.json',dict(status='COMPLETE',training_inputs_sha256=sha(out/'training_inputs.json'),gpu_receipt_sha256=sha(out/'gpu_receipt.json'),arms=receipt['arms'],
            train_only=True,cal_supervision='thresholds only',validation_selected_checkpoint=False,checkpoint_steps=600,normalization=info['normalization']))
        cal=calibrate(out,sensor,info,check);(out/'evaluation').mkdir(exist_ok=True)
        results={}
        for name,path in cohorts.items():results[name]=evaluate(name,path,heads[name],out,info,cal,old,check)
        write_csv(out/'summary.csv',[dict(cohort=n,arm=a,**v['summary']['all']) for n,res in results.items() for a,v in res['arms'].items()])
        complete=True
    except Exception as exc:failure=repr(exc);raise
    finally:
        if gpu:cpu+=time.perf_counter()-cpu_phase
        elif (out/'gpu_receipt.json').exists():gpu=load(out/'gpu_receipt.json')['gpu_allocation_wall_s']
        write(out/'completion_receipt.json',dict(status='COMPLETE' if complete else 'PARTIAL',failure=failure,cpu_auxiliary_wall_s=cpu,gpu_allocation_wall_s=gpu,
            total_wall_s=time.perf_counter()-wall,download_bytes=0,depthpro_calls=0,source_sha256=sha(__file__),execution_plan_sha256=sha(out/'execution_plan.json'),resources_released=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cpu-budget-s',type=float,default=400);p.add_argument('--gpu-budget-s',type=float,default=700)
    a=p.parse_args();run(a.repo.resolve(),a.root.resolve(),a.cpu_budget_s,a.gpu_budget_s)
