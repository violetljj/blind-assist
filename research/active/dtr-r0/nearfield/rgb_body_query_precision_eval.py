"""Frozen 16-frame precision comparison, evaluator-only Development.

Public roster and predictions are sealed before reference reads. Native sampling
uses only existing color/depth K. Raw metric diagnosis and unchanged train-affine
query readout are distinct; no new labels, thresholds or model calls.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from rgb_body_query_input_diagnostic import sha
from rgb_body_query_interval_distribution import load, utc
from rgb_body_query_3rscan import color_coordinates, sample_prediction
from rgb_body_query_negative_frozen_score import paths, numeric, summaries
from rgb_body_query_reference_eval import rays, ray_interval
from rgb_body_query_query_calibration_probe import FREE, nth_score, write
from rgb_body_query_scene_diagnostic import write_csv

BANDS=('0.3-0.8m','0.8-1.5m','1.5-3m')
QUANTILES=(0.,.01,.05,.25,.5,.75,.95,.99,1.)
NAMES=('min','q01','q05','q25','q50','q75','q95','q99','max')

def qstats(v):
    return dict(zip(NAMES,np.quantile(v,QUANTILES).tolist())) if len(v) else None

def metric(gt,pred,low,high,band):
    lo,hi=map(float,band.removesuffix('m').split('-'))
    n=len(gt); inside=(pred>=lo-1e-12)&(pred<=hi+1e-12)
    return dict(status='EVALUABLE' if n else 'NOT_EVALUABLE_NO_POSITIVE_NATIVE_POINTS',
        native_point_units=n,low_saturated_units=int(low.sum()),high_saturated_units=int(high.sum()),
        pred_below_gt=int((pred<gt).sum()),pred_equal_gt=int((pred==gt).sum()),pred_above_gt=int((pred>gt).sum()),
        pred_inside_z_band=int(inside.sum()),pred_before_z_band=int((pred<lo-1e-12).sum()),pred_after_z_band=int((pred>hi+1e-12).sum()),
        signed_z_error_m=qstats(pred-gt),log_z_ratio=qstats(np.log(pred/gt)),z_ratio=qstats(pred/gt),gt_z_m=qstats(gt),predicted_z_m=qstats(pred))

def flat_metric(sm):
    measures=('signed_z_error_m','log_z_ratio','z_ratio','gt_z_m','predicted_z_m')
    out={k:v for k,v in sm.items() if k not in measures}
    for measure in measures:
        for name in NAMES:out[measure+'_'+name]=sm[measure][name] if sm[measure] else None
    return out

def intercept_check(inv,depth):
    """CPU arithmetic with actual tensor dtype; allow one reciprocal ULP."""
    inv=np.asarray(inv).squeeze()
    if inv.dtype not in (np.dtype('float16'),np.dtype('float32')):raise ValueError('Unexpected preclamp dtype')
    if inv.shape!=depth.shape or not np.isfinite(inv).all():raise ValueError('Invalid preclamp shape/value')
    dtype=inv.dtype;lo=np.asarray(1e-4,dtype=dtype);hi=np.asarray(1e4,dtype=dtype)
    clamped=np.maximum(lo,np.minimum(hi,inv));expected=(np.float64(1.)/clamped.astype(np.float64)).astype(dtype)
    actual=np.asarray(depth,dtype=np.float32);exp32=expected.astype(np.float32)
    ulp=np.abs(np.spacing(expected)).astype(np.float32)
    error=np.abs(actual-exp32);scaled=np.divide(error,ulp,out=np.zeros_like(error),where=ulp>0)
    if not np.isfinite(actual).all() or not (actual>0).all() or float(scaled.max())>1.:raise ValueError('Intercept reciprocal differs by >1 original dtype ULP')
    return inv,dict(status='PASS',inverse_dtype=str(dtype),lower_actual_dtype=float(lo),upper_actual_dtype=float(hi),
        reciprocal_lower_actual_dtype=float(np.asarray(1./float(lo),dtype=dtype)),reciprocal_upper_actual_dtype=float(np.asarray(1./float(hi),dtype=dtype)),
        max_abs_reciprocal_error=float(error.max()),max_reciprocal_dtype_ULP=float(scaled.max()),
        strict_lower_clipped_pixels=int((inv<lo).sum()),lower_endpoint_pixels=int((inv<=lo).sum()),
        strict_upper_clipped_pixels=int((inv>hi).sum()),upper_endpoint_pixels=int((inv>=hi).sum()),
        pixels=int(inv.size)),inv<=lo,inv>=hi

def run(repo,out,numerics,budget_s,prior_attempt=None):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():raise FileExistsError('Preserve completed/failed evaluation')
    prior=load(prior_attempt/'terminal.json')['wall_s'] if prior_attempt else 0.
    started=time.perf_counter();receipt=dict(status='STARTING',prior_attempt_wall_s=prior)
    inputs=[];checks=Counter()
    def check():
        if time.perf_counter()-started+prior>=budget_s:raise TimeoutError('Cumulative evaluator CPU wall budget')
    def registered(path,expected=None):
        check();h=sha(path)
        if expected is not None and h!=expected:raise ValueError(f'Input SHA differs: {path}')
        inputs.append(dict(path=str(path),sha256=h));return Path(path)
    try:
        rosterpath=numerics/'roster.json';roster=load(registered(rosterpath));rows=roster['rows'];assert len(rows)==16 and len({(r['scan'],r['frame']) for r in rows})==16
        manifests={};predictions={};statuses={}
        for precision in ('fp16','fp32'):
            terminal=load(registered(numerics/precision/'terminal.json'));status=terminal['status'];statuses[precision]=status
            if status in ('RUNNING','STARTING'):raise ValueError('Prediction arm is not frozen')
            if (numerics/precision/'writer.json').exists():raise ValueError('Prediction writer retained')
            manifests[precision]=load(registered(numerics/precision/'predictions.json'))
            assert terminal['roster_sha256']==sha(rosterpath) and terminal['weight_sha256']=='3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce'
            assert terminal['provided_focal'] and not terminal['evaluator_read'] and terminal['clamp_min']==1e-4 and terminal['clamp_max']==1e4
            predictions[precision]={(r['scan'],r['frame']):r for r in manifests[precision]['rows']}
        allids={(r['scan'],r['frame']) for r in rows};assert all(set(v)<=allids for v in predictions.values())
        matched=[r for r in rows if all((r['scan'],r['frame']) in predictions[p] for p in predictions)]
        write(out/'plan.json',dict(frozen_utc=utc(),source_sha256=sha(__file__),roster_sha256=sha(rosterpath),
            numerics_statuses=statuses,roster_frames=16,matched_frames=len(matched),parent_plan_sha256=sha(out.parent/'plan.json'),
            input_roles='Prediction roster frozen before evaluator reads; actual completed matched identities, no replacement',
            mapping='Public color_K/depth_K bilinear remap only; no GT projection or hole filling',
            saturation='Native remapped low/high endpoint indicator has positive interpolation mass; independent of GT; raw color counts also saved',
            geometry='Native POSITIVE>=16 query known-positive union per distance band; common prediction availability; no UNKNOWN changes',
            query='Unchanged train-affine only, frozen global588/band5 absolute and normalized affine cutoffs; raw has no inherited calibrated threshold',
            prior_attempt_wall_s=prior,budget_s=budget_s,training=0,GPU=0,downloads=0))
        shutil.copyfile(__file__,out/'executed_precision_eval.py')
        if not matched:
            write(out/'results.json',dict(status='NOT_EVALUABLE_NO_MATCHED_PRECISIONS',matched_frames=0,numerics_statuses=statuses));receipt['status']='NOT_EVALUABLE';return
        work,oldroot,candidate,folders=paths(repo);calroot=work/'rgb-body-query-arkit-cal-dev-20261010';info=load(registered(candidate/'training_inputs.json'))
        refs={};fixed=None
        for cohort,folder in list(folders.items())+[('cal_arkit',calroot/'reference')]:
            m=load(registered(folder/'dataset_manifest.json'))
            if fixed is None:fixed=m['queries']
            assert m['queries']==fixed
            for r in m['rows']:refs[r['scan'],r['frame']]=(cohort,r)
        cutoffsets={}
        for readout in ('absolute','normalized'):
            cutoffsets[readout,'global588']=load(registered(calroot/f'score/{readout}/thresholds.json'))['arms']['affine']['cutoff']
            cutoffsets[readout,'band5']=load(registered(work/f'rgb-body-query-distance-cal-dev-20261010/score/{readout}/thresholds.json'))['arms']['affine']
        (out/'native-values').mkdir();metric_rows=[];point_rows=[];primitive_rows=[];scoredata=defaultdict(list);parts=defaultdict(list);paired_native=[]
        for selected in matched:
            check();sid=selected['scan'],selected['frame'];cohort,ref=refs[sid];assert selected['rgb_sha256']==ref['rgb_sha256'] and selected['color_K']==ref['color_K'] and selected['depth_K']==ref['depth_K']
            registered(ref['reference_path'],ref['reference_sha256'])
            with np.load(ref['reference_path']) as f:gt=f['depth'];labels=f['labels'];observed=f['observed'];k=f['depth_K'];ck=f['color_K']
            assert np.array_equal(k,ref['depth_K']) and np.array_equal(ck,ref['color_K']) and labels.shape==(27,*gt.shape)
            assert list(gt.shape)==selected['depth_shape'] and observed.shape==gt.shape
            mx,my=color_coordinates(np.array(selected['depth_K']),np.array(selected['color_K']),selected['depth_shape'])
            native={};sat={}
            for precision in ('fp16','fp32'):
                p=predictions[precision][sid];registered(p['path'],p['sha256'])
                assert p['rgb_sha256']==selected['rgb_sha256']
                with np.load(p['path']) as f:
                    full=f['depth'];inv=f['inverse_preclamp'];rawdepth=f['depth_raw'];focal=np.asarray(f['focallength_px'])
                assert list(full.shape)==selected['color_shape'] and rawdepth.dtype==inv.dtype
                assert np.array_equal(rawdepth.astype(np.float32),full)
                assert focal.size==1 and float(focal.ravel()[0])==float(np.asarray(selected['color_K'][0][0],dtype=focal.dtype))
                inv,ic,low,high=intercept_check(inv,full);primitive_rows.append(dict(cohort=cohort,environment=ref['environment'],scan=sid[0],frame=sid[1],precision=precision,**ic));checks['original_dtype_intercept_reciprocal_frames']+=1
                dp=sample_prediction(full,mx,my);valid=np.isfinite(dp)&(dp>0);affine=np.full(dp.shape,np.nan);affine[valid]=np.exp(info['affine']['a']*np.log(dp[valid].astype(float))+info['affine']['b'])
                native[precision]=dict(raw=dp,affine=affine,valid=valid)
                sat[precision]={key:cv2.remap(values.astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT) for key,values in [('low',low),('high',high)]}
            common=native['fp16']['valid']&native['fp32']['valid'];rx,ry=rays(k,gt.shape)
            for band in BANDS:
                union=np.zeros(gt.shape,bool)
                for j,q in enumerate(fixed):
                    if f'{q["low"][2]:g}-{q["high"][2]:g}m'==band and ref['queries'][j]['state']=='POSITIVE':union|=labels[j]==1
                assert (np.isfinite(gt[union])&(gt[union]>0)&observed[union]).all();mask=union&common;indices=np.flatnonzero(mask);gv=gt[mask].astype(float)
                payload=dict(native_flat_indices=indices,gt_unique=gv)
                for precision in ('fp16','fp32'):
                    payload[precision+'_low_mass']=sat[precision]['low'][mask];payload[precision+'_high_mass']=sat[precision]['high'][mask]
                    low=sat[precision]['low'][mask]>0;high=sat[precision]['high'][mask]>0
                    for model in ('raw','affine'):
                        pv=native[precision][model][mask].astype(float);payload[precision+'_'+model]=pv
                        for subset,ss in [('all',np.ones(len(gv),bool)),('low_saturated',low),('high_saturated',high),('unclamped',~(low|high))]:
                            sm=metric(gv[ss],pv[ss],low[ss],high[ss],band)
                            for group in (cohort,'all_selected_matched'):
                                parts[group,precision,model,band,subset].append((gv[ss],pv[ss],low[ss],high[ss]))
                            metric_rows.append(dict(cohort=cohort,environment=ref['environment'],scan=sid[0],frame=sid[1],precision=precision,model=model,band=band,subset=subset,**flat_metric(sm)))
                path=out/'native-values'/f'{sid[0]}_{sid[1]:06d}_{band}.npz';np.savez_compressed(path,**payload)
                point_rows.append(dict(cohort=cohort,environment=ref['environment'],scan=sid[0],frame=sid[1],band=band,reference_union_pixels=int(union.sum()),common_available_points=len(gv),unavailable_prediction_pixels=int((union&~common).sum()),path=str(path),sha256=sha(path)))
                for model in ('raw','affine'):
                    p16,p32=payload['fp16_'+model],payload['fp32_'+model];inside16=(p16>=float(band.split('-')[0])-1e-12)&(p16<=float(band.removesuffix('m').split('-')[1])+1e-12);inside32=(p32>=float(band.split('-')[0])-1e-12)&(p32<=float(band.removesuffix('m').split('-')[1])+1e-12)
                    paired_native.append(dict(cohort=cohort,scan=sid[0],frame=sid[1],band=band,model=model,common_native_points=len(gv),low_endpoint_removed=int(((payload['fp16_low_mass']>0)&(payload['fp32_low_mass']==0)).sum()),low_endpoint_added=int(((payload['fp32_low_mass']>0)&(payload['fp16_low_mass']==0)).sum()),inside_z_band_rescue=int((inside32&~inside16).sum()),inside_z_band_loss=int((inside16&~inside32).sum()),depth_delta_quantiles=qstats(p32-p16),log_depth_delta_quantiles=qstats(np.log(p32/p16))))
            for j,q in enumerate(fixed):
                entry,exit,domain=ray_interval(rx,ry,q);width=exit-entry;band=f'{q["low"][2]:g}-{q["high"][2]:g}m';state=ref['queries'][j]['state'];lab=labels[j]
                expected='POSITIVE' if (lab==1).sum()>=16 else FREE if (lab==1).sum()==0 and (lab==2).sum()==0 and (lab==0).sum()>=16 else 'UNKNOWN';assert expected==state
                for precision in ('fp16','fp32'):
                    z=native[precision]['affine'];valid=native[precision]['valid']
                    for readout in ('absolute','normalized'):
                        mask=valid&domain&(width>1e-12 if readout=='normalized' else True);margin=np.full(gt.shape,-np.inf);margin[mask]=np.minimum(z[mask]-entry[mask],exit[mask]-z[mask]);score=margin if readout=='absolute' else np.divide(2*margin,width,out=np.full(gt.shape,-np.inf),where=mask)
                        scoredata[precision,readout].append(dict(cohort=cohort,environment=ref['environment'],scan=sid[0],frame=sid[1],query=q['name'],distance_band=band,reference_state=state,valid_ray_count=int(mask.sum()),query_score=nth_score(score),known_positive_score=nth_score(score[lab==1])))
            checks['matched_reference_frames']=checks['matched_reference_frames']+1
        metric_results={}
        for key,pp in parts.items():
            check();g,precision,model,band,subset=key;gv,pv,lo,hi=[np.concatenate([p[i] for p in pp]) for i in range(4)];metric_results['/'.join(key)]=metric(gv,pv,lo,hi,band)
        decisions={};query_pairs=[];query_summary={};pair_summary=defaultdict(Counter)
        for (precision,readout),rr in scoredata.items():
            write(out/f'{precision}_{readout}_query_scores.json',rr)
            for mode in ('global588','band5'):
                evaluated=[]
                for r in rr:
                    c=cutoffsets[readout,mode] if mode=='global588' else cutoffsets[readout,mode][r['distance_band']]['cutoff']
                    evaluated.append(dict(r,precision=precision,readout=readout,mode=mode,cutoff=c,predicted_positive=c is not None and r['query_score']>=c,positive_known_witness=c is not None and r['known_positive_score']>=c))
                decisions[precision,readout,mode]=evaluated;query_summary[f'{precision}/{readout}/{mode}']=summaries(evaluated);write(out/f'{precision}_{readout}_{mode}_evaluation.json',evaluated)
        for readout in ('absolute','normalized'):
            for mode in ('global588','band5'):
                rr=decisions['fp32',readout,mode];bb=decisions['fp16',readout,mode]
                for r,b in zip(rr,bb,strict=True):
                    assert (r['scan'],r['frame'],r['query'],r['reference_state'])==(b['scan'],b['frame'],b['query'],b['reference_state']);pos=r['reference_state']=='POSITIVE';free=r['reference_state']==FREE
                    values=dict(positive_rescue=pos and r['positive_known_witness'] and not b['positive_known_witness'],positive_loss=pos and b['positive_known_witness'] and not r['positive_known_witness'],free_removed=free and b['predicted_positive'] and not r['predicted_positive'],free_added=free and r['predicted_positive'] and not b['predicted_positive'])
                    p=dict(cohort=r['cohort'],environment=r['environment'],scan=r['scan'],frame=r['frame'],query=r['query'],distance_band=r['distance_band'],reference_state=r['reference_state'],readout=readout,mode=mode,cutoff=r['cutoff'],fp16_support=b['predicted_positive'],fp32_support=r['predicted_positive'],fp16_known_witness=b['positive_known_witness'],fp32_known_witness=r['positive_known_witness'],**values);query_pairs.append(p)
                    for group in ('all','cohort/'+r['cohort'],'environment/'+r['environment'],'band/'+r['distance_band'],'cohort_band/'+r['cohort']+'/'+r['distance_band']):pair_summary[f'{readout}/{mode}/{group}'].update(values)
        write_csv(out/'intercept_checks.csv',primitive_rows);write_csv(out/'point_files.csv',point_rows);write_csv(out/'metric_frame_rows.csv',metric_rows);write_csv(out/'query_pairs.csv',query_pairs)
        write(out/'native_paired.json',paired_native);write(out/'metric_results.json',metric_results);write(out/'query_results.json',dict(summaries=query_summary,pairs={k:dict(v) for k,v in pair_summary.items()}))
        write(out/'inputs.json',inputs);write(out/'results.json',dict(status='COMPLETE' if len(matched)==16 else 'PARTIAL_MATCHED_PRECISION_DESCRIPTION',matched_frames=len(matched),roster_frames=16,numerics_statuses=statuses,
            metric_groups=len(metric_results),query_pairs=len(query_pairs),affine=info['affine'],cutoffs=dict(global588={r:cutoffsets[r,'global588'] for r in ('absolute','normalized')},band5={r:cutoffsets[r,'band5'] for r in ('absolute','normalized')}),
            limits='Selected16 cached Development diagnostic; cal positives descriptive; low/high endpoint interpolation masks not GT repairs; native UNKNOWN untouched; no new calibration or raw calibrated working point'))
        receipt.update(status='COMPLETE',matched_frames=len(matched),checks=dict(checks),metric_groups=len(metric_results),query_pairs=len(query_pairs))
    except Exception as exc:
        receipt.update(status='FAILED_PARTIAL',error=repr(exc));raise
    finally:
        receipt.update(wall_s=time.perf_counter()-started,source_sha256=sha(__file__),GPU=0,training=0,downloads=0);write(out/'terminal.json',receipt);print(receipt,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--numerics',type=Path,required=True);p.add_argument('--budget-s',type=float,default=180.);p.add_argument('--prior-attempt',type=Path)
    a=p.parse_args();run(a.repo.resolve(),a.output.resolve(),a.numerics.resolve(),a.budget_s,a.prior_attempt)
