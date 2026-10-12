"""Four frozen residual readout/calibration combinations, Development only.

No forward/training. Absolute/global preserves prior query scores. Bands are
public query bounds, never GT distance. Missing calibration remains N/E.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import time
import numpy as np
from rgb_body_query_interval_distribution import load, utc
from rgb_body_query_input_diagnostic import sha
from rgb_body_query_bounded_evaluate import ARMS, manifest_rows, TARGET_COST
from rgb_body_query_negative_frozen_score import paths, EVAL, numeric
from rgb_body_query_query_calibration_probe import FREE, COHORTS, nth_score, select_cutoff, write
from rgb_body_query_reference_eval import rays, ray_interval
from rgb_body_query_bounded_residual_proxy import cutoff_at_cost
from rgb_body_query_scene_diagnostic import write_csv

BANDS=('0.3-0.8m','0.8-1.5m','1.5-3m')
COMBINATIONS=('absolute-global','absolute-band','normalized-global','normalized-band')


def summary(rows,pairs=False):
    out={}
    for title,keys in [('all',[]),('band',['distance_band']),('environment',['environment']),
                       ('environment_band',['environment','distance_band'])]:
        groups=defaultdict(list)
        for r in rows:groups[tuple(r[k] for k in keys)].append(r)
        for key,rr in groups.items():
            available=[r for r in rr if r['paired_calibrated' if pairs else 'calibrated']]
            if pairs:
                counts=dict(query_pairs=len(rr),paired_calibrated=len(available),paired_uncalibrated=len(rr)-len(available),
                    **{k:sum(r[k] for r in available) if available else None
                       for k in ('positive_rescue','positive_loss','free_removed','free_added')})
            else:
                counts=dict(query_total=len(rr),frames=len({(r['scan'],r['frame']) for r in rr}),
                    calibrated_queries=len(available),uncalibrated_queries=len(rr)-len(available))
                for state,prefix in [('POSITIVE','positive'),(FREE,'free'),('UNKNOWN','unknown')]:
                    total=[r for r in rr if r['reference_state']==state]
                    selected=[r for r in available if r['reference_state']==state]
                    counts.update({prefix+'_total':len(total),prefix+'_calibrated_total':len(selected),
                        prefix+'_uncalibrated_total':len(total)-len(selected),
                        prefix+'_support':sum(r['predicted_positive'] for r in selected) if selected or not total else None})
                    if state=='POSITIVE':counts['positive_known_witness']=sum(r['positive_known_witness'] for r in selected) if selected or not total else None
                counts['free_support_rate']=counts['free_support']/counts['free_calibrated_total'] if counts['free_calibrated_total'] else None
            out['/'.join([title]+list(key))]=counts
    return out


def evaluate(rows,choices,band):
    ev=[]
    for r in rows:
        choice=choices[r['distance_band']] if band else choices
        cutoff=choice['cutoff'];available=cutoff is not None
        ev.append(dict(r,cutoff=cutoff,calibrated=available,
            predicted_positive=r['query_score']>=cutoff if available else None,
            positive_known_witness=r['known_positive_score']>=cutoff if available else None))
    return ev


def paired(rows,baseline,cohort,arm,baseline_arm,mode):
    pairs=[]
    for r,b in zip(rows,baseline,strict=True):
        assert all(r[k]==b[k] for k in ('scan','frame','query','reference_state','distance_band'))
        available=r['calibrated'] and b.get('calibrated',b['cutoff'] is not None)
        pos=r['reference_state']=='POSITIVE';free=r['reference_state']==FREE
        pairs.append(dict(cohort=cohort,arm=arm,baseline_arm=baseline_arm,mode=mode,scan=r['scan'],frame=r['frame'],
            environment=r['environment'],query=r['query'],distance_band=r['distance_band'],reference_state=r['reference_state'],
            paired_calibrated=available,candidate_calibrated=r['calibrated'],baseline_calibrated=b.get('calibrated',True),
            candidate_cutoff=r['cutoff'],baseline_cutoff=b['cutoff'],candidate_support=r['predicted_positive'],baseline_support=b['predicted_positive'],
            candidate_known_witness=r['positive_known_witness'],baseline_known_witness=b['positive_known_witness'],
            positive_rescue=bool(pos and r['positive_known_witness'] and not b['positive_known_witness']) if available else None,
            positive_loss=bool(pos and b['positive_known_witness'] and not r['positive_known_witness']) if available else None,
            free_removed=bool(free and b['predicted_positive'] and not r['predicted_positive']) if available else None,
            free_added=bool(free and r['predicted_positive'] and not b['predicted_positive']) if available else None))
    return pairs


def select(rows,band):
    free=[r for r in rows if r['reference_state']==FREE]
    if band:return {b:select_cutoff([r['query_score'] for r in free if r['distance_band']==b]) for b in BANDS}
    return select_cutoff([r['query_score'] for r in free])


def normalized(name,manifest,pm,info,check,inputs,checks,widths):
    refs={(r['scan'],r['frame']):r for r in manifest['rows']};data={a:[] for a in ARMS}
    assert len(pm['rows'])==len(refs)
    for row in pm['rows']:
        check();ref=refs[row['scan'],row['frame']]
        for p,digest in [(row['sampled_depth_path'],row['sampled_depth_sha256']),(ref['reference_path'],ref['reference_sha256'])]:
            assert sha(p)==digest;inputs.append(dict(path=str(p),sha256=digest))
        with np.load(row['sampled_depth_path']) as f:dp=f['depth']
        with np.load(ref['reference_path']) as f:labels=f['labels']
        valid=np.isfinite(dp)&(dp>0);loga=np.zeros(dp.shape,np.float64)
        loga[valid]=info['affine']['a']*np.log(dp[valid].astype(np.float64))+info['affine']['b']
        depths={'affine':np.exp(loga)}
        for a in ARMS[1:]:
            p=row['distributions'][a];assert sha(p['path'])==p['sha256'];inputs.append(dict(path=p['path'],sha256=p['sha256']))
            with np.load(p['path']) as f:mu,mask=f['mu'],f['valid']
            assert np.array_equal(valid,mask) and np.isfinite(mu[valid]).all()
            depths[a]=np.exp(mu);checks['frozen_trained_masks']+=1
        rx,ry=rays(row['depth_K'],row['depth_shape'])
        for j,q in enumerate(manifest['queries']):
            check();entry,exit,domain=ray_interval(rx,ry,q);width=exit-entry
            oldmask=valid&domain;mask=oldmask&(width>1e-12);lab=labels[j]
            pos,free,unk=[int((lab==n).sum()) for n in (1,0,2)]
            state='POSITIVE' if pos>=16 else FREE if pos==0 and unk==0 and free>=16 else 'UNKNOWN'
            assert state==ref['queries'][j]['state'];checks['unchanged_reference_states']+=1
            widths.append(dict(cohort=name,scan=row['scan'],frame=row['frame'],query=q['name'],
                zero_width_domain=int((domain&(width==0)).sum()),small_or_zero_width_domain=int((domain&(width<=1e-12)).sum()),
                old_valid_domain=int(oldmask.sum()),normalized_valid_domain=int(mask.sum()),
                query_availability_changed=(int(oldmask.sum())>=16)!=(int(mask.sum())>=16)))
            assert not (domain&(width<0)).any()
            for a,z in depths.items():
                margin=np.minimum(z[mask]-entry[mask],exit[mask]-z[mask]);s=2*margin/width[mask]
                assert np.isfinite(s).all() and (s<=1+2e-14).all()
                assert np.array_equal(s>=0,margin>=0) and np.array_equal(np.sign(s),np.sign(margin))
                checks['max_bound_zero_sign_queries']+=1;checks['zero_sign_pixels']+=len(s)
                grid=np.full(dp.shape,-np.inf);grid[mask]=s
                data[a].append(dict(cohort=name,environment=ref['environment'],scan=row['scan'],frame=row['frame'],query=q['name'],
                    distance_band=f'{q["low"][2]:g}-{q["high"][2]:g}m',reference_state=state,valid_ray_count=len(s),
                    query_score=nth_score(grid),known_positive_score=nth_score(grid[lab==1])))
    return data


def save(out,thresholds,evaluation,comparisons):
    out.mkdir(parents=True,exist_ok=True);cohorts={};rows=[];pairs=[];summaries={}
    for c,arms in evaluation.items():
        write(out/f'{c}_evaluation.json',arms);cohorts[c]={a:summary(rr) for a,rr in arms.items()}
        for a,ss in cohorts[c].items():
            for g,s in ss.items():rows.append(dict(cohort=c,arm=a,group=g,**s))
        for a,baseline,baseline_arm,mode in comparisons[c]:
            pp=paired(arms[a],baseline,c,a,baseline_arm,mode);pairs+=pp
            summaries[f'{c}/{a}/{mode}']=summary(pp,pairs=True)
    write(out/'results.json',dict(status='COMPLETE',thresholds=thresholds,cohorts=cohorts,pairs=summaries,
        unavailable='N/E cutoff=None; unavailable queries keep denominators, support/witness=None; pairs only common calibrated set',
        limits='Consumed Development; public query bands only; related frames; sampled FREE not volume clearance'))
    write_csv(out/'summary.csv',rows);write_csv(out/'query_pairs.csv',pairs)


def calibrate(repo,oldroot,out,info,check,inputs,checks,widths):
    manifest,sourcepaths=manifest_rows(repo);ppath=oldroot/'predictions/cal/predictions.json'
    inputs.extend(dict(path=str(p),sha256=sha(p)) for p in sourcepaths+[ppath])
    norm=normalized('cal_fit_description',manifest,load(ppath),info,check,inputs,checks,widths)
    apath=oldroot/'evaluation/calibration/scores.json';inputs.append(dict(path=str(apath),sha256=sha(apath)))
    absolute={a:numeric(rr) for a,rr in load(apath)['arms'].items()}
    assert all(len(rr)==8208 for rr in norm.values())
    data={'absolute':absolute,'normalized':norm};choices={}
    for combo in COMBINATIONS:
        readout,cal=combo.split('-');choices[combo]={a:select(rr,cal=='band') for a,rr in data[readout].items()}
    # All pooled combinations freeze before any new evaluation references/scores.
    write(out/'thresholds.json',dict(status='FROZEN_BEFORE_EVALUATION',frozen_utc=utc(),combinations=choices,
        calibration_frames=304,strict_free_queries=588,band_FREE_counts=dict(zip(BANDS,(395,180,13))),
        band_allowed_costs=dict(zip(BANDS,(19,9,0))),selection='strict FREE only; each arm/readout public band; <=5%; ties retained',evaluation_truth_read=False))
    for readout,ss in data.items():write(out/f'{readout}_scores.json',dict(arms=ss))
    previous=load(oldroot/'evaluation/calibration/thresholds.json')['arms']
    assert choices['absolute-global']==previous;checks['absolute_global_cal_cutoffs_exact']+=3
    for combo in COMBINATIONS:
        readout,cal=combo.split('-');ev={a:evaluate(rr,choices[combo][a],cal=='band') for a,rr in data[readout].items()}
        oldaff=evaluate(absolute['affine'],choices['absolute-global']['affine'],False)
        comparisons=[(a,ev['affine'],'same_combo_affine','vs_same_combo_affine') for a in ARMS[1:]]
        comparisons.append(('trained_context',ev['trained_depth_ray'],'same_combo_depth_ray','context_vs_depth_ray'))
        comparisons.extend((a,oldaff,'absolute_global_strong_affine','vs_absolute_global_strong_affine') for a in ARMS)
        comparisons.extend((a,evaluate(absolute[a],choices['absolute-global'][a],False),
                            f'absolute_global_same_arm_{a}','vs_absolute_global_same_arm') for a in ARMS)
        save(out/combo,choices[combo],{'cal_fit_description':ev},{'cal_fit_description':comparisons})


def execute_final(repo,oldroot,out,calibration,info,check,inputs,checks,widths):
    assert load(calibration/'terminal.json')['status']=='COMPLETE'
    pooled=load(calibration/'thresholds.json')['combinations']
    # Add same-arm cal contrasts as a derived artifact if the first calibration
    # snapshot predates this reporting addition; never replace frozen outputs.
    cal_cross=[];cal_cross_summary={}
    baseline_path=calibration/'absolute-global/cal_fit_description_evaluation.json'
    calbaseline=load(baseline_path);inputs.append(dict(path=str(baseline_path),sha256=sha(baseline_path)))
    for combo in COMBINATIONS:
        p=calibration/combo/'cal_fit_description_evaluation.json';ev=load(p)
        inputs.append(dict(path=str(p),sha256=sha(p)))
        for a in ARMS:
            pp=paired(ev[a],calbaseline[a],'cal_fit_description',a,f'absolute_global_same_arm_{a}',f'{combo}_vs_absolute_global_same_arm')
            cal_cross+=pp;cal_cross_summary[f'{combo}/{a}']=summary(pp,pairs=True)
    write_csv(out/'cal_derived_samearm_pairs.csv',cal_cross)
    write(out/'cal_derived_samearm_pair_summary.json',cal_cross_summary)
    data={'absolute':{},'normalized':{}};_,_,_,cohorts=paths(repo)
    for name,folder in cohorts.items():
        check();mpath=folder/'dataset_manifest.json';ppath=oldroot/f'predictions/{name}/predictions.json'
        apath=oldroot/f'evaluation/final/{name}_scores.json'
        inputs.extend(dict(path=str(p),sha256=sha(p)) for p in [mpath,ppath,apath])
        data['absolute'][name]={a:numeric(rr) for a,rr in load(apath)['arms'].items()}
        data['normalized'][name]=normalized(name,load(mpath),load(ppath),info,check,inputs,checks,widths)
        write(out/f'{name}_normalized_scores.json',dict(arms=data['normalized'][name]))
    mainchoices={};mainev={};supp={}
    for combo in COMBINATIONS:
        readout,cal=combo.split('-');band=cal=='band';mainchoices[combo]={};mainev[combo]={}
        supp[combo]={c:{a:evaluate(rr,pooled[combo][a],band) for a,rr in ss.items()} for c,ss in data[readout].items()}
        for held in COHORTS:
            choices={a:select([r for c in COHORTS if c!=held for r in data[readout][c][a]],band) for a in ARMS}
            mainchoices[combo][held]=choices
            write(out/f'{combo}_{held}_thresholds.json',dict(status='FROZEN_BEFORE_HELD_EVALUATION',held=held,
                other_captures=[c for c in COHORTS if c!=held],arms=choices))
            mainev[combo][held]={a:evaluate(rr,choices[a],band) for a,rr in data[readout][held].items()}
            if combo=='absolute-global':
                prior=load(oldroot/f'evaluation/final/{held}_main_LOCO_thresholds.json')['arms']
                assert choices==prior;checks['absolute_global_LOCO_cutoffs_exact']+=3
    for combo in COMBINATIONS:
        for protocol,ev,choices,strong in [('main-loco',mainev[combo],mainchoices[combo],mainev['absolute-global']),
                                           ('pooled-supplemental',supp[combo],pooled[combo],supp['absolute-global'])]:
            comparisons={}
            for c,arms in ev.items():
                comparisons[c]=[(a,arms['affine'],'same_combo_affine','vs_same_combo_affine') for a in ARMS[1:]]
                comparisons[c].append(('trained_context',arms['trained_depth_ray'],'same_combo_depth_ray','context_vs_depth_ray'))
                comparisons[c].extend((a,strong[c]['affine'],'absolute_global_strong_affine','vs_absolute_global_strong_affine') for a in ARMS)
                comparisons[c].extend((a,strong[c][a],f'absolute_global_same_arm_{a}','vs_absolute_global_same_arm') for a in ARMS)
            save(out/combo/protocol,choices,ev,comparisons)
    curves=[];points=[]
    for combo in COMBINATIONS:
        readout,cal=combo.split('-');band=cal=='band';post={};choices={};comparisons={}
        for held in COHORTS:
            check();post[held]={};choices[held]={};baseline=mainev['absolute-global'][held]['affine'];comparisons[held]=[]
            for a in ARMS:
                ss=data[readout][held][a]
                if not band:
                    free=[r['query_score'] for r in ss if r['reference_state']==FREE]
                    cutoff=cutoff_at_cost(free,TARGET_COST[held]);choice=dict(cutoff=cutoff)
                    ev=evaluate(ss,choice,False);choices[held][a]=choice
                    for cost in range(len(free)+1):
                        check();cut=cutoff_at_cost(free,cost);sm=summary(evaluate(ss,dict(cutoff=cut),False))['all']
                        assert sm['free_support']<=cost
                        curves.append(dict(combination=combo,cohort=held,arm=a,negative_count_budget=cost,cutoff=cut,
                            actual_free_support=sm['free_support'],free_total=sm['free_total'],positive_known_witness=sm['positive_known_witness'],positive_total=sm['positive_total']))
                else:
                    selected={}
                    for b in BANDS:
                        free=[r['query_score'] for r in ss if r['reference_state']==FREE and r['distance_band']==b]
                        target=sum(r['predicted_positive'] for r in baseline if r['reference_state']==FREE and r['distance_band']==b)
                        main_available=mainchoices[combo][held][a][b]['cutoff'] is not None
                        selected[b]=dict(status='POSTHOC_HELD_FREE_ONLY' if free and main_available else 'NOT_EVALUABLE',
                            cutoff=cutoff_at_cost(free,target) if free and main_available else None,held_free_queries=len(free),requested_cost=target)
                    ev=evaluate(ss,selected,True);choices[held][a]=selected
                post[held][a]=ev;points.append(dict(combination=combo,cohort=held,arm=a,**summary(ev)['all']))
                comparisons[held].append((a,baseline,'fixed_original_absolute_affine','posthoc_vs_fixed_original_affine'))
            if not band:
                comparisons[held].extend((a,post[held]['affine'],f'{combo}_curve_plateau_affine','posthoc_vs_same_combo_curve_affine') for a in ARMS)
                absrows=data['absolute'][held]['affine']
                absfree=[r['query_score'] for r in absrows if r['reference_state']==FREE]
                absplateau=evaluate(absrows,dict(cutoff=cutoff_at_cost(absfree,TARGET_COST[held])),False)
                comparisons[held].extend((a,absplateau,'absolute_global_curve_plateau_affine',
                                          'posthoc_vs_absolute_global_curve_affine') for a in ARMS)
        save(out/combo/'posthoc',choices,post,comparisons)
    write_csv(out/'posthoc_curves.csv',curves);write_csv(out/'posthoc_points.csv',points)
    write(out/'results.json',dict(status='COMPLETE',combinations=COMBINATIONS,main='Three original captures LOCO',
        supplemental='304 pooled FREE before eval',global_posthoc='held FREE fixed old absolute affine5/12/10 and full cost curves',
        band_posthoc='Fixed original absolute affine actual per-band costs; missing held/cal FREE is N/E, no cross-band allocation; total exact match not guaranteed',
        missing_calibration='cutoff None; explicit calibrated and uncalibrated denominators; no fallback',training=0,gpu=0,new_forward=0))


def run(repo,runroot,out,stage,budget,calibration=None,prior_attempt=None):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():raise FileExistsError('Preserve existing stage payload')
    prior=0.
    if prior_attempt:
        previous=load(prior_attempt/'terminal.json');assert previous['status']=='FAILED_PARTIAL' and previous['stage']==stage
        prior=previous['stage_wall_s']+previous.get('prior_attempt_wall_s',0.);budget-=prior;assert budget>0
    parent=runroot/'plan.json';assert parent.exists();parent_info=load(parent)
    oldroot=repo/'artifacts.local/work/rgb-body-query-bounded-residual-dev-20261010'
    assert load(oldroot/'train_terminal.json')['completed_steps']==dict(trained_depth_ray=600,trained_context=600)
    write(out/'plan.json',dict(stage=stage,frozen_utc=utc(),parent_plan_sha256=sha(parent),source_sha256=sha(__file__),
        frozen_model_sha256={a:sha(oldroot/f'{a}_step600.pt') for a in ARMS[1:]},combinations=COMBINATIONS,
        budget_cpu_wall_s_remaining=budget,prior_attempt_wall_s=prior,training=0,gpu=0,new_forward=0,downloads=0,
        norm='2*min(z-entry,exit-z)/(exit-entry), frozen valid & public domain & width>1e-12; other rays -Infinity',
        public_bands=BANDS,missing_calibration='N/E None, not zero or fallback',all_arms_all_captures=True))
    shutil.copyfile(__file__,out/'executed_residual_readout.py');start=time.perf_counter();receipt=dict(status='STARTING',stage=stage,prior_attempt_wall_s=prior)
    inputs=[];checks=Counter();widths=[]
    def check():
        if time.perf_counter()-start>=budget:raise TimeoutError('Readout cumulative stage CPU budget reached')
    try:
        assert [c.replace('_','-') for c in parent_info['combinations']]==list(COMBINATIONS)
        assert parent_info['frozen_arms']==list(ARMS)
        assert sha(oldroot/'trained_depth_ray_step600.pt')==parent_info['models']['depth_ray_step600_sha256']
        assert sha(oldroot/'trained_context_step600.pt')==parent_info['models']['context_step600_sha256']
        ipath=oldroot/'training_inputs.json';info=load(ipath);inputs.extend(dict(path=str(p),sha256=sha(p)) for p in [parent,ipath])
        if stage=='calibrate':calibrate(repo,oldroot,out,info,check,inputs,checks,widths)
        else:
            assert calibration is not None
            assert load(calibration/'plan.json')['parent_plan_sha256']==sha(parent)
            execute_final(repo,oldroot,out,calibration,info,check,inputs,checks,widths)
        write(out/'inputs.json',inputs);write(out/'checks.json',dict(checks));write_csv(out/'width_diagnostics.csv',widths)
        receipt.update(status='COMPLETE',checks=dict(checks),widths=dict(query_rows=len(widths),
            small_or_zero_width_domain=sum(r['small_or_zero_width_domain'] for r in widths),
            query_availability_changed=sum(r['query_availability_changed'] for r in widths)))
    except Exception as exc:receipt.update(status='FAILED_PARTIAL',error=repr(exc));raise
    finally:
        receipt.update(stage_wall_s=time.perf_counter()-start,source_sha256=sha(__file__),completed_utc=utc(),training=0,gpu=0,new_forward=0,downloads=0)
        write(out/'terminal.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--runroot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=('calibrate','final'),required=True)
    p.add_argument('--budget-s',type=float,required=True);p.add_argument('--calibration',type=Path);p.add_argument('--prior-attempt',type=Path);a=p.parse_args()
    run(a.repo.resolve(),a.runroot.resolve(),a.output.resolve(),a.stage,a.budget_s,a.calibration,a.prior_attempt)
