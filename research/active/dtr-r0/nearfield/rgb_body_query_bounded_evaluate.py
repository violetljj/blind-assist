"""Frozen bounded-residual query scoring; no training or model selection.

304-frame pooled FREE calibration freezes before evaluation truth is read.
Three original ARKit leave-one-capture-out folds remain the main protocol;
held-capture FREE same-cost points/curves are posthoc descriptions only.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import shutil
import time
import numpy as np
from rgb_body_query_interval_distribution import load, utc
from rgb_body_query_input_diagnostic import sha
from rgb_body_query_negative_frozen_score import paths, EVAL, summaries, compare, numeric
from rgb_body_query_arkit_cal_frozen import pair_groups
from rgb_body_query_query_calibration_probe import FREE, COHORTS, nth_score, select_cutoff, write
from rgb_body_query_reference_eval import rays, ray_interval
from rgb_body_query_bounded_residual_proxy import evaluate, cutoff_at_cost
from rgb_body_query_scene_diagnostic import write_csv

ARMS=('affine','trained_depth_ray','trained_context')
MODEL_KEYS={'trained_depth_ray':'trained_depth_ray','trained_context':'trained_context'}
TARGET_COST={'arkit16':5,'arkit_40777060':12,'arkit_40777065':10}


def manifest_rows(repo):
    work,root,_,_=paths(repo)
    sources=[(root/'train-cal-sensor/dataset_manifest.json','cal'),
             (work/'rgb-body-query-negative-reference-dev-20261010/dataset_manifest.json','additional_cal'),
             (work/'rgb-body-query-arkit-cal-dev-20261010/reference/dataset_manifest.json','additional_cal')]
    rows=[]; queries=None
    for p,split in sources:
        m=load(p)
        if queries is None: queries=m['queries']
        assert queries==m['queries']
        rows.extend(r for r in m['rows'] if r['split']==split)
    assert len(rows)==304 and len({(r['scan'],r['frame']) for r in rows})==304
    return dict(rows=rows,queries=queries),[p for p,_ in sources]


def score(name,manifest,predictions,info,check,inputs,checks,bound=None):
    refs={(r['scan'],r['frame']):r for r in manifest['rows']}
    assert len(refs)==len(manifest['rows'])
    assert len(predictions['rows'])==len(refs)
    data={a:[] for a in ARMS};seen=set()
    for row in predictions['rows']:
        check(); key=(row['scan'],row['frame']); assert key in refs and key not in seen;seen.add(key)
        ref=refs[key]
        for p,expected in [(row['sampled_depth_path'],row['sampled_depth_sha256']),
                           (ref['reference_path'],ref['reference_sha256'])]:
            assert sha(p)==expected;inputs.append(dict(path=str(p),sha256=expected))
        assert row['depth_K']==ref['depth_K'] and row['depth_shape']==ref['depth_shape']
        assert row['rgb_sha256']==ref['rgb_sha256']
        with np.load(row['sampled_depth_path']) as f: dp=f['depth']
        with np.load(ref['reference_path']) as f: labels=f['labels']
        valid=np.isfinite(dp)&(dp>0); loga=np.zeros(dp.shape,np.float64)
        loga[valid]=info['affine']['a']*np.log(dp[valid].astype(np.float64))+info['affine']['b']
        depths={'affine':np.exp(loga)}
        for a,k in MODEL_KEYS.items():
            item=row['distributions'][k] if k in row['distributions'] else row['distributions'][a]
            assert sha(item['path'])==item['sha256'];inputs.append(dict(path=str(item['path']),sha256=item['sha256']))
            with np.load(item['path']) as f:
                mu,mask=f['mu'],f['valid'];residual=f['residual']
                affine_mu=f['affine_mu'] if 'affine_mu' in f.files else None
            assert mu.shape==dp.shape and np.array_equal(valid,mask)
            # Invalid native DP is masked and may retain NaN model outputs;
            # only the frozen public valid domain supplies finite evidence.
            assert np.isfinite(mu[valid]).all() and np.isfinite(residual[valid]).all()
            assert np.allclose(mu[valid],loga[valid]+residual[valid].astype(np.float64),rtol=0,atol=2e-6)
            if affine_mu is not None: assert np.array_equal(affine_mu[valid],loga[valid])
            if bound is not None: assert (np.abs(residual[valid])<=bound+2e-6).all()
            depths[a]=np.exp(mu);checks['frozen_bounded_prediction_arrays']+=1
        rx,ry=rays(row['depth_K'],row['depth_shape'])
        for j,q in enumerate(manifest['queries']):
            check();entry,exit,domain=ray_interval(rx,ry,q);lab=labels[j]
            pos,free,unk=[int((lab==n).sum()) for n in (1,0,2)]
            state='POSITIVE' if pos>=16 else (FREE if pos==0 and unk==0 and free>=16 else 'UNKNOWN')
            assert state==ref['queries'][j]['state'];checks['strict_reference_states']+=1
            mask=valid&domain
            for a,z in depths.items():
                s=np.full(dp.shape,-np.inf);s[mask]=np.minimum(z[mask]-entry[mask],exit[mask]-z[mask])
                qs,ws=nth_score(s),nth_score(s[lab==1])
                data[a].append(dict(cohort=name,environment=ref['environment'],scan=row['scan'],frame=row['frame'],
                    query=q['name'],distance_band=f'{q["low"][2]:g}-{q["high"][2]:g}m',reference_state=state,
                    valid_ray_count=int(np.isfinite(s).sum()),query_score=qs,known_positive_score=ws))
    return data


def affine_reproduction(data,previous,checks):
    lookup={(r['scan'],r['frame'],r['query']):r for r in numeric(previous)}
    diffs=[]
    for r in data['affine']:
        old=lookup[(r['scan'],r['frame'],r['query'])]
        assert r['reference_state']==old['reference_state'] and r['valid_ray_count']==old['valid_ray_count']
        for k in ('query_score','known_positive_score'):
            if r[k]!=old[k]:
                assert np.isfinite(r[k]) and np.isfinite(old[k])
                diffs.append(dict(scan=r['scan'],frame=r['frame'],query=r['query'],field=k,
                    previous=old[k],current=r[k],absolute_difference=abs(r[k]-old[k])))
            else:checks['exact_frozen_affine_score_reproduction']+=1
    return diffs


def calibration(repo,runroot,out,info,check,inputs,checks,bound):
    manifest,sourcepaths=manifest_rows(repo)
    ppath=runroot/'predictions/cal/predictions.json';pm=load(ppath)
    assert pm['status']=='COMPLETE'
    calids={(r['scan'],r['frame']) for r in manifest['rows']}
    trainids={(r['scan'],r['frame']) for r in info['frames']}
    assert calids.isdisjoint(trainids)
    inputs.extend(dict(path=str(p),sha256=sha(p)) for p in sourcepaths+[ppath])
    data=score('cal_fit_description',manifest,pm,info,check,inputs,checks,bound)
    oldpath=repo/'artifacts.local/work/rgb-body-query-arkit-cal-dev-20261010/score/absolute/cal_scores.json'
    inputs.append(dict(path=str(oldpath),sha256=sha(oldpath)))
    write(out/'affine_reproduction_differences.json',affine_reproduction(data,load(oldpath)['arms']['affine'],checks))
    assert all(len(r)==8208 for r in data.values())
    free=[r for r in data['affine'] if r['reference_state']==FREE]
    assert len(free)==588
    cuts={a:select_cutoff([r['query_score'] for r in data[a] if r['reference_state']==FREE]) for a in ARMS}
    write(out/'scores.json',dict(arms=data))
    write(out/'thresholds.json',dict(status='FROZEN_BEFORE_EVALUATION',frozen_utc=utc(),arms=cuts,
        protocol='304 cal pooled FREE only: supplemental working point, not original LOCO main',
        cal_frames=304,cal_queries=8208,cal_free_queries=588,negative_environments=len({r['environment'] for r in free}),
        negative_band_counts={b:dict(queries=sum(r['distance_band']==b for r in free),
            environments=len({r['environment'] for r in free if r['distance_band']==b}))
            for b in ('0.3-0.8m','0.8-1.5m','1.5-3m')},
        cal_POS_descriptive_only=True,evaluation_truth_read=False,all_ties_retained=True))
    ev={a:evaluate(r,cuts[a]['cutoff']) for a,r in data.items()}
    write(out/'evaluation.json',ev);sm={a:summaries(r) for a,r in ev.items()};write(out/'summary.json',sm)
    write_csv(out/'summary.csv',[dict(cohort='cal_fit_description',arm=a,group=g,cutoff=cuts[a]['cutoff'],**s)
        for a,groups in sm.items() for g,s in groups.items()])


def save_protocol(out,name,cutoffs,evaluation,pairs,pair_results):
    folder=out/name;folder.mkdir(exist_ok=True)
    cohorts={c:{a:summaries(rr) for a,rr in arms.items()} for c,arms in evaluation.items()}
    for c,arms in evaluation.items(): write(folder/f'{c}_evaluation.json',arms)
    write(folder/'results.json',dict(status='COMPLETE',thresholds=cutoffs,cohorts=cohorts,pairs=pair_results,
        limits='Consumed Development; sampled FREE not volume truth; related capture frames; no hardware/safety claim'))
    write_csv(folder/'summary.csv',[dict(cohort=c,arm=a,group=g,cutoff=evaluation[c][a][0]['cutoff'],**s)
        for c,arms in cohorts.items() for a,groups in arms.items() for g,s in groups.items()])
    write_csv(folder/'query_pairs.csv',pairs)
    return cohorts


def final(repo,runroot,out,calibration_path,info,check,inputs,checks,bound):
    calreceipt=load(calibration_path/'terminal.json');assert calreceipt['status']=='COMPLETE'
    calcuts=load(calibration_path/'thresholds.json')['arms']
    assert load(calibration_path/'plan.json')['parent_plan_sha256']==sha(runroot/'plan.json')
    inputs.extend(dict(path=str(p),sha256=sha(p)) for p in [calibration_path/'thresholds.json',calibration_path/'scores.json'])
    _,_,_,cohorts=paths(repo);data={}
    calrows=load(calibration_path/'scores.json')['arms']['affine']
    calids={(r['scan'],r['frame']) for r in calrows}
    trainids={(r['scan'],r['frame']) for r in info['frames']};evalids=set()
    for name,folder in cohorts.items():
        check();mpath=folder/'dataset_manifest.json';ppath=runroot/f'predictions/{name}/predictions.json'
        inputs.extend(dict(path=str(p),sha256=sha(p)) for p in [mpath,ppath])
        pm=load(ppath);assert pm['status']=='COMPLETE'
        ids={(r['scan'],r['frame']) for r in pm['rows']}
        assert ids.isdisjoint(calids|trainids|evalids);evalids.update(ids)
        data[name]=score(name,load(mpath),pm,info,check,inputs,checks,bound)
        oldpath=repo/f'artifacts.local/work/rgb-body-query-negative-score-dev-20261010/frozen-score/cached/{name}_scores.json'
        inputs.append(dict(path=str(oldpath),sha256=sha(oldpath)))
        write(out/f'{name}_affine_reproduction_differences.json',affine_reproduction(data[name],load(oldpath)['arms']['affine'],checks))
        write(out/f'{name}_scores.json',dict(arms=data[name]))
    assert sum(len(data[c]['affine']) for c in EVAL)==3672
    assert len(evalids)==136
    pooled={c:{a:evaluate(rr,calcuts[a]['cutoff']) for a,rr in ss.items()} for c,ss in data.items()}
    pooled_pairs=[];pooled_pair_results={}
    for c,ev in pooled.items():
        for a in ARMS[1:]:
            pp=compare(ev[a],ev['affine'],c,a,'pooled_affine','pooled_cal_vs_affine')
            pooled_pairs+=pp;pooled_pair_results[f'{c}/{a}/vs_affine']=pair_groups(pp)
        pp=compare(ev['trained_context'],ev['trained_depth_ray'],c,'trained_context','pooled_trained_depth_ray','pooled_context_vs_depth_ray')
        pooled_pairs+=pp;pooled_pair_results[f'{c}/trained_context/vs_depth_ray']=pair_groups(pp)
    save_protocol(out,'pooled-supplemental',calcuts,pooled,pooled_pairs,pooled_pair_results)
    maincuts={};main={};mainpairs=[];main_pair_results={};post={};postpairs=[];curves=[];matched=[]
    for held in COHORTS:
        check();cal_cohorts=[c for c in COHORTS if c!=held]
        choices={a:select_cutoff([r['query_score'] for c in cal_cohorts for r in data[c][a] if r['reference_state']==FREE]) for a in ARMS}
        maincuts[held]=choices
        write(out/f'{held}_main_LOCO_thresholds.json',dict(status='FROZEN_BEFORE_HELD_EVALUATION',frozen_utc=utc(),
            held_capture=held,cal_cohorts=cal_cohorts,arms=choices,rule='Other two captures strict FREE only; <=5%; ties retained'))
        main[held]={a:evaluate(rr,choices[a]['cutoff']) for a,rr in data[held].items()}
        assert summaries(main[held]['affine'])['all']['free_support']==TARGET_COST[held]
        for a in ARMS[1:]:
            pp=compare(main[held][a],main[held]['affine'],held,a,'LOCO_affine','main_LOCO_vs_affine')
            mainpairs+=pp;main_pair_results[f'{held}/{a}/vs_affine']=pair_groups(pp)
        pp=compare(main[held]['trained_context'],main[held]['trained_depth_ray'],held,'trained_context','LOCO_trained_depth_ray','main_LOCO_context_vs_depth_ray')
        mainpairs+=pp;main_pair_results[f'{held}/trained_context/vs_depth_ray']=pair_groups(pp)
        post[held]={}
        for a in ARMS:
            free=[r['query_score'] for r in data[held][a] if r['reference_state']==FREE]
            cutoff=cutoff_at_cost(free,TARGET_COST[held])
            # Fixed original affine main point, never its lowest threshold on a plateau.
            post[held][a]=main[held]['affine'] if a=='affine' else evaluate(data[held][a],cutoff)
            pp=compare(post[held][a],main[held]['affine'],held,a,'fixed_main_LOCO_affine','posthoc_fixed_affine_cost')
            postpairs+=pp
            matched.append(dict(cohort=held,arm=a,requested_free_budget=TARGET_COST[held],
                cutoff=post[held][a][0]['cutoff'],**summaries(post[held][a])['all']))
            for cost in range(len(free)+1):
                check();cut=cutoff_at_cost(free,cost);s=summaries(evaluate(data[held][a],cut))['all'];assert s['free_support']<=cost
                curves.append(dict(cohort=held,arm=a,negative_count_budget=cost,cutoff=cut,
                    actual_free_support=s['free_support'],free_total=s['free_total'],positive_known_witness=s['positive_known_witness'],positive_total=s['positive_total']))
    save_protocol(out,'main-loco',maincuts,main,mainpairs,main_pair_results)
    save_protocol(out,'posthoc-fixed-affine-cost',dict(affine_targets=TARGET_COST),post,postpairs,
        {f'{c}/{a}/vs_fixed_affine':pair_groups([r for r in postpairs if r['cohort']==c and r['arm']==a]) for c in COHORTS for a in ARMS})
    write_csv(out/'posthoc_matched_points.csv',matched);write_csv(out/'posthoc_curves.csv',curves)
    write(out/'results.json',dict(status='COMPLETE',main='main-loco',supplement='pooled-supplemental',posthoc='posthoc-fixed-affine-cost',
        main_LOCO_capture_count=3,pooled_cal_frames=304,eval_frames=136,eval_queries=3672,
        readout='absolute meter interval margin',normalized='NOT_RUN',best_arm_or_delta_selection=False))


def run(repo,runroot,out,stage,budget,calibration_path=None,prior_attempt=None,bound=None):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'plan.json').exists():raise FileExistsError('Preserve stage payload; resume failed stage at distinct attempt path')
    parent=load(runroot/'plan.json');prior_wall=0.
    assert parent['query_evaluation']['arms']==list(ARMS)
    if bound is None: bound=parent['residual']['bound_B']
    assert bound==parent['residual']['bound_B']
    if prior_attempt:
        earlier=load(prior_attempt/'terminal.json');assert earlier['status']=='FAILED_PARTIAL' and earlier['stage']==stage
        prior_wall=earlier['stage_wall_s']+earlier.get('prior_attempt_wall_s',0.);budget-=prior_wall;assert budget>0
    write(out/'plan.json',dict(stage=stage,frozen_utc=utc(),source_sha256=sha(__file__),parent_plan_sha256=sha(runroot/'plan.json'),
        arms=ARMS,model_keys=MODEL_KEYS,budget_cpu_wall_s_remaining=budget,prior_attempt_wall_s=prior_wall,
        main='Three original ARKit LOCO other-two strict sampled FREE only',supplement='304 pooled cal FREE-only <=5%, frozen before eval truth',
        posthoc='Held FREE matches fixed affine5/12/10, all attainable curves; never main or selection',
        normalized=False,training=0,gpu=0,download=0,eval_bound_adaptation=False,
        query_score='16th largest min(z-entry,exit-z) over public interval domain & frozen prediction valid',
        query_positive_definition='At least16 supported valid domain rays at frozen cutoff; may include UNKNOWN rays',
        positive_witness_definition='At least16 evaluator-known positive rays supported at frozen cutoff; query POS state requires16 known positive rays',
        negative_definition='Strict FREE: zero positive, zero UNKNOWN and at least16 free rays; UNKNOWN never negative',
        UNKNOWN_negative=False,cal_POS_descriptive_only=True,frozen_bound_assert=bound,
        finite_prediction_contract='Finite mu/residual required only on frozen public valid mask; invalid values never evidence',
        dependency_sha256={str(Path(f.__code__.co_filename).name):sha(f.__code__.co_filename)
                           for f in (select_cutoff,compare,summaries,nth_score,cutoff_at_cost,pair_groups)}))
    shutil.copyfile(__file__,out/'executed_bounded_evaluate.py');started=time.perf_counter()
    receipt=dict(status='STARTING',stage=stage,prior_attempt_wall_s=prior_wall)
    inputs=[];checks=Counter()
    def check():
        if time.perf_counter()-started>=budget:raise TimeoutError('Bounded evaluation cumulative CPU budget reached')
    try:
        _,_,candidate,_=paths(repo);ipath=candidate/'training_inputs.json';info=load(ipath)
        assert info['affine']==parent['affine']
        producer_info=runroot/'training_inputs.json'
        if producer_info.exists():
            ti=load(producer_info);assert ti['affine']==info['affine']
            assert ti.get('B',ti.get('bound_B'))==bound
            inputs.append(dict(path=str(producer_info),sha256=sha(producer_info)))
        inputs.extend(dict(path=str(p),sha256=sha(p)) for p in [runroot/'plan.json',ipath])
        if stage=='calibrate':calibration(repo,runroot,out,info,check,inputs,checks,bound)
        else:
            assert calibration_path is not None
            final(repo,runroot,out,calibration_path,info,check,inputs,checks,bound)
        write(out/'inputs.json',inputs);write(out/'checks.json',dict(checks));receipt.update(status='COMPLETE',checks=dict(checks))
    except Exception as exc:
        receipt.update(status='FAILED_PARTIAL',error=repr(exc));raise
    finally:
        receipt.update(stage_wall_s=time.perf_counter()-started,source_sha256=sha(__file__),completed_utc=utc(),gpu=0,training=0,download=0)
        write(out/'terminal.json',receipt);print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--runroot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--stage',choices=('calibrate','final'),required=True)
    p.add_argument('--budget-s',type=float,required=True);p.add_argument('--calibration',type=Path);p.add_argument('--prior-attempt',type=Path)
    p.add_argument('--bound',type=float);a=p.parse_args()
    run(a.repo.resolve(),a.runroot.resolve(),a.output.resolve(),a.stage,a.budget_s,a.calibration,a.prior_attempt,a.bound)
