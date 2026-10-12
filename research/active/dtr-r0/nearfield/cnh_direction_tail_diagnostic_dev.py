"""Frozen calibration tails and score→threshold decomposition; no inference.

Tail units are jointly-clear time slots or any-pass/no-contact full clips.
Every selected-active unit and every predecessor-level tie is retained, with
all maximising query/frame exposures and their causal last-five contributors.
Metadata describes calibration constraints, never photon-source attribution.
"""
import argparse
import csv
from pathlib import Path
import time

import numpy as np
import cnh_direction_aggregation_evaluate_dev as A
import cnh_counterfactual_eval_dev as E
import cnh_pass_workpoints_dev as W

ROOT=A.ROOT
OUT=ROOT/'artifacts.local/work/cnh-direction-anchor-dev-20261009'
PRIOR=A.OUT
POINTS=('pass_20pct','pass_30pct','pass_40pct')
STAGES=('center_at_single_theta','aggregator_at_single_theta','aggregator_at_own_theta')

def write_csv(path,rows):
    if not rows:raise ValueError('Explicit nonempty schema required')
    with Path(path).open('w',encoding='utf8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def costs(scores,category,theta):
    clear,passed=W.exposure(category)
    return dict(clear_slots=int((scores[clear].max(-1)>=theta).sum()),
                pass_clips=int((scores[passed].max((-2,-1))>=theta).sum()))

def predecessor_record(scores,category,record):
    """Validate the frozen grid choice; never select a replacement threshold."""
    levels=np.r_[np.unique(scores),np.inf];theta=E.threshold(record['threshold'])
    index=int(record['selected_grid_index'])
    if len(levels)!=record['grid_levels'] or levels[index]!=theta:
        raise AssertionError('Frozen threshold/grid identity differs')
    selected=costs(scores,category,theta)
    for key in selected:
        if selected[key]!=record[key]:raise AssertionError('Frozen cal cost differs')
    caps=dict(clear_slots=int(record['clear_cap']),pass_clips=int(record['pass_cap']))
    if any(selected[k]>caps[k] for k in caps):raise AssertionError('Selected level infeasible')
    previous=float(levels[index-1]) if index else None
    pc=costs(scores,category,previous) if previous is not None else dict(clear_slots=0,pass_clips=0)
    blockers=[k for k in caps if previous is not None and pc[k]>caps[k]]
    if index and not blockers:raise AssertionError('Predecessor must be infeasible')
    return theta,previous,selected,pc,blockers

def contributions(raw,method,scene,replica,peak_frame,q):
    """Exact raw-first direction aggregation and inherited causal weights."""
    start=max(0,peak_frame-4);weights=2.**np.arange(peak_frame-start+1);weights/=weights.sum()
    rows=[]
    for t,weight in zip(range(start,peak_frame+1),weights):
        values=raw[scene,replica,t,q];chosen=int(A.highest_direction(values))
        dw=np.array([0.,1.,0.]) if method=='single' else A.WEIGHTS if method=='weighted' else np.eye(3)[chosen]
        parts=values*dw*weight
        rows.append(dict(contributing_frame=t+3,temporal_weight=float(weight),
            raw_minus=float(values[0]),raw_center=float(values[1]),raw_plus=float(values[2]),
            raw_argmax_relative=int(A.RELATIVE[chosen]),raw_argmax_ties='|'.join(str(int(A.RELATIVE[x])) for x in np.flatnonzero(values==values.max())),
            direction_weight_minus=float(dw[0]),direction_weight_center=float(dw[1]),direction_weight_plus=float(dw[2]),
            contribution_minus=float(parts[0]),contribution_center=float(parts[1]),contribution_plus=float(parts[2]),
            total_contribution=float(parts.sum())))
    return rows

def metadata(row):
    return {key:row.get(key,'') for key in ('scene_uid','shape_family','background_family','background_id','target_id','presence','placement','rho','physical_key','occurrence_draw_id')}|dict(
        target_side='N/A' if not row['presence'] else 'negative_x' if row['side']==-1 else 'positive_x')

def tail_rows(raw,scores,cohort,identity,theta,previous,blockers):
    c=cohort['category'];clear,passed=W.exposure(c);units=[];peaks=[];frames=[]
    for kind,keep in (('clear_joint_slot',clear),('pass_clip',passed)):
        for i in np.flatnonzero(keep):
            for k in range(scores.shape[1]):
                values=scores[i,k].max(-1) if kind=='clear_joint_slot' else np.array([scores[i,k].max()])
                for t,value in enumerate(values):
                    active=bool(value>=theta);ptie=previous is not None and bool(value==previous)
                    if not (active or ptie):continue
                    unit_id=':'.join(map(str,(*identity.values(),kind,int(i),k,t if kind=='clear_joint_slot' else 'clip')))
                    positions=np.array([(t,int(q)) for q in np.flatnonzero(scores[i,k,t]==value)]) if kind=='clear_joint_slot' else np.argwhere(scores[i,k]==value)
                    constraint='clear_slots' if kind=='clear_joint_slot' else 'pass_clips'
                    common=dict(identity,unit_id=unit_id,unit_kind=kind,scene_id=int(cohort['scene_ids'][i]),replica=k,
                        unit_frame=t+3 if kind=='clear_joint_slot' else '',unit_score=float(value),threshold=theta if np.isfinite(theta) else '',
                        predecessor=previous if previous is not None else '',active_at_selected=active,
                        selected_level_tie=bool(value==theta),predecessor_level_tie=ptie,predecessor_added=ptie and not active,
                        constraint_violates_at_predecessor=constraint in blockers,maximising_exposures=len(positions),**metadata(cohort['rows'][i]))
                    units.append(common)
                    for peak_index,(tf,q) in enumerate(positions):
                        tf,q=int(tf),int(q);pv=raw[i,k,tf,q];chosen=int(A.highest_direction(pv));peak_id=f'{unit_id}:peak{peak_index}'
                        p=dict(identity,unit_id=unit_id,peak_id=peak_id,scene_id=int(cohort['scene_ids'][i]),replica=k,
                            peak_frame=tf+3,height=E.HEIGHTS[q],query_category=str(c[i,q]),smoothed_peak=float(value),
                            peak_raw_argmax_relative=int(A.RELATIVE[chosen]),peak_raw_argmax_absolute=int(A.RELATIVE[chosen]),
                            temporal_start_frame=max(0,tf-4)+3,temporal_end_frame=tf+3,
                            explanation='raw direction argmax at peak frame only; temporal mixture retained')
                        peaks.append(p);cr=contributions(raw,identity['aggregator'],i,k,tf,q)
                        if not np.isclose(sum(x['total_contribution'] for x in cr),value,rtol=1e-12,atol=1e-12):raise AssertionError('Contributor sum differs')
                        frames.extend(dict(identity,unit_id=unit_id,peak_id=peak_id,scene_id=int(cohort['scene_ids'][i]),replica=k,height=E.HEIGHTS[q],peak_frame=tf+3,**r) for r in cr)
    return units,peaks,frames

def decompose(scores,cohort,identity,single_theta,own_theta):
    single,agg=scores;flags=(single>=single_theta,agg>=single_theta,agg>=own_theta)
    metrics=[];timely=[]
    for f in flags:
        m,t=E.summary(f,cohort['category']);metrics.append(m);timely.append(t)
    mask=np.array([r['placement']=='in1cm' for r in cohort['rows']]);allstrata={};summary=[]
    for stratum,keep in (('all',np.ones(len(mask),bool)),('in1cm',mask)):
        category=cohort['category'][keep];ids=cohort['scene_ids'][keep]
        pairs={name:E.paired(timely[a][keep],timely[b][keep],category,ids) for name,a,b in (('score_step',0,1),('threshold_step',1,2),('total',0,2))}
        sm=[E.summary(f[keep],category)[0] for f in flags]
        allstrata[stratum]=dict(stages=dict(zip(STAGES,sm)),paired=pairs)
        for j,stage in enumerate(STAGES):
            previous=0 if j==0 else j-1;ps=E.paired(timely[previous][keep],timely[j][keep],category,ids)
            row=dict(identity,stratum=stratum,stage=stage,threshold=single_theta if j<2 else own_theta)
            for key in ('clear_slots','clear_denominator','clear_segments','clear_clips','clear_clip_denominator','pass_clips','pass_clip_denominator','physical_contact_any_height','physical_contact_denominator'):row[key]=sm[j][key]
            for q,h in enumerate(E.HEIGHTS):
                row[h+'_count']=sm[j]['counts'][q];row[h+'_late_count']=sm[j]['late_counts'][q];row[h+'_denominator']=sm[j]['contact_event_denominators'][q]
                for key in ('gain','loss','net'):row[h+'_vs_previous_'+key]=ps[q][key]
                full=E.paired(timely[0][keep],timely[j][keep],category,ids)
                for key in ('gain','loss','net'):row[h+'_vs_center_'+key]=full[q][key]
            summary.append(row)
    ledger=[]
    for i,q in np.argwhere(cohort['category']=='contact'):
        for k in range(flags[0].shape[1]):
            values=[bool(t[i,k,q]) for t in timely]
            row=dict(identity,scene_id=int(cohort['scene_ids'][i]),replica=k,height=E.HEIGHTS[q],one_cm=bool(mask[i]),single_theta=single_theta,own_theta=own_theta,**metadata(cohort['rows'][i]))
            for j,stage in enumerate(STAGES):
                row[stage+'_timely']=values[j];row[stage+'_first_frame']=A.first_frame(flags[j],i,k,q)
                row[stage+'_first_timely_frame']=A.first_frame(flags[j],i,k,q,end=11)
                row[stage+'_first_late_frame']=A.first_frame(flags[j],i,k,q,start=11)
            for name,a,b in (('score_step',0,1),('threshold_step',1,2),('total',0,2)):
                row[name+'_gain']=bool(not values[a] and values[b]);row[name+'_loss']=bool(values[a] and not values[b]);row[name+'_net']=int(values[b])-int(values[a])
            ledger.append(row)
    check=dict(identity,min_smoothed_delta=float((agg-single).min()),same_theta_query_slot_alarm_losses=int((flags[0]&~flags[1]).sum()),
        same_theta_query_slot_alarm_gains=int((~flags[0]&flags[1]).sum()),theta_shift=float(own_theta-single_theta),
        full_contact_total_losses=sum(r['loss'] for r in allstrata['all']['paired']['total']))
    if identity['aggregator']=='max':
        if check['min_smoothed_delta']<0 or check['same_theta_query_slot_alarm_losses']!=0:raise AssertionError('Max monotonicity violated')
        if check['full_contact_total_losses'] and own_theta<=single_theta:raise AssertionError('Max total losses require theta increase')
    return allstrata,summary,ledger,check

def run(plan_path,manifest_path,threshold_path,output):
    started=time.monotonic();plan=W.read(plan_path);cap=float(plan['allocations_cpu']['tail_diagnostic'])
    def check():
        if time.monotonic()-started>=cap:raise TimeoutError('Tail diagnostic allocation reached')
    if output.exists():raise FileExistsError('Preserve previous diagnostic outputs')
    if not output.resolve().is_relative_to((ROOT/'artifacts.local').resolve()):raise ValueError('Canonical artifact output required')
    output.mkdir(parents=True);receipt=dict(status='RUNNING',budget_kind='cpu',source_sha256=W.sha(__file__),plan_sha256=W.sha(plan_path),inference=0,training=0)
    try:
        if (plan['arms'],plan['seeds'],plan['branch_angles'],plan['weights_relative_minus_center_plus'])!=(list(A.ARMS),list(A.SEEDS),{'ideal':[-3,0,3],'yaw_plus3':[0,3,6]},A.WEIGHTS.tolist()):raise ValueError('Frozen scope differs')
        jobs,cohorts,_,sources=A.load_manifest(manifest_path.resolve(),plan,check);thresholds=W.read(threshold_path)['arms']
        sources.update((Path(__file__).resolve(),Path(A.__file__).resolve(),Path(E.__file__).resolve(),Path(W.__file__).resolve(),Path(E.F.__file__).resolve(),plan_path.resolve(),threshold_path.resolve()))
        thresholdrows=[];units=[];peaks=[];contributors=[];decomposition={};scalars=[];ledger=[];checks=[]
        for arm in A.ARMS:
            decomposition[arm]={}
            for seed in A.SEEDS:
                decomposition[arm][str(seed)]={};raws={};smooths={}
                for split,branch in (('cal','ideal'),('validation','ideal'),('validation','yaw_plus3')):
                    check();raw=np.stack([jobs[arm,seed,split,a] for a in plan['branch_angles'][branch]],-1);raws[split,branch]=raw
                    for agg in A.AGGREGATORS:smooths[agg,split,branch]=E.smooth(A.aggregate(raw,agg))
                cal=cohorts['cal']
                for agg in A.AGGREGATORS:
                    for point in POINTS:
                        check();record=thresholds[arm][str(seed)][agg][point];scores=smooths[agg,'cal','ideal']
                        theta,previous,selected,pc,blockers=predecessor_record(scores,cal['category'],record)
                        single_theta=E.threshold(thresholds[arm][str(seed)]['single'][point]['threshold']);identity=dict(arm=arm,seed=seed,aggregator=agg,point=point)
                        gridlabels={}
                        for label,level in (('selected',theta),('predecessor',previous)):
                            for category in ('contact','clear','pass'):
                                gridlabels[label+'_level_'+category+'_query_exposures']=0 if level is None else int(((scores==level)&(cal['category'][:,None,None,:]==category)).sum())
                        thresholdrows.append(dict(identity,threshold=theta if np.isfinite(theta) else '',threshold_positive_infinity=bool(np.isposinf(theta)),
                            single_threshold=single_theta if np.isfinite(single_theta) else '',theta_shift_from_single=float(theta-single_theta) if np.isfinite(theta) and np.isfinite(single_theta) else '',
                            selected_grid_index=int(record['selected_grid_index']),grid_levels=int(record['grid_levels']),predecessor=previous if previous is not None else '',
                            clear_cap=record['clear_cap'],pass_cap=record['pass_cap'],selected_clear_slots=selected['clear_slots'],selected_pass_clips=selected['pass_clips'],
                            predecessor_clear_slots=pc['clear_slots'],predecessor_pass_clips=pc['pass_clips'],blocking_constraints='|'.join(blockers),**gridlabels))
                        u,p,f=tail_rows(raws['cal','ideal'],scores,cal,identity,theta,previous,blockers);units.extend(u);peaks.extend(p);contributors.extend(f)
                for agg in ('max','weighted'):
                    decomposition[arm][str(seed)][agg]={}
                    for branch in A.BRANCHES:
                        decomposition[arm][str(seed)][agg][branch]={}
                        for point in POINTS:
                            check();single_theta=E.threshold(thresholds[arm][str(seed)]['single'][point]['threshold']);own_theta=E.threshold(thresholds[arm][str(seed)][agg][point]['threshold'])
                            identity=dict(arm=arm,seed=seed,aggregator=agg,branch=branch,point=point)
                            r,s,l,c=decompose((smooths['single','validation',branch],smooths[agg,'validation',branch]),cohorts['validation'],identity,single_theta,own_theta)
                            decomposition[arm][str(seed)][agg][branch][point]=dict(single_threshold=E.record_threshold(single_theta),own_threshold=E.record_threshold(own_theta),strata=r)
                            scalars.extend(s);ledger.extend(l);checks.append(c)
        write_csv(output/'tail_thresholds.csv',thresholdrows);write_csv(output/'tail_units.csv',units);write_csv(output/'tail_peak_frames.csv',peaks);write_csv(output/'contributor_frames.csv',contributors)
        write_csv(output/'decomposition_summary.csv',scalars);write_csv(output/'decomposition_event_ledger.csv',ledger)
        W.save(output/'decomposition_metrics.json',dict(arms=decomposition,scope='Frozen validation events, stages descriptive; no threshold selection',correlation='384 fixtures / 324 unique physical worlds; K4, frames and seeds correlated'))
        W.save(output/'same_theta_checks.json',dict(rows=checks,max_monotonicity='All max smoothed scores >= center; zero same-theta query alarm loss',weighted_monotonicity='No monotonicity assumed',scale='Within same arm/checkpoint only; no cross-model logit causal claim'))
        W.save(output/'input_manifest.json',dict(files=[dict(path=str(p),sha256=W.sha(p)) for p in sorted(sources,key=str)]))
        receipt.update(status='COMPLETE',counts=dict(thresholds=len(thresholdrows),tail_units=len(units),maximising_exposures=len(peaks),contributor_frames=len(contributors),decomposition_paths=len(checks),stage_stratum_rows=len(scalars),contact_ledger=len(ledger)),
            statement='Consumed simulated Development; frozen prior cal thresholds; all selected-active units plus predecessor ties, all maxima and last5 contributions. Constraint metadata is not photon attribution.')
    except Exception as error:
        receipt.update(status='FAILED',error=repr(error));raise
    finally:
        receipt['seconds']=time.monotonic()-started;W.save(output/'run_receipt.json',receipt)
    print(receipt)

def focused_checks(output):
    started=time.monotonic();results=[]
    # Selected score level can come from contact, while predecessor is blocked by pass.
    s=np.zeros((3,1,13,2));c=np.array([['clear','clear'],['pass','clear'],['contact','clear']]);s[1,0,0]=2;s[2,0,0]=3
    record=W.select_threshold(s,c,clear_cap=0,pass_cap=0)
    theta,previous,selected,pc,b=predecessor_record(s,c,record)
    assert theta==3 and previous==2 and b==['pass_clips'] and pc['pass_clips']==1
    assert not (s[1].max()==theta);results.append('contact_grid_level_with_pass_predecessor_blocker')
    # Whole-query ties form one clear slot, but full pass frames form one clip.
    s[0,0,:2]=4;s[1,0,:2]=4;co=costs(s,c,4);assert co==dict(clear_slots=2,pass_clips=1);results.append('whole_ties_slot_vs_clip')
    raw=np.zeros((1,1,13,2,3));raw[0,0,0,0]=[9,0,0];raw[0,0,1,0]=[0,0,9]
    maximum=E.smooth(A.aggregate(raw,'max'));center=E.smooth(A.aggregate(raw,'single'))
    assert (maximum>=center).all() and not ((center>=1)&~(maximum>=1)).any();results.append('raw_max_then_smooth_monotonicity')
    for method in A.AGGREGATORS:
        score=E.smooth(A.aggregate(raw,method))
        for t in (0,1,7):
            parts=contributions(raw,method,0,0,t,0)
            np.testing.assert_allclose(sum(r['total_contribution'] for r in parts),score[0,0,t,0],rtol=1e-12,atol=1e-12)
    results.append('causal_last5_contributions_including_startup')
    raw[:]=0;raw[...,1]=8
    assert (A.aggregate(raw,'weighted')<A.aggregate(raw,'single')).all();results.append('weighted_not_monotone')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError('Preserve old checks')
    W.save(output,dict(status='PASS',checks=results,budget_kind='cpu',seconds=time.monotonic()-started,source_sha256=W.sha(__file__)))
    print('Focused checks PASS',len(results))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,default=OUT/'PLAN.json');parser.add_argument('--manifest',type=Path,default=PRIOR/'scores/manifest.json')
    parser.add_argument('--thresholds',type=Path,default=PRIOR/'evaluation/thresholds.json');parser.add_argument('--output',type=Path,default=OUT/'tails');parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    if args.check:focused_checks(OUT/'tail_focused_checks.json')
    else:run(args.plan,args.manifest,args.thresholds,args.output)
