"""Fixed-score event/control and physical-pair ledgers; CPU, no model fitting."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
WORK=ROOT/'artifacts.local/work'
OUT=WORK/'cnh-event-ledger-dev-20261007'
GROUP=WORK/'cnh-group-calibration-dev-20261007'
STABILITY=WORK/'cnh-model-stability-dev-20261007'
POLICIES=('actual','fa_aligned')
CAPS=('0.025','0.050')
TAGS=('original','HB')
METHODS=('raw_hb_aug','original_center')


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def json_value(value):
    if isinstance(value,np.ndarray):return json_value(value.tolist())
    if isinstance(value,np.generic):return json_value(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return 'Infinity' if value>0 else '-Infinity' if value<0 else None
    if isinstance(value,dict):return {str(k):json_value(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_value(v) for v in value]
    return value


def save(path,value):
    with Path(path).open('x',encoding='utf8') as stream:
        json.dump(json_value(value),stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def line(stream,value):
    stream.write(json.dumps(json_value(value),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')


def control_ties(values):
    values=np.asarray(values,float).ravel()
    if not len(values) or not np.isfinite(values).all():raise ValueError('Nonempty finite control scores required')
    unique,counts=np.unique(values,return_counts=True)
    ge=np.cumsum(counts[::-1])[::-1]
    return [dict(score=float(v),count=int(n),alarm_count_at_score=int(c),
                 alarm_count_just_above_score=int(c-n)) for v,n,c in zip(unique,counts,ge)]


def integer_budget_threshold(values,budget):
    """Lowest extended-float threshold satisfying integer alarm budget under >=.

    For K<N, sorted[N-K-1] is a forbidden boundary: any theta <= it yields
    at least K+1 alarms. Its nextafter(+inf) removes the entire boundary tie
    and is the smallest representable feasible threshold. For K=N, -inf
    includes all scores. No quantile interpolation or floating-rate floor.
    """
    values=np.asarray(values,float).ravel()
    if not len(values) or not np.isfinite(values).all():raise ValueError('Finite control sample required')
    if isinstance(budget,(bool,np.bool_)) or int(budget)!=budget or not 0<=budget<=len(values):
        raise ValueError('Budget must be an integer in0..N')
    budget=int(budget)
    boundary=None if budget==len(values) else float(np.sort(values)[len(values)-budget-1])
    theta=-np.inf if boundary is None else float(np.nextafter(boundary,np.inf))
    actual=int((values>=theta).sum())
    predecessor_count=None if boundary is None else int((values>=np.nextafter(theta,-np.inf)).sum())
    if actual>budget or (predecessor_count is not None and predecessor_count<=budget):
        raise AssertionError('Whole-tie threshold minimality failed')
    return dict(threshold=theta,target_fa_count=budget,actual_fa_count=actual,
        residual_fa_count=budget-actual,control_intervals=len(values),boundary_score=boundary,
        boundary_tie_count=0 if boundary is None else int((values==boundary).sum()),
        predecessor_fa_count=predecessor_count,minimality_verified=True)


def first_index(alarm,start=0,end=13):
    found=np.flatnonzero(np.asarray(alarm,bool)[start:end])
    return int(start+found[0]) if len(found) else None


def event_trace(scores,theta,deadline):
    scores=np.asarray(scores,float)
    if scores.shape!=(13,) or not np.isfinite(scores).all():raise ValueError('13 finite event scores required')
    if not 0<=int(deadline)<=12:raise ValueError('Contact deadline must be0..12')
    deadline=int(deadline);alarm=scores>=theta
    first=first_index(alarm);any_before=first_index(alarm,0,deadline+1)
    primary=first_index(alarm,2,deadline+1) if deadline>=2 else None
    return dict(scores=scores,threshold=theta,alarms=alarm,deadline_index=deadline,
        timely_excluding_warmup=primary is not None,any_before_timely=any_before is not None,
        first_alarm_output=first,first_any_before_output=any_before,first_primary_output=primary,
        first_alarm_native_frame=None if first is None else first+3,
        first_alarm_seconds_from_native0=None if first is None else .2*(first+3),
        first_primary_native_frame=None if primary is None else primary+3,
        first_primary_seconds_from_native0=None if primary is None else .2*(primary+3),
        first_any_before_native_frame=None if any_before is None else any_before+3,
        first_any_before_seconds_from_native0=None if any_before is None else .2*(any_before+3))


def outcome(candidate,baseline):
    return 'both' if candidate and baseline else 'rescued' if candidate else 'lost' if baseline else 'neither'


def physical_membership(original,hb):
    reasons=[]
    for tag,row in (('original',original),('HB',hb)):
        if not row['valid']:reasons.append(tag+'/invalid')
        if not row['evaluable']:reasons.append(tag+'/not_evaluable')
        if row['contact'] is not True:reasons.append(tag+'/not_contact')
    both_contact=original['contact'] is True and hb['contact'] is True
    deadline_equal=original['deadline_index']==hb['deadline_index']
    if both_contact and not deadline_equal:reasons.append('contact_deadline_mismatch')
    return dict(anchor_id=original['anchor_id'],unit=original['unit'],config=original['config'],
        eligible=not reasons,reasons=reasons,contact_labels=[original['contact'],hb['contact']],
        deadlines=[original['deadline_index'],hb['deadline_index']],deadline_equal=deadline_equal,
        contact_query_original=original['contact_query'],contact_query_hb=hb['contact_query'],
        contact_query_changed=original['contact_query']!=hb['contact_query'])


def nondecrease_diagnostic(original_pair,hb_pair,first,deadline,original_theta,hb_theta):
    a,b=np.asarray(original_pair,float),np.asarray(hb_pair,float)
    if a.shape!=(13,2) or b.shape!=(13,2):raise ValueError('Paired13x2 raw logits required')
    indices=np.arange(2,int(deadline)+1)
    delta=b-a;both=(delta>=0).all(1)
    changed=bool(original_theta!=hb_theta)
    return dict(first_original_primary_output=first,
        at_original_first_alarm=None if first is None else dict(output=first,native_frame=first+3,
            original_logits=a[first],hb_logits=b[first],delta=delta[first],both_nondecreasing=bool(both[first])),
        timely_window=[dict(output=int(i),native_frame=int(i+3),original_logits=a[i],hb_logits=b[i],
            delta=delta[i],head_nondecreasing=bool(delta[i,0]>=0),body_nondecreasing=bool(delta[i,1]>=0),
            both_nondecreasing=bool(both[i])) for i in indices],
        any_window_both_nondecreasing=bool(both[indices].any()) if len(indices) else None,
        all_window_both_nondecreasing=bool(both[indices].all()) if len(indices) else None,
        exact_comparison_tolerance=0.,threshold_changed=changed,original_threshold=original_theta,hb_threshold=hb_theta,
        interpretation=('Threshold changed across tags: flip cannot be assigned to geometry or readout suppression.' if changed else
          'Pending explanation only: paired logits do not describe the full current136 input; no causal suppression claim.'))


def freeze(outdir=OUT):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    if read(GROUP/'analysis_terminal.json')['status']!='COMPLETE':raise ValueError('Group evaluation incomplete')
    geometry=read(STABILITY/'geometry_manifest.json');result=read(STABILITY/'result.json')
    if geometry['status']!='COMPLETE':raise ValueError('Stability geometry incomplete')
    contact_ids={i for i,r in enumerate(geometry['rows']) if r['valid'] and r['evaluable'] and r['contact'] is True}
    if len(contact_ids)!=280:raise ValueError('Expected all280 contact records')
    observations={r['path']:r['sha256'] for r in result['observations'] if r['row_index'] in contact_ids}
    if len(observations)!=280:raise ValueError('Contact observations incomplete')
    paths=[Path(__file__),Path(__file__).with_name('test_cnh_event_ledger_dev.py'),
        GROUP/'ledger.npz',GROUP/'result.json',GROUP/'thresholds.json',GROUP/'analysis_terminal.json',
        STABILITY/'geometry_manifest.json',STABILITY/'ledger.npz',STABILITY/'result.json']
    plan=dict(status='frozen',cpu_budget_seconds=600,single_run=True,no_gpu=True,no_fit=True,
        policies=list(POLICIES),caps=list(CAPS),models=[0,1,2],candidate='raw_hb_aug',baseline='original_center',
        actual='Inherited group_calibrated thresholds; one threshold across original/HB for each model/cap',
        fa_aligned='Per tag keep M3 group theta, take its actual integer control FAcount[2:12] as candidate budget; lowest whole-tie feasible threshold, evaluator-picked descriptive only',
        tie_rule='>=, complete finite control-score groups, no interpolation; nextafter(sorted[N-K-1],+inf), K=N permits -inf',
        primary='Timely outputs2..deadline; any-before0..deadline separate. All280 contacts, all strict controls, including zero-alarm rows.',
        physical_pair='All1920 anchors retain membership/exclusion reasons; pair diagnostics require both valid/evaluable/contact and equal deadline. contact_query can change.',
        diagnostic='At original first primary alarm and every2..deadline output compare both raw logits without decreases. Record any/all, no single-frame attribution. Tag-specific threshold changes are explicit confounding.',
        observations='Stored smooth pair and current136 trajectories, contact-only once per event. No new histogram inference.',
        expected_event_rows=280*2*2*3,scope='Consumed synthetic Development, descriptive evaluator diagnostics; no deployment threshold or causal explanation claim',
        hashes={p.relative_to(ROOT).as_posix():sha(p) for p in paths},observation_sha256=observations)
    save(outdir/'PLAN.json',plan)
    return plan


def run(outdir=OUT):
    outdir=Path(outdir);start=time.monotonic();plan=read(outdir/'PLAN.json')
    if (outdir/'RUN_STARTED.json').exists():raise FileExistsError('Single frozen run; preserve prior outputs')
    save(outdir/'RUN_STARTED.json',dict(unix=time.time(),budget_seconds=plan['cpu_budget_seconds']))
    def check():
        if time.monotonic()-start>plan['cpu_budget_seconds']:raise TimeoutError('600s event-ledger CPU budget reached')
    try:
        for path,expected in plan['hashes'].items():
            check()
            if sha(ROOT/path)!=expected:raise ValueError(f'Frozen input changed: {path}')
        group_result=read(GROUP/'result.json')
        if sha(GROUP/'ledger.npz')!=group_result['ledger_sha256']:raise ValueError('Parent ledger receipt mismatch')
        geometry=read(STABILITY/'geometry_manifest.json')['rows']
        metadata=('unit','config','tag','mode','mirror','family','valid','evaluable','contact','control','deadline','row_id','anchor_id','contact_query')
        with np.load(GROUP/'ledger.npz',allow_pickle=False) as saved:
            z={k:saved[k] for k in metadata}
            scores={(f,m):saved[f'score/fold{f}/{m}'] for f in range(3) for m in METHODS}
            frozen_alarm={(f,c,m):saved[f'alarm/fold{f}/group_calibrated/{c}/{m}']
                for f in range(3) for c in CAPS for m in METHODS}
        with np.load(STABILITY/'ledger.npz',allow_pickle=False) as saved:
            for key in metadata:np.testing.assert_array_equal(z[key],saved[key])
            pair,current=saved['pair'],saved['current']
            for key in scores:np.testing.assert_array_equal(scores[key],saved[f'score/fold{key[0]}/{key[1]}'])
        n=len(z['unit']);ev=z['valid']&z['evaluable'];contact=ev&z['contact'];control=ev&z['control']
        if n!=3840 or int(contact.sum())!=280 or int(control.sum())!=728:raise ValueError('Full source denominators changed')
        if [(int(z['unit'][i]),int(z['config'][i]),str(z['tag'][i])) for i in range(n)]!=[(r['unit'],r['config'],r['tag']) for r in geometry]:
            raise ValueError('Geometry and evaluation identities differ')
        thresholds={(r['fold'],r['cap'],r['method']):r['threshold'] for r in read(GROUP/'thresholds.json')['group']}
        threshold_records=[];policy_data={};tie_records=[]
        for f in range(3):
            for tag in TAGS:
                mask=control&(z['tag']==tag)
                for method in METHODS:
                    tie_records.append(dict(tie_group_id=f'fold{f}/{tag}/{method}',fold=f,tag=tag,method=method,
                        control_sequences=int(mask.sum()),control_intervals=int(mask.sum())*10,
                        groups=control_ties(scores[f,method][mask,2:12])))
                for cap in CAPS:
                    base_theta=thresholds[f,cap,'original_center']
                    target=int((scores[f,'original_center'][mask,2:12]>=base_theta).sum())
                    for policy in POLICIES:
                        raw_theta=thresholds[f,cap,'raw_hb_aug']
                        aligned=integer_budget_threshold(scores[f,'raw_hb_aug'][mask,2:12],target) if policy=='fa_aligned' else None
                        if aligned is not None:raw_theta=aligned['threshold']
                        entry={}
                        for method,theta in (('raw_hb_aug',raw_theta),('original_center',base_theta)):
                            alarm=ev[:,None]&(scores[f,method]>=theta)
                            if policy=='actual':np.testing.assert_array_equal(alarm,frozen_alarm[f,cap,method])
                            entry[method]=dict(theta=theta,alarm=alarm)
                            actual=int(alarm[mask,2:12].sum())
                            threshold_records.append(dict(policy=policy,fold=f,tag=tag,cap=cap,method=method,
                                threshold=theta,target_fa_count=target,actual_fa_count=actual,residual_fa_count=target-actual,
                                control_sequences=int(mask.sum()),control_intervals=int(mask.sum())*10,
                                target_fa=target/(int(mask.sum())*10),actual_fa=actual/(int(mask.sum())*10),
                                tie_group_id=f'fold{f}/{tag}/{method}',selection='evaluator_integer_budget' if policy=='fa_aligned' and method=='raw_hb_aug' else 'frozen_group_calibration',
                                integer_budget_details=aligned if method=='raw_hb_aug' else None))
                        policy_data[f,policy,cap,tag]=entry
        save(outdir/'thresholds.json',dict(records=threshold_records,fa_aligned_deployable=False))
        with (outdir/'control_score_ties.jsonl').open('x',encoding='utf8') as stream:
            for value in tie_records:line(stream,value)

        event_indices=np.flatnonzero(contact);control_indices=np.flatnonzero(control)
        def identity(i):
            return {k:json_value(z[k][i]) for k in ('row_id','anchor_id','unit','config','tag','mode','mirror','family','contact_query')}
        observation_paths={r['row_index']:r['path'] for r in read(STABILITY/'result.json')['observations']}
        obs_csv_fields=['row_id','output','native_frame','primary_window','HEAD_logit','BODY_logit','current_signed_mean','current_signed_min','current_signed_max','ambient_log_sector_mean']
        with (outdir/'observation_trajectories.jsonl').open('x',encoding='utf8') as stream, (outdir/'observation_trajectories.csv').open('x',encoding='utf8',newline='') as cs:
            writer=csv.DictWriter(cs,fieldnames=obs_csv_fields);writer.writeheader()
            for i in event_indices:
                check();path=observation_paths[int(i)];expected=plan['observation_sha256'][path]
                if sha(ROOT/path)!=expected:raise ValueError('Frozen event observation changed')
                with np.load(ROOT/path,allow_pickle=False) as obs:np.testing.assert_array_equal(obs['smooth'],pair[i])
                if not np.isfinite(current[i]).all():raise ValueError('Missing current136 event trajectory')
                d=int(z['deadline'][i]);line(stream,dict(**identity(i),deadline_index=d,source_path=path,source_sha256=expected,
                    output_indices=list(range(13)),native_frames=list(range(3,16)),primary_window=list(range(2,d+1)),
                    smooth_pair=pair[i],current136=current[i],current_layout='8 sectors x16 signed radial means; then8 mean(log1p ambient)'))
                for t in range(13):writer.writerow(dict(row_id=str(z['row_id'][i]),output=t,native_frame=t+3,primary_window=2<=t<=d,
                    HEAD_logit=float(pair[i,t,0]),BODY_logit=float(pair[i,t,1]),current_signed_mean=float(current[i,t,:128].mean()),
                    current_signed_min=float(current[i,t,:128].min()),current_signed_max=float(current[i,t,:128].max()),ambient_log_sector_mean=float(current[i,t,128:].mean())))

        summary={};loss_sets={};event_count=control_count=0
        csv_fields=['row_id','unit','config','tag','fold','policy','cap','deadline','primary_outcome','any_before_outcome',
                    'candidate_first_primary','m3_first_primary','candidate_theta','m3_theta']
        with (outdir/'event_ledger.jsonl').open('x',encoding='utf8') as es, (outdir/'event_ledger.csv').open('x',encoding='utf8',newline='') as ec, (outdir/'control_ledger.jsonl').open('x',encoding='utf8') as cs, (outdir/'control_ledger.csv').open('x',encoding='utf8',newline='') as cc:
            ew=csv.DictWriter(ec,fieldnames=csv_fields);ew.writeheader()
            cw=csv.DictWriter(cc,fieldnames=['row_id','unit','config','tag','fold','policy','cap','candidate_startup_FA','m3_startup_FA','candidate_main_FA','m3_main_FA']);cw.writeheader()
            for f in range(3):
                for policy in POLICIES:
                    for cap in CAPS:
                        for tag in TAGS:
                            check();entry=policy_data[f,policy,cap,tag];key=f'fold{f}/{policy}/{cap}/{tag}'
                            primary=Counter();anybefore=Counter();loss_sets[key]=set();control_totals=Counter()
                            for i in event_indices[z['tag'][event_indices]==tag]:
                                traces={m:event_trace(scores[f,m][i],entry[m]['theta'],z['deadline'][i]) for m in METHODS}
                                raw,base=traces['raw_hb_aug'],traces['original_center']
                                state=outcome(raw['timely_excluding_warmup'],base['timely_excluding_warmup'])
                                any_state=outcome(raw['any_before_timely'],base['any_before_timely'])
                                primary[state]+=1;anybefore[any_state]+=1
                                if state=='lost':loss_sets[key].add(str(z['row_id'][i]))
                                line(es,dict(**identity(i),fold=f,policy=policy,cap=cap,primary_outcome=state,any_before_outcome=any_state,
                                    candidate=raw,m3=base,observation_trajectory_ref=str(z['row_id'][i])))
                                ew.writerow(dict(row_id=str(z['row_id'][i]),unit=int(z['unit'][i]),config=int(z['config'][i]),tag=tag,fold=f,policy=policy,cap=cap,
                                    deadline=int(z['deadline'][i]),primary_outcome=state,any_before_outcome=any_state,
                                    candidate_first_primary=raw['first_primary_output'],m3_first_primary=base['first_primary_output'],candidate_theta=entry['raw_hb_aug']['theta'],m3_theta=entry['original_center']['theta']))
                                event_count+=1
                            for i in control_indices[z['tag'][control_indices]==tag]:
                                if not (np.asarray(geometry[i]['frame_category'])=='clear').all():raise ValueError('Non-strict control in ledger')
                                views={}
                                for m in METHODS:
                                    alarm=entry[m]['alarm'][i];startup=int(alarm[:2].sum());main=int(alarm[2:12].sum())
                                    views[m]=dict(scores=scores[f,m][i],threshold=entry[m]['theta'],alarms=alarm,
                                        startup_alarm_count=startup,main_alarm_count=main,excluded_final_output_alarm=bool(alarm[12]),
                                        first_alarm_output=first_index(alarm))
                                    control_totals[m+'/startup']+=startup;control_totals[m+'/main']+=main
                                line(cs,dict(**identity(i),fold=f,policy=policy,cap=cap,strict_control=True,
                                    candidate=views['raw_hb_aug'],m3=views['original_center']))
                                cw.writerow(dict(row_id=str(z['row_id'][i]),unit=int(z['unit'][i]),config=int(z['config'][i]),tag=tag,fold=f,policy=policy,cap=cap,
                                    candidate_startup_FA=views['raw_hb_aug']['startup_alarm_count'],m3_startup_FA=views['original_center']['startup_alarm_count'],
                                    candidate_main_FA=views['raw_hb_aug']['main_alarm_count'],m3_main_FA=views['original_center']['main_alarm_count']))
                                control_count+=1
                            summary[key]=dict(primary=dict(primary),any_before=dict(anybefore),controls=int((control&(z['tag']==tag)).sum()),control_alarms=dict(control_totals))

        memberships=[];paired_stats={};paired_count=0
        with (outdir/'physical_pair_membership.jsonl').open('x',encoding='utf8') as ms, (outdir/'physical_pair_ledger.jsonl').open('x',encoding='utf8') as ps:
            for i in range(0,n,2):
                check();a,b=geometry[i],geometry[i+1]
                if a['tag']!='original' or b['tag']!='HB' or a['anchor_id']!=b['anchor_id']:raise ValueError('Physical pair order mismatch')
                membership=physical_membership(a,b);line(ms,membership);memberships.append(membership)
                if not membership['eligible']:continue
                for f in range(3):
                    for policy in POLICIES:
                        for cap in CAPS:
                            methods={}
                            for method in METHODS:
                                ta=policy_data[f,policy,cap,'original'][method]['theta'];tb=policy_data[f,policy,cap,'HB'][method]['theta']
                                if policy=='actual' and ta!=tb:raise ValueError('Actual policy must use the same threshold across tags')
                                va=event_trace(scores[f,method][i],ta,z['deadline'][i]);vb=event_trace(scores[f,method][i+1],tb,z['deadline'][i+1])
                                flip=va['timely_excluding_warmup'] and not vb['timely_excluding_warmup']
                                state=outcome(vb['timely_excluding_warmup'],va['timely_excluding_warmup'])
                                statkey=f'fold{f}/{policy}/{cap}/{method}'
                                paired_stats.setdefault(statkey,Counter())[state]+=1
                                methods[method]=dict(original_timely=va['timely_excluding_warmup'],hb_timely=vb['timely_excluding_warmup'],
                                    original_any_before=va['any_before_timely'],hb_any_before=vb['any_before_timely'],
                                    original_first_primary=va['first_primary_output'],hb_first_primary=vb['first_primary_output'],
                                    threshold_changed=bool(ta!=tb),original_threshold=ta,hb_threshold=tb,primary_transition=state,
                                    original_timely_hb_miss=flip,
                                    pending_explanation=nondecrease_diagnostic(pair[i],pair[i+1],va['first_primary_output'],z['deadline'][i],ta,tb) if flip else None)
                            line(ps,dict(**membership,fold=f,policy=policy,cap=cap,methods=methods,
                                observation_refs=[str(z['row_id'][i]),str(z['row_id'][i+1])]))
                            paired_count+=1
        if event_count!=plan['expected_event_rows'] or control_count!=728*12:raise ValueError('Event/control ledger dropped rows')
        overlaps=[];losskeys=list(loss_sets)
        for ia,ka in enumerate(losskeys):
            for kb in losskeys[ia+1:]:
                a,b=loss_sets[ka],loss_sets[kb];union=a|b;intersection=a&b
                overlaps.append(dict(a=ka,b=kb,a_losses=len(a),b_losses=len(b),intersection=len(intersection),union=len(union),
                    jaccard=None if not union else len(intersection)/len(union),shared_row_ids=sorted(intersection)))
        save(outdir/'loss_set_overlap.json',dict(loss_sets={k:sorted(v) for k,v in loss_sets.items()},pairwise=overlaps,
            note='Exact event identities, separate original/HB rows; same physical anchor across tags is not counted as identical event'))
        check()
        final=dict(status='COMPLETE',events=280,controls=728,event_ledger_rows=event_count,control_ledger_rows=control_count,
            physical_anchors=len(memberships),physical_pairs_eligible=sum(v['eligible'] for v in memberships),physical_pair_ledger_rows=paired_count,
            pair_exclusion_reasons=dict(Counter(r for v in memberships for r in v['reasons'])),metrics=summary,
            paired_transitions={k:dict(v) for k,v in paired_stats.items()},
            threshold_records=len(threshold_records),complete_tie_group_records=len(tie_records),
            fa_aligned_note='Evaluator-picked tag-specific thresholds; residual FA budgets preserve whole ties; not deployment calibration',
            mechanism_note='First-frame and all-window comparisons are pending explanations, not causal attribution. Different tag thresholds explicitly confound flips.',
            plan_sha256=sha(outdir/'PLAN.json'),seconds=time.monotonic()-start)
        save(outdir/'summary.json',final);save(outdir/'TERMINAL.json',dict(status='COMPLETE',seconds=time.monotonic()-start))
        print(json.dumps({k:final[k] for k in ('status','event_ledger_rows','control_ledger_rows','physical_pairs_eligible','seconds')},indent=2))
        return final
    except BaseException as exc:
        save(outdir/'TERMINAL.json',dict(status='FAILED_OR_BUDGET_STOP',seconds=time.monotonic()-start,error=repr(exc)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','run']);args=parser.parse_args()
    freeze() if args.stage=='freeze' else run()
