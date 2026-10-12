"""Independent all-tie calibration audit for additive HEAD cost fractions.

Reuses independently implemented clocks/cost/report definitions from the previous
audit. This does not import the producer or select any threshold using validation.
"""
import json
from pathlib import Path
import time

import numpy as np
import audit_cnh_graded_peak_body_only_dev as A
import cnh_graded_corridor_eval_dev as C
import cnh_graded_evidence_dev as G
import cnh_counterfactual_eval_dev as E

OUT = C.ROOT/'artifacts.local/work/cnh-graded-peak-head-cal-dev-20261010'
JOINT = C.ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010'
BODY = C.ROOT/'artifacts.local/work/cnh-graded-peak-body-only-dev-20261010'
POLICIES = ('baseline','both','body_only','head25','head50','head75')
FRACTIONS = (.25,.5,.75)


def independent_cal_mask(rows):
    families = {}
    for row in rows: families.setdefault(row['background_family'],set()).add(row['background_id'])
    assert all(len(values)==2 for values in families.values())
    cal_ids = {max(values) for values in families.values()}
    fit_ids = {min(values) for values in families.values()}
    assert not (cal_ids&fit_ids)
    mask = np.array([row['background_id'] in cal_ids for row in rows])
    return mask,sorted(fit_ids),sorted(cal_ids)


def independent_cutoff(score, eligible, category, mask, parent, fraction):
    clear = mask&(category=='clear').all(1)
    passed = mask&(category=='pass').any(1)&~(category=='contact').any(1)
    old_flags = eligible&(score>=parent)
    clear_old,pass_old = int(old_flags[clear].sum()),int(old_flags[passed].sum())
    caps = (int(np.floor(fraction*clear_old)),int(np.floor(fraction*pass_old)))
    finite = np.concatenate((score[clear][eligible[clear]],score[passed][eligible[passed]]))
    # Every score tie has a discontinuity directly above it, and thresholds
    # below the fixed parent are disallowed. Evaluating all breakpoints gives
    # the smallest feasible threshold without calling producer at_most.
    candidates = np.unique(np.r_[parent,np.nextafter(np.unique(finite),np.inf)])
    candidates = candidates[candidates>=parent]
    records = []
    for theta in candidates:
        flags = eligible&(score>=theta)
        counts = (int(flags[clear].sum()),int(flags[passed].sum()))
        records.append((float(theta),counts))
        if counts[0]<=caps[0] and counts[1]<=caps[1]:
            return dict(theta=float(theta),old_clear=clear_old,old_pass=pass_old,
                cap_clear=caps[0],cap_pass=caps[1],actual_clear=counts[0],actual_pass=counts[1],
                breakpoints_examined=len(records),calibration_scene_count=int(mask.sum()))
    raise ValueError('No feasible cutoff among all score breakpoints')


def individual_cut(values, cap):
    if len(values)<=cap: return None
    for theta in np.nextafter(np.unique(values),np.inf):
        if int((values>=theta).sum())<=cap: return float(theta)
    raise ValueError('No feasible individual tie cut')


def check_cut(actual, record, fraction, parent, fit_ids, cal_ids, key, clear_values, pass_values,
              clear_denominator, pass_denominator):
    expected=dict(fraction=fraction,parent_theta=parent,theta=record['theta'],
        old_HEAD_incremental_clear_slots=record['old_clear'],old_HEAD_incremental_pass_slots=record['old_pass'],
        target_clear_slots=record['cap_clear'],target_pass_slots=record['cap_pass'],
        actual_clear_slots=record['actual_clear'],actual_pass_slots=record['actual_pass'],
        clear_unused=record['cap_clear']-record['actual_clear'],pass_unused=record['cap_pass']-record['actual_pass'],
        clear_cut=individual_cut(clear_values,record['cap_clear']),
        pass_cut=individual_cut(pass_values,record['cap_pass']),
        clear_denominator=clear_denominator,pass_denominator=pass_denominator)
    A.equal(actual,expected,key+'/calibration')


def check_summary(actual, split, seed, policy, r, key):
    p,costs=r['paired_any']['prior_light'],r['new_costs_vs_baseline']
    expected=dict(split=split,seed=seed,policy=policy,
        HEAD=r['any']['counts'][0],BODY=r['any']['counts'][1],
        HEAD_rescue=p[0]['rescue'],BODY_rescue=p[1]['rescue'],HEAD_earlier=p[0]['earlier'],BODY_earlier=p[1]['earlier'],
        HEAD_loss=p[0]['loss'],BODY_loss=p[1]['loss'],HEAD_later=p[0]['later'],BODY_later=p[1]['later'],
        HEAD_foregone_rescue=r['paired_vs_both'][0]['loss'],HEAD_foregone_advance=r['paired_vs_both'][0]['later'],
        clear_slots=r['any']['clear_slots'],clear_clips=r['any']['clear_clips'],pass_clips=r['any']['pass_clips'],
        extra_clear_slots=costs['clear']['new_slots'],extra_clear_clips=costs['clear']['new_clips'],
        extra_pass_slots=costs['pass']['new_slots'],extra_pass_clips=costs['pass']['new_clips'],
        HEAD_incremental_clear_slots=r['new_costs_vs_body_only']['clear']['new_slots'],
        HEAD_incremental_pass_slots=r['new_costs_vs_body_only']['pass']['new_slots'])
    A.equal(actual,{k:str(v) for k,v in expected.items()},key+'/summary')


def run():
    started = time.monotonic()
    audit = OUT/'audit'
    if (audit/'PLAN.json').exists(): raise FileExistsError('Preserve original audit attempt')
    paths = [OUT/name for name in ('PLAN.json','metrics.json','calibrations.json','cal_partition.json','summary.csv','ledger.csv',
        'cal_grades.npz','validation_grades.npz','receipt.json')]
    C.save(audit/'PLAN.json',dict(task='INDEPENDENT_HEAD_FRACTION_CALIBRATION_AUDIT',
        budget_CPU_command_wall_seconds=90,GPU_seconds=0,fit=0,prediction=0,new_raw=0,
        scope='All36 cells,9 all-tie cuts, grades, independent metrics and110592 ledger rows',
        definitions='Only cal9/11 for actual added union cost; all policy HEAD additions permitted regardless BODY silence',
        producer_imported=False,source_sha256=C.sha(Path(__file__)),
        independent_helpers_sha256=C.sha(Path(A.__file__)),
        inputs_sha256={str(p.relative_to(C.ROOT)):C.sha(p) for p in paths}))
    try:
        plan = C.read(OUT/'PLAN.json')
        for path,digest in plan['inputs_sha256'].items(): A.equal(C.sha(C.ROOT/path),digest,'input/'+path)
        producer = Path(__file__).with_name('cnh_graded_peak_head_cal_dev.py')
        A.equal(C.sha(producer),plan['source_sha256'],'producer_source')
        A.equal(plan['seeds'],list(G.SEEDS),'seeds')
        A.equal(plan['policies'],list(POLICIES),'policies')
        A.equal(plan['fractions'],list(FRACTIONS),'declared_fractions')
        data = C.load()
        thresholds = C.read(C.PARENT/'thresholds.json')
        parent_cuts = C.read(JOINT/'calibrations.json')
        cuts,published = C.read(OUT/'calibrations.json'),C.read(OUT/'metrics.json')
        previous_metrics = C.read(BODY/'metrics.json')
        mask,fit_ids,cal_ids = independent_cal_mask(data['cal']['rows'])
        A.equal(fit_ids,[8,10],'fit_ids')
        A.equal(cal_ids,[9,11],'cal_ids')
        A.equal(C.read(OUT/'cal_partition.json'),dict(fit_background_ids=fit_ids,calibrate_background_ids=cal_ids),
            'cal_partition')
        # Construct thresholds once from calibration. Validation never enters
        # candidate breakpoints, old contribution, caps or feasible-cut search.
        rebuilt,cut_records = {},{}
        for si,seed in enumerate(G.SEEDS):
            d=data['cal'];th=thresholds[str(seed)]
            ordinary=d['candidates'][0,si]
            strong=(d['m3']>=E.OLD_RAISED)|(d['local']>=E.OLD_LOCAL)|(ordinary>=th['addition'])
            base=np.where(strong,2,np.where(ordinary>=th['single'],1,0)).astype(np.int8)
            with np.load(JOINT/'cal_scores.npz',allow_pickle=False) as ar:
                score=ar['scores'][ar['keys'].tolist().index(f'{seed}/score_current')]
            parent=parent_cuts[f'{seed}/score_current/c15_p64']
            A.equal(parent['nonbinding'],False,f'{seed}/finite_parent_required')
            pt=-np.inf if parent['nonbinding'] else parent['theta']
            body=base.copy();body[...,1][(base[...,1]==0)&np.isfinite(score[...,1])&(score[...,1]>=pt)]=1
            eligible=(base[...,0]==0)&np.isfinite(score[...,0])&(score[...,0]>=pt)&~(body>0).any(-1)
            clear=mask&(d['category']=='clear').all(1)
            passed=mask&(d['category']=='pass').any(1)&~(d['category']=='contact').any(1)
            clear_values=score[...,0][eligible&clear[:,None,None]]
            pass_values=score[...,0][eligible&passed[:,None,None]]
            for fraction,policy in zip(FRACTIONS,POLICIES[3:]):
                record=independent_cutoff(score[...,0],eligible,d['category'],mask,pt,fraction)
                key=f'{seed}/{policy}';rebuilt[key]=record['theta'];cut_records[key]=record
                check_cut(cuts[key],record,fraction,pt,fit_ids,cal_ids,key,clear_values,pass_values,
                    int(clear.sum())*4*13,int(passed.sum())*4*13)
        A.equal(set(cuts),set(rebuilt),'complete_calibration_keys')
        summary_csv,ledger_csv=A.read_csv(OUT/'summary.csv'),A.read_csv(OUT/'ledger.csv')
        A.equal(len(summary_csv),36,'summary_rows')
        A.equal(len(ledger_csv),110592,'ledger_rows')
        ledger_by_key,summary_by_key={},{}
        for i,row in enumerate(ledger_csv):
            key=(row['split'],int(row['seed']),row['policy'],int(row['scene']),int(row['replica']),row['height'])
            assert key not in ledger_by_key,('Duplicate ledger',key)
            ledger_by_key[key]=(i,row)
        for row in summary_csv:
            key=(row['split'],int(row['seed']),row['policy'])
            assert key not in summary_by_key,('Duplicate summary',key)
            summary_by_key[key]=row
        used_ledger,expected_keys=set(),set()
        for split,d in data.items():
            A.equal(d['m3'].shape,(384,4,13,2),split+'/population')
            category=d['category']
            with np.load(JOINT/f'{split}_scores.npz',allow_pickle=False) as ar:
                score_archive=dict(zip(ar['keys'].tolist(),ar['scores']))
            with np.load(BODY/f'{split}_grades.npz',allow_pickle=False) as ar:
                old_grade=dict(zip(ar['keys'].tolist(),ar['grades']))
            with np.load(OUT/f'{split}_grades.npz',allow_pickle=False) as ar:
                A.equal(ar['scene_ids'],d['scene_ids'],split+'/scene_ids')
                A.equal(ar['category'],category,split+'/category')
                A.equal(ar['keys'].tolist(),[f'{seed}/{p}' for seed in G.SEEDS for p in POLICIES],split+'/grade_keys')
                actual_grades=dict(zip(ar['keys'].tolist(),ar['grades']))
            for si,seed in enumerate(G.SEEDS):
                if time.monotonic()-started>=90: raise TimeoutError('Audit90s cap')
                th=thresholds[str(seed)];ordinary=d['candidates'][0,si]
                old=(d['m3']>=E.OLD_RAISED)|(d['local']>=E.OLD_LOCAL)
                strong=old|(ordinary>=th['addition'])
                base=np.where(strong,2,np.where(ordinary>=th['single'],1,0)).astype(np.int8)
                score=score_archive[f'{seed}/score_current']
                parent=parent_cuts[f'{seed}/score_current/c15_p64']
                pt=-np.inf if parent['nonbinding'] else parent['theta']
                add=(base==0)&np.isfinite(score)&(score>=pt)
                both=base.copy();both[add]=1
                body=base.copy();body[...,1][add[...,1]]=1
                grades=dict(baseline=base,both=both,body_only=body)
                for policy in POLICIES[3:]:
                    grade=body.copy()
                    grade[...,0][(base[...,0]==0)&np.isfinite(score[...,0])&(score[...,0]>=rebuilt[f'{seed}/{policy}'])]=1
                    grades[policy]=grade
                refs=dict(prior_light=base>0,ordinary_OR=strong,M3=d['m3']>=E.M3_THETA,old_fusion=old)
                before_full,before_timely=A.clock(base>0),A.clock(base>0,11)
                both_full,both_timely=A.clock(both>0),A.clock(both>0,11)
                for policy in POLICIES:
                    key=f'{split}/{seed}/{policy}';expected_keys.add(key)
                    grade=grades[policy]
                    A.equal(actual_grades[f'{seed}/{policy}'],grade,key+'/grades')
                    A.equal(grade==2,strong,key+'/strong')
                    A.equal(grade[base>0],base[base>0],key+'/retained_baseline')
                    if policy!='baseline': A.equal(grade[...,1],both[...,1],key+'/complete_BODY')
                    if policy in POLICIES[3:]:
                        assert np.all(grade<=both),key+'/subset_of_both'
                        assert np.all(grade>=body),key+'/superset_of_body'
                    r=A.describe(grade,d,refs)
                    r['paired_vs_both']=A.paired(both>0,grade>0,category)
                    r['physical_paired_vs_baseline']=A.physical_pair(base>0,grade>0,category)
                    r['physical_paired_vs_both']=A.physical_pair(both>0,grade>0,category)
                    r['physical_contact_timely']=int((A.joint_clock(grade>0,category)>=0).sum())
                    r['addition_costs']=A.addition_costs((grade>0)&(base==0),category)
                    r['new_costs_vs_baseline']=A.new_costs(base>0,grade>0,category)
                    r['new_costs_vs_body_only']=A.new_costs(body>0,grade>0,category)
                    A.equal(published[key],r,key+'/metrics')
                    if policy in POLICIES[:3]:
                        A.equal(grade,old_grade[f'{seed}/{policy}'],key+'/previous_control_grades')
                        previous={name:value for name,value in r.items() if name!='new_costs_vs_body_only'}
                        A.equal(previous,previous_metrics[key],key+'/previous_control_metrics')
                    check_summary(summary_by_key[split,seed,policy],split,seed,policy,r,key)
                    clocks=dict(first_any=A.clock(grade>0),first_timely=A.clock(grade>0,11),
                        first_strong=A.clock(grade==2),first_light=A.clock(grade==1),
                        before_first_any=before_full,before_first_timely=before_timely,
                        both_first_any=both_full,both_first_timely=both_timely)
                    for n,row in enumerate(d['rows']):
                        for k in range(4):
                            for q,height in enumerate(E.HEIGHTS):
                                identity=(split,seed,policy,int(d['scene_ids'][n]),k,height)
                                index,actual=ledger_by_key[identity];used_ledger.add(index)
                                expected=dict(split=split,seed=seed,policy=policy,scene=int(d['scene_ids'][n]),
                                    replica=k,height=height,category=category[n,q],shape_family=row['shape_family'],
                                    background_family=row['background_family'],
                                    **{name:int(value[n,k,q]) for name,value in clocks.items()})
                                A.equal(actual,{k:str(v) for k,v in expected.items()},key+'/ledger/'+str(index))
        A.equal(set(published),expected_keys,'metric_keys')
        A.equal(len(used_ledger),len(ledger_csv),'all_ledger_rows_checked')
        for path,digest in plan['inputs_sha256'].items(): A.equal(C.sha(C.ROOT/path),digest,'post_input/'+path)
        receipt=C.read(OUT/'receipt.json')
        for name,value in dict(status='COMPLETE',cells=36,cuts=9,ledger_rows=110592,GPU_seconds=0,fit=0,prediction=0,new_raw=0,
            source_sha256=C.sha(producer)).items(): A.equal(receipt[name],value,'receipt/'+name)
        result=dict(status='PASS',cells=36,ledger_rows=110592,summary_rows=36,calibrations=9,
            scalar_and_array_checks=A.CHECKS,seconds=time.monotonic()-started,GPU_seconds=0,
            source_sha256=C.sha(Path(__file__)),inputs_sha256_verified=len(plan['inputs_sha256']),
            calibration_breakpoints_examined={key:r['breakpoints_examined'] for key,r in cut_records.items()},
            independent='Producer never imported; independently enumerated all score-tie breakpoints; own clock/cost/metrics')
        C.save(audit/'result.json',result);print(json.dumps(result,ensure_ascii=False))
    except BaseException as error:
        C.save(audit/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-started,
            scalar_and_array_checks=A.CHECKS))
        raise


if __name__=='__main__': run()
