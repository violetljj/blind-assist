"""Independent finite-window alert-chain audit; no new notification policy.

All stream clocks, sequences and summaries are read-only descriptions of frozen
saved grades. No upgrade does not mean free, and later frames remain unknown.
"""
import csv
import json
from pathlib import Path
import time

import numpy as np
import audit_cnh_graded_peak_body_only_dev as A
import cnh_graded_corridor_eval_dev as C
import cnh_graded_evidence_dev as G
import cnh_counterfactual_eval_dev as E

OUT = C.ROOT/'artifacts.local/work/cnh-graded-alert-chain-dev-20261010'
PARENT = C.ROOT/'artifacts.local/work/cnh-graded-peak-head-cal-dev-20261010'
POLICIES = ('baseline','both','head50')


def series_stats(values):
    active = [i for i,value in enumerate(values) if value]
    runs = []
    length = 0
    for value in values:
        if value: length += 1
        elif length:
            runs.append(length);length=0
    if length: runs.append(length)
    return dict(slots=len(active),segments=len(runs),longest_run_frames=max(runs,default=0),
        first=active[0]+3 if active else -1)


def scan(grade, baseline):
    grade = [int(value) for value in grade]
    baseline = [int(value) for value in baseline]
    any_flags = [value>0 for value in grade]
    light_flags = [value==1 for value in grade]
    strong_flags = [value==2 for value in grade]
    any_stats,light,strong = [series_stats(values) for values in (any_flags,light_flags,strong_flags)]
    before = series_stats([value>0 for value in baseline])['first']
    added = series_stats([value==1 and old==0 for value,old in zip(grade,baseline)])
    first = any_stats['first']
    initial = grade[first-3] if first>=0 else 0
    if initial==0: chain='silent'
    elif initial==2: chain='direct_strong'
    elif strong['first']>=0: chain='light_then_strong'
    else: chain='light_no_strong'
    upgrade = chain=='light_then_strong'
    interior = []
    if first>=0:
        last = max(i for i,value in enumerate(grade) if value>0)
        interior = [grade[i]==0 for i in range(first-3,last+1)]
    gaps = series_stats(interior)
    timely = first if 3<=first<=13 else -1
    before_timely = before if 3<=before<=13 else -1
    return dict(grades=''.join(str(value) for value in grade),chain=chain,initial_grade=initial,
        first_any=first,first_light=light['first'],first_strong=strong['first'],
        first_timely=timely,
        first_strong_timely=strong['first'] if 3<=strong['first']<=13 else -1,
        baseline_first_any=before,baseline_first_timely=before_timely,
        any_slots=any_stats['slots'],any_segments=any_stats['segments'],any_longest_frames=any_stats['longest_run_frames'],
        light_slots=light['slots'],light_segments=light['segments'],light_longest_frames=light['longest_run_frames'],
        strong_slots=strong['slots'],strong_segments=strong['segments'],strong_longest_frames=strong['longest_run_frames'],
        internal_silent_gaps=gaps['segments'],internal_silent_longest_frames=gaps['longest_run_frames'],
        light_to_strong_gap_frames=strong['first']-first if upgrade else -1,
        contiguous_upgrade=int(upgrade and all(grade[i]>0 for i in range(first-3,strong['first']-2))),
        added_light_first=added['first'],added_light_slots=added['slots'],added_light_segments=added['segments'],
        added_light_longest_frames=added['longest_run_frames'],
        timely_rescue=int(before_timely<0 and timely>=0),
        timely_earlier=int(before_timely>=0 and timely>=0 and timely<before_timely),
        new_alert_clip=int(before<0 and first>=0),
        earlier_existing_clip=int(before>=0 and first>=0 and first<before))


def aggregate(rows):
    summed = ('any_slots','light_slots','strong_slots','any_segments','light_segments','strong_segments',
        'internal_silent_gaps','added_light_slots','added_light_segments','timely_rescue','timely_earlier',
        'new_alert_clip','earlier_existing_clip')
    maxima = ('any_longest_frames','light_longest_frames','strong_longest_frames',
        'internal_silent_longest_frames','added_light_longest_frames')
    counts = ('active_streams','timely_streams','timely_strong_streams','silent','direct_strong',
        'light_then_strong','light_no_strong','contiguous_upgrades','gapped_upgrades',
        'fragmented_any_streams','fragmented_added_light_streams')
    result = {name:0 for name in summed+maxima+counts}
    result['streams'] = len(rows)
    gaps = []
    for row in rows:
        for name in summed: result[name] += row[name]
        for name in maxima: result[name] = max(result[name],row[name])
        result['active_streams'] += int(row['first_any']>=0)
        result['timely_streams'] += int(row['first_timely']>=0)
        result['timely_strong_streams'] += int(row['first_strong_timely']>=0)
        result[row['chain']] += 1
        result['contiguous_upgrades'] += row['contiguous_upgrade']
        result['gapped_upgrades'] += int(row['chain']=='light_then_strong' and row['contiguous_upgrade']==0)
        result['fragmented_any_streams'] += int(row['any_segments']>1)
        result['fragmented_added_light_streams'] += int(row['added_light_segments']>1)
        if row['light_to_strong_gap_frames']>=0: gaps.append(row['light_to_strong_gap_frames'])
    result['median_upgrade_gap_frames'] = float(np.median(gaps)) if gaps else None
    return result


def csv_values(row):
    return {name:'' if value is None else str(value) for name,value in row.items()}


def csv_map(rows, identity):
    mapped={}
    for row in rows:
        key=tuple(row[name] for name in identity)
        assert key not in mapped,('Duplicate CSV identity',identity,key)
        mapped[key]=row
    return mapped


def parent_metrics_check(rows, grade, category, report, label):
    # Parent counts and clocks were evaluated earlier using different code.
    # Compare query contact timing and joint costs without importing policies.
    for q,height in enumerate(E.HEIGHTS):
        selected=rows['query',height,'contact'];stats=aggregate(selected)
        A.equal(stats['timely_streams'],report['any']['counts'][q],label+'/contact_any/'+height)
        A.equal(stats['timely_strong_streams'],report['strong']['counts'][q],label+'/contact_strong/'+height)
        A.equal(stats['timely_rescue'],report['paired_any']['prior_light'][q]['rescue'],label+'/rescue/'+height)
        A.equal(stats['timely_earlier'],report['paired_any']['prior_light'][q]['earlier'],label+'/earlier/'+height)
    for category_name in ('clear','pass'):
        selected=rows['joint','ALL',category_name];stats=aggregate(selected)
        A.equal(stats['active_streams'],report['any'][category_name+'_clips'],label+'/'+category_name+'_anyclips')
        if category_name=='clear':
            A.equal(stats['any_slots'],report['any']['clear_slots'],label+'/clear_slots')
            A.equal(stats['any_segments'],report['any']['clear_segments'],label+'/clear_segments')
        for level in ('strong','light'):
            expected=report['joint_costs'][category_name][level]
            got=dict(slots=stats[level+'_slots'],slot_denominator=len(selected)*13,
                clips=sum(row[level+'_slots']>0 for row in selected),clip_denominator=len(selected),
                segments=stats[level+'_segments'],longest_run_frames=stats[level+'_longest_frames'])
            A.equal(got,expected,label+'/'+category_name+'/'+level+'_cost')
    A.equal(aggregate(rows['joint','ALL','contact'])['timely_streams'],
        report['any']['physical_contact_any_height'],label+'/legacy_any_height_contact')
    A.equal(int((A.joint_clock(grade>0,category)>=0).sum()),report['physical_contact_timely'],
        label+'/true_contact_query_joint')


def run():
    began=time.monotonic()
    audit=OUT/'audit'
    if (audit/'PLAN.json').exists(): raise FileExistsError('Preserve original audit attempt')
    paths=[OUT/name for name in ('PLAN.json','streams.csv','summary.csv','effect_summary.csv','receipt.json')]
    C.save(audit/'PLAN.json',dict(task='INDEPENDENT_FIXED_GRADE_ALERT_CHAIN_AUDIT',
        budget_CPU_command_wall_seconds=90,GPU_seconds=0,fit=0,prediction=0,new_raw=0,
        source_sha256=C.sha(Path(__file__)),helper_sha256=C.sha(Path(A.__file__)),producer_imported=False,
        definitions='Independent sequential run/gap scan; first nonzero grade determines chain; no new state rule',
        inputs_sha256={str(p.relative_to(C.ROOT)):C.sha(p) for p in paths}))
    try:
        plan=C.read(OUT/'PLAN.json')
        A.equal(plan['seeds'],list(G.SEEDS),'seeds')
        A.equal(plan['policies'],list(POLICIES),'policies')
        for path,digest in plan['inputs_sha256'].items(): A.equal(C.sha(C.ROOT/path),digest,'input/'+path)
        inherited=C.read(PARENT/'PLAN.json')['inputs_sha256']
        for path,digest in inherited.items(): A.equal(C.sha(C.ROOT/path),digest,'inherited/'+path)
        producer=Path(__file__).with_name('cnh_graded_alert_chain_dev.py')
        A.equal(C.sha(producer),plan['source_sha256'],'producer_source_sha256')
        stream_rows=A.read_csv(OUT/'streams.csv')
        summary_rows=A.read_csv(OUT/'summary.csv')
        effect_rows=A.read_csv(OUT/'effect_summary.csv')
        A.equal(len(stream_rows),82944,'stream_rows')
        A.equal(len(summary_rows),162,'summary_rows')
        A.equal(len(effect_rows),144,'effect_rows')
        identity=('split','seed','policy','level','height','scene','replica')
        streams=csv_map(stream_rows,identity)
        group_identity=('split','seed','policy','level','height','category')
        summaries=csv_map(summary_rows,group_identity)
        effects=csv_map(effect_rows,group_identity+('effect',))
        parent_reports=C.read(PARENT/'metrics.json')
        stream_used,group_used,effect_used=set(),set(),set()
        chains=('silent','direct_strong','light_then_strong','light_no_strong')
        for split in ('cal','validation'):
            with np.load(PARENT/f'{split}_grades.npz',allow_pickle=False) as ar:
                grades=dict(zip(ar['keys'].tolist(),ar['grades']))
                ids,category=ar['scene_ids'],ar['category']
            A.equal(category.shape,(384,2),split+'/category_shape')
            assert set(category.flatten())<=set(('contact','pass','clear'))
            assert len(set(ids.tolist()))==384
            joint_category=['contact' if 'contact' in row else 'pass' if 'pass' in row else 'clear' for row in category]
            for seed in G.SEEDS:
                base=grades[f'{seed}/baseline']
                A.equal(base.shape,(384,4,13,2),split+f'/{seed}/base_shape')
                for policy in POLICIES:
                    if time.monotonic()-began>=90: raise TimeoutError('Audit90s cap')
                    grade=grades[f'{seed}/{policy}']
                    label=f'{split}/{seed}/{policy}'
                    A.equal(grade==2,base==2,label+'/strong_slots_unchanged')
                    A.equal(A.clock(grade==2),A.clock(base==2),label+'/strong_first_unchanged')
                    groups={(level,height,cat):[] for level,height in
                        (('query','HEAD'),('query','BODY'),('joint','ALL')) for cat in ('contact','pass','clear')}
                    for n,scene in enumerate(ids):
                        for k in range(4):
                            items=[('query',E.HEIGHTS[q],category[n,q],grade[n,k,:,q],base[n,k,:,q]) for q in range(2)]
                            items.append(('joint','ALL',joint_category[n],grade[n,k].max(-1),base[n,k].max(-1)))
                            for level,height,cat,values,old_values in items:
                                result=scan(values,old_values)
                                assert result['chain'] in chains
                                row=dict(split=split,seed=seed,policy=policy,level=level,height=height,
                                    scene=int(scene),replica=k,category=cat,**result)
                                key=tuple(str(row[name]) for name in identity)
                                stream_used.add(key)
                                A.equal(streams[key],csv_values(row),label+'/stream/'+str(scene)+'/'+str(k)+'/'+height)
                                groups[level,height,cat].append(row)
                    for (level,height,cat),selected in groups.items():
                        row=dict(split=split,seed=seed,policy=policy,level=level,height=height,category=cat,
                            **aggregate(selected))
                        key=tuple(str(row[name]) for name in group_identity);group_used.add(key)
                        A.equal(summaries[key],csv_values(row),label+'/summary/'+str((level,height,cat)))
                        if level=='query' and cat=='contact':
                            selection=(('rescue','timely_rescue'),('earlier','timely_earlier'))
                        elif level=='joint' and cat!='contact':
                            selection=(('new_clip','new_alert_clip'),('earlier_clip','earlier_existing_clip'))
                        else: selection=()
                        for effect,flag in selection:
                            chosen=[r for r in selected if r[flag]==1]
                            er=dict(split=split,seed=seed,policy=policy,level=level,height=height,category=cat,
                                effect=effect,**aggregate(chosen))
                            ekey=key+(effect,);effect_used.add(ekey)
                            A.equal(effects[ekey],csv_values(er),label+'/effect/'+str((level,height,cat,effect)))
                    parent_metrics_check(groups,grade,category,parent_reports[label],label)
        A.equal(set(streams),stream_used,'all_stream_rows_checked')
        A.equal(set(summaries),group_used,'all_summary_rows_checked')
        A.equal(set(effects),effect_used,'all_effect_rows_checked')
        for path,digest in plan['inputs_sha256'].items(): A.equal(C.sha(C.ROOT/path),digest,'post_input/'+path)
        for path,digest in inherited.items(): A.equal(C.sha(C.ROOT/path),digest,'post_inherited/'+path)
        receipt=C.read(OUT/'receipt.json')
        for name,value in dict(status='COMPLETE',streams=82944,summaries=162,effect_summaries=144,
            source_sha256=C.sha(producer),GPU_seconds=0,fit=0,prediction=0,new_raw=0,grade_policy_changes=0).items():
            A.equal(receipt[name],value,'receipt/'+name)
        result=dict(status='PASS',streams=82944,summaries=162,effect_summaries=144,
            scalar_and_array_checks=A.CHECKS,seconds=time.monotonic()-began,GPU_seconds=0,
            source_sha256=C.sha(Path(__file__)),inputs_sha256_verified=len(plan['inputs_sha256']),
            inherited_hashes_verified=len(inherited),
            independent='Producer never imported; sequential clock/run/gap scans and independent aggregates; parent metrics reconciled')
        C.save(audit/'result.json',result);print(json.dumps(result,ensure_ascii=False))
    except BaseException as error:
        C.save(audit/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-began,
            scalar_and_array_checks=A.CHECKS))
        raise


if __name__=='__main__': run()
