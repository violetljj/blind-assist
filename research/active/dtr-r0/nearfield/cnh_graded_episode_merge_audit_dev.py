"""Independent causal episode-notification audit over fixed cached grades.

Notifications describe emission counts, not occupancy/free or deployed app
behaviour. Input grading and query heights are retained without cross merging.
"""
import json
from pathlib import Path
import time

import numpy as np
import audit_cnh_graded_peak_body_only_dev as A
import cnh_graded_corridor_eval_dev as C
import cnh_graded_evidence_dev as G
import cnh_counterfactual_eval_dev as E

OUT = C.ROOT/'artifacts.local/work/cnh-graded-episode-merge-dev-20261010'
PARENT = C.ROOT/'artifacts.local/work/cnh-graded-peak-head-cal-dev-20261010'
DETECTORS = ('baseline','both','head50')
POLICIES = ('framewise','episode0','episode1')


def emit_stream(values, gap):
    """A sequential oracle on one query, only previously seen inputs."""
    peak,quiet,identifier=0,0,0
    emitted,ids,onsets,upgrades=[],[],[],[]
    for value in values:
        value=int(value);code=0;onset=0;upgrade=0
        if value==0:
            quiet+=1
            if quiet>gap: peak=0
        else:
            quiet=0
            if peak==0:
                identifier+=1;onset=1
            if value>peak:
                code=value;upgrade=int(peak>0);peak=value
        emitted.append(code);ids.append(identifier if peak>0 else 0)
        onsets.append(onset);upgrades.append(upgrade)
    return tuple(np.asarray(result,dtype=np.int16) for result in (emitted,ids,onsets,upgrades))


def boundary_fixtures(producer):
    cases=[
        ([1,0,1],0,[1,0,1]),([1,0,1],1,[1,0,0]),
        ([1,0,0,1],1,[1,0,0,1]),([1,2,1],0,[1,2,0]),
        ([2,1,0,1],0,[2,0,0,1]),([2,1,0,1],1,[2,0,0,0]),
        ([0,0,2,0,0,1],1,[0,0,2,0,0,1]),([0,0,0],1,[0,0,0])]
    for values,gap,expected in cases:
        emitted,ids,onsets,upgrades=emit_stream(values,gap)
        A.equal(emitted,np.asarray(expected),f'fixture/{values}/{gap}/codes')
        actual=producer.replay(np.asarray(values,np.int8),f'episode{gap}')
        for got,want in zip(actual,(emitted,ids,onsets,upgrades)):
            A.equal(got,want,f'producer_fixture/{values}/{gap}')
        for stop in range(1,len(values)+1):
            prefix=emit_stream(values[:stop],gap)
            for actual,full in zip(prefix,(emitted,ids,onsets,upgrades)):
                A.equal(actual,full[:stop],f'fixture/{values}/{gap}/prefix{stop}')
    return len(cases)


def prefix_oracle(values, gap, end):
    """A separately arranged vector oracle for all streams in one prefix."""
    values=np.asarray(values)
    peak=np.zeros(len(values),np.int8)
    quiet=np.zeros(len(values),np.int16)
    identifier=np.zeros(len(values),np.int16)
    codes=[];ids=[];onsets=[];upgrades=[]
    for t in range(end):
        current=values[:,t]
        quiet=np.where(current==0,quiet+1,0)
        peak=np.where((current==0)&(quiet>gap),0,peak)
        onset=(current>0)&(peak==0)
        upgrade=(current>peak)&(peak>0)
        identifier+=onset.astype(np.int16)
        code=np.where(current>peak,current,0)
        peak=np.maximum(peak,current)
        codes.append(code);ids.append(np.where(peak>0,identifier,0))
        onsets.append(onset.astype(np.int8));upgrades.append(upgrade.astype(np.int8))
    return tuple(np.stack(result,axis=1) for result in (codes,ids,onsets,upgrades))


def describe(grade,emitted,ids,onsets,upgrades):
    def first(values,strong=False,end=13):
        chosen=[i+3 for i,value in enumerate(values[:end]) if value==2 or (not strong and value>0)]
        return min(chosen) if chosen else -1
    ga,na=first(grade),first(emitted)
    gs,ns=first(grade,True),first(emitted,True)
    total=int((emitted>0).sum());strong=int((emitted==2).sum())
    return dict(grades=''.join(str(int(v)) for v in grade),notifications=''.join(str(int(v)) for v in emitted),
        episode_ids=','.join(str(int(v)) for v in ids),onset_flags=''.join(str(int(v)) for v in onsets),
        upgrade_flags=''.join(str(int(v)) for v in upgrades),notification_count=total,
        light_notifications=int((emitted==1).sum()),strong_notifications=strong,
        starts=int(onsets.sum()),upgrades=int(upgrades.sum()),repeat_notifications=total-int(na>=0),
        repeat_strong_notifications=strong-int(ns>=0),grade_first_any=ga,notification_first_any=na,
        grade_first_strong=gs,notification_first_strong=ns,grade_first_timely=first(grade,end=11),
        notification_first_timely=first(emitted,end=11),grade_first_strong_timely=first(grade,True,11),
        notification_first_strong_timely=first(emitted,True,11))


def aggregate(rows):
    summed=('notification_count','light_notifications','strong_notifications','starts','upgrades',
        'repeat_notifications','repeat_strong_notifications')
    counts=('active_streams','timely_streams','timely_strong_streams','first_any_changed','first_strong_changed')
    result={name:0 for name in summed+counts};result['streams']=len(rows)
    for row in rows:
        for name in summed: result[name]+=row[name]
        result['active_streams']+=int(row['notification_first_any']>=0)
        result['timely_streams']+=int(row['notification_first_timely']>=0)
        result['timely_strong_streams']+=int(row['notification_first_strong_timely']>=0)
        result['first_any_changed']+=int(row['grade_first_any']!=row['notification_first_any'])
        result['first_strong_changed']+=int(row['grade_first_strong']!=row['notification_first_strong'])
    return result


def csv_map(rows,identity):
    result={}
    for row in rows:
        key=tuple(row[name] for name in identity)
        assert key not in result,('Duplicate CSV identity',key)
        result[key]=row
    return result


def tensor_counts(emitted,onsets,upgrades,category):
    masks=dict(contact=(category=='contact').any(-1),
        **{'pass':(category=='pass').any(-1)&~(category=='contact').any(-1)},
        clear=(category=='clear').all(-1))
    result={}
    for name,mask in masks.items():
        selected=emitted[mask];joint=selected.max(-1)
        result[name]=dict(clip_denominator=int(mask.sum())*4,
            query_notifications=int((selected>0).sum()),query_light=int((selected==1).sum()),
            query_strong=int((selected==2).sum()),query_starts=int(onsets[mask].sum()),
            query_upgrades=int(upgrades[mask].sum()),joint_notifications=int((joint>0).sum()),
            joint_light=int((joint==1).sum()),joint_strong=int((joint==2).sum()),
            joint_clips=int((joint>0).any(-1).sum()))
    return result


def run():
    started=time.monotonic();audit=OUT/'audit'
    if (audit/'PLAN.json').exists(): raise FileExistsError('Preserve original audit attempt')
    paths=[OUT/name for name in ('PLAN.json','streams.csv','summary.csv','metrics.json',
        'cal_notifications.npz','validation_notifications.npz','receipt.json')]
    C.save(audit/'PLAN.json',dict(task='INDEPENDENT_FIXED_EPISODE_NOTIFICATION_AUDIT',
        budget_CPU_command_wall_seconds=90,GPU_seconds=0,fit=0,prediction=0,new_raw=0,
        producer_used='Only fixed boundary replay fixtures; never for expected real-stream states/metrics',
        source_sha256=C.sha(Path(__file__)),helper_sha256=C.sha(Path(A.__file__)),
        definitions='Independent scalar state replay plus independent vector replay of all13prefixes; original grades unchanged',
        inputs_sha256={str(path.relative_to(C.ROOT)):C.sha(path) for path in paths}))
    try:
        plan=C.read(OUT/'PLAN.json')
        A.equal(plan['seeds'],list(G.SEEDS),'seeds')
        A.equal(plan['grade_policies'],list(DETECTORS),'grade_policies')
        A.equal(plan['notification_policies'],list(POLICIES),'notify_policies')
        for path,digest in plan['inputs_sha256'].items(): A.equal(C.sha(C.ROOT/path),digest,'input/'+path)
        inherited=C.read(PARENT/'PLAN.json')['inputs_sha256']
        for path,digest in inherited.items(): A.equal(C.sha(C.ROOT/path),digest,'inherited/'+path)
        producer_source=Path(__file__).with_name('cnh_graded_episode_merge_dev.py')
        A.equal(C.sha(producer_source),plan['source_sha256'],'producer_source')
        # Executing producer replay only on explicit fixtures tests its boundary
        # implementation; all actual-data expected values below are independent.
        import cnh_graded_episode_merge_dev as fixture_producer
        fixtures=boundary_fixtures(fixture_producer)
        rows=A.read_csv(OUT/'streams.csv');summary_rows=A.read_csv(OUT/'summary.csv')
        A.equal(len(rows),165888,'stream_rows');A.equal(len(summary_rows),324,'summary_rows')
        identity=('split','seed','grade_policy','notify_policy','scene','replica','height')
        group_identity=('split','seed','grade_policy','notify_policy','height','category')
        streams=csv_map(rows,identity);summaries=csv_map(summary_rows,group_identity)
        published=C.read(OUT/'metrics.json');parent_metrics=C.read(PARENT/'metrics.json')
        used_streams,used_groups,metric_keys=set(),set(),set()
        matrices={};prefix_comparisons=0;immediate_light_onsets=0
        for split in ('cal','validation'):
            with np.load(PARENT/f'{split}_grades.npz',allow_pickle=False) as ar:
                grades=dict(zip(ar['keys'].tolist(),ar['grades']))
                ids,category=ar['scene_ids'],ar['category']
            with np.load(OUT/f'{split}_notifications.npz',allow_pickle=False) as ar:
                A.equal(ar['scene_ids'],ids,split+'/scene_ids')
                A.equal(ar['category'],category,split+'/category')
                keys=[f'{seed}/{detector}/{policy}' for seed in G.SEEDS for detector in DETECTORS for policy in POLICIES]
                A.equal(ar['keys'].tolist(),keys,split+'/keys')
                notify=dict(zip(keys,ar['notifications']));saved_ids=dict(zip(keys,ar['episode_ids']))
            A.equal(category.shape,(384,2),split+'/category_shape')
            for seed in G.SEEDS:
                base=grades[f'{seed}/baseline']
                for detector in DETECTORS:
                    grade=grades[f'{seed}/{detector}'];A.equal(grade.shape,(384,4,13,2),split+f'/{seed}/{detector}/shape')
                    A.equal(grade==2,base==2,split+f'/{seed}/{detector}/strong_input_preserved')
                    parent=parent_metrics[f'{split}/{seed}/{detector}']
                    flat=grade.transpose(0,1,3,2).reshape(-1,13)
                    controls={}
                    for policy in POLICIES:
                        if time.monotonic()-started>=90: raise TimeoutError('Audit90s cap')
                        key=f'{seed}/{detector}/{policy}';label=f'{split}/{key}';metric_keys.add(label)
                        expected=[np.zeros_like(grade) for _ in range(4)]
                        groups={(height,cat):[] for height in E.HEIGHTS for cat in ('contact','pass','clear')}
                        for n,scene in enumerate(ids):
                            for k in range(4):
                                for q,height in enumerate(E.HEIGHTS):
                                    values=grade[n,k,:,q]
                                    if policy=='framewise':
                                        active=(values>0).astype(np.int16)
                                        outputs=(values.copy(),np.where(values>0,np.cumsum(active),0),active,np.zeros_like(values))
                                    else: outputs=emit_stream(values,int(policy[-1]))
                                    for arr,value in zip(expected,outputs): arr[n,k,:,q]=value
                                    result=describe(values,*outputs)
                                    A.equal(result['notification_first_any'],result['grade_first_any'],label+'/first_any')
                                    A.equal(result['notification_first_strong'],result['grade_first_strong'],label+'/first_strong')
                                    # A positive grade1 onset is emitted on that same
                                    # frame, even when a later strong arrives.
                                    light_onset=(outputs[2]>0)&(values==1)
                                    A.equal(outputs[0][light_onset],values[light_onset],label+'/immediate_light_onset')
                                    immediate_light_onsets+=int(light_onset.sum())
                                    row=dict(split=split,seed=seed,grade_policy=detector,notify_policy=policy,
                                        scene=int(scene),replica=k,height=height,category=category[n,q],**result)
                                    skey=tuple(str(row[name]) for name in identity);used_streams.add(skey)
                                    A.equal(streams[skey],{name:str(value) for name,value in row.items()},label+'/stream/'+str(scene)+'/'+str(k)+'/'+height)
                                    groups[height,category[n,q]].append(row)
                        emitted,eids,onsets,upgrades=expected
                        A.equal(notify[key],emitted,label+'/saved_codes')
                        A.equal(saved_ids[key],eids,label+'/saved_episode_ids')
                        A.equal(emitted[grade==0],np.zeros_like(emitted[grade==0]),label+'/no_silent_emit')
                        assert np.all((emitted==0)|(emitted==grade)),label+'/only_current_input_grade_emits'
                        controls[policy]=emitted
                        if policy=='episode1':
                            for level in (1,2):
                                assert np.all(~(emitted==level)|(controls['episode0']==level)),label+'/gap1_subset_gap0'
                        # All thirteen full-cohort prefixes are independently
                        # replayed, including carried episode IDs on gap1 zero.
                        full=tuple(value.transpose(0,1,3,2).reshape(-1,13) for value in expected)
                        for stop in range(1,14):
                            if policy!='framewise':
                                prefix=prefix_oracle(flat,int(policy[-1]),stop)
                            else:
                                positive=(flat[:,:stop]>0).astype(np.int16)
                                prefix=(flat[:,:stop],np.where(positive>0,positive.cumsum(1),0),positive,np.zeros_like(positive))
                            for value,complete in zip(prefix,full):
                                A.equal(value,complete[:,:stop],label+f'/causal_prefix{stop}')
                            prefix_comparisons+=1
                        for (height,cat),selected in groups.items():
                            row=dict(split=split,seed=seed,grade_policy=detector,notify_policy=policy,
                                height=height,category=cat,**aggregate(selected))
                            gkey=tuple(str(row[name]) for name in group_identity);used_groups.add(gkey)
                            A.equal(summaries[gkey],{name:str(value) for name,value in row.items()},label+'/summary/'+height+'/'+cat)
                        counts=tensor_counts(emitted,onsets,upgrades,category)
                        report=dict(counts=counts,paired_any_vs_grade=A.paired(grade>0,emitted>0,category),
                            paired_strong_vs_grade=A.paired(grade==2,emitted==2,category))
                        A.equal(published[label],report,label+'/metrics')
                        # Same first clocks must reconcile timely detections and
                        # baseline paired timing with the saved detector report.
                        for q,height in enumerate(E.HEIGHTS):
                            group=aggregate(groups[height,'contact'])
                            A.equal(group['timely_streams'],parent['any']['counts'][q],label+'/parent_any/'+height)
                            A.equal(group['timely_strong_streams'],parent['strong']['counts'][q],label+'/parent_strong/'+height)
                        A.equal(A.paired(base>0,emitted>0,category),parent['paired_any']['prior_light'],label+'/parent_paired_any')
                        A.equal(A.paired(base>0,emitted==2,category),parent['paired_strong']['prior_light'],label+'/parent_paired_strong')
                        A.equal(counts['clear']['joint_clips'],parent['any']['clear_clips'],label+'/parent_clear_clips')
                        A.equal(counts['pass']['joint_clips'],parent['any']['pass_clips'],label+'/parent_pass_clips')
                        if policy=='framewise':
                            A.equal(counts['clear']['joint_notifications'],parent['any']['clear_slots'],label+'/framewise_clear_slots')
                            for kind in ('clear','pass'):
                                for level in ('light','strong'):
                                    A.equal(counts[kind]['joint_'+level],parent['joint_costs'][kind][level]['slots'],
                                        label+'/framewise_'+kind+'_'+level)
                        matrix=np.zeros((3,3),np.int64)
                        for old_level in range(3):
                            for new_level in range(3): matrix[old_level,new_level]=int(((grade==old_level)&(emitted==new_level)).sum())
                        matrices[label]=matrix.tolist()
        A.equal(set(streams),used_streams,'all_stream_rows_checked')
        A.equal(set(summaries),used_groups,'all_summary_rows_checked')
        A.equal(set(published),metric_keys,'all_metrics_checked')
        A.equal(prefix_comparisons,702,'all_prefix_cells_checked')
        for path,digest in plan['inputs_sha256'].items(): A.equal(C.sha(C.ROOT/path),digest,'post_input/'+path)
        for path,digest in inherited.items(): A.equal(C.sha(C.ROOT/path),digest,'post_inherited/'+path)
        receipt=C.read(OUT/'receipt.json')
        for name,value in dict(status='COMPLETE',streams=165888,summaries=324,cells=54,
            source_sha256=C.sha(producer_source),GPU_seconds=0,fit=0,prediction=0,new_raw=0,detector_grade_changes=0).items():
            A.equal(receipt[name],value,'receipt/'+name)
        C.save(audit/'notification_matrices.json',dict(rows='input_grade0/1/2',columns='notification0/1/2',matrices=matrices))
        result=dict(status='PASS',streams=165888,summaries=324,cells=54,fixtures=fixtures,
            all_stream_prefix_cells=prefix_comparisons,immediate_light_onsets_checked=immediate_light_onsets,
            scalar_and_array_checks=A.CHECKS,seconds=time.monotonic()-started,GPU_seconds=0,
            inputs_sha256_verified=len(plan['inputs_sha256']),inherited_hashes_verified=len(inherited),
            source_sha256=C.sha(Path(__file__)),
            producer_use='Boundary replay fixtures only; actual-data expectation independently recomputed')
        C.save(audit/'result.json',result);print(json.dumps(result,ensure_ascii=False))
    except BaseException as error:
        C.save(audit/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-started,
            scalar_and_array_checks=A.CHECKS));raise


if __name__=='__main__': run()
