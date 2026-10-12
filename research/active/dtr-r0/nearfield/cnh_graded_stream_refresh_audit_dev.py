"""Independent timestamp/lifecycle adapter audit over unchanged grade caches.

Fixed lazy expiry1500ms and strong refresh12000ms; nominal200ms clip clocks are
proxies. Engineering fixtures are not additional scientific scenes or App proof.
"""
import json
from pathlib import Path
import time

import numpy as np
import audit_cnh_graded_peak_body_only_dev as A
import cnh_graded_corridor_eval_dev as C
import cnh_graded_evidence_dev as G
import cnh_counterfactual_eval_dev as E

OUT=C.ROOT/'artifacts.local/work/cnh-graded-stream-refresh-dev-20261010'
PARENT=C.ROOT/'artifacts.local/work/cnh-graded-peak-head-cal-dev-20261010'
EPISODE=C.ROOT/'artifacts.local/work/cnh-graded-episode-merge-dev-20261010'
DETECTORS=('baseline','both','head50')
POLICIES=('refresh_none','refresh12')


class Oracle:
    def __init__(self, refresh):
        self.refresh=refresh;self.reset()

    def reset(self):
        self.peak=0;self.quiet=0;self.last=None;self.strong_at=None
        self.counter=0;self.current_id=0

    def update(self, timestamp, grade, usable=True, live=True):
        if not usable or not live:
            self.reset();return 0,'inactive' if not live else 'unusable'
        if self.last is not None and timestamp==self.last: return 0,'duplicate'
        if self.last is not None and (timestamp<self.last or timestamp-self.last>1500): self.reset()
        self.last=timestamp
        if grade==0:
            self.quiet+=1
            if self.quiet>1:
                self.peak=0;self.current_id=0;self.strong_at=None
            return 0,'none'
        self.quiet=0
        reason='onset' if self.peak==0 else 'upgrade'
        if self.peak==0:
            self.counter+=1;self.current_id=self.counter
        if grade>self.peak:
            self.peak=grade
            if grade==2: self.strong_at=timestamp
            return grade,reason
        if grade==2 and self.refresh is not None and timestamp-self.strong_at>=self.refresh:
            self.strong_at=timestamp;return 2,'refresh'
        return 0,'none'


def fixture_inputs():
    def normal(times,grades): return [(t,g,True,True,False) for t,g in zip(times,grades)]
    return dict(strong30s=normal(range(0,30001,200),[2]*151),light30s=normal(range(0,30001,200),[1]*151),
        deadline=normal(range(0,12000,1000),[2]*12)+normal((11999,12000,12001),(2,2,2)),
        upgrade_immediate=normal((0,200,400),(1,1,2)),one_quiet=normal((0,200,400),(2,0,2)),
        two_quiet=normal((0,200,400,600),(2,0,0,2)),
        downgrade_at_deadline=normal(range(0,12000,1000),[2]*12)+normal((12000,12001),(1,2)),
        unusable=[(0,2,True,True,False),(200,2,True,False,False),(400,2,True,True,False)],
        stop=[(0,2,True,True,False),(200,2,False,True,False),(400,2,True,True,False)],
        explicit_reset=[(0,2,True,True,False),(200,2,True,True,True)],
        rollback=normal((1000,1200,100),(2,2,2)),duplicate_quiet=normal((0,200,200,400),(2,0,0,2)),
        stale_exact=normal((0,1500),(2,2)),stale_exceeded=normal((0,1501),(2,2)))


def fixture_audit(producer, rows, published):
    mapped={}
    for row in rows:
        key=(row['fixture'],row['policy'],int(row['index']))
        assert key not in mapped,key
        mapped[key]=row
    seen=set();events_by_key={}
    for name,inputs in fixture_inputs().items():
        for policy in POLICIES:
            oracle=Oracle(12000 if policy=='refresh12' else None)
            machine=producer.StreamNotifier(policy=='refresh12');events=[]
            for i,(now,grade,live,usable,reset) in enumerate(inputs):
                if reset: oracle.reset();machine.reset()
                expected=oracle.update(now,grade,usable,live)
                actual=machine.update(now,grade,live=live,usable=usable)
                A.equal(actual,expected,f'fixture_source/{name}/{policy}/{i}')
                code,reason=expected
                row=dict(fixture=name,policy=policy,index=i,now_ms=now,grade=grade,live=live,usable=usable,
                    explicit_reset=reset,notification=code,reason=reason)
                key=(name,policy,i);seen.add(key)
                A.equal(mapped[key],{k:str(v) for k,v in row.items()},f'fixture_csv/{key}')
                if code: events.append(dict(index=i,now_ms=now,code=code,reason=reason))
            key=f'{name}/{policy}';events_by_key[key]=events
            A.equal(published[key],dict(samples=len(inputs),events=events),'fixture_metrics/'+key)
    A.equal(set(mapped),seen,'all_fixture_rows_checked')
    A.equal(set(published),set(events_by_key),'all_fixture_metric_keys')
    # Explicit engineering boundaries, independently of tabulation.
    A.equal([(e['now_ms'],e['reason']) for e in events_by_key['strong30s/refresh12']],
        [(0,'onset'),(12000,'refresh'),(24000,'refresh')],'strong30s_boundaries')
    for policy in POLICIES:
        A.equal([e['now_ms'] for e in events_by_key['light30s/'+policy]],[0],'no_light_refresh/'+policy)
        A.equal([e['now_ms'] for e in events_by_key['stale_exact/'+policy]],[0],'1500_no_expiry/'+policy)
        A.equal([e['now_ms'] for e in events_by_key['stale_exceeded/'+policy]],[0,1501],'1501_expiry/'+policy)
        A.equal([e['now_ms'] for e in events_by_key['duplicate_quiet/'+policy]],[0],'duplicate_does_not_accumulate_quiet/'+policy)
    A.equal([e['now_ms'] for e in events_by_key['deadline/refresh12']],[0,12000],'12000_exact_refresh')
    A.equal(events_by_key['downgrade_at_deadline/refresh12'][-1]['now_ms'],12001,'light_preserves_strong_clock')
    # Additional API boundary cases, not extra scientific clips.
    extra=dict(duplicate_upgrade=[(0,1,True,True,False),(0,2,True,True,False),(200,2,True,True,False)],
        inactive_same_timestamp=[(0,2,True,True,False),(0,2,False,True,False),(0,2,True,True,False)],
        unusable_same_timestamp=[(0,2,True,True,False),(0,2,True,False,False),(0,2,True,True,False)])
    for name,inputs in extra.items():
        for policy in POLICIES:
            oracle=Oracle(12000 if policy=='refresh12' else None);machine=producer.StreamNotifier(policy=='refresh12')
            for i,(now,grade,live,usable,reset) in enumerate(inputs):
                expected=oracle.update(now,grade,usable,live)
                A.equal(machine.update(now,grade,live=live,usable=usable),expected,f'additional_fixture/{name}/{policy}/{i}')
    return dict(declared_fixtures=14,additional_API_fixtures=3,fixture_rows=len(rows),
        boundaries='strong refresh0/12000/24000ms; no light refresh;1500 retained/1501 expires; duplicate skips quiet and grade; lifecycle checked before duplicate')


def counts(emitted,category):
    result={}
    masks={'contact':(category=='contact').any(-1),'pass':(category=='pass').any(-1)&~(category=='contact').any(-1),
        'clear':(category=='clear').all(-1)}
    for name,mask in masks.items():
        selected=emitted[mask];joint=selected.max(-1)
        result[name]=dict(clip_denominator=int(mask.sum())*4,query_notifications=int((selected>0).sum()),
            query_light=int((selected==1).sum()),query_strong=int((selected==2).sum()),
            joint_notifications=int((joint>0).sum()),joint_light=int((joint==1).sum()),joint_strong=int((joint==2).sum()),
            joint_clips=int((joint>0).any(-1).sum()))
    return result


def run():
    began=time.monotonic();audit=OUT/'audit'
    if (audit/'PLAN.json').exists(): raise FileExistsError('Preserve original stream audit attempt')
    paths=[OUT/name for name in ('PLAN.json','metrics.json','summary.csv','fixtures.csv','fixture_metrics.json',
        'cal_notifications.npz','validation_notifications.npz','receipt.json')]
    C.save(audit/'PLAN.json',dict(task='INDEPENDENT_TIMESTAMP_STREAM_REFRESH_AUDIT',
        budget_CPU_command_wall_seconds=60,GPU_seconds=0,fit=0,prediction=0,new_raw=0,
        producer_use='Only explicit synthetic lifecycle fixtures, never actual-cache expectation or metrics',
        source_sha256=C.sha(Path(__file__)),helper_sha256=C.sha(Path(A.__file__)),
        inputs_sha256={str(p.relative_to(C.ROOT)):C.sha(p) for p in paths}))
    try:
        plan=C.read(OUT/'PLAN.json')
        A.equal(plan['policies'],list(POLICIES),'policies')
        A.equal(plan['grade_policies'],list(DETECTORS),'grade_policies')
        A.equal(plan['refresh_ms'],12000,'fixed_refresh_ms');A.equal(plan['stale_ms'],1500,'fixed_lazy_expiry_ms')
        for path,digest in plan['inputs_sha256'].items(): A.equal(C.sha(C.ROOT/path),digest,'input/'+path)
        inherited=C.read(PARENT/'PLAN.json')['inputs_sha256']
        for path,digest in inherited.items(): A.equal(C.sha(C.ROOT/path),digest,'inherited/'+path)
        producer_path=Path(__file__).with_name('cnh_graded_stream_refresh_dev.py')
        A.equal(C.sha(producer_path),plan['source_sha256'],'producer_source')
        import cnh_graded_stream_refresh_dev as fixture_producer
        fixture_report=fixture_audit(fixture_producer,A.read_csv(OUT/'fixtures.csv'),C.read(OUT/'fixture_metrics.json'))
        rows=A.read_csv(OUT/'summary.csv');A.equal(len(rows),216,'summary_rows')
        summary={}
        for row in rows:
            key=tuple(row[name] for name in ('split','seed','grade_policy','policy','height','category'))
            assert key not in summary,key
            summary[key]=row
        published=C.read(OUT/'metrics.json');parent_metrics=C.read(PARENT/'metrics.json')
        previous_metrics=C.read(EPISODE/'metrics.json');used_summary,metric_keys=set(),set()
        for split in ('cal','validation'):
            with np.load(PARENT/f'{split}_grades.npz',allow_pickle=False) as ar:
                grades=dict(zip(ar['keys'].tolist(),ar['grades']));ids,category=ar['scene_ids'],ar['category']
            with np.load(EPISODE/f'{split}_notifications.npz',allow_pickle=False) as ar:
                previous=dict(zip(ar['keys'].tolist(),ar['notifications']))
                A.equal(ar['scene_ids'],ids,split+'/episode_scene_ids');A.equal(ar['category'],category,split+'/episode_category')
            with np.load(OUT/f'{split}_notifications.npz',allow_pickle=False) as ar:
                keys=[f'{seed}/{detector}/{policy}' for seed in G.SEEDS for detector in DETECTORS for policy in POLICIES]
                A.equal(ar['keys'].tolist(),keys,split+'/keys');A.equal(ar['scene_ids'],ids,split+'/scene_ids')
                A.equal(ar['category'],category,split+'/category')
                saved=dict(zip(keys,ar['notifications']))
            for seed in G.SEEDS:
                for detector in DETECTORS:
                    grade=grades[f'{seed}/{detector}'];A.equal(grade.shape,(384,4,13,2),split+f'/{seed}/{detector}/shape')
                    for policy in POLICIES:
                        if time.monotonic()-began>=60: raise TimeoutError('Audit60s cap')
                        emitted=np.zeros_like(grade);reason_counts={};group={}
                        for n in range(384):
                            for k in range(4):
                                for q,height in enumerate(E.HEIGHTS):
                                    state=Oracle(12000 if policy=='refresh12' else None)
                                    for t,value in enumerate(grade[n,k,:,q]):
                                        code,reason=state.update(t*200,int(value));emitted[n,k,t,q]=code
                                        reason_counts[reason]=reason_counts.get(reason,0)+1
                                    group.setdefault((height,category[n,q]),[]).append(emitted[n,k,:,q])
                        key=f'{seed}/{detector}/{policy}';label=f'{split}/{key}';metric_keys.add(label)
                        A.equal(saved[key],emitted,label+'/saved_notifications')
                        A.equal(emitted,previous[f'{seed}/{detector}/episode1'],label+'/episode1_exact')
                        report=dict(counts=counts(emitted,category),refresh_count=reason_counts.get('refresh',0),reasons=reason_counts,
                            paired_any_vs_grade=A.paired(grade>0,emitted>0,category),
                            paired_strong_vs_grade=A.paired(grade==2,emitted==2,category))
                        A.equal(report['refresh_count'],0,label+'/cache_refresh_zero')
                        A.equal(published[label],report,label+'/metrics')
                        old=previous_metrics[f'{split}/{seed}/{detector}/episode1']
                        for kind in ('contact','pass','clear'):
                            for name,value in report['counts'][kind].items():
                                A.equal(value,old['counts'][kind][name],label+'/previous_'+kind+'/'+name)
                        parent=parent_metrics[f'{split}/{seed}/{detector}']
                        A.equal(A.clock(emitted>0),A.clock(grade>0),label+'/first_any')
                        A.equal(A.clock(emitted==2),A.clock(grade==2),label+'/first_strong')
                        for q,height in enumerate(E.HEIGHTS):
                            mask=category[:,q]=='contact'
                            A.equal(int((A.clock(emitted>0,11)[mask,:,q]>=0).sum()),parent['any']['counts'][q],label+'/parent_timely/'+height)
                            A.equal(int((A.clock(emitted==2,11)[mask,:,q]>=0).sum()),parent['strong']['counts'][q],label+'/parent_strong_timely/'+height)
                            for cat in ('contact','pass','clear'):
                                values=np.asarray(group.get((height,cat),[]))
                                row=dict(split=split,seed=seed,grade_policy=detector,policy=policy,height=height,category=cat,
                                    streams=len(values),notifications=int((values>0).sum()),light=int((values==1).sum()),
                                    strong=int((values==2).sum()),refreshes=0)
                                skey=tuple(str(row[name]) for name in ('split','seed','grade_policy','policy','height','category'));used_summary.add(skey)
                                A.equal(summary[skey],{name:str(value) for name,value in row.items()},label+'/summary/'+height+'/'+cat)
        A.equal(set(published),metric_keys,'all_metric_keys');A.equal(set(summary),used_summary,'all_summary_rows_checked')
        for path,digest in plan['inputs_sha256'].items(): A.equal(C.sha(C.ROOT/path),digest,'post_input/'+path)
        for path,digest in inherited.items(): A.equal(C.sha(C.ROOT/path),digest,'post_inherited/'+path)
        receipt=C.read(OUT/'receipt.json')
        for name,value in dict(status='COMPLETE',cells=36,query_streams=110592,summaries=216,fixtures=14,
            fixture_rows=fixture_report['fixture_rows'],source_sha256=C.sha(producer_path),GPU_seconds=0,
            fit=0,prediction=0,new_raw=0,cache_changes=0).items(): A.equal(receipt[name],value,'receipt/'+name)
        result=dict(status='PASS',cells=36,query_streams=110592,summaries=216,fixture_report=fixture_report,
            scalar_and_array_checks=A.CHECKS,seconds=time.monotonic()-began,GPU_seconds=0,
            inputs_sha256_verified=len(plan['inputs_sha256']),inherited_hashes_verified=len(inherited),
            source_sha256=C.sha(Path(__file__)),
            evidence='Cached200ms nominal proxy; synthetic lifecycle fixtures; no App/device/persistent obstacle effectiveness claim')
        C.save(audit/'result.json',result);print(json.dumps(result,ensure_ascii=False))
    except BaseException as error:
        C.save(audit/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-began,
            scalar_and_array_checks=A.CHECKS));raise


if __name__=='__main__': run()
