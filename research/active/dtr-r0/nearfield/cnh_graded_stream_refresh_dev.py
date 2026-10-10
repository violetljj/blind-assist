"""Timestamp-aware offline CNH notification adapter; no Android dispatch.

Cached clips retain their fixed grades. Long signals below are engineering
fixtures, never new obstacle trajectories or detector evaluation samples.
"""
from pathlib import Path
import time
import numpy as np
import cnh_graded_corridor_eval_dev as C
import cnh_graded_evidence_dev as G
import cnh_graded_episode_merge_dev as P

ROOT=C.ROOT
OUT=ROOT/'artifacts.local/work/cnh-graded-stream-refresh-dev-20261010'
GRADES=P.PARENT
EPISODES=P.OUT
POLICIES=('refresh_none','refresh12')
REFRESH_MS=12000
STALE_MS=1500


class StreamNotifier:
    """Caller monotonic processing clock, fresh grade and availability only."""
    def __init__(self, refresh=False):
        self.refresh=refresh
        self.reset()

    def reset(self):
        self.peak=self.quiet=0
        self.last_now=self.last_strong=None

    def update(self, now_ms, grade, *, live=True, usable=True):
        if grade not in (0,1,2): raise ValueError('grade0/1/2 only')
        if not live or not usable:
            self.reset()
            return 0,'inactive' if not live else 'unusable'
        if self.last_now is not None:
            if now_ms<self.last_now or now_ms-self.last_now>STALE_MS:
                self.reset()
            elif now_ms==self.last_now:
                return 0,'duplicate'
        self.last_now=now_ms
        if grade==0:
            self.quiet+=1
            if self.quiet>1:
                self.peak=0
                self.last_strong=None
            return 0,'none'
        self.quiet=0
        code,reason=0,'none'
        if grade>self.peak:
            code,reason=grade,'onset' if self.peak==0 else 'upgrade'
            self.peak=grade
        elif self.refresh and grade==2 and now_ms-self.last_strong>=REFRESH_MS:
            code,reason=2,'refresh'
        if code==2: self.last_strong=now_ms
        return code,reason


def replay(values, policy, times=None):
    clock=np.arange(len(values))*200 if times is None else times
    machine=StreamNotifier(policy=='refresh12')
    records=[machine.update(int(t),int(v)) for t,v in zip(clock,values)]
    return np.array([r[0] for r in records],np.int8),[r[1] for r in records]


def fixtures():
    # Tuple inputs: now_ms, grade, live, usable, explicit reset before update.
    normal=lambda ts,vs:[(t,v,True,True,False) for t,v in zip(ts,vs)]
    strong=normal(range(0,30001,200),[2]*151)
    light=normal(range(0,30001,200),[1]*151)
    boundary=normal(range(0,12000,1000),[2]*12)+normal((11999,12000,12001),(2,2,2))
    return {
        'strong30s':strong,'light30s':light,'deadline':boundary,
        'upgrade_immediate':normal((0,200,400),(1,1,2)),
        'one_quiet':normal((0,200,400),(2,0,2)),
        'two_quiet':normal((0,200,400,600),(2,0,0,2)),
        'downgrade_at_deadline':normal(range(0,12000,1000),[2]*12)+normal((12000,12001),(1,2)),
        'unusable':[(0,2,True,True,False),(200,2,True,False,False),(400,2,True,True,False)],
        'stop':[(0,2,True,True,False),(200,2,False,True,False),(400,2,True,True,False)],
        'explicit_reset':[(0,2,True,True,False),(200,2,True,True,True)],
        'rollback':normal((1000,1200,100),(2,2,2)),
        'duplicate_quiet':normal((0,200,200,400),(2,0,0,2)),
        'stale_exact':normal((0,1500),(2,2)),
        'stale_exceeded':normal((0,1501),(2,2)),
    }


def run():
    began=time.monotonic()
    if (OUT/'PLAN.json').exists(): raise FileExistsError('Preserve stream attempt')
    source_paths=[GRADES/f'{s}_grades.npz' for s in ('cal','validation')]
    source_paths += [EPISODES/f'{s}_notifications.npz' for s in ('cal','validation')]
    source_paths += [EPISODES/'PLAN.json',EPISODES/'metrics.json',Path(P.__file__),
        ROOT/'core/assist/src/main/java/com/linnan/blindassist/feedback/HardwareDemoFeedbackPolicy.kt',
        ROOT/'app/src/main/java/com/linnan/blindassist/HardwareDemoActivity.kt']
    hashes={str(p.relative_to(ROOT)):C.sha(p) for p in source_paths}
    C.save(OUT/'PLAN.json',dict(task='CNH_GRADED_STREAM_REFRESH_DEV_20261010',
        authorization='User继续 timestamp reset and persistent strong reminder candidate',
        lane='EXPLORE consumed simulated Development plus separate synthetic engineering fixtures',
        budget=dict(main_CPU_command_wall_seconds=90,audit_CPU_command_wall_seconds=60,
                    interface_CPU_command_wall_seconds=60,integration_CPU_command_wall_seconds=120,total=330,GPU_seconds=0),
        backend='TASK_NOT_GPU_SUITABLE scalar timestamp state and fixed cached grade accounting',
        inputs_sha256=hashes,source_sha256=C.sha(Path(__file__)),
        policies=POLICIES,refresh_ms=REFRESH_MS,stale_ms=STALE_MS,grade_policies=P.GRADE_POLICIES,
        clock='Caller processing monotonic ms, cached nominal200ms proxy; not sensor timestamps or physical latency',
        state='episode1 grades0 twice reset; live/usable false and explicit reset clear; rollback or >1500ms observation gap clear before processing fresh sample; exact duplicate timestamp ignores sample; positive onset/upgrade immediate; only currentgrade2 refresh>=12000ms after last strong',
        decision_check='No cached clip spans12s: refresh gain cannot be evaluated; zero cache changes checks adapter retention. Long fixtures establish lifecycle mechanics, not persistent-obstacle benefit. Fixed one interval referenced from separate ToF Demo, no sweep.',
        stop='36 cached cells and14 lifecycle fixtures, relevant independent checks, report and delivery; no new raw/model/App/device or clip concatenation',
        limits='Lazy gap reset at next update is not active watchdog; upstream usable flag/reset needed without new samples. Main App event gate and Demo boolean feedback are distinct; no graded CNH integration. No distance closer cue or light refresh.'))
    try:
        metrics,summary={},[]
        for split in ('cal','validation'):
            with np.load(GRADES/f'{split}_grades.npz') as a:
                grades=dict(zip(a['keys'].tolist(),a['grades']));category=a['category'];scene_ids=a['scene_ids']
            with np.load(EPISODES/f'{split}_notifications.npz') as a:
                parent=dict(zip(a['keys'].tolist(),a['notifications']))
            saved,keys=[],[]
            for seed in G.SEEDS:
                for gp in P.GRADE_POLICIES:
                    grade=grades[f'{seed}/{gp}']
                    for policy in POLICIES:
                        if time.monotonic()-began>90: raise TimeoutError('main90s')
                        emitted=np.zeros_like(grade);refresh_count=0;reason_counts={}
                        group=[]
                        for n in range(len(scene_ids)):
                            for k in range(4):
                                for q,height in enumerate(('HEAD','BODY')):
                                    e,reasons=replay(grade[n,k,:,q],policy)
                                    emitted[n,k,:,q]=e
                                    for reason in reasons: reason_counts[reason]=reason_counts.get(reason,0)+1
                                    refresh_count+=reasons.count('refresh')
                                    row=dict(height=height,category=category[n,q],notifications=int((e>0).sum()),
                                        light=int((e==1).sum()),strong=int((e==2).sum()),refreshes=reasons.count('refresh'))
                                    group.append(row)
                        assert np.array_equal(emitted,parent[f'{seed}/{gp}/episode1'])
                        assert refresh_count==0
                        counts={}
                        for cat,mask in dict(contact=(category=='contact').any(-1),
                            pass_=(category=='pass').any(-1)&~(category=='contact').any(-1),clear=(category=='clear').all(-1)).items():
                            selected=emitted[mask];joint=selected.max(-1)
                            counts[cat.rstrip('_')]=dict(clip_denominator=int(mask.sum())*4,
                                query_notifications=int((selected>0).sum()),query_light=int((selected==1).sum()),query_strong=int((selected==2).sum()),
                                joint_notifications=int((joint>0).sum()),joint_light=int((joint==1).sum()),joint_strong=int((joint==2).sum()),joint_clips=int((joint>0).any(-1).sum()))
                        key=f'{seed}/{gp}/{policy}';keys.append(key);saved.append(emitted)
                        metrics[f'{split}/{key}']=dict(counts=counts,refresh_count=refresh_count,reasons=reason_counts,
                            paired_any_vs_grade=G.paired_timing(grade>0,emitted>0,category),
                            paired_strong_vs_grade=G.paired_timing(grade==2,emitted==2,category))
                        for height in ('HEAD','BODY'):
                            for cat in ('contact','pass','clear'):
                                rows=[r for r in group if r['height']==height and r['category']==cat]
                                summary.append(dict(split=split,seed=seed,grade_policy=gp,policy=policy,height=height,category=cat,
                                    streams=len(rows),**{f:sum(r[f] for r in rows) for f in ('notifications','light','strong','refreshes')}))
            np.savez_compressed(OUT/f'{split}_notifications.npz',keys=np.array(keys),notifications=np.array(saved),category=category,scene_ids=scene_ids)
        records,fixture_metrics=[],{}
        for name,inputs in fixtures().items():
            for policy in POLICIES:
                state=StreamNotifier(policy=='refresh12');events=[]
                for index,(now,grade,live,usable,reset) in enumerate(inputs):
                    if reset: state.reset()
                    code,reason=state.update(now,grade,live=live,usable=usable)
                    if code: events.append(dict(index=index,now_ms=now,code=code,reason=reason))
                    records.append(dict(fixture=name,policy=policy,index=index,now_ms=now,grade=grade,
                        live=live,usable=usable,explicit_reset=reset,notification=code,reason=reason))
                fixture_metrics[f'{name}/{policy}']=dict(samples=len(inputs),events=events)
        for path,digest in hashes.items(): assert C.sha(ROOT/path)==digest,path
        C.save(OUT/'metrics.json',metrics);C.save(OUT/'fixture_metrics.json',fixture_metrics)
        G.write_csv(OUT/'summary.csv',summary);G.write_csv(OUT/'fixtures.csv',records)
        C.save(OUT/'receipt.json',dict(status='COMPLETE',seconds=time.monotonic()-began,cells=len(metrics),
            query_streams=len(metrics)*384*4*2,summaries=len(summary),fixtures=len(fixtures()),fixture_rows=len(records),
            source_sha256=C.sha(Path(__file__)),GPU_seconds=0,fit=0,prediction=0,new_raw=0,cache_changes=0))
        print(f'COMPLETE {len(metrics)}cells {len(summary)}summaries {len(fixtures())}fixtures {time.monotonic()-began:.3f}s')
    except BaseException as error:
        C.save(OUT/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-began));raise


if __name__=='__main__': run()
