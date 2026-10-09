"""CPU-only frozen center-anchored OR and fixed height hybrid evaluation.

No new inference or thresholds chosen from validation. The OR keeps the exact
previous center threshold and adds zero clear-slot/pass-clip ideal-cal cost.
Hybrid HEAD uses raw weighted directions; BODY uses raw center, followed by the
unchanged smoother and one shared, ideal-cal-only whole-tie threshold.
"""
import argparse
import csv
from pathlib import Path
import time

import numpy as np
import cnh_direction_aggregation_evaluate_dev as A

E, W = A.E, A.W
ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-direction-anchor-dev-20261009'
PRIOR = ROOT/'artifacts.local/work/cnh-direction-aggregation-dev-20261009'
POLICIES = ('center_single','anchored_zero_extra','head_weighted_body_center')
POINTS = ('pass_20pct','pass_30pct','pass_40pct')
PASS_CAPS = (51,76,102)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    A.save(path,value)


class Budget:
    def __init__(self, stage):
        self.start=time.monotonic();self.stage=stage;self.identifier=time.time_ns()
        self.plan=A.read(OUT/'PLAN.json')
        self.folder=OUT/'stage_receipts';self.folder.mkdir(parents=True,exist_ok=True)
        receipts=[A.read(p) for p in self.folder.glob('*.json')]
        self.previous=sum(float(r['seconds']) for r in receipts if r.get('budget_kind','cpu')=='cpu')
        self.cap=float(self.plan['cpu_stage_cap_seconds'])
        self.allocation=float(self.plan.get('allocations_cpu',{}).get('policies',200))
        self.own_previous=sum(float(r['seconds']) for r in receipts if r.get('stage','').startswith('anchor_policy'))

    def check(self):
        elapsed=time.monotonic()-self.start
        if self.previous+elapsed>=self.cap or self.own_previous+elapsed>=self.allocation:
            raise TimeoutError('Anchor policy cumulative CPU stage wall reached')

    def finish(self,status,**details):
        receipt=dict(stage='anchor_policy_'+self.stage,budget_kind='cpu',status=status,
                     seconds=time.monotonic()-self.start,previous_seconds=self.previous,
                     allocation_seconds=self.allocation,cap_seconds=self.cap,
                     source_sha256=W.sha(__file__),plan_sha256=W.sha(OUT/'PLAN.json'),**details)
        save(self.folder/f'runstage_cpu_policy_{self.identifier}.json',receipt)
        return receipt


def policy_scores(raw):
    raw=np.asarray(raw,np.float64)
    center=E.smooth(A.aggregate(raw,'single'))
    off=E.smooth(np.maximum(raw[...,0],raw[...,2]))
    hybrid_raw=A.aggregate(raw,'weighted').copy()
    hybrid_raw[...,1]=raw[...,1,1]
    return center,off,E.smooth(hybrid_raw)


def zero_extra_threshold(off,center_flags,category,scene_ids=None):
    """Unoccupied jointclear slot and wholly unoccupied pass clip, f3..15.

    Other-query center alarms occupy a jointclear slot; any center alarm in
    either query at any retained frame occupies a pass clip. No remaining
    clear/pass budget is spent. Contact scores do not select this threshold.
    """
    off=np.asarray(off,np.float64);center_flags=np.asarray(center_flags,bool)
    if off.shape!=center_flags.shape or off.ndim!=4 or off.shape[-2:]!=(13,2) or not np.isfinite(off).all():
        raise ValueError('Finite matching scores/flags [scene,K,13,2] required')
    if category.shape!=(off.shape[0],2):raise ValueError('Query category axes differ')
    scene_ids=np.arange(len(category)) if scene_ids is None else np.asarray(scene_ids)
    clear,passed=W.exposure(category)
    joint=off.max(-1);occupied=center_flags.any(-1)
    eligible_clear=clear[:,None,None]&~occupied
    eligible_pass=passed[:,None]&~occupied.any(-1)
    cv=joint[eligible_clear];pv=joint.max(-1)[eligible_pass]
    cm=float(cv.max()) if cv.size else None
    pm=float(pv.max()) if pv.size else None
    values=[v for v in (cm,pm) if v is not None]
    maximum=max(values) if values else None
    theta=float(np.nextafter(maximum,np.inf)) if values else np.inf
    limiting=[]
    for kind,value in (('jointclear_slot',cm),('pass_clip',pm)):
        if value is None:continue
        indices=np.argwhere(eligible_clear&(joint==value)) if kind=='jointclear_slot' else np.argwhere(eligible_pass&(joint.max(-1)==value))
        for ix in indices:
            if kind=='jointclear_slot':i,k,f=(int(v) for v in ix)
            else:
                i,k=(int(v) for v in ix);f=int(np.argmax(joint[i,k]))
            q=int(np.argmax(off[i,k,f]))
            limiting.append(dict(kind=kind,scene=int(scene_ids[i]),scene_index=i,replica=k,
                                 frame=f+3,height=('HEAD','BODY')[q],query_index=q,
                                 max_score=float(value),global_limiter=bool(value==maximum),
                                 tie_frames_in_clip=int(np.count_nonzero(joint[i,k]==value)) if kind=='pass_clip' else 1,
                                 center_occupied_slot=False,center_occupied_clip=bool(occupied[i,k].any()),
                                 cost_scope='joint queries and ALL13 frames f3..15',
                                 query_tie_rule='HEAD first; diagnostic only'))
    result=dict(status='CALIBRATED_ZERO_EXTRA_CAL_CLEAR_AND_PASS' if values else 'NO_ELIGIBLE_CAL_CLEAR_OR_PASS',
                threshold=E.record_threshold(theta),maximum_eligible_score=maximum,
                maximum_eligible_clear_score=cm,maximum_eligible_pass_score=pm,
                eligible_jointclear_slots=int(cv.size),eligible_pass_clips=int(pv.size),
                threshold_rule='nextafter(max eligible jointclear slots and center-unoccupied passclip all-frame/query maxima,+inf); +inf if none',
                zero_extra_scope='ideal-cal jointclear slots and passclips; f3..15 both queries',
                stricter_than_full_caps=True,limiting_exposures=limiting)
    union=center_flags|(off>=theta)
    before,_=E.summary(center_flags,category);after,_=E.summary(union,category)
    if not np.all(~center_flags|union):raise AssertionError('OR revoked a center alarm')
    if (before['clear_slots'],before['pass_clips'])!=(after['clear_slots'],after['pass_clips']):
        raise AssertionError('Zero-extra ideal-cal clear/pass costs changed')
    result.update(center_calibration_metrics=before,calibration_metrics=after)
    return result


def fixture():
    """Structural checks only; no experiment arrays or inference opened."""
    start=time.monotonic()
    category=np.array([['clear','clear'],['pass','clear'],['contact','clear']])
    off=np.zeros((3,1,13,2));center=np.zeros_like(off,dtype=bool)
    # A slot already occupied by the other query cannot impose an OR cost.
    center[0,0,0,1]=True;off[0,0,0,0]=1000
    off[0,0,12,0]=5
    # f15 must constrain passclip costs, despite lying outside timely f3..13.
    off[1,0,12,1]=7;off[2,0,4,0]=100
    record=zero_extra_threshold(off,center,category)
    assert E.threshold(record['threshold'])==np.nextafter(7.,np.inf)
    assert record['eligible_jointclear_slots']==12 and record['eligible_pass_clips']==1
    union=center|(off>=E.threshold(record['threshold']))
    assert np.all(~center|union) and union[2,0,4,0]
    assert not union[1].any() and E.summary(union,category)[0]['pass_clips']==0
    # Any center alarm, even f15/opposite query, occupies the whole pass clip.
    center[1,0,12,1]=True
    record=zero_extra_threshold(off,center,category)
    assert E.threshold(record['threshold'])==np.nextafter(5.,np.inf)
    center[:]=True
    inf=E.threshold(zero_extra_threshold(off,center,category)['threshold'])
    assert inf==np.inf
    np.testing.assert_array_equal(center|(off>=inf),center)
    raw=np.zeros((3,1,13,2,3));raw[...,0,0]=8;raw[...,0,2]=4;raw[...,1,0]=100
    central,offset,hybrid=policy_scores(raw)
    expected=E.smooth(np.stack((raw[...,0,0]*.25+raw[...,0,1]*.5+raw[...,0,2]*.25,raw[...,1,1]),-1))
    np.testing.assert_array_equal(hybrid,expected)
    assert np.all(hybrid[...,0]==3) and np.all(hybrid[...,1]==0)
    selected=W.select_threshold(hybrid,category,clear_cap=0,pass_cap=0)
    assert selected['clear_slots']==selected['pass_clips']==0
    rec=dict(status='PASS',seconds=time.monotonic()-start,source_sha256=W.sha(__file__),
             checks=['jointclear opposite-query occupancy','passclip f15 whole-query blocker','nextafter exact whole ties',
                     'occupied passclip wholeclip exclusion','no eligible => infinity','OR contains center',
                     'HEAD raw weighted/BODY raw center then smooth','one shared hybrid threshold joint clear/pass costs'],
             inference=0,experiment_data_opened=False)
    path=OUT/'focused_policy_check.json'
    if path.exists():raise FileExistsError('Preserve focused structural check')
    save(path,rec);print(rec)
    return rec


def run():
    budget=Budget('run');folder=OUT/'policies'
    try:
        if folder.exists():raise FileExistsError('Preserve completed/partial policy outputs')
        folder.mkdir()
        plan=budget.plan
        if (tuple(plan['arms'])!=A.ARMS or tuple(plan['seeds'])!=A.SEEDS
                or tuple(plan['policies'])!=POLICIES[1:] or tuple(plan['pass_caps'])!=PASS_CAPS
                or plan['workpoint_pass_rates']!=[.2,.3,.4] or plan['clear_cal_slot_cap']!=65
                or plan['weights_relative_minus_center_plus']!=[.25,.5,.25]):
            raise ValueError('Frozen all-model identities differ')
        # Freeze exact policy source before reading any scientific score arrays.
        snapshot=folder/'source_snapshot.py'
        snapshot.write_bytes(Path(__file__).read_bytes())
        sources_frozen={str(p):W.sha(p) for p in (Path(__file__),Path(A.__file__),Path(E.__file__),Path(W.__file__),Path(E.F.__file__))}
        save(folder/'source_snapshot.json',dict(files=sources_frozen,source_snapshot_sha256=W.sha(snapshot)))
        previous_plan=A.read(PRIOR/'PLAN.json')
        manifest_path=PRIOR/'scores/manifest.json'
        old_threshold_path=PRIOR/'evaluation/thresholds.json'
        old_thresholds=A.read(old_threshold_path)['arms']
        jobs,cohorts,baselines,sources=A.load_manifest(manifest_path,previous_plan,budget.check)
        sources.update((OUT/'PLAN.json',PRIOR/'PLAN.json',old_threshold_path,Path(__file__),
                        Path(A.__file__),Path(E.__file__),Path(W.__file__),Path(E.F.__file__)))
        scores={};thresholds={};limits=[]
        for arm in A.ARMS:
            thresholds[arm]={}
            for seed in A.SEEDS:
                thresholds[arm][str(seed)]={}
                for split,branch in (('cal','ideal'),('validation','ideal'),('validation','yaw_plus3')):
                    budget.check()
                    raw=np.stack([jobs[arm,seed,split,a] for a in previous_plan['branch_angles'][branch]],-1)
                    scores[arm,seed,split,branch]=policy_scores(raw)
                central,off,hybrid=scores[arm,seed,'cal','ideal']
                cal=cohorts['cal'];c=cal['category']
                for point,cap in zip(POINTS,PASS_CAPS):
                    old=old_thresholds[arm][str(seed)]['single'][point]
                    if old['threshold'] is None:raise ValueError('Frozen center threshold absent')
                    center_flags=central>=E.threshold(old['threshold'])
                    cm=E.summary(center_flags,c)[0]
                    if cm!=old['calibration_metrics']:raise ValueError('Saved center calibration metrics differ')
                    rec=zero_extra_threshold(off,center_flags,c,cal['scene_ids'])
                    rec.update(center_threshold=old['threshold'],center_source='exact prior single ideal-cal threshold, unchanged',
                               offscore='smooth(raw max of relative lower/upper); excludes middle',
                               clear_full_cap=65,pass_full_cap=cap,remaining_full_clear_headroom=65-cm['clear_slots'],
                               remaining_full_pass_headroom=cap-cm['pass_clips'])
                    for limit in rec['limiting_exposures']:
                        limits.append(dict(arm=arm,seed=seed,point=point,**limit))
                    mix=W.select_threshold(hybrid,c,clear_cap=65,pass_cap=cap)
                    if mix['threshold'] is not None:mix['calibration_metrics']=E.summary(hybrid>=E.threshold(mix['threshold']),c)[0]
                    mix.update(raw_hybrid='HEAD .25/.5/.25 directions; BODY center; then inherited smooth',shared_query_threshold=True)
                    thresholds[arm][str(seed)][point]=dict(center_single=old,anchored_zero_extra=rec,head_weighted_body_center=mix)
        save(folder/'thresholds.json',dict(status='COMPLETE',arms=thresholds,threshold_origin='ideal cal only; unchanged validation branches',
                                          policy_names=list(POLICIES),calibration_records=54))
        header=['dataset','arm','seed','policy','point','scene','replica','height','placement','shape_family','background_family','target_side','rho',
                'timely','late_only','center_timely','M3_timely','gain_vs_center','loss_vs_center','gain_vs_M3','loss_vs_M3',
                'first_alarm_frame','first_timely_frame','first_late_frame','center_first_alarm_frame','M3_first_alarm_frame']
        scalars=[];event_count=cell_count=0;datasets={}
        with (folder/'event_ledger.csv').open('x',newline='',encoding='utf8') as handle:
            writer=csv.DictWriter(handle,fieldnames=header);writer.writeheader()
            for branch in A.BRANCHES:
                cohort=cohorts['validation'];c=cohort['category'];m3=baselines['validation',branch]
                dataset='validation/'+branch;datasets[dataset]=dict(baseline_M3=E.summary(m3,c)[0],arms={})
                for arm in A.ARMS:
                    datasets[dataset]['arms'][arm]={}
                    for seed in A.SEEDS:
                        bypoint={};datasets[dataset]['arms'][arm][str(seed)]=bypoint
                        central,off,hybrid=scores[arm,seed,'validation',branch]
                        for point in POINTS:
                            recs=thresholds[arm][str(seed)][point]
                            center=central>=E.threshold(recs['center_single']['threshold'])
                            st=E.summary(center,c)[1];mt=E.summary(m3,c)[1]
                            bypolicy={};bypoint[point]=bypolicy
                            for policy in POLICIES:
                                budget.check();cell_count+=1;rec=recs[policy]
                                if rec['threshold'] is None:
                                    bypolicy[policy]=dict(status=rec['status'],metrics=None,paired=None,groups={});continue
                                if policy=='center_single':flags=center
                                elif policy=='anchored_zero_extra':flags=center|(off>=E.threshold(rec['threshold']))
                                else:flags=hybrid>=E.threshold(rec['threshold'])
                                if policy=='anchored_zero_extra' and not np.all(~center|flags):raise AssertionError('OR loses center flag')
                                report,timely=A.describe(flags,cohort,dict(center_same_arm_seed_point=center,M3=m3))
                                report.update(status=rec['status'],threshold=rec['threshold'])
                                if policy=='anchored_zero_extra' and any(row['loss'] for row in report['paired']['center_same_arm_seed_point']):
                                    raise AssertionError('OR loses center timely event')
                                bypolicy[policy]=report
                                row=A.summary_row(dataset,arm,seed,policy,point,'all',report);row['policy']=row.pop('aggregator');scalars.append(row)
                                for group,gr in report['groups'].items():
                                    row=A.summary_row(dataset,arm,seed,policy,point,group,gr);row['policy']=row.pop('aggregator');scalars.append(row)
                                for i,k,q in np.argwhere(np.broadcast_to((c=='contact')[:,None,:],timely.shape)):
                                    meta=cohort['rows'][i];a=bool(timely[i,k,q]);b=bool(st[i,k,q]);mm=bool(mt[i,k,q])
                                    writer.writerow(dict(dataset=dataset,arm=arm,seed=seed,policy=policy,point=point,
                                        scene=int(cohort['scene_ids'][i]),replica=int(k),height=('HEAD','BODY')[q],placement=meta['placement'],
                                        shape_family=meta['shape_family'],background_family=meta['background_family'],
                                        target_side='negative_x' if meta['side']==-1 else 'positive_x',rho=meta['rho'],
                                        timely=int(a),late_only=int(not a and flags[i,k,11:,q].any()),center_timely=int(b),M3_timely=int(mm),
                                        gain_vs_center=int(a and not b),loss_vs_center=int(b and not a),gain_vs_M3=int(a and not mm),loss_vs_M3=int(mm and not a),
                                        first_alarm_frame=A.first_frame(flags,i,k,q),first_timely_frame=A.first_frame(flags,i,k,q,end=11),
                                        first_late_frame=A.first_frame(flags,i,k,q,start=11),center_first_alarm_frame=A.first_frame(center,i,k,q),M3_first_alarm_frame=A.first_frame(m3,i,k,q)))
                                    event_count+=1
        if cell_count!=108 or event_count!=82944:raise AssertionError('Missing all-policy/contact event cells')
        for name,records in (('summary.csv',scalars),('calibration_tail_limits.csv',limits)):
            with (folder/name).open('x',newline='',encoding='utf8') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        save(folder/'metrics.json',dict(status='COMPLETE',policy_names=list(POLICIES),arms=list(A.ARMS),seeds=list(A.SEEDS),
              datasets=datasets,policy_cells=108,candidate_cells=72,center_baseline_cells=36,event_ledger_rows=event_count,
              calibration_records=54,scope='Consumed simulated Development; all fixed seeds/workpoints and original truth retained',
              zero_extra_scope='exact center ideal-cal jointclear slot/passclip cost; validation cost can increase; stricter than spending residual fullcaps',
              thresholds='Frozen prior center; eligible-cal-only OR nextafter; hybrid lowest feasible whole-tie idealcal level; one shared scalar per policy'))
        budget.check()
        if any(W.sha(p)!=digest for p,digest in sources_frozen.items()):raise ValueError('Scientific helper source changed during run')
        save(folder/'input_manifest.json',{str(p):dict(bytes=p.stat().st_size,sha256=W.sha(p)) for p in sorted(sources)})
        receipt=budget.finish('COMPLETE',policy_cells=108,candidate_cells=72,center_baseline_cells=36,event_ledger_rows=event_count,
                              calibration_records=54,tail_limit_rows=len(limits),input_jobs=len(jobs),inference=0,training=0)
        save(folder/'run_receipt.json',receipt)
        print(receipt);return receipt
    except Exception as error:
        receipt=budget.finish('FAILED',error=repr(error),inference=0,training=0)
        save(folder/'run_receipt.json',receipt)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=('check','run'))
    args=parser.parse_args()
    if args.stage=='check':
        budget=Budget('check')
        try:
            fixture();budget.check();print(budget.finish('COMPLETE',experiment_data_opened=False,inference=0))
        except Exception as error:
            budget.finish('FAILED',error=repr(error),experiment_data_opened=False,inference=0);raise
    else:run()
