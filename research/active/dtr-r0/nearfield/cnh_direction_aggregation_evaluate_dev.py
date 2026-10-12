"""Frozen raw-direction aggregation, then inherited smoothing/calibration.

The replay manifest contains individual arm/seed/split/absolute-angle NPZs:
raw[19968,2], original_shape[384,4,13,2], scalar arm/seed/split/angle_degrees.
Only ideal cal selects each shared HEAD/BODY threshold. All validation cells
and original truth are retained. This evaluator performs no model inference.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path
import time

import numpy as np
import cnh_counterfactual_eval_dev as E
import cnh_pass_workpoints_dev as W

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'artifacts.local/work/cnh-direction-aggregation-dev-20261009'
PREVIOUS=ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009'
CONTINUATION=ROOT/'artifacts.local/work/cnh-pass-boundary-dev-20261009'
ARMS=('control','weak_pass')
SEEDS=(2026100955,2026100956,2026100957)
AGGREGATORS=('single','max','weighted')
BRANCHES=('ideal','yaw_plus3')
RELATIVE=np.array([-3,0,3],np.int64)
WEIGHTS=np.array([.25,.5,.25],np.float64)

def read(path):return E.read_json(path)
def save(path,value):W.save(path,value)
def aggregate(raw,method):
    """Direction is the last axis; output retains original frame/query axes."""
    raw=np.asarray(raw,np.float64)
    if raw.shape[-3:]!=(13,2,3) or not np.isfinite(raw).all():
        raise ValueError('Finite raw[scene,K,13,2,relative-direction3] required')
    if method=='single':return raw[...,1]
    if method=='max':return raw.max(-1)
    if method=='weighted':return np.sum(raw*WEIGHTS,axis=-1)
    raise ValueError('Unknown frozen aggregator')

def highest_direction(raw):
    """Raw argmax only for description. Whole ties prefer center, minus, plus."""
    priority=np.array([1,0,2],np.int64)
    return priority[np.argmax(np.asarray(raw)[...,priority],axis=-1)]

def validate_raw(archive,entry):
    for key in ('arm','seed','split','angle_degrees','checkpoint_sha256'):
        if key not in archive or np.asarray(archive[key]).item()!=entry[key]:
            raise ValueError(('Replay identity mismatch',key))
    shape=tuple(np.asarray(archive['original_shape'],dtype=np.int64).tolist())
    raw=archive['raw']
    if shape!=(384,4,13,2) or raw.shape!=(int(np.prod(shape[:-1])),2) or not np.isfinite(raw).all():
        raise ValueError('Complete finite replay rows in scene→replica→frame order required')
    if 'original_shape' in entry and entry['original_shape']!=list(shape):
        raise ValueError('Manifest and replay shape differ')
    return raw.reshape(shape).astype(np.float64)

def load_manifest(path,plan,check):
    manifest=read(path);base=path.parent
    if manifest['arm_names']!=list(ARMS) or manifest['seed_ids']!=list(SEEDS):
        raise ValueError('All frozen arms and seeds must be present')
    if manifest['angles_by_split']!=plan['replay_angles']:
        raise ValueError('Replay directions differ from frozen PLAN')
    jobs={};sources={path}
    expected={(a,s,split,angle) for a in ARMS for s in SEEDS
              for split in ('cal','validation') for angle in plan['replay_angles'][split]}
    for entry in manifest['outputs']:
        check();key=(entry['arm'],int(entry['seed']),entry['split'],int(entry['angle_degrees']))
        if key in jobs or key not in expected:raise ValueError('Duplicate/unexpected replay job')
        file=(base/entry['path']).resolve()
        if W.sha(file)!=entry['sha256']:raise ValueError(('Replay SHA mismatch',file))
        checkpoint=CONTINUATION/'models'/f'{key[0]}_seed{key[1]}.pt'
        if W.sha(checkpoint)!=entry['checkpoint_sha256']:raise ValueError('Replay checkpoint binding differs')
        sources.add(checkpoint)
        with np.load(file,allow_pickle=False) as archive:jobs[key]=validate_raw(archive,entry)
        sources.add(file)
        if key[-1] in (0,3):
            suffix='' if key[-1]==0 else '_yaw_plus3'
            frozen=CONTINUATION/'scores'/f'{key[0]}_seed{key[1]}_{key[2]}{suffix}.npz'
            with np.load(frozen,allow_pickle=False) as archive:original=archive['raw']
            np.testing.assert_array_equal(jobs[key],original)
            sources.add(frozen)
    if set(jobs)!=expected:raise ValueError('Every declared full-cohort replay job required')
    metadata=read(PREVIOUS/'scene_rows.json');sources.add(PREVIOUS/'scene_rows.json')
    cohorts={}
    for split in ('cal','validation'):
        physics=PREVIOUS/'data'/split/'physics.npz';sources.add(physics)
        with np.load(physics,allow_pickle=False) as archive:
            category,ids,fc=archive['category'],archive['scene_ids'],archive['frame_category']
        rows=metadata[split]
        if category.shape!=(384,2) or not np.isin(category,['contact','pass','clear']).all():raise ValueError('Frozen query truth required')
        np.testing.assert_array_equal(fc,np.broadcast_to(category[:,None],(384,13,2)))
        np.testing.assert_array_equal(ids,[r['scene_id'] for r in rows])
        groups={}
        for field in ('shape_family','background_family','placement'):
            for value in sorted({str(r[field]) for r in rows}):
                label=('family' if field=='shape_family' else field)+':'+value
                groups[label]=np.array([str(r[field])==value for r in rows])
        cohorts[split]=dict(category=category,scene_ids=ids,rows=rows,groups=groups)
    baselines={}
    for split,branch in (('cal','ideal'),('validation','ideal'),('validation','yaw_plus3')):
        file=PREVIOUS/'baselines'/f'{split}_{branch}.npz';sources.add(file)
        with np.load(file,allow_pickle=False) as archive:raw=archive['m3_raw']
        if raw.shape!=(384,4,13,2) or not np.isfinite(raw).all():raise ValueError('M3 cohort identity differs')
        baselines[split,branch]=E.smooth(raw)>=E.M3_THETA
    return jobs,cohorts,baselines,sources

def summary_row(dataset,arm,seed,agg,point,stratum,report):
    m=report['metrics'];row=dict(dataset=dataset,arm=arm,seed=seed,aggregator=agg,point=point,stratum=stratum,status=report.get('status','EVALUATED'))
    for q,height in enumerate(('HEAD','BODY')):
        row[height]=m['counts'][q];row[height+'_denominator']=m['contact_event_denominators'][q];row[height+'_late_only']=m['late_counts'][q]
        for name,p in report['paired'].items():
            for field in ('gain','loss','net'):row[height+'_'+field+'_vs_'+name]=p[q][field]
    for field in ('clear_slots','clear_denominator','clear_segments','clear_clips','clear_clip_denominator','pass_clips','pass_clip_denominator','physical_contact_any_height','physical_contact_denominator'):row[field]=m[field]
    return row

def describe(flags,cohort,references):
    c=cohort['category'];metrics,timely=E.summary(flags,c)
    paired={name:E.paired(E.summary(f,c)[1],timely,c,cohort['scene_ids']) for name,f in references.items()}
    groups={}
    for label,keep in cohort['groups'].items():
        gm,gt=E.summary(flags[keep],c[keep])
        groups[label]=dict(metrics=gm,paired={name:E.paired(E.summary(f[keep],c[keep])[1],gt,c[keep],cohort['scene_ids'][keep]) for name,f in references.items()})
    return dict(metrics=metrics,paired=paired,groups=groups),timely

def first_frame(flags,i,k,q,start=0,end=13):
    indices=np.flatnonzero(flags[i,k,start:end,q])
    return int(indices[0]+start+3) if indices.size else ''

def direction_rows(raw,cohort,arm,seed,branch):
    """Query-slot descriptions, never interpreted as clear cost or correction."""
    chosen=highest_direction(raw);c=cohort['category'];metadata=cohort['rows']
    kind=np.where((c=='contact').any(1),'contact',np.where((c=='pass').any(1),'pass','clear'))
    sides=np.array(['N/A' if not r['presence'] else 'negative_x' if r['side']==-1 else 'positive_x' for r in metadata])
    center=0 if branch=='ideal' else 3
    result=[]
    for q,height in enumerate(('HEAD','BODY')):
        for category,clipkind,side in itertools.product(('contact','pass','clear'),('contact','pass','clear'),('all','negative_x','positive_x','N/A')):
            keep=(c[:,q]==category)&(kind==clipkind)
            if side!='all':keep&=sides==side
            values=chosen[keep,:,:,q];den=int(values.size);timeden=int(values[...,:11].size)
            for direction,relative in enumerate(RELATIVE):
                count=int((values==direction).sum());timelycount=int((values[...,:11]==direction).sum())
                result.append(dict(dataset='validation/'+branch,arm=arm,seed=seed,height=height,query_category=category,clip_kind=clipkind,target_side=side,
                    absolute_degrees=int(center+relative),relative_degrees=int(relative),endpoint=bool(relative!=0),raw_slot_count=count,raw_slot_denominator=den,
                    timely_raw_slot_count=timelycount,timely_raw_slot_denominator=timeden,
                    unit='scene×replica×frame×query; raw f3..15 / timely-window raw f3..13',tie_priority='center,minus,plus',use='description only; no direction correction'))
    return result

def run(plan_path,manifest_path,output):
    started=time.monotonic();plan=read(plan_path)
    if (plan['arms'],plan['seeds'],plan['aggregators'])!=(list(ARMS),list(SEEDS),list(AGGREGATORS)):raise ValueError('Frozen full scope differs')
    if (plan['branch_angles'],plan['weights_relative_minus_center_plus'],plan['clear_cal_slot_cap'])!=({'ideal':[-3,0,3],'yaw_plus3':[0,3,6]},WEIGHTS.tolist(),65):raise ValueError('Frozen aggregation/calibration differs')
    cap=float(plan['allocations_cpu']['evaluation'])
    def check():
        if time.monotonic()-started>=cap:raise TimeoutError('Evaluation CPU allocation reached; preserve partials')
    output=output.resolve()
    if not output.is_relative_to((ROOT/'artifacts.local').resolve()):raise ValueError('Canonical artifact output required')
    if output.exists():raise FileExistsError('Preserve prior evaluation outputs')
    output.mkdir(parents=True)
    receipt=dict(status='RUNNING',plan_sha256=W.sha(plan_path),source_sha256=W.sha(__file__),training=0,inference=0)
    try:
        jobs,cohorts,baselines,sources=load_manifest(manifest_path.resolve(),plan,check)
        sources.update((plan_path.resolve(),Path(__file__).resolve(),Path(E.__file__).resolve(),Path(W.__file__).resolve(),Path(E.F.__file__).resolve()))
        cal=cohorts['cal'];cm=E.summary(baselines['cal','ideal'],cal['category'])[0]
        if (cm['clear_slots'],cm['clear_denominator'],cm['pass_clip_denominator'])!=(65,6656,256):raise ValueError('Frozen actual M3 cal denominator/cost differs')
        thresholds={};smoothed={};directional={}
        for arm in ARMS:
            thresholds[arm]={}
            for seed in SEEDS:
                thresholds[arm][str(seed)]={}
                for split,branch in (('cal','ideal'),('validation','ideal'),('validation','yaw_plus3')):
                    check();angles=plan['branch_angles'][branch]
                    raw=np.stack([jobs[arm,seed,split,int(a)] for a in angles],-1)
                    if split=='validation':directional[arm,seed,branch]=raw
                    for agg in AGGREGATORS:smoothed[arm,seed,agg,split,branch]=E.smooth(aggregate(raw,agg))
                for agg in AGGREGATORS:
                    bypoint={};thresholds[arm][str(seed)][agg]=bypoint
                    for rate in plan['workpoint_pass_rates']:
                        point=f'pass_{round(rate*100)}pct';score=smoothed[arm,seed,agg,'cal','ideal']
                        rec=W.select_threshold(score,cal['category'],clear_cap=65,pass_cap=int(np.floor(rate*cm['pass_clip_denominator'])))
                        rec.update(pass_rate_budget=rate,source='cal/ideal only',aggregator=agg)
                        if rec['threshold'] is not None:rec['calibration_metrics']=E.summary(score>=E.threshold(rec['threshold']),cal['category'])[0]
                        bypoint[point]=rec
        save(output/'thresholds.json',dict(status='COMPLETE',arms=thresholds,calibration_M3=cm,shared_query_threshold=True,selection='minimum feasible complete score-tie level; contact not optimized'))
        report=dict(status='COMPLETE',arms=list(ARMS),seeds=list(SEEDS),aggregators=list(AGGREGATORS),datasets={},
            scope='Consumed simulated Development; all predeclared seeds/points/aggregators retained; no validation selection',
            aggregation='raw first, then inherited smooth',weights=WEIGHTS.tolist(),threshold_origin='cal/ideal only; fixed on both validation branches')
        scalars=[];directions=[];event_count=0;cell_count=0
        header=['dataset','arm','seed','aggregator','point','scene','replica','height','placement','shape_family','background_family','target_side','rho','timely','late_only',
            'single_timely','M3_timely','gain_vs_single','loss_vs_single','gain_vs_M3','loss_vs_M3','first_alarm_frame','first_timely_frame','first_late_frame',
            'single_first_alarm_frame','M3_first_alarm_frame','peak_timely_frame','raw_argmax_absolute_at_peak','raw_argmax_relative_at_peak','raw_argmax_endpoint_at_peak']
        with (output/'event_ledger.csv').open('x',encoding='utf8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=header);writer.writeheader()
            for branch in BRANCHES:
                check();cohort=cohorts['validation'];c=cohort['category'];m3=baselines['validation',branch]
                dataset='validation/'+branch;dr=dict(baseline_M3=E.summary(m3,c)[0],arms={});report['datasets'][dataset]=dr
                for arm in ARMS:
                    dr['arms'][arm]={}
                    for seed in SEEDS:
                        dr['arms'][arm][str(seed)]={};raw=directional[arm,seed,branch];choices=highest_direction(raw)
                        directions.extend(direction_rows(raw,cohort,arm,seed,branch))
                        for agg in AGGREGATORS:
                            points={};dr['arms'][arm][str(seed)][agg]=points
                            for point,calibration in thresholds[arm][str(seed)][agg].items():
                                check();cell_count+=1
                                if calibration['threshold'] is None:
                                    points[point]=dict(status=calibration['status'],threshold=None,metrics=None,paired=None,groups={})
                                    continue
                                score=smoothed[arm,seed,agg,'validation',branch];flags=score>=E.threshold(calibration['threshold'])
                                single_cal=thresholds[arm][str(seed)]['single'][point]
                                single=smoothed[arm,seed,'single','validation',branch]>=E.threshold(single_cal['threshold'])
                                rr,timely=describe(flags,cohort,dict(single_same_arm_seed_point=single,M3=m3))
                                rr.update(status=calibration['status'],threshold=calibration['threshold']);points[point]=rr
                                scalars.append(summary_row(dataset,arm,seed,agg,point,'all',rr))
                                for group,gr in rr['groups'].items():scalars.append(summary_row(dataset,arm,seed,agg,point,group,gr))
                                st=E.summary(single,c)[1];mt=E.summary(m3,c)[1]
                                for i,k,q in np.argwhere(np.broadcast_to((c=='contact')[:,None,:],timely.shape)):
                                    meta=cohort['rows'][i];t=int(np.argmax(score[i,k,:11,q]));direction=int(choices[i,k,t,q]);relative=int(RELATIVE[direction]);a=bool(timely[i,k,q]);b=bool(st[i,k,q]);mm=bool(mt[i,k,q])
                                    writer.writerow(dict(dataset=dataset,arm=arm,seed=seed,aggregator=agg,point=point,scene=int(cohort['scene_ids'][i]),replica=int(k),height=('HEAD','BODY')[q],placement=meta['placement'],
                                        shape_family=meta['shape_family'],background_family=meta['background_family'],target_side='negative_x' if meta['side']==-1 else 'positive_x',rho=meta['rho'],timely=int(a),late_only=int(not a and flags[i,k,11:,q].any()),
                                        single_timely=int(b),M3_timely=int(mm),gain_vs_single=int(a and not b),loss_vs_single=int(b and not a),gain_vs_M3=int(a and not mm),loss_vs_M3=int(mm and not a),
                                        first_alarm_frame=first_frame(flags,i,k,q),first_timely_frame=first_frame(flags,i,k,q,end=11),first_late_frame=first_frame(flags,i,k,q,start=11),
                                        single_first_alarm_frame=first_frame(single,i,k,q),M3_first_alarm_frame=first_frame(m3,i,k,q),peak_timely_frame=t+3,
                                        raw_argmax_absolute_at_peak=relative+(0 if branch=='ideal' else 3),raw_argmax_relative_at_peak=relative,raw_argmax_endpoint_at_peak=int(relative!=0)))
                                    event_count+=1
        if cell_count!=108:raise AssertionError('Every validation policy cell must be retained')
        report.update(policy_cells=cell_count,event_ledger_rows=event_count,calibration_records=54,
            direction_definition='Raw argmax among available directions; center/minus/plus tie rule. Query-slot descriptive categories and scene clip_kind are separate from jointclear alarm costs; absent targets have side N/A')
        save(output/'metrics.json',report)
        for name,records in (('summary.csv',scalars),('direction_description.csv',directions)):
            with (output/name).open('x',encoding='utf8',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        check();save(output/'input_manifest.json',{str(p):dict(bytes=p.stat().st_size,sha256=W.sha(p)) for p in sorted(sources)})
        receipt.update(status='COMPLETE',policy_cells=108,calibration_records=54,event_ledger_rows=event_count,direction_description_rows=len(directions),input_jobs=len(jobs))
    except Exception as error:
        receipt.update(status='FAILED',error=repr(error));raise
    finally:
        receipt.update(seconds=time.monotonic()-started,cpu_allocation_seconds=cap)
        save(output/'run_receipt.json',receipt)
    print(json.dumps(receipt))
    return report

def check_fixture():
    started=time.monotonic()
    raw=np.zeros((1,1,13,2,3));raw[...,0,0,0]=10;raw[...,1,0,1]=10
    assert E.smooth(aggregate(raw,'max'))[0,0,1,0]==10
    assert np.stack([E.smooth(raw[...,j]) for j in range(3)],-1).max(-1)[0,0,1,0]<10
    np.testing.assert_array_equal(aggregate(raw,'single'),raw[...,1])
    np.testing.assert_array_equal(aggregate(raw,'weighted'),raw[...,0]*.25+raw[...,1]*.5+raw[...,2]*.25)
    ties=np.ones((1,1,13,2,3));assert (highest_direction(ties)==1).all()
    ties[...,1]=0;assert (highest_direction(ties)==0).all()
    fake=dict(raw=np.zeros((19968,2)),original_shape=np.array([384,4,13,2]),arm=np.array('control'),seed=np.array(SEEDS[0]),split=np.array('cal'),angle_degrees=np.array(-3),checkpoint_sha256=np.array('fixture'))
    entry=dict(arm='control',seed=SEEDS[0],split='cal',angle_degrees=-3,checkpoint_sha256='fixture')
    assert validate_raw(fake,entry).shape==(384,4,13,2)
    rejected=False
    try:validate_raw(fake,dict(entry,angle_degrees=3))
    except ValueError:rejected=True
    assert rejected
    rec=dict(status='PASS',checks=['raw-then-smooth max counterexample','single identity','fixed .25/.5/.25 weights','center-minus-plus ties','flatrow shape identity','angle identity rejection'],seconds=time.monotonic()-started,source_sha256=W.sha(__file__))
    target=OUT/'focused_evaluator_check.json'
    if target.exists():raise FileExistsError('Preserve focused check receipt')
    save(target,rec);print(json.dumps(rec))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,default=OUT/'PLAN.json');parser.add_argument('--manifest',type=Path,default=OUT/'scores/manifest.json')
    parser.add_argument('--output',type=Path,default=OUT/'evaluation');parser.add_argument('--check',action='store_true');args=parser.parse_args()
    check_fixture() if args.check else run(args.plan,args.manifest,args.output)
