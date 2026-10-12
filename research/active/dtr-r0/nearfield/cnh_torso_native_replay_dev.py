"""Same-native-trajectory frozen-M3 replay, future1.5m query and geometry.

The common sensor has torso-aligned proxy yaw, not measured head yaw. Raw native
positions retain metres; each scene is rendered once and shared across all arms.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import cnh_torso_bias_replay_dev as B
import cnh_real_head_confirm as RC

OUT=B.OUT/'native_motion'
ARMS=('exact','e1','torso','corrected','corrected_gait')
SEED=2026100747
RUN_BUDGET=1800.
ANALYSIS_BUDGET=600.


def module():
    import cnh_torso_native_motion_dev as M
    return M


def spent():
    return sum(float(B.A.read(p)['seconds']) for p in (OUT/'attempts').glob('*.json'))


def freeze():
    M=module()
    paths=[Path(__file__),HERE/'cnh_torso_native_motion_dev.py',HERE/'cnh_torso_bias_dev.py',
        HERE/'cnh_torso_gait_bias_dev.py',HERE/'cnh_real_head_confirm.py',HERE/'cnh_extrinsic_aug_evaluate.py',
        HERE/'cnh_sequence_observed_geometry.py',OUT/'bank.npz',OUT/'windows.json',OUT/'PLAN.json',*B.A.M3_MODELS]
    plan=dict(task='CNH_TORSO_NATIVE_REPLAY_DEV_20261007',lane='POSTHOC EXPLORE',units=B.A.UNITS,
        arms=ARMS,run_budget_seconds=RUN_BUDGET,analysis_budget_seconds=ANALYSIS_BUDGET,seed=SEED,
        input='Same raw-metres native pelvis/head horizontal positions; torso-aligned proxy sensor yaw,pitch-10; calibration P01..P05,evaluation P06..P10',
        pairing='Each config one sensor/noisy/z/photon scene; all arms change only fullSE3 inv(estimatedtravel)@sensor query. Same futurepelvis1.5m direction target and origin in exact query and geometry',
        requested_groups='config%4 fixes cold/onset/slowturn/straight source group; preserve actual deadline group and missing cold/event support',
        endpoint='Recomputed original all-object surface labels and0.9m target-front crossing in future-direction frame; censored and undefined retained. Not swept curved-path collision.',
        primary='Evaluation E1 whole-tie minimumthreshold with <=2.5% all13 strict-control outputs; candidate threshold at actual E1 integercount; all residuals reported',
        secondary='Per-arm independent cal-only <=2.5% all13 thresholds; report actual evaluation cost, not sameFA',
        cost='Startupoutputs0:2/main2:12/all13, with counts/denominators. Timely full and excludingstartup2..deadline.',
        uncertainty='No simulated-unit CI; pernominalPxx and leaveoneprefix counts preserve source-dependency limits',
        privileged='Future labels enter source sampling and future-conditioned synthetic scene placement, as well as evaluator exactquery/geometry. Commonphi/anchor cancels in causalarm queries; this is not a pure whole-world coordinate change. Common pelvisqueryorigin is ideal currentposition proxy.',
        budget='Cumulative every attempt including renderer,geometry,inference,unit save,and engine setup/failures/resume; no newunits or budget on stop',
        hashes={str(p.relative_to(B.A.ROOT)):B.A.sha(p) for p in paths},
        limits='Previously consumed/posthoc source; native motion drives simulated boxes,not measured obstacles or glasses sensor; sameXsens position model and proxy torso orientation')
    dest=OUT/'replay_plan.json'
    if dest.exists():
        if B.A.read(dest)!=B.json_value(plan):raise ValueError('Frozen native replay inputs/source changed')
    else:B.save(dest,plan)


def pose_pack(bank,windows,unit,C):
    M=module();items=[]
    for c in range(C):
        row,sign=M.draw(bank,windows,unit,c);wi=int(row['window'])
        value=M.poses(bank,row,sign)
        if not isinstance(value,dict):raise ValueError('Motion poses must return dictionary sensor/travel/queries +metadata')
        for key in ('sensor','travel'):
            if value[key].shape!=(16,4,4) or not np.isfinite(value[key]).all():raise ValueError('16 finite native poses required')
        for a in ARMS:
            if value['queries'][a].shape!=(16,4,4) or not np.isfinite(value['queries'][a]).all():raise ValueError('FullquerySE3 missing')
        np.testing.assert_allclose(value['queries']['exact'],np.linalg.inv(value['travel'])@value['sensor'],atol=1e-10)
        items.append(dict(row=row,window=wi,sign=sign,**value))
    return items


def run_unit(rn,bank,windows,unit,available):
    import cnh_margin_confirm as MC
    import cnh_extrinsic_aug_data as D
    tick=time.monotonic();status='FAILED';error=None;dest=OUT/'units'/f'unit{unit}.npz'
    def check():
        if time.monotonic()-tick>=available:raise TimeoutError('1800-second native replay cumulative budget reached')
    try:
        scenes=MC.scenes_for(unit);C=len(scenes);poses=pose_pack(bank,windows,unit,C)
        boxes=[D.mirror_boxes(s['boxes'],B.A.mirrored(unit)) for s in scenes]
        sensor=np.stack([p['sensor'] for p in poses]);travel=np.stack([p['travel'] for p in poses])
        noisy=np.stack([RC.noisy_for(p['sensor'],unit,c) for c,p in enumerate(poses)])
        queries={a:np.stack([p['queries'][a] for p in poses]) for a in ARMS}
        zs=[];hist=[];ambient=[];geometry=[];backends=[]
        for c in range(C):
            check();h,amb,z,backend=D.render(boxes[c],sensor[c],list(B.A.ANGLES),unit,c,{})
            hist.append(h);ambient.append(amb);zs.append(z)
            backends.append(json.dumps(B.json_value(backend),separators=(',',':')))
            check();geometry.append(RC.geometry(boxes[c],travel[c]))
        z=np.stack(zs,1);hist=np.stack(hist,1);ambient=np.stack(ambient,1)
        result=dict(unit=unit,mode=unit%3,mirror=B.A.mirrored(unit),sensor=sensor,travel=travel,noisy=noisy,
            z=z,hist=hist,ambient=ambient,window=np.array([p['window'] for p in poses]),sign=np.array([p['sign'] for p in poses]),
            requested_group=np.array([p['row']['requested_group'] for p in poses]),
            clip_index=np.array([p['row']['clip_index'] for p in poses]),start=np.array([p['row']['start'] for p in poses]),
            source_pid=np.array([p.get('pid',p['row'].get('pid','')) for p in poses]),
            source_clip=np.array([p.get('clip',p['row'].get('clip','')) for p in poses]),
            boxes_json=np.array([json.dumps(b,separators=(',',':')) for b in boxes]),renderer_backend=np.array(backends))
        for key in ('nativeframes','native_frame_index','turn_group','updated','bias','bias_deg','first_updated','ever_updated','seen','heading_error'):
            if all(key in p for p in poses):result[key]=np.stack([p[key] for p in poses])
        for key in ('category','clear_all','covered','fraction','censor'):
            result[key]=np.array([g[key] for g in geometry])
        for a in ARMS:
            check();raw=np.empty((3,C,13,2),np.float32)
            for branch,angle in enumerate(B.A.ANGLES):
                check();ex=rn.eng.N.extrinsic(angle)
                raw[branch]=rn.raw(z[branch],noisy@ex,queries[a]@ex)
            result[a+'_raw']=raw;result[a+'_query']=queries[a]
        check();result['seconds_before_save']=time.monotonic()-tick
        B.atomic_npz(dest,**result);status='COMPLETE'
        print('native unit',unit,'seconds',round(time.monotonic()-tick,2),flush=True)
    except BaseException as exc:error=repr(exc);raise
    finally:
        attempt=len(list((OUT/'attempts').glob('attempt*.json')))
        B.save(OUT/'attempts'/f'attempt{attempt:04d}_unit{unit}.json',dict(unit=unit,status=status,error=error,seconds=time.monotonic()-tick))


def run(limit):
    M=module();bank,windows=M.load_bank();freeze();tick=time.monotonic();rn=None;error=None
    try:
        B.A.OUT=OUT/'runtime';rn=B.H.Runner()
    except BaseException as exc:error=repr(exc);raise
    finally:
        B.save(OUT/'attempts'/f'setup{int(time.time()*1000)}.json',dict(seconds=time.monotonic()-tick,error=error))
    try:
        for u in B.A.UNITS[:limit] if limit else B.A.UNITS:
            if (OUT/'units'/f'unit{u}.npz').exists():continue
            remaining=RUN_BUDGET-spent()
            prior=[B.A.read(p)['seconds'] for p in (OUT/'attempts').glob('attempt*.json') if B.A.read(p)['status']=='COMPLETE']
            if remaining<=0 or (prior and remaining<max(prior[-3:])):
                print('BUDGET_STOP remaining',remaining,flush=True);break
            run_unit(rn,bank,windows,u,remaining)
    finally:
        if rn is not None:rn.eng.torch.cuda.empty_cache()


def compare(candidate,base,mask):
    count=int(mask.sum());gains=int((mask&candidate&~base).sum());losses=int((mask&~candidate&base).sum())
    return dict(events=count,candidate=int(candidate[mask].sum()),baseline=int(base[mask].sum()),gains=gains,losses=losses,
                diff=gains-losses,rate_diff=None if not count else (gains-losses)/count)


def analyze(limit):
    tick=time.monotonic();wanted=B.A.UNITS[:limit] if limit else B.A.UNITS
    completed=[u for u in wanted if (OUT/'units'/f'unit{u}.npz').exists()]
    missing=[u for u in wanted if u not in completed]
    if not completed:raise ValueError('No completed native units')
    suffix=f'_smoke{limit}' if limit else ''
    ledger=OUT/f'ledger{suffix}.npz';output=OUT/f'replay_result{suffix}.json'
    if ledger.exists() or output.exists():raise FileExistsError('Preserve native analysis')
    def check():
        if time.monotonic()-tick>=ANALYSIS_BUDGET:raise TimeoutError('600-second native analysis budget reached')
    unit=[];config=[];scores=[];meta={k:[] for k in ('window','sign','requested_group','source_pid','source_clip','clip_index','start','category','clear_all','covered','fraction','censor')}
    dynamic={};input_hashes={};query_diffs=[]
    for u in completed:
        check();path=OUT/'units'/f'unit{u}.npz'
        with np.load(path,allow_pickle=False) as f:
            C=f['sensor'].shape[0];unit.extend([u]*C);config.extend(range(C));s=[]
            for a in ARMS:
                sm=B.R.smooth(f[a+'_raw']).max(-1)
                s.append(np.stack((sm[0],sm[1:].max(0)),0))
            scores.append(np.stack(s,1))
            for k in meta:meta[k].append(f[k])
            query_diffs.append(np.abs(f['corrected_gait_query']-f['torso_query']).max(axis=(-1,-2)))
            for key in ('nativeframes','native_frame_index','turn_group','updated','bias','bias_deg','first_updated','ever_updated','seen','heading_error'):
                if key in f:dynamic.setdefault(key,[]).append(f[key])
        input_hashes[str(u)]=B.A.sha(path)
    uid=np.array(unit);cfg=np.array(config);n=len(uid);score=np.concatenate(scores,2)
    meta={k:np.concatenate(v) for k,v in meta.items()};dynamic={k:np.concatenate(v) for k,v in dynamic.items()}
    dynamic['gait_torso_query_max_abs']=np.concatenate(query_diffs)
    np.testing.assert_array_equal(meta['covered'][:,0],meta['covered'][:,1])
    contact_query=meta['covered']&np.isin(meta['category'],B.E.CATS);contact=contact_query.any(1)
    control=meta['clear_all'].all(1);deadline=B.E.causal_index(meta['fraction'])
    if np.any((deadline[contact]<0)|(deadline[contact]>12)):raise ValueError('Native contactdeadline unavailable')
    ev=uid>=99000;cal=uid<99000;evt=contact&ev;ctl=control&ev;cc=control&cal
    frames=dynamic.get('native_frame_index',dynamic.get('nativeframes'))
    if frames is None:raise ValueError('Native frame provenance absent')
    di=np.clip(deadline+3,0,15);age=frames[np.arange(n),di]/60.
    groups={'all':evt,'age_under1':evt&(age<1),'age_1to2':evt&(age>=1)&(age<2),
            'age_2to4':evt&(age>=2)&(age<4),'age_4plus':evt&(age>=4)}
    if 'turn_group' in dynamic:
        tg=dynamic['turn_group'][np.arange(n),di]
        groups.update({name:evt&(tg==i) for i,name in enumerate(('straight','slowturn','onset','other'))})
    if 'ever_updated' in dynamic or 'seen' in dynamic:
        updated=dynamic.get('ever_updated',dynamic.get('seen'))[np.arange(n),di]
        groups.update({'ever_updated_at_deadline':evt&updated,'never_updated_at_deadline':evt&~updated})
    elif 'first_updated' in dynamic:
        first=dynamic['first_updated'];first=first[:,0] if first.ndim>1 else first
        updated=(first>=0)&(first<=frames[np.arange(n),di])
        groups.update({'ever_updated_at_deadline':evt&updated,'never_updated_at_deadline':evt&~updated})
    prefixes=sorted(set(meta['source_pid'][ev]))
    groups.update({f'pid/{p}':evt&(meta['source_pid']==p) for p in prefixes})
    groups.update({f'leaveone/{p}':evt&(meta['source_pid']!=p) for p in prefixes})
    metrics={};thresholds=[];saved={};loss_records=[]
    for si,sensor in enumerate(('single','dual')):
        for selection in ('matched_actual','calibrated'):
            check();policy=f'{selection}/{sensor}/0.025';mask=ctl if selection=='matched_actual' else cc
            if not mask.any():
                metrics[policy]=dict(status='UNDEFINED',reason='No strictcontrols in thresholdselection cohort');continue
            e1=B.integer_budget_threshold(score[si,1,mask],int(np.floor(.025*int(mask.sum())*13)))
            target=e1['actual_fa_count'];alarms=[];fulls=[];mains=[];cell={}
            for ai,a in enumerate(ARMS):
                budget=target if selection=='matched_actual' else int(np.floor(.025*int(mask.sum())*13))
                th=e1 if a=='e1' else B.integer_budget_threshold(score[si,ai,mask],budget)
                thresholds.append(dict(policy=policy,arm=a,**th))
                alarm,full,main=B.outcomes(score[si,ai],th['threshold'],contact,deadline)
                alarms.append(alarm);fulls.append(full);mains.append(main)
                cell[a]=dict(events=int(evt.sum()),timely_all=int(full[evt].sum()),timely_main=int(main[evt].sum()),
                    false_alarm=B.alarm_cost(alarm,ctl),calibration_false_alarm=B.alarm_cost(alarm,cc))
            for ai,a in enumerate(ARMS):
                comps={}
                for b,bi in [('e1',1),('torso',2),('corrected',3)]:
                    if a==b:continue
                    comps[b]=dict(all_before=compare(fulls[ai],fulls[bi],evt),main=compare(mains[ai],mains[bi],evt),
                        groups={g:dict(all_before=compare(fulls[ai],fulls[bi],m),main=compare(mains[ai],mains[bi],m)) for g,m in groups.items()})
                    saved[policy+'/'+a+'/'+b+'/gain_main']=evt&mains[ai]&~mains[bi]
                    saved[policy+'/'+a+'/'+b+'/loss_main']=evt&~mains[ai]&mains[bi]
                cell[a]['comparisons']=comps
            candidate_index=4;baseline_index=2
            loss_any=evt&((~mains[candidate_index]&mains[baseline_index])|(~fulls[candidate_index]&fulls[baseline_index]))
            selected_threshold={t['arm']:t['threshold'] for t in thresholds if t['policy']==policy}
            for j in np.flatnonzero(loss_any):
                d=int(deadline[j]);end=d+1
                prefix_diff=dynamic['gait_torso_query_max_abs'][j,:d+4]
                record=dict(policy=policy,unit=int(uid[j]),config=int(cfg[j]),window=int(meta['window'][j]),
                    source_pid=str(meta['source_pid'][j]),source_clip=str(meta['source_clip'][j]),
                    deadline_output=d,deadline_native_frame=int(frames[j,d+3]),native_age_seconds=float(age[j]),
                    requested_group=int(meta['requested_group'][j]),
                    lost_main=bool(not mains[candidate_index][j] and mains[baseline_index][j]),
                    lost_full=bool(not fulls[candidate_index][j] and fulls[baseline_index][j]),
                    gait_threshold=selected_threshold['corrected_gait'],torso_threshold=selected_threshold['torso'],
                    thresholds_differ=selected_threshold['corrected_gait']!=selected_threshold['torso'],
                    gait_score=score[si,candidate_index,j].tolist(),torso_score=score[si,baseline_index,j].tolist(),
                    same_query_all_native_frames_through_deadline=bool(np.all(prefix_diff<=1e-12)),
                    max_query_difference_through_deadline=float(prefix_diff.max()),
                    interpretation='Paired loss trace only; differing thresholds can change outcome even with identical queries. Not evidence of turn absorption.')
                for key in ('turn_group','updated','seen','bias_deg'):
                    if key in dynamic:record[key]=dynamic[key][j].tolist()
                if 'bias_deg' in dynamic:record['absolute_bias_at_deadline']=abs(float(dynamic['bias_deg'][j,d+3]))
                loss_records.append(record)
            metrics[policy]=cell;saved[policy+'/alarm']=np.asarray(alarms);saved[policy+'/timely_all']=np.asarray(fulls);saved[policy+'/timely_main']=np.asarray(mains)
    check();B.atomic_npz(ledger,unit=uid,config=cfg,score=score,arms=np.array(ARMS),contact=contact,control=control,
        contact_query=contact_query,deadline=deadline,evaluation=ev,calibration=cal,**meta,**dynamic,**saved)
    result=dict(status='BUDGET_PARTIAL' if missing else 'SMOKE_COMPLETE' if limit else 'COMPLETE',
        completed_units=completed,missing_units=missing,sequences=n,evaluation_units=len(np.unique(uid[ev])),
        evaluation_events=int(evt.sum()),evaluation_controls=int(ctl.sum()),calibration_events=int((contact&cal).sum()),
        calibration_controls=int(cc.sum()),requested_group_counts={str(g):int(((meta['requested_group']==g)&ev).sum()) for g in range(4)},
        deadline_group_counts={g:int(m.sum()) for g,m in groups.items()},
        censored={str(c):int((meta['censor']==c).sum()) for c in np.unique(meta['censor'])},
        metrics=metrics,thresholds=thresholds,gait_minus_torso_losses=loss_records,
        run_cumulative_seconds=spent(),analysis_seconds=time.monotonic()-tick,
        provenance=dict(ledger_sha256=B.A.sha(ledger),source_sha256=B.A.sha(__file__),unit_sha256=input_hashes,
                        plan_sha256=B.A.sha(OUT/'replay_plan.json')),
        limits=['Native trajectories drive simulated scenes; torso-aligned sensor yaw is a proxy,not measured headyaw or glasses torso observation',
            'Raw metre positions and futurepelvis1.5m query target shared by exactquery and geometry;0.9m targetfront deadline is not curved sweptpath collision',
            'Future labels enter source sampling and future-conditioned synthetic scene placement; motion-only reexpression changes observation relative to fixedboxes. Commonphi/anchor cancels in causalquery algebra but exactquery/geometry usefuturetruth. Common pelvisqueryorigin is ideal currentposition proxy; yaw comparison does not establish glasses observationcontract',
            'Development/source consumed and configs frozen from prior pilots; no fresh participant confirmation',
            'Overlapping windows and common Xsens jointpositions; no independent-source or participant-generalization claim',
            'No simulatedunitCI; perPxx and leaveoneprefix counts are descriptive dependent-source diagnostics',
            'Cal-only targetFA may differ from actual evalFA; matched evaluationthresholds not deploymentcalibration',
            'Absent event/control/earlycold support remains undefined; censored rows remain in fullledger',
            'No validated coveragegate/UNKNOWN/sidequery or hardware safety claim'])
    check();B.save(output,result)
    print(json.dumps({k:result[k] for k in ('status','evaluation_events','evaluation_controls','calibration_events','calibration_controls','run_cumulative_seconds','analysis_seconds')},indent=2),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('run','analyze'));parser.add_argument('--limit',type=int,choices=(1,2))
    args=parser.parse_args();run(args.limit) if args.stage=='run' else analyze(args.limit)


if __name__=='__main__':main()
