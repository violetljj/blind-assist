"""CPU-only evaluator for fixed cached spatial-notice queries.

Truth stays here, outside the inference/notice generator. Whole-cohort prompt
budget matching is descriptive Development analysis, not deployable calibration.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import cnh_sector_notice_dev as S
import cnh_sector_geometry_dev as G

B=S.B; Y=S.Y; E=S.E; OUT=S.OUT


def paired(new, old, contact):
    return dict(rescued=int((contact & new & ~old).sum()),
        lost=int((contact & ~new & old).sum()), diff=int(new[contact].sum()-old[contact].sum()))


def summarize(notice, truth, contact, control, deadline, point):
    e=S.events(notice,truth,contact,deadline)
    timely=e['timely']; rr=np.flatnonzero(timely); ff=e['first'][rr]
    first_labels=notice[rr,ff]; first_truth=truth[rr,ff]
    correct_labels=(first_labels & first_truth).sum()
    incorrect_labels=(first_labels & ~first_truth).sum()
    main=(np.arange(13)[None,:]>=2)&(np.arange(13)[None,:]<=deadline[:,None])&contact[:,None]
    full=notice.any(-1)&(np.arange(13)[None,:]<=deadline[:,None])&contact[:,None]
    unclassified=~contact&~control
    record=dict(working_point=point,events=int(contact.sum()),timely_main=int(timely.sum()),
        timely_including_startup=int(full.any(-1).sum()),
        first_supported=int(e['first_supported'].sum()),first_clean=int(e['first_clean'].sum()),
        first_unique=int(e['first_unique'].sum()),later_or_first_supported=int(e['ever_supported'].sum()),
        first_supported_fraction=None if not len(rr) else float(e['first_supported'].sum()/len(rr)),
        first_clean_fraction=None if not len(rr) else float(e['first_clean'].sum()/len(rr)),
        first_timed_command_correct_labels=int(correct_labels),first_timed_command_incorrect_labels=int(incorrect_labels),
        first_ambiguous_truth_events=int((first_truth.sum(-1)>1).sum()),
        total_notices=int(notice.sum()), contact_window_notices=int(notice[contact].sum()),
        strict_clear_notices=int(notice[control].sum()),other_window_notices=int(notice[unclassified].sum()),
        strict_clear_windows=int(control.sum()),other_windows=int(unclassified.sum()),
        contact_predeadline_main_supported_notices=int((notice & truth & main[...,None]).sum()),
        contact_predeadline_main_unsupported_notices=int((notice & ~truth & main[...,None]).sum()),
        label_removed_timely=int(timely.sum()),
        labeling_ablation='Same command times/count with text label removed: timely detection identical; no user-benefit result')
    return record,e


def analyze():
    with S.stage('analyze') as check:
        missing=[u for u in Y.UNITS if not (OUT/'units'/f'unit{u}.npz').exists()]
        if missing:
            B.save(OUT/'result.json',dict(status='INCOMPLETE_NO_RANKING',missing=missing));return
        if B.A.sha(Path(G.__file__))!=B.A.read(OUT/'prepare_result.json')['geometry_source_sha256']:
            raise ValueError('Geometry evaluator changed after prepare')
        with np.load(E.OUT/'ledger.npz') as z: baseline={k:z[k] for k in ('score','unit','config','contact','control','deadline')}
        uid=baseline['unit'];cfg=baseline['config']; contact=baseline['contact'];control=baseline['control'];deadline=baseline['deadline']
        assert (len(uid),int(contact.sum()),int(control.sum()))==(3840,229,384)
        score=np.empty((5,2,3840,13,3),np.float64)
        truth=np.zeros((5,3840,13,3),bool);target_ids=np.full(3840,-1,int);geometry_rows=[];hashes={}
        for u in Y.UNITS:
            check(); m=np.flatnonzero(uid==u); path=OUT/'units'/f'unit{u}.npz'; hashes[str(u)]=B.A.sha(path)
            assert np.array_equal(cfg[m],np.arange(40))
            with np.load(path) as f,np.load(Y.M.OUT/'units'/f'unit{u}.npz') as native,np.load(Y.OUT/'units'/f'unit{u}.npz') as physical:
                for ni,name in enumerate(Y.NAMES):
                    raw=f[name+'/raw']
                    if raw.shape!=(3,3,40,13,2):raise ValueError('Frozen sector raw shape changed')
                    sm=B.R.smooth(raw).max(-1)
                    combined=np.stack((sm[:,0],sm[:,1:].max(1))).transpose(0,2,3,1)
                    score[ni,:,m]=combined.transpose(1,0,2,3)  # numpy advanced row index goes first
                    sensor=native['sensor'] if name=='zero' else physical[name+'/sensor']
                    for c,row in enumerate(m):
                        if not contact[row]:continue
                        boxes=json.loads(str(native['boxes_json'][c]))
                        details=G.localization_truth(boxes,native['travel'][c],sensor[c],deadline[row],True,return_details=True)
                        truth[ni,row]=details['mask']
                        if ni==0:
                            if len(details['target_ids'])!=1:raise ValueError('Save all-target sets if multi-target cohort appears')
                            target_ids[row]=details['target_ids'][0]
                        geometry_rows.append(dict(unit=u,config=c,name=name,target_ids=details['target_ids'],
                            ambiguous_frames=int(details['ambiguous_frames'].sum()),unsupported_frames=int(details['unsupported_frames'].sum())))
        metrics={};saved={};table=[];parent=B.A.read(E.OUT/'result.json')
        for ni,name in enumerate(Y.NAMES):
            for si,sensor in enumerate(('single','dual')):
                check();key=name+'/'+sensor;cell={};baseline_events={}
                e1_score=baseline['score'][ni,si,1]
                old_theta=parent['metrics'][key]['e1']['threshold']['threshold']
                anchor=e1_score>=old_theta;budget=int(anchor.sum())
                for arm,ai in (('e1',1),('ema',4)):
                    data=baseline['score'][ni,si,ai]
                    th=B.integer_budget_threshold(data,budget)
                    alarm,full,main=B.outcomes(data,th['threshold'],contact,deadline)
                    if arm=='e1':np.testing.assert_array_equal(alarm,anchor)
                    point=dict(threshold=th['threshold'],target_notices=budget,
                        actual_notices=int(alarm.sum()),unused_notices=budget-int(alarm.sum()))
                    cell[arm]=dict(working_point=point,timely_main=int(main[contact].sum()),
                        timely_including_startup=int(full[contact].sum()),total_notices=int(alarm.sum()),
                        strict_clear_notices=int(alarm[control].sum()),
                        contact_window_notices=int(alarm[contact].sum()),other_window_notices=int(alarm[~contact&~control].sum()),
                        events=229, strict_clear_windows=384, other_windows=int((~contact&~control).sum()))
                    baseline_events[arm]=main
                    saved[key+'/'+arm+'/timely']=main
                reference='ema' if cell['ema']['timely_main']>=cell['e1']['timely_main'] else 'e1'
                for policy in ('top1','all_sectors'):
                    point=S.select_threshold(score[ni,si],budget,policy)
                    notice=S.emitted(score[ni,si],point['threshold'],policy)
                    rec,e=summarize(notice,truth[ni],contact,control,deadline,point)
                    if policy=='top1':
                        # Evaluator-only diagnostic: can an unselected sector at
                        # the first command support the target at this same theta?
                        rr=np.flatnonzero(e['timely']); ff=e['first'][rr]
                        supported_above=((score[ni,si,rr,ff]>=point['threshold']) & truth[ni,rr,ff]).any(-1)
                        rec['first_unclean_with_supported_query_above_threshold']=int((~e['first_clean'][rr] & supported_above).sum())
                        rec['diagnostic_identity']='Evaluator-only same-time ranking diagnostic, no extra emitted notice or achievable benefit claim'
                    rec['against']={arm:dict(detection=paired(e['timely'],base,contact),
                        first_clean=paired(e['first_clean'],base,contact)) for arm,base in baseline_events.items()}
                    rec['reference_arm']=reference
                    rec['first_clean_minus_reference']=rec['first_clean']-cell[reference]['timely_main']
                    cell[policy]=rec
                    for metric,value in e.items():saved[key+'/'+policy+'/'+metric]=value
                    saved[key+'/'+policy+'/notice']=notice
                    for row in np.flatnonzero(contact):
                        first=int(e['first'][row]);labels=[] if first<0 else [S.LABELS[i] for i in np.flatnonzero(notice[row,first])]
                        actual=[] if first<0 else [S.LABELS[i] for i in np.flatnonzero(truth[ni,row,first])]
                        table.append(dict(name=name,sensor=sensor,policy=policy,unit=int(uid[row]),config=int(cfg[row]),
                            target_box=int(target_ids[row]),deadline=int(deadline[row]),first_output=first,
                            first_input_frame=None if first<0 else first+3,reported='|'.join(labels),supported='|'.join(actual),
                            timely=bool(e['timely'][row]),first_clean=bool(e['first_clean'][row]),
                            first_unique=bool(e['first_unique'][row]),ever_supported=bool(e['ever_supported'][row]),
                            e1_timely=bool(baseline_events['e1'][row]),ema_timely=bool(baseline_events['ema'][row])))
                metrics[key]=cell
        decisions={}
        for sensor in ('single','dual'):
            diffs=[metrics[name+'/'+sensor]['top1']['first_clean_minus_reference'] for name in Y.NAMES[1:]]
            decisions[sensor]=dict(diffs=diffs,decision=S.decision(diffs))
        B.atomic_npz(OUT/'ledger.npz',score=score,truth=truth,unit=uid,config=cfg,contact=contact,control=control,
            deadline=deadline,target_box=target_ids,**saved)
        with (OUT/'events.csv').open('x',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        B.save(OUT/'geometry.json',dict(rows=geometry_rows,events=229,
            targets={str(i):int((target_ids[contact]==i).sum()) for i in np.unique(target_ids[contact])},
            by_condition={name:dict(ambiguous_frames=int((truth[ni,contact].sum(-1)>1).sum()),
                unsupported_frames=int((~truth[ni,contact].any(-1)).sum()),output_frames=229*13) for ni,name in enumerate(Y.NAMES)}))
        B.save(OUT/'result.json',dict(status='COMPLETE',events=229,controls=384,sequences=3840,
            metrics=metrics,decisions=decisions,labels=S.LABELS,angles=S.ANGLES,
            source_sha256=B.A.sha(Path(__file__)),geometry_source_sha256=B.A.sha(Path(G.__file__)),
            input_units_sha256=hashes,ledger_sha256=B.A.sha(OUT/'ledger.npz'),
            limits=['Consumed Development artificial physical yaw, future-conditioned simulation, overlapping source windows, not real obstacles/device/new participants',
                'Shared ideal pelvis origin/head position adapter; E1 fullclip ideal60Hz, EMA noisy5Hz window reset, sector instantaneous noisy-head yaw; whole schemes under existing inputs',
                'Whole3840 scene-mixture command budget describes13output windows; not measured audio duration, long-term rate or deployment calibration',
                'Queries overlap, labels are fixed +/-10deg geometry; set-valued box footprint support is not unique object attribution. Unique-only metric reported separately',
                'All-object contact at inherited0.9m reference, not swept-route collision or closed-loop user action',
                'No distance readout, coverage/UNKNOWN/CLEAR validation, statistics/safety claims or long-sequence dedup evidence']))
        print('SECTOR_COMPLETE',json.dumps(decisions),flush=True)


if __name__=='__main__':analyze()
