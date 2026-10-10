"""Predeclared strict event matching for frozen joint-current middle-budget alerts.

Evaluation-only category and author background/family joins locate comparable
events. Runtime descriptors are unchanged audited peak/track/score observations.
No new rules, fitting, thresholds, raw readouts or sampling are introduced.
"""
import argparse
from pathlib import Path
import time
import traceback

import numpy as np

import cnh_graded_corridor_eval_dev as C
import cnh_graded_evidence_dev as G
import cnh_counterfactual_eval_dev as E

ROOT=C.ROOT
JOINT=ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010'
OUTPUT=ROOT/'artifacts.local/work/cnh-graded-peak-match-dev-20261010'
SEEDS=G.SEEDS
Q=1
MATCH_FIELDS=('rank0_depth_m','rank0_support_x_extent_m','rank0_support_y_extent_m',
              'rank0_support_z_extent_m','inner_weighted_sum_share','inner_supported_candidate_count')
TOLERANCES=np.array([.3002784,.15,.15,.15,.2,1.],float)
SCORE_NAMES=('ordinary_smooth_margin','ordinary_raw_slope5','ordinary_raw_detrended_fluctuation5')


def first(flags,deadline=15):
    allowed=np.array(flags[...,:deadline-2],bool)
    return np.where(allowed.any(-1),allowed.argmax(-1)+3,-1)


def first_height(flags,deadline=15):
    return np.stack([first(flags[...,q],deadline) for q in range(2)],-1)


def baseline(d,seed_index,thresholds):
    seed=SEEDS[seed_index]
    ordinary=d['candidates'][0,seed_index]
    strong=E.old_fusion(d['m3'],d['local'])|(ordinary>=thresholds[str(seed)]['addition'])
    return np.where(strong,2,np.where(ordinary>=thresholds[str(seed)]['single'],1,0)).astype(np.int8)


def event_row(kind,n,k,frame,d,seed,before_all,after_all,first_added,archive,score):
    r=d['rows'][n]
    return dict(event_id=f'{d["split"]}/{seed}/{kind}/{int(d["scene_ids"][n])}/{k}/{frame}',
        kind=kind,split=d['split'],seed=seed,scene=int(d['scene_ids'][n]),replica=k,height='BODY',frame=frame,
        timely=int(frame<=13),category=d['category'][n,Q],
        background_family=r['background_family'],background_id=r['background_id'],shape_family=r['shape_family'],
        before_first=int(before_all[n,k,Q]),after_first=int(after_all[n,k,Q]),
        body_first_added=int(first_added[n,k]),
        **{f'current_{name}':float(archive['current_features'][n,k,frame-3,Q,j])
           if archive['current_valid'][n,k,frame-3,Q,j] else '' for j,name in enumerate(archive['current_names'])},
        **{f'score_{name}':float(score[n,k,frame-3,Q,j]) for j,name in enumerate(SCORE_NAMES)},
        **{f'{name}':float(archive['temporal_features'][n,k,frame-3,Q,j])
           if archive['temporal_valid'][n,k,frame-3,Q,j] else ''
           for j,name in enumerate(archive['temporal_names']) if str(name).startswith('track_')})


def strict_match(anchors,controls,panel):
    """Fixed condition cascade, L1 normalized distance, scene/K anchor order."""
    remaining=set(range(len(controls)))
    attempts,pairs=[],[]
    for anchor in sorted(anchors,key=lambda r:(r['scene'],r['replica'],r['frame'])):
        keep=np.ones(len(controls),bool)
        stages=dict(initial_controls=len(controls))
        for identity in ('background_family','background_id','shape_family'):
            keep &= np.array([r[identity]==anchor[identity] for r in controls],bool)
            stages[identity]=int(keep.sum())
        av=np.array([float(anchor[f'current_{field}']) if anchor[f'current_{field}']!='' else np.nan for field in MATCH_FIELDS])
        cv=np.array([[float(r[f'current_{field}']) if r[f'current_{field}']!='' else np.nan for field in MATCH_FIELDS] for r in controls],float).reshape(-1,len(MATCH_FIELDS))
        finite=np.isfinite(cv).all(-1)&np.isfinite(av).all()
        keep &= finite
        stages['finite_descriptors']=int(keep.sum())
        delta=abs(cv-av)
        for j,field in enumerate(MATCH_FIELDS):
            keep &= delta[:,j]<=TOLERANCES[j]
            stages[field]=int(keep.sum())
        eligible=[i for i in np.flatnonzero(keep) if i in remaining]
        stages['available_without_replacement']=len(eligible)
        attempt=dict(panel=panel,anchor_event_id=anchor['event_id'],**stages)
        if eligible:
            distance=(delta/TOLERANCES).sum(-1)
            pick=min(eligible,key=lambda i:(distance[i],controls[i]['scene'],controls[i]['replica'],controls[i]['frame']))
            control=controls[pick]
            remaining.remove(pick)
            pair=dict(panel=panel,anchor_event_id=anchor['event_id'],control_event_id=control['event_id'],
                split=anchor['split'],seed=anchor['seed'],anchor_scene=anchor['scene'],anchor_replica=anchor['replica'],
                anchor_frame=anchor['frame'],control_scene=control['scene'],control_replica=control['replica'],control_frame=control['frame'],
                control_minus_anchor_frame=control['frame']-anchor['frame'],distance=float(distance[pick]),
                **{f'abs_delta_{name}':float(delta[pick,j]) for j,name in enumerate(MATCH_FIELDS)})
            pairs.append(pair)
            attempt.update(status='MATCHED',control_event_id=control['event_id'],distance=float(distance[pick]))
        else:
            attempt.update(status='NOT_EVALUABLE',control_event_id='',distance='')
        attempts.append(attempt)
    return pairs,attempts


def declare(out):
    C.save(out/'PLAN.json',dict(task='CNH_GRADED_PEAK_MATCH_DEV_20261010',lane='EXPLORE consumed simulated Development',
        CPU_command_wall_seconds_cap=180,GPU_seconds_cap=0,
        source_rule='Frozen score_current/c15_p64, all3seeds cal/validation; preserve baseline grades and thresholds',
        primary='BODY contact baseline lacks timely first(f3..13), current gains timely; once per sceneK at first BODY added alert',
        secondary='BODY contact already timely under baseline, current first is earlier; once per sceneK at earliest added BODY frame; separate from rescue',
        controls=dict(clear_candidate='Joint clear truth, first BODY added grade1 with baselineBODY0, may have baselineHEAD alert',
            clear_jointnew='Joint clear, first BODY added frame where baseline both heights0',
            pure_pass_body='No contact in either height and BODY categorypass, first BODY added alert'),
        time='All first events recordedf3..15; rescue/earlier anchors timelyf3..13; controls include entire13frames, report frame difference',
        matching='Within split/seed and background_family+background_id+shape_family; six public descriptor tolerances; standardizedL1 nearest, tie controlscene/K/frame; anchororder scene/K/frame',
        fields=list(MATCH_FIELDS),tolerances=TOLERANCES.tolist(),
        panels='rescue and earlier each independently match all3controltypes, one-to-one without replacement inside eachpanel; independentpanel controls mayrecur',
        missing='NOT_EVALUABLE without relaxation; retain counts after each condition, plus available control count',
        outputs='All contact first times, anchor/control runtime descriptor ledger, matchedpairs+attempts, clear slot/clip decompositions',
        restrictions='No rawhist/observation rebuild, fitting, threshold changes, new sampling, alternate cut or matched-subset rule effectiveness claim',
        source_sha256=C.sha(Path(__file__))))


def run(out):
    began=time.monotonic()
    out=out.resolve()
    if not out.is_relative_to((ROOT/'artifacts.local').resolve()):raise ValueError('Use canonical artifacts')
    if (out/'PLAN.json').exists():raise FileExistsError('Preserve existing matching PLAN')
    out.mkdir(parents=True,exist_ok=True)
    declare(out)
    phase='join'
    try:
        data=C.load();thresholds=C.read(C.PARENT/'thresholds.json')
        all_events,all_contacts,all_clips,all_pairs,all_attempts=[],[],[],[],[]
        stats={};input_hashes={}
        for split,d in data.items():
            with np.load(JOINT/'features'/f'{split}_features.npz',allow_pickle=False) as a:
                archive={key:a[key] for key in ('current_features','current_valid','current_names','temporal_features','temporal_valid','temporal_names')}
                np.testing.assert_array_equal(a['scene_ids'],d['scene_ids'])
            with np.load(JOINT/f'{split}_grades.npz',allow_pickle=False) as a:
                grade_keys=a['keys'].tolist();grades=a['grades']
            input_hashes[split]={str(p.relative_to(ROOT)):C.sha(p) for p in (
                JOINT/'features'/f'{split}_features.npz',JOINT/f'{split}_grades.npz')}
            for si,seed in enumerate(SEEDS):
                if time.monotonic()-began>=180:raise TimeoutError('180s commandcap reached')
                phase=f'{split}/{seed}'
                key=f'{seed}/score_current/c15_p64'
                grade=grades[grade_keys.index(key)]
                base=baseline(d,si,thresholds)
                assert np.all(grade>=base)
                np.testing.assert_array_equal(grade==2,base==2)
                raw_path=C.SOURCE/f'scores/ordinary_seed{seed}_{split}.npz'
                raw=E.read_job(raw_path,'ordinary',seed,split,'ideal')
                score=C.build_score_features(raw,d['candidates'][0,si],thresholds[str(seed)]['single'])
                added=(grade==1)&(base==0)
                before,after=first_height(base>0),first_height(grade>0)
                before_time,after_time=first_height(base>0,13),first_height(grade>0,13)
                af=first(added[...,Q])
                contact=d['category'][:,Q]=='contact'
                clear=(d['category']=='clear').all(1)
                purepass=(d['category']=='pass').any(1)&~(d['category']=='contact').any(1)&(d['category'][:,Q]=='pass')
                jointnewbody=first(added[...,Q]&~(base>0).any(-1))
                rescue=contact[:,None]&(before_time[...,Q]<0)&(after_time[...,Q]>=0)
                earlier=contact[:,None]&(before_time[...,Q]>=0)&(after_time[...,Q]>=0)&(after_time[...,Q]<before_time[...,Q])
                selections=dict(rescue=(rescue,af),earlier=(earlier,af),
                    clear_candidate=(clear[:,None]&(af>=0),af),
                    clear_jointnew=(clear[:,None]&(jointnewbody>=0),jointnewbody),
                    pure_pass_body=(purepass[:,None]&(af>=0),af))
                events={}
                for kind,(mask,frame_values) in selections.items():
                    events[kind]=[event_row(kind,n,k,int(frame_values[n,k]),d,seed,before,after,af,archive,score)
                                  for n,k in zip(*np.where(mask))]
                    all_events.extend(events[kind])
                for n in np.flatnonzero(contact):
                    for k in range(4):
                        all_contacts.append(dict(split=split,seed=seed,scene=int(d['scene_ids'][n]),replica=k,height='BODY',
                            before_first=int(before[n,k,Q]),after_first=int(after[n,k,Q]),
                            before_timely_first=int(before_time[n,k,Q]),after_timely_first=int(after_time[n,k,Q]),
                            first_added=int(af[n,k]),rescue=int(rescue[n,k]),earlier=int(earlier[n,k]),
                            lost=int((before_time[n,k,Q]>=0)&(after_time[n,k,Q]<0))))
                base_joint=(base>0).any(-1);new_joint=(grade>0).any(-1)
                extra_joint=new_joint&~base_joint
                headonly=extra_joint&added[...,0]&~added[...,1]
                bodyonly=extra_joint&added[...,1]&~added[...,0]
                both=extra_joint&added[...,0]&added[...,1]
                np.testing.assert_array_equal(headonly|bodyonly|both,extra_joint)
                bf,nf=first(base_joint),first(new_joint)
                for n in np.flatnonzero(clear):
                    for k in range(4):
                        all_clips.append(dict(split=split,seed=seed,scene=int(d['scene_ids'][n]),replica=k,
                            before_first=int(bf[n,k]),after_first=int(nf[n,k]),first_body_added=int(af[n,k]),
                            first_jointnew_body_contributor=int(jointnewbody[n,k]),
                            extra_headonly_slots=int(headonly[n,k].sum()),extra_bodyonly_slots=int(bodyonly[n,k].sum()),
                            extra_both_slots=int(both[n,k].sum()),extra_joint_slots=int(extra_joint[n,k].sum()),
                            body_added_candidate_slots=int(added[n,k,:,Q].sum()),
                            head_added_candidate_slots=int(added[n,k,:,0].sum())))
                panel_stats={}
                for anchors in ('rescue','earlier'):
                    for controls in ('clear_candidate','clear_jointnew','pure_pass_body'):
                        panel=f'{split}/{seed}/{anchors}_vs_{controls}'
                        pairs,attempts=strict_match(events[anchors],events[controls],panel)
                        all_pairs.extend(pairs);all_attempts.extend(attempts)
                        panel_stats[f'{anchors}_vs_{controls}']=dict(anchors=len(events[anchors]),controls=len(events[controls]),
                            matched=len(pairs),not_evaluable=len(attempts)-len(pairs),
                            frame_delta=[p['control_minus_anchor_frame'] for p in pairs],
                            surviving_anchor_count_by_stage={stage:int(sum(a[stage]>0 for a in attempts))
                                for stage in ('initial_controls','background_family','background_id','shape_family','finite_descriptors',*MATCH_FIELDS,'available_without_replacement')})
                stats[f'{split}/{seed}']=dict(events={k:len(v) for k,v in events.items()},matching=panel_stats,
                    clear=dict(sceneK_clips=int(clear.sum())*4,joint_slot_denominator=int(clear.sum())*4*13,
                        baseline_slots=int(base_joint[clear].sum()),current_slots=int(new_joint[clear].sum()),
                        extra_joint_slots=int(extra_joint[clear].sum()),extra_headonly_slots=int(headonly[clear].sum()),
                        extra_bodyonly_slots=int(bodyonly[clear].sum()),extra_both_slots=int(both[clear].sum()),
                        body_candidate_added_slots=int(added[clear,...,Q].sum()),
                        baseline_clips=int((bf[clear]>=0).sum()),current_clips=int((nf[clear]>=0).sum()),
                        new_clips=int(((bf[clear]<0)&(nf[clear]>=0)).sum()),
                        existing_clips_earlier=int(((bf[clear]>=0)&(nf[clear]>=0)&(nf[clear]<bf[clear])).sum()),
                        existing_clips_same_first=int(((bf[clear]>=0)&(nf[clear]==bf[clear])).sum())))
        for name,rows in (('events',all_events),('body_contact_events',all_contacts),('clear_clips',all_clips),
                          ('matched_pairs',all_pairs),('matching_attempts',all_attempts)):
            if rows:G.write_csv(out/f'{name}.csv',rows)
            else:(out/f'{name}.csv').write_text('',encoding='utf8')
        C.save(out/'stats.json',stats)
        C.save(out/'schema.json',dict(event_runtime_columns='current_22+score_3+track_35; exact runtime values at selected BODY frame, no evaluator box descriptor',
            panel_units='One scene/K/BODY event, not frame count; each panel independently one-to-one',
            clear_cost='Joint union slots with baseline neither height alert, divided HEADonly/BODYonly/both; BODY candidate may coincide baseline HEAD alert and is separately counted',
            files=dict(events='Anchor/control descriptor ledger with stable eventIDs',body_contact_events='Every BODY contact sceneK with full and timely first',
                clear_clips='Every joint-clear sceneK clip with slot contribution and first times',matched_pairs='Strict selected eventID pairs and frame deltas',
                matching_attempts='All anchors including NOT_EVALUABLE, condition-cascade counts'),
            limitations='Author background and family are evaluator-only matching strata. Existing selected-alert controls cannot estimate a new rule detection or interruption reduction. Peak depth differences are observations, not background visible/free evidence.',
            input_sha256=input_hashes,thresholds_sha256=C.sha(C.PARENT/'thresholds.json'),eval_manifest_sha256=C.sha(C.SOURCE/'eval_manifest.json'),
            source_sha256=C.sha(Path(__file__))))
        C.save(out/'result.json',dict(status='COMPLETE',CPU_seconds=time.monotonic()-began,GPU_seconds=0,
            event_rows=len(all_events),contact_rows=len(all_contacts),clear_clip_rows=len(all_clips),
            matched_pair_rows=len(all_pairs),matching_attempt_rows=len(all_attempts),phase=phase))
    except BaseException as error:
        C.save(out/f'failure_{time.time_ns()}.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),CPU_seconds=time.monotonic()-began))
        raise
    finally:
        C.save(out/'execution_receipt.json',dict(seconds=time.monotonic()-began,GPU_seconds=0,
            source_sha256=C.sha(Path(__file__)),outputs_sha256={p.name:C.sha(p) for p in out.iterdir()
                if p.is_file() and p.name!='execution_receipt.json'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    run(parser.parse_args().output)
