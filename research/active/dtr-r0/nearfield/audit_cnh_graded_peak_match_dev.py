"""Independent event selection, strict matching and clear-slot replay.

Reads consumed grade/feature caches and evaluator categories only. No producer
helpers, fitting, model prediction, raw histogram, boxes or relaxed matching.
"""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import audit_cnh_graded_corridor_eval_dev as A

OUT = A.ROOT / 'artifacts.local/work/cnh-graded-peak-match-dev-20261010'
JOINT = A.ROOT / 'artifacts.local/work/cnh-graded-peak-joint-dev-20261010'
FIELDS = ['rank0_depth_m', 'rank0_support_x_extent_m', 'rank0_support_y_extent_m',
          'rank0_support_z_extent_m', 'inner_weighted_sum_share', 'inner_supported_candidate_count']
TOL = np.array([.3002784, .15, .15, .15, .2, 1.])
STAGES = ['initial_controls', 'background_family', 'background_id', 'shape_family',
          'finite_descriptors', *FIELDS, 'available_without_replacement']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


def compare_row(actual, expected, where):
    assert actual.keys() == expected.keys(), (where, 'keys')
    for name, value in expected.items():
        if isinstance(value, (float, np.floating)):
            assert abs(float(actual[name])-value) <= 1e-12, (where, name, actual[name], value)
        else:
            A.eq(actual[name], str(value), where+'/'+name)


def match(anchors, controls, panel):
    controls = list(controls)
    used = set();pairs = [];attempts = [];reasons = {}
    for anchor in sorted(anchors, key=lambda a: (int(a['scene']), int(a['replica']), int(a['frame']))):
        surviving = list(range(len(controls)))
        counts = {'initial_controls': len(surviving)}
        for identity in ('background_family', 'background_id', 'shape_family'):
            surviving = [i for i in surviving if controls[i][identity] == anchor[identity]]
            counts[identity] = len(surviving)
        av = np.array([float(anchor['current_'+f]) if anchor['current_'+f] else np.nan for f in FIELDS])
        cv = np.array([[float(c['current_'+f]) if c['current_'+f] else np.nan
                        for f in FIELDS] for c in controls]).reshape(-1, len(FIELDS))
        surviving = [i for i in surviving if np.isfinite(av).all() and np.isfinite(cv[i]).all()]
        counts['finite_descriptors'] = len(surviving)
        delta = np.abs(cv-av)
        for j, field in enumerate(FIELDS):
            surviving = [i for i in surviving if delta[i,j] <= TOL[j]]
            counts[field] = len(surviving)
        available = [i for i in surviving if i not in used]
        counts['available_without_replacement'] = len(available)
        attempt = dict(panel=panel, anchor_event_id=anchor['event_id'], **counts)
        if available:
            distance = (delta/TOL).sum(-1)
            choice = min(available, key=lambda i: (distance[i], int(controls[i]['scene']),
                                                    int(controls[i]['replica']), int(controls[i]['frame'])))
            assert choice not in used;used.add(choice);control = controls[choice]
            pairs.append(dict(panel=panel, anchor_event_id=anchor['event_id'], control_event_id=control['event_id'],
                              split=anchor['split'], seed=int(anchor['seed']), anchor_scene=int(anchor['scene']),
                              anchor_replica=int(anchor['replica']), anchor_frame=int(anchor['frame']),
                              control_scene=int(control['scene']), control_replica=int(control['replica']),
                              control_frame=int(control['frame']),
                              control_minus_anchor_frame=int(control['frame'])-int(anchor['frame']),
                              distance=float(distance[choice]),
                              **{'abs_delta_'+f:float(delta[choice,j]) for j,f in enumerate(FIELDS)}))
            attempt.update(status='MATCHED', control_event_id=control['event_id'], distance=float(distance[choice]))
        else:
            attempt.update(status='NOT_EVALUABLE', control_event_id='', distance='')
            failing = next(name for name in STAGES if counts[name] == 0)
            if failing == 'finite_descriptors' and not np.isfinite(av).all():
                failing = 'anchor_missing_descriptor'
            if failing == 'available_without_replacement':
                failing = 'feasible_controls_exhausted_by_prior_anchors'
            reasons[failing] = reasons.get(failing, 0)+1
        attempts.append(attempt)
    return pairs, attempts, reasons


def run():
    began = time.monotonic();target = OUT/'audit';target.mkdir(exist_ok=True)
    if (target/'result.json').exists():
        raise FileExistsError('Preserve matching audit')
    plan = A.read(OUT/'PLAN.json');schema = A.read(OUT/'schema.json')
    producer = Path(__file__).with_name('cnh_graded_peak_match_dev.py')
    A.eq(sha(producer), plan['source_sha256'], 'source hash')
    A.eq(schema['source_sha256'], plan['source_sha256'], 'schema source hash')
    A.eq(plan['fields'], FIELDS, 'match fields');A.eq(plan['tolerances'], TOL.tolist(), 'fixed tolerances')
    receipt = A.read(OUT/'execution_receipt.json')
    for name, value in receipt['outputs_sha256'].items():
        A.eq(sha(OUT/name), value, 'producer output hash')
    for identity in schema['input_sha256'].values():
        for path, value in identity.items():A.eq(sha(A.ROOT/path), value, 'input hash')
    source = A.ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009'
    A.eq(sha(source/'eval_manifest.json'), schema['eval_manifest_sha256'], 'eval manifest')
    threshold_path = A.ROOT/'artifacts.local/work/cnh-graded-evidence-dev-20261010/thresholds.json'
    A.eq(sha(threshold_path), schema['thresholds_sha256'], 'threshold input')
    ths = A.read(threshold_path);ds = A.load()
    expected_events = {};expected_contacts = {};expected_clips = {};groups = {};statistics = {}
    for split, d in ds.items():
        with np.load(JOINT/'features'/f'{split}_features.npz', allow_pickle=False) as archive:
            features = {name:archive[name] for name in ('current_features','current_valid','current_names',
                         'temporal_features','temporal_valid','temporal_names')}
            np.testing.assert_array_equal(archive['scene_ids'], d['ids'])
        with np.load(JOINT/f'{split}_grades.npz', allow_pickle=False) as archive:
            saved = dict(zip(archive['keys'].tolist(),archive['grades']))
        for seed in A.SEEDS:
            th = ths[str(seed)];ordinary = A.smooth(d['raw'][seed])
            score = A.score_features(d['raw'][seed], th['single'])
            old = (d['m3']>=.9404184587540165)|(d['local']>=4.625390338985158)
            strong = old|(ordinary>=th['addition'])
            base = np.where(strong,2,np.where(ordinary>=th['single'],1,0))
            grade = saved[f'{seed}/score_current/c15_p64']
            assert np.all(grade>=base);np.testing.assert_array_equal(grade==2,strong)
            added = (grade==1)&(base==0)
            before,after = A.first(base>0),A.first(grade>0)
            bt,at = A.first(base>0,11),A.first(grade>0,11)
            af = A.first(added)[...,1]
            joint_base = (base>0).any(-1);joint_after = (grade>0).any(-1)
            extra = joint_after&~joint_base
            body_jointnew = np.zeros_like(added)
            body_jointnew[...,1] = added[...,1]&~joint_base
            jointnew_first = A.first(body_jointnew)[...,1]
            cat = d['category'];body_contact = cat[:,1]=='contact'
            clear = (cat=='clear').all(-1)
            passed = (cat[:,1]=='pass')&~(cat=='contact').any(-1)
            rescue = body_contact[:,None]&(bt[...,1]<0)&(at[...,1]>=0)
            earlier = body_contact[:,None]&(bt[...,1]>=0)&(at[...,1]>=0)&(at[...,1]<bt[...,1])
            kinds = dict(rescue=(rescue,af), earlier=(earlier,af),
                         clear_candidate=(clear[:,None]&(af>=0),af),
                         clear_jointnew=(clear[:,None]&(jointnew_first>=0),jointnew_first),
                         pure_pass_body=(passed[:,None]&(af>=0),af))
            selections = {}
            for kind, (mask, fvalues) in kinds.items():
                entries = []
                for n,k in zip(*np.where(mask)):
                    frame = int(fvalues[n,k]);r = d['rows'][n]
                    event = dict(event_id=f'{split}/{seed}/{kind}/{int(d["ids"][n])}/{k}/{frame}',
                                 kind=kind, split=split, seed=seed, scene=int(d['ids'][n]),replica=int(k),
                                 height='BODY',frame=frame,timely=int(frame<=13),category=cat[n,1],
                                 background_family=r['background_family'],background_id=r['background_id'],
                                 shape_family=r['shape_family'], before_first=int(before[n,k,1]),after_first=int(after[n,k,1]),
                                 body_first_added=int(af[n,k]))
                    for label in ('current','temporal'):
                        for j,name in enumerate(features[label+'_names']):
                            if label=='temporal' and not str(name).startswith('track_'):continue
                            column = 'current_'+str(name) if label=='current' else str(name)
                            valid = features[label+'_valid'][n,k,frame-3,1,j]
                            event[column] = float(features[label+'_features'][n,k,frame-3,1,j]) if valid else ''
                    for j,name in enumerate(('ordinary_smooth_margin','ordinary_raw_slope5','ordinary_raw_detrended_fluctuation5')):
                        event['score_'+name] = float(score[n,k,frame-3,1,j])
                    assert event['event_id'] not in expected_events
                    expected_events[event['event_id']] = event;entries.append(event)
                selections[kind] = entries
            groups[(split,seed)] = selections
            for n in np.flatnonzero(body_contact):
                for k in range(4):
                    identity = (split,seed,int(d['ids'][n]),k)
                    expected_contacts[identity] = dict(split=split,seed=seed,scene=int(d['ids'][n]),replica=k,height='BODY',
                        before_first=int(before[n,k,1]),after_first=int(after[n,k,1]),
                        before_timely_first=int(bt[n,k,1]),after_timely_first=int(at[n,k,1]),
                        first_added=int(af[n,k]),rescue=int(rescue[n,k]),earlier=int(earlier[n,k]),
                        lost=int(bt[n,k,1]>=0 and at[n,k,1]<0))
            headonly = extra&added[...,0]&~added[...,1]
            bodyonly = extra&added[...,1]&~added[...,0]
            both = extra&added[...,0]&added[...,1]
            np.testing.assert_array_equal(headonly.astype(int)+bodyonly.astype(int)+both.astype(int),extra.astype(int))
            base_joint_q = np.zeros_like(added);base_joint_q[...,0] = joint_base
            after_joint_q = np.zeros_like(added);after_joint_q[...,0] = joint_after
            bf = A.first(base_joint_q)[...,0];nf = A.first(after_joint_q)[...,0]
            for n in np.flatnonzero(clear):
                for k in range(4):
                    identity = (split,seed,int(d['ids'][n]),k)
                    expected_clips[identity] = dict(split=split,seed=seed,scene=int(d['ids'][n]),replica=k,
                        before_first=int(bf[n,k]),after_first=int(nf[n,k]),first_body_added=int(af[n,k]),
                        first_jointnew_body_contributor=int(jointnew_first[n,k]),extra_headonly_slots=int(headonly[n,k].sum()),
                        extra_bodyonly_slots=int(bodyonly[n,k].sum()),extra_both_slots=int(both[n,k].sum()),
                        extra_joint_slots=int(extra[n,k].sum()),body_added_candidate_slots=int(added[n,k,:,1].sum()),
                        head_added_candidate_slots=int(added[n,k,:,0].sum()))
            statistics[f'{split}/{seed}'] = dict(events={name:len(value) for name,value in selections.items()},
                clear=dict(sceneK_clips=int(clear.sum())*4,joint_slot_denominator=int(clear.sum())*4*13,
                    baseline_slots=int(joint_base[clear].sum()),current_slots=int(joint_after[clear].sum()),
                    extra_joint_slots=int(extra[clear].sum()),extra_headonly_slots=int(headonly[clear].sum()),
                    extra_bodyonly_slots=int(bodyonly[clear].sum()),extra_both_slots=int(both[clear].sum()),
                    body_candidate_added_slots=int(added[clear,...,1].sum()),baseline_clips=int((bf[clear]>=0).sum()),
                    current_clips=int((nf[clear]>=0).sum()),new_clips=int(((bf[clear]<0)&(nf[clear]>=0)).sum()),
                    existing_clips_earlier=int(((bf[clear]>=0)&(nf[clear]>=0)&(nf[clear]<bf[clear])).sum()),
                    existing_clips_same_first=int(((bf[clear]>=0)&(nf[clear]==bf[clear])).sum())))
    seen = set()
    actual_events = rows(OUT/'events.csv')
    for event in actual_events:
        identity = event['event_id'];assert identity not in seen;seen.add(identity)
        compare_row(event, expected_events[identity], 'event/'+identity)
    A.eq(seen,set(expected_events),'complete events')
    for name, expected in (('body_contact_events',expected_contacts),('clear_clips',expected_clips)):
        seen = set()
        for row in rows(OUT/(name+'.csv')):
            identity = (row['split'],int(row['seed']),int(row['scene']),int(row['replica']))
            assert identity not in seen;seen.add(identity)
            compare_row(row,expected[identity],name)
        A.eq(seen,set(expected),'complete/'+name)
    actual_attempts = rows(OUT/'matching_attempts.csv');actual_pairs = rows(OUT/'matched_pairs.csv')
    pair_lookup = {(r['panel'],r['anchor_event_id']):r for r in actual_pairs}
    attempt_lookup = {(r['panel'],r['anchor_event_id']):r for r in actual_attempts}
    assert len(pair_lookup)==len(actual_pairs) and len(attempt_lookup)==len(actual_attempts)
    expected_pair_ids = set();expected_attempt_ids = set();unmatched = {}
    for (split,seed), events in groups.items():
        panel_stats = {}
        for anchors in ('rescue','earlier'):
            for controls in ('clear_candidate','clear_jointnew','pure_pass_body'):
                panel = f'{split}/{seed}/{anchors}_vs_{controls}'
                # Matching works on string-valued records exactly as the persisted CSV.
                converted = lambda records:[{k:str(v) for k,v in r.items()} for r in records]
                pairs, attempts, reasons = match(converted(events[anchors]),converted(events[controls]),panel)
                unmatched[panel] = reasons
                for pair in pairs:
                    identity = (panel,pair['anchor_event_id']);expected_pair_ids.add(identity)
                    compare_row(pair_lookup[identity],pair,'pair/'+panel)
                for attempt in attempts:
                    identity = (panel,attempt['anchor_event_id']);expected_attempt_ids.add(identity)
                    compare_row(attempt_lookup[identity],attempt,'attempt/'+panel)
                panel_stats[anchors+'_vs_'+controls] = dict(anchors=len(events[anchors]),controls=len(events[controls]),
                    matched=len(pairs),not_evaluable=len(attempts)-len(pairs),
                    frame_delta=[p['control_minus_anchor_frame'] for p in pairs],
                    surviving_anchor_count_by_stage={stage:sum(a[stage]>0 for a in attempts) for stage in STAGES})
        statistics[f'{split}/{seed}']['matching'] = panel_stats
    A.eq(expected_pair_ids,set(pair_lookup),'complete matching pairs')
    A.eq(expected_attempt_ids,set(attempt_lookup),'complete matching attempts')
    A.eq(statistics,A.read(OUT/'stats.json'),'complete stats')
    seconds = time.monotonic()-began;assert seconds<=90
    result = dict(status='PASS',CPU_seconds=seconds,CPU_seconds_cap=90,GPU_seconds=0,
        events=len(expected_events),BODY_contact_events=len(expected_contacts),clear_clips=len(expected_clips),
        matching_attempts=len(attempt_lookup),matched_pairs=len(pair_lookup),scalar_checks=A.CHECKS,
        audit_source_sha256=sha(Path(__file__)),producer_source_sha256=sha(producer),
        helpers_sha256=sha(Path(A.__file__)),
        independence='Independent baseline smoothing/first-event selection and runtime ledger values; one-to-one ordered greedy exact strata/calipers/ties/distances, full cascade and unmatched reasons, clear HEAD/BODY/both slot disjoint decomposition. No producer helpers/models/refits/raw hist/boxes.')
    (target/'unmatched_reasons.json').write_text(json.dumps(dict(
        limitation='First failing condition is cascade-order-dependent; no match is NOT_EVALUABLE, not evidence against a mechanism. No-reuse is per split/seed/panel; physical observations may recur across seeds or panels.',
        reasons=unmatched),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (target/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    with threadpool_limits(limits=2):run()
