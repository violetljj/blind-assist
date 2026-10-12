"""Independent cached-score audit for fixed BODY-only additive light alerts.

No producer import, fit, prediction, raw reconstruction or threshold adjustment.
Clocks, cost transitions, contact unions and report counts are recomputed here.
"""
import csv
import json
from pathlib import Path
import time

import numpy as np
import cnh_graded_corridor_eval_dev as C
import cnh_graded_evidence_dev as G
import cnh_counterfactual_eval_dev as E

OUT = C.ROOT/'artifacts.local/work/cnh-graded-peak-body-only-dev-20261010'
PARENT = C.ROOT/'artifacts.local/work/cnh-graded-peak-joint-dev-20261010'
POLICIES = ('baseline', 'both', 'body_only')
CHECKS = 0


def equal(actual, expected, label):
    global CHECKS
    if isinstance(expected, dict):
        assert set(actual) == set(expected), (label, set(actual)^set(expected))
        for key, value in expected.items(): equal(actual[key], value, label+'/'+str(key))
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected), (label, len(actual), len(expected))
        for i, value in enumerate(expected): equal(actual[i], value, label+'/'+str(i))
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected, err_msg=label)
        CHECKS += expected.size
    else:
        if isinstance(expected, (float, np.floating)):
            assert np.isclose(actual, expected, rtol=1e-12, atol=1e-12), (label, actual, expected)
        else: assert actual == expected, (label, actual, expected)
        CHECKS += 1


def clock(flags, end=13):
    frames = np.arange(3, 3+end)
    minimum = np.where(flags[..., :end, :], frames[:, None], 99).min(-2)
    return np.where(minimum == 99, -1, minimum)


def joint_clock(flags, category, end=11):
    true_query = category[:, None, None, :] == 'contact'
    values = clock(flags & true_query, end)
    minimum = np.where(values >= 0, values, 99).min(-1)
    return np.where(minimum == 99, -1, minimum)


def pair_row(a, b, height=None, physical=False):
    both = (a >= 0)&(b >= 0)
    delta = a[both]-b[both]
    result = dict(denominator=int(a.size), rescue=int(((a<0)&(b>=0)).sum()),
        loss=int(((a>=0)&(b<0)).sum()), both_timely=int(both.sum()),
        earlier=int((delta>0).sum()), same=int((delta==0).sum()), later=int((delta<0).sum()),
        median_advance_frames=float(np.median(delta)) if len(delta) else None,
        advance_frame_histogram={str(int(n)):int((delta==n).sum()) for n in sorted(set(delta.tolist()))})
    if height is not None: result = dict(height=height, **result)
    if physical:
        result['baseline'] = int((a>=0).sum())
        result['candidate'] = int((b>=0).sum())
    return result


def paired(before, after, category):
    a, b = clock(before, 11), clock(after, 11)
    return [pair_row(a[category[:, q]=='contact', :, q],
        b[category[:, q]=='contact', :, q], E.HEIGHTS[q]) for q in range(2)]


def physical_pair(before, after, category):
    mask = (category=='contact').any(1)
    a, b = joint_clock(before, category), joint_clock(after, category)
    return pair_row(a[mask], b[mask], physical=True)


def cost(flags, mask):
    selected = flags[mask]
    segments = 0
    longest = 0
    for row in selected.reshape(-1, 13):
        length = 0
        for value in row:
            if value:
                if length == 0: segments += 1
                length += 1
                longest = max(longest, length)
            else: length = 0
    return dict(slots=int(selected.sum()), slot_denominator=int(selected.size),
        clips=int(selected.any(-1).sum()), clip_denominator=int(np.prod(selected.shape[:-1])),
        segments=segments, longest_run_frames=longest)


def summary(flags, category):
    contact = category=='contact'
    clear = (category=='clear').all(1)
    pure_pass = (category=='pass').any(1)&~contact.any(1)
    timely = (clock(flags, 11)>=0)&contact[:, None, :]
    late = (clock(flags)>=14)&contact[:, None, :]
    joint = flags.any(-1)
    cc = cost(joint, clear)
    return dict(counts=timely.sum((0, 1)).tolist(), late_counts=late.sum((0, 1)).tolist(),
        clear_slots=cc['slots'], clear_denominator=cc['slot_denominator'],
        clear_segments=cc['segments'], clear_clips=cc['clips'],
        physical_contact_any_height=int(joint[contact.any(1), :, :11].any(-1).sum()),
        pass_clips=int(joint[pure_pass].any(-1).sum()),
        contact_event_denominators=(contact.sum(0)*flags.shape[1]).tolist(),
        physical_contact_denominator=int(contact.any(1).sum()*flags.shape[1]),
        clear_clip_denominator=cc['clip_denominator'],
        pass_clip_denominator=int(pure_pass.sum()*flags.shape[1]))


def describe(grade, data, references):
    category = data['category']
    result = {name:summary(flags, category) for name, flags in
        (('any',grade>0), ('strong',grade==2), ('light',grade==1))}
    for label, flags in (('paired_any',grade>0), ('paired_strong',grade==2)):
        result[label] = {name:paired(value, flags, category) for name, value in references.items()}
    clear = (category=='clear').all(1)
    pure_pass = (category=='pass').any(1)&~(category=='contact').any(1)
    joint = grade.max(-1)
    result['joint_costs'] = {kind:{name:cost(joint==number, mask) for name,number in
        (('light',1), ('strong',2))} for kind,mask in (('clear',clear), ('pass',pure_pass))}
    strong, light = clock(grade==2,11)>=0, clock(grade==1,11)>=0
    result['light_only_timely_contact'] = ((light&~strong)&(category=='contact')[:,None,:]).sum((0,1)).tolist()
    result['old_alert_query_slots_to_light'] = int((references['old_fusion']&(grade==1)).sum())
    result['ordinary_OR_query_slots_to_light'] = int((references['ordinary_OR']&(grade==1)).sum())
    result['contact_old_timely_to_light_only'] = ((clock(references['old_fusion'],11)>=0)&light&~strong&
        (category=='contact')[:,None,:]).sum((0,1)).tolist()
    result['groups'] = {name:dict(any=summary((grade>0)[mask],category[mask]),
        vs_old_fusion=paired(references['old_fusion'][mask],(grade>0)[mask],category[mask]))
        for name,mask in data['groups'].items()}
    return result


def read_csv(path):
    with path.open(encoding='utf8', newline='') as stream: return list(csv.DictReader(stream))


def addition_costs(added, category):
    masks = {'clear':(category=='clear').all(1),
        'pass':(category=='pass').any(1)&~(category=='contact').any(1)}
    return {name:dict(joint=cost(added.any(-1), mask),
        **{height:cost(added[..., q], mask) for q,height in enumerate(E.HEIGHTS)})
        for name,mask in masks.items()}


def new_costs(before, after, category):
    a, b = before.any(-1), after.any(-1)
    fa, fb = clock(a[...,None])[...,0], clock(b[...,None])[...,0]
    masks = {'clear':(category=='clear').all(1),
        'pass':(category=='pass').any(1)&~(category=='contact').any(1)}
    return {name:dict(new_slots=int((b[mask]&~a[mask]).sum()),
        new_clips=int(((fa[mask]<0)&(fb[mask]>=0)).sum()),
        existing_clips_earlier=int(((fa[mask]>=0)&(fb[mask]>=0)&(fa[mask]>fb[mask])).sum()))
        for name,mask in masks.items()}


def summary_row(split, seed, policy, r):
    p, n = r['paired_any']['prior_light'], r['new_costs_vs_baseline']
    return dict(split=split, seed=seed, policy=policy,
        HEAD=r['any']['counts'][0], BODY=r['any']['counts'][1],
        HEAD_strong=r['strong']['counts'][0], BODY_strong=r['strong']['counts'][1],
        HEAD_light_only=r['light_only_timely_contact'][0], BODY_light_only=r['light_only_timely_contact'][1],
        HEAD_rescue=p[0]['rescue'], BODY_rescue=p[1]['rescue'], HEAD_loss=p[0]['loss'], BODY_loss=p[1]['loss'],
        HEAD_earlier=p[0]['earlier'], BODY_earlier=p[1]['earlier'], HEAD_later=p[0]['later'], BODY_later=p[1]['later'],
        physical_contact_query_timely=r['physical_contact_timely'],
        physical_rescue=r['physical_paired_vs_baseline']['rescue'],
        clear_slots=r['any']['clear_slots'], clear_clips=r['any']['clear_clips'],
        pass_slots=r['joint_costs']['pass']['strong']['slots']+r['joint_costs']['pass']['light']['slots'],
        pass_clips=r['any']['pass_clips'], extra_clear_slots=n['clear']['new_slots'],
        extra_clear_clips=n['clear']['new_clips'], extra_pass_slots=n['pass']['new_slots'],
        extra_pass_clips=n['pass']['new_clips'],
        clear_light_longest_frames=r['joint_costs']['clear']['light']['longest_run_frames'],
        pass_light_longest_frames=r['joint_costs']['pass']['light']['longest_run_frames'],
        added_clear_longest_frames=r['addition_costs']['clear']['joint']['longest_run_frames'],
        added_pass_longest_frames=r['addition_costs']['pass']['joint']['longest_run_frames'])


def run():
    started = time.monotonic()
    audit = OUT/'audit'
    if (audit/'PLAN.json').exists(): raise FileExistsError('Preserve original audit attempt')
    inputs = [OUT/name for name in ('PLAN.json','metrics.json','summary.csv','ledger.csv',
        'cal_grades.npz','validation_grades.npz','receipt.json')]
    C.save(audit/'PLAN.json', dict(task='INDEPENDENT_FIXED_BODY_ONLY_AUDIT',
        budget_CPU_command_wall_seconds=90, GPU_seconds=0, fit=0, prediction=0, new_raw=0,
        scope='Independent grades, first times, costs, contact groups, physical unions and complete ledgers; all18 cells',
        definitions='Timely f3..13; full f3..15; physical contact only true-contact queries, legacy any-height preserved',
        producer_imported=False, source_sha256=C.sha(Path(__file__)),
        inputs_sha256={str(p.relative_to(C.ROOT)):C.sha(p) for p in inputs}))
    try:
        plan = C.read(OUT/'PLAN.json')
        for path,digest in plan['inputs_sha256'].items():
            equal(C.sha(C.ROOT/path), digest, 'inherited_hash/'+path)
        producer = Path(__file__).with_name('cnh_graded_peak_body_only_dev.py')
        equal(C.sha(producer), plan['source_sha256'], 'producer_source_sha256')
        equal(plan['seeds'], list(G.SEEDS), 'declared_seeds')
        equal(plan['policies'], list(POLICIES), 'declared_policies')
        data = C.load()
        thresholds = C.read(C.PARENT/'thresholds.json')
        cuts = C.read(PARENT/'calibrations.json')
        published = C.read(OUT/'metrics.json')
        summary_csv, ledger_csv = read_csv(OUT/'summary.csv'), read_csv(OUT/'ledger.csv')
        equal(len(summary_csv), 18, 'summary_rows')
        equal(len(ledger_csv), 55296, 'ledger_rows')
        used_ledger, expected_keys = set(), set()
        ledger_by_key = {}
        for i,row in enumerate(ledger_csv):
            key = (row['split'], int(row['seed']), row['policy'], int(row['scene']), int(row['replica']), row['height'])
            assert key not in ledger_by_key, ('Duplicate ledger identity',key)
            ledger_by_key[key] = (i,row)
        summary_by_key = {}
        for row in summary_csv:
            key = (row['split'],int(row['seed']),row['policy'])
            assert key not in summary_by_key, ('Duplicate summary',key)
            summary_by_key[key] = row
        for split,d in data.items():
            equal(d['m3'].shape, (384,4,13,2), split+'/population')
            category = d['category']
            with np.load(PARENT/f'{split}_scores.npz',allow_pickle=False) as archive:
                score_archive = dict(zip(archive['keys'].tolist(),archive['scores']))
            with np.load(PARENT/f'{split}_grades.npz',allow_pickle=False) as archive:
                parent_grade = dict(zip(archive['keys'].tolist(),archive['grades']))
            with np.load(OUT/f'{split}_grades.npz',allow_pickle=False) as archive:
                equal(archive['scene_ids'],d['scene_ids'],split+'/scene_ids')
                equal(archive['category'],category,split+'/category')
                equal(archive['keys'].tolist(),[f'{seed}/{p}' for seed in G.SEEDS for p in POLICIES],split+'/grade_keys')
                actual_grades = dict(zip(archive['keys'].tolist(),archive['grades']))
            for si,seed in enumerate(G.SEEDS):
                if time.monotonic()-started >= 90: raise TimeoutError('Independent audit90s cap')
                th = thresholds[str(seed)]
                ordinary = d['candidates'][0,si]
                old = (d['m3']>=E.OLD_RAISED)|(d['local']>=E.OLD_LOCAL)
                strong = old|(ordinary>=th['addition'])
                baseline = np.zeros(strong.shape,np.int8)
                baseline[ordinary>=th['single']] = 1
                baseline[strong] = 2
                score = score_archive[f'{seed}/score_current']
                cut = cuts[f'{seed}/score_current/c15_p64']
                theta = -np.inf if cut['nonbinding'] else cut['theta']
                added = (baseline==0)&np.isfinite(score)&(score>=theta)
                both = baseline.copy();both[added]=1
                body = baseline.copy();body[...,1][added[...,1]]=1
                equal(both,parent_grade[f'{seed}/score_current/c15_p64'],split+f'/{seed}/parent_grade')
                refs = dict(prior_light=baseline>0,ordinary_OR=strong,M3=d['m3']>=E.M3_THETA,old_fusion=old)
                before_full,before_timely = clock(baseline>0),clock(baseline>0,11)
                both_full,both_timely = clock(both>0),clock(both>0,11)
                for policy,grade in zip(POLICIES,(baseline,both,body)):
                    key = f'{split}/{seed}/{policy}';expected_keys.add(key)
                    equal(actual_grades[f'{seed}/{policy}'],grade,key+'/grades')
                    equal(grade==2,strong,key+'/strong_identity')
                    equal(grade[baseline>0],baseline[baseline>0],key+'/baseline_retained')
                    if policy=='body_only':
                        equal(grade[...,0],baseline[...,0],key+'/HEAD_exact_baseline')
                        equal(grade[...,1],both[...,1],key+'/BODY_exact_both')
                    expected = describe(grade,d,refs)
                    expected['paired_vs_both'] = paired(both>0,grade>0,category)
                    expected['physical_paired_vs_baseline'] = physical_pair(baseline>0,grade>0,category)
                    expected['physical_paired_vs_both'] = physical_pair(both>0,grade>0,category)
                    expected['physical_contact_timely'] = int((joint_clock(grade>0,category)>=0).sum())
                    expected['addition_costs'] = addition_costs((grade>0)&(baseline==0),category)
                    expected['new_costs_vs_baseline'] = new_costs(baseline>0,grade>0,category)
                    equal(published[key],expected,key+'/metrics')
                    sr = {k:str(v) for k,v in summary_row(split,seed,policy,expected).items()}
                    equal(summary_by_key[split,seed,policy],sr,key+'/summary')
                    clocks = dict(first_any=clock(grade>0),first_timely=clock(grade>0,11),
                        first_strong=clock(grade==2),first_light=clock(grade==1),
                        before_first_any=before_full,before_first_timely=before_timely,
                        both_first_any=both_full,both_first_timely=both_timely)
                    for n,row in enumerate(d['rows']):
                        for k in range(4):
                            for q,height in enumerate(E.HEIGHTS):
                                identity=(split,seed,policy,int(d['scene_ids'][n]),k,height)
                                index,actual = ledger_by_key[identity];used_ledger.add(index)
                                er=dict(split=split,seed=seed,policy=policy,scene=int(d['scene_ids'][n]),
                                    replica=k,height=height,category=category[n,q],shape_family=row['shape_family'],
                                    background_family=row['background_family'],
                                    **{name:int(value[n,k,q]) for name,value in clocks.items()})
                                equal(actual,{name:str(value) for name,value in er.items()},key+'/ledger/'+str(index))
        equal(set(published),expected_keys,'complete_metric_keys')
        equal(len(used_ledger),len(ledger_csv),'all_ledger_rows_checked')
        for path,digest in plan['inputs_sha256'].items():
            equal(C.sha(C.ROOT/path),digest,'post_audit_input_hash/'+path)
        receipt=C.read(OUT/'receipt.json')
        for name,value in dict(status='COMPLETE',cells=18,ledger_rows=55296,GPU_seconds=0,fit=0,prediction=0,new_raw=0,
            source_sha256=C.sha(producer)).items(): equal(receipt[name],value,'receipt/'+name)
        result=dict(status='PASS',cells=18,ledger_rows=55296,summary_rows=18,
            scalar_and_array_checks=CHECKS,seconds=time.monotonic()-started,GPU_seconds=0,
            source_sha256=C.sha(Path(__file__)),inputs_sha256_verified=len(plan['inputs_sha256']),
            independent='Producer never imported; clocks/minima, loops/transitions, paired and physical metrics independently implemented')
        C.save(audit/'result.json',result)
        print(json.dumps(result,ensure_ascii=False))
    except BaseException as error:
        C.save(audit/f'failure_{time.time_ns()}.json',dict(error=repr(error),seconds=time.monotonic()-started,
            scalar_and_array_checks=CHECKS))
        raise


if __name__ == '__main__': run()
