"""Observed alert-chain diagnosis on frozen baseline/both/head50 grade caches.

No new state policy, notification dispatch, accumulation, fitting or thresholds.
"""
from pathlib import Path
import time
import numpy as np
import cnh_graded_corridor_eval_dev as C
import cnh_graded_evidence_dev as G

ROOT = C.ROOT
PARENT = ROOT/'artifacts.local/work/cnh-graded-peak-head-cal-dev-20261010'
OUT = ROOT/'artifacts.local/work/cnh-graded-alert-chain-dev-20261010'
POLICIES = ('baseline', 'both', 'head50')
CATEGORIES = ('contact', 'pass', 'clear')


def first(flags, end=13):
    indices = np.flatnonzero(flags[:end])
    return int(indices[0]+3) if len(indices) else -1


def runs(flags):
    flags = np.asarray(flags, bool)
    starts = np.flatnonzero(flags & ~np.r_[False, flags[:-1]])
    ends = np.flatnonzero(flags & ~np.r_[flags[1:], False])
    return int(len(starts)), int((ends-starts+1).max()) if len(starts) else 0


def scan(grade, base):
    any_alert, light, strong = grade > 0, grade == 1, grade == 2
    fa, fl, fs = first(any_alert), first(light), first(strong)
    initial = int(grade[fa-3]) if fa >= 0 else 0
    chain = 'silent' if initial == 0 else 'direct_strong' if initial == 2 else 'light_then_strong' if fs >= 0 else 'light_no_strong'
    escalation = chain == 'light_then_strong'
    segments, longest = runs(any_alert)
    lsegments, llongest = runs(light); ssegments, slongest = runs(strong)
    added = light & (base == 0)
    asegments, alongest = runs(added)
    visible = np.flatnonzero(any_alert)
    gaps, gap_longest = runs(~any_alert[visible[0]:visible[-1]+1]) if len(visible) else (0, 0)
    before_full, before_timely = first(base > 0), first(base > 0, 11)
    timely = first(any_alert, 11)
    return dict(grades=''.join(map(str, grade.tolist())), first_any=fa, first_light=fl, first_strong=fs,
        first_timely=timely, first_strong_timely=first(strong, 11), initial_grade=initial, chain=chain,
        light_to_strong_gap_frames=fs-fa if escalation else -1,
        contiguous_upgrade=int(escalation and any_alert[fa-3:fs-2].all()),
        any_slots=int(any_alert.sum()), light_slots=int(light.sum()), strong_slots=int(strong.sum()),
        any_segments=segments, any_longest_frames=longest, light_segments=lsegments, light_longest_frames=llongest,
        strong_segments=ssegments, strong_longest_frames=slongest, internal_silent_gaps=gaps,
        internal_silent_longest_frames=gap_longest,
        added_light_first=first(added), added_light_slots=int(added.sum()), added_light_segments=asegments,
        added_light_longest_frames=alongest, baseline_first_any=before_full, baseline_first_timely=before_timely,
        timely_rescue=int(before_timely < 0 <= timely),
        timely_earlier=int(before_timely >= 0 and timely >= 0 and timely < before_timely),
        new_alert_clip=int(before_full < 0 <= fa),
        earlier_existing_clip=int(before_full >= 0 and fa >= 0 and fa < before_full))


def aggregate(rows):
    names = ('any_slots', 'light_slots', 'strong_slots', 'any_segments', 'light_segments', 'strong_segments',
             'internal_silent_gaps', 'added_light_slots', 'added_light_segments',
             'timely_rescue', 'timely_earlier', 'new_alert_clip', 'earlier_existing_clip')
    result = dict(streams=len(rows), active_streams=sum(r['first_any'] >= 0 for r in rows),
        timely_streams=sum(r['first_timely'] >= 0 for r in rows),
        timely_strong_streams=sum(r['first_strong_timely'] >= 0 for r in rows),
        **{name:sum(r[name] for r in rows) for name in names},
        **{chain:sum(r['chain'] == chain for r in rows) for chain in ('silent', 'direct_strong', 'light_then_strong', 'light_no_strong')},
        contiguous_upgrades=sum(r['contiguous_upgrade'] for r in rows),
        gapped_upgrades=sum(r['chain'] == 'light_then_strong' and not r['contiguous_upgrade'] for r in rows),
        fragmented_any_streams=sum(r['any_segments'] > 1 for r in rows),
        fragmented_added_light_streams=sum(r['added_light_segments'] > 1 for r in rows),
        any_longest_frames=max((r['any_longest_frames'] for r in rows), default=0),
        light_longest_frames=max((r['light_longest_frames'] for r in rows), default=0),
        strong_longest_frames=max((r['strong_longest_frames'] for r in rows), default=0),
        internal_silent_longest_frames=max((r['internal_silent_longest_frames'] for r in rows), default=0),
        added_light_longest_frames=max((r['added_light_longest_frames'] for r in rows), default=0))
    gaps = [r['light_to_strong_gap_frames'] for r in rows if r['light_to_strong_gap_frames'] >= 0]
    result['median_upgrade_gap_frames'] = float(np.median(gaps)) if gaps else None
    return result


def run():
    began = time.monotonic()
    if (OUT/'PLAN.json').exists(): raise FileExistsError('Preserve prior chain attempt')
    inputs = {str((PARENT/name).relative_to(ROOT)):C.sha(PARENT/name) for name in
        ('PLAN.json', 'metrics.json', 'cal_grades.npz', 'validation_grades.npz')}
    C.save(OUT/'PLAN.json', dict(task='CNH_GRADED_ALERT_CHAIN_DEV_20261010',
        authorization='User 继续 fixed light-to-strong event-chain diagnosis', lane='EXPLORE consumed simulated Development',
        budget=dict(main_CPU_command_wall_seconds=90, audit_CPU_command_wall_seconds=90,
                    integration_CPU_command_wall_seconds=120, total_CPU_command_wall_seconds=300, GPU_seconds=0),
        goal='Determine observed early-light value, later strong support and fragmentation, separately contact/pass/clear',
        backend='TASK_NOT_GPU_SUITABLE: cached13-frame integer stream scanning and CSV',
        inputs_sha256=inputs, source_sha256=C.sha(Path(__file__)), seeds=list(G.SEEDS), policies=POLICIES,
        population='cal/validation each384x4x13x2, query HEAD/BODY and joint(maxgrade), all categories including silent',
        chain='first observed alert atf3..15 determines initial grade: silent/directstrong/lightthenstrong/lightnostrong; light after earlierstrong not lightfirst',
        gap='firststrong minus firstany for lightfirst, else-1; contiguous when all grades from firstany through firststrong positive; silent gaps only between first/last any, boundaries excluded',
        addition='query: grade1 and baseline0; joint: maxgrade1 and baseline maxgrade0, not hidden lower-grade other query',
        effects='querycontact baseline timely miss->rescue or both timely earlier; jointclear/purepass first full-window miss->new_clip or existing full first earlier. Separate group denominators, not a newly evaluated rule.',
        decision_check='contact query384 per height, jointcontact768/purepass256/clear512 per split;1 event resolution. Never strong in finite window is not negative evidence; useful light need not upgrade. No outcome-selected filtering or success margin.',
        limits='First observed in13 frames, not first lifetime alert; same-query stream not same-object track. Joint escalation may cross height; no actual audio/haptic count, real latency or risk label over time.',
        adjustable_scope='Implementation/schema repair and scoped verification; fixed grades/policies/thresholds',
        stop='Complete82944 stream rows/162 summaries/144 effect summaries; no new notification/hysteresis or score rule'))
    try:
        for path, digest in inputs.items():
            if C.sha(ROOT/path) != digest: raise ValueError(f'Input drift: {path}')
        streams, summaries, effects = [], [], []
        for split in ('cal', 'validation'):
            with np.load(PARENT/f'{split}_grades.npz', allow_pickle=False) as a:
                grades = dict(zip(a['keys'].tolist(), a['grades']))
                ids, category = a['scene_ids'], a['category']
            assert category.shape == (384, 2)
            joint_category = np.where((category == 'contact').any(-1), 'contact',
                np.where((category == 'pass').any(-1), 'pass', 'clear'))
            for seed in G.SEEDS:
                base = grades[f'{seed}/baseline']
                for policy in POLICIES:
                    if time.monotonic()-began >= 90: raise TimeoutError('Main90s cap')
                    grade = grades[f'{seed}/{policy}']
                    np.testing.assert_array_equal(grade == 2, base == 2)
                    group = []
                    for n, scene in enumerate(ids):
                        for k in range(4):
                            for q, height in enumerate(('HEAD', 'BODY')):
                                row = dict(split=split, seed=seed, policy=policy, level='query', height=height,
                                    scene=int(scene), replica=k, category=category[n, q], **scan(grade[n, k, :, q], base[n, k, :, q]))
                                streams.append(row); group.append(row)
                            row = dict(split=split, seed=seed, policy=policy, level='joint', height='ALL',
                                scene=int(scene), replica=k, category=joint_category[n], **scan(grade[n, k].max(-1), base[n, k].max(-1)))
                            streams.append(row); group.append(row)
                    for level, height in (('query', 'HEAD'), ('query', 'BODY'), ('joint', 'ALL')):
                        for cat in CATEGORIES:
                            selected = [r for r in group if r['level'] == level and r['height'] == height and r['category'] == cat]
                            identity = dict(split=split, seed=seed, policy=policy, level=level, height=height, category=cat)
                            summaries.append(dict(**identity, **aggregate(selected)))
                            if level == 'query' and cat == 'contact':
                                for effect, flag in (('rescue', 'timely_rescue'), ('earlier', 'timely_earlier')):
                                    effects.append(dict(**identity, effect=effect, **aggregate([r for r in selected if r[flag]])))
                            elif level == 'joint' and cat != 'contact':
                                for effect, flag in (('new_clip', 'new_alert_clip'), ('earlier_clip', 'earlier_existing_clip')):
                                    effects.append(dict(**identity, effect=effect, **aggregate([r for r in selected if r[flag]])))
        G.write_csv(OUT/'streams.csv', streams); G.write_csv(OUT/'summary.csv', summaries)
        G.write_csv(OUT/'effect_summary.csv', effects)
        C.save(OUT/'receipt.json', dict(status='COMPLETE', seconds=time.monotonic()-began, streams=len(streams),
            summaries=len(summaries), effect_summaries=len(effects), source_sha256=C.sha(Path(__file__)),
            GPU_seconds=0, fit=0, prediction=0, new_raw=0, grade_policy_changes=0))
        print(f'COMPLETE: {len(streams)} streams, {len(summaries)} groups, {len(effects)} effects, {time.monotonic()-began:.3f}s')
    except BaseException as error:
        C.save(OUT/f'failure_{time.time_ns()}.json', dict(error=repr(error), seconds=time.monotonic()-began)); raise


if __name__ == '__main__': run()
