"""Causal per-query notification episodes on fixed grade caches, gap0 versus1.

Detector grades never change. These are offline notification counts, not App
dispatch or distinct-object measurements.
"""
from pathlib import Path
import time
import numpy as np
import cnh_graded_corridor_eval_dev as C
import cnh_graded_evidence_dev as G

ROOT = C.ROOT
PARENT = ROOT/'artifacts.local/work/cnh-graded-peak-head-cal-dev-20261010'
OUT = ROOT/'artifacts.local/work/cnh-graded-episode-merge-dev-20261010'
GRADE_POLICIES = ('baseline', 'both', 'head50')
NOTIFY_POLICIES = ('framewise', 'episode0', 'episode1')


class EpisodeNotifier:
    """Only current grade and task-local state; zeros emit nothing."""
    def __init__(self, allowed_gap):
        if allowed_gap not in (0, 1): raise ValueError('Fixed gap0/1 only')
        self.gap = allowed_gap
        self.peak = self.quiet = self.counter = self.current_id = 0

    def step(self, grade):
        if grade not in (0, 1, 2): raise ValueError('Grade must be0/1/2')
        onset = upgrade = emission = 0
        if grade == 0:
            self.quiet += 1
            if self.quiet > self.gap:
                self.peak = self.current_id = 0
        else:
            self.quiet = 0
            if self.peak == 0:
                self.counter += 1
                self.current_id = self.counter
                onset = 1
            if grade > self.peak:
                upgrade = int(self.peak == 1 and grade == 2)
                emission = grade
                self.peak = grade
        return emission, self.current_id, onset, upgrade


def replay(values, policy):
    values = np.asarray(values, np.int8)
    if policy == 'framewise':
        # Counterfactual per-positive-frame dispatcher, not current App evidence.
        onset = (values > 0).astype(np.int8)
        return values.copy(), np.where(values > 0, onset.cumsum(), 0).astype(np.int8), onset, np.zeros_like(values)
    machine = EpisodeNotifier(int(policy[-1]))
    result = np.array([machine.step(int(v)) for v in values], np.int8)
    return tuple(result[:, i] for i in range(4))


def first(values, strong=False, end=13):
    indices = np.flatnonzero((values[:end] == 2) if strong else (values[:end] > 0))
    return int(indices[0]+3) if len(indices) else -1


def describe_stream(grade, emitted, ids, onsets, upgrades):
    before, after = first(grade), first(emitted)
    before_strong, after_strong = first(grade, True), first(emitted, True)
    before_timely, after_timely = first(grade, end=11), first(emitted, end=11)
    before_strong_timely, after_strong_timely = first(grade, True, 11), first(emitted, True, 11)
    return dict(grades=''.join(map(str, grade.tolist())), notifications=''.join(map(str, emitted.tolist())),
        episode_ids=','.join(map(str, ids.tolist())), onset_flags=''.join(map(str, onsets.tolist())),
        upgrade_flags=''.join(map(str, upgrades.tolist())),
        notification_count=int((emitted > 0).sum()), light_notifications=int((emitted == 1).sum()),
        strong_notifications=int((emitted == 2).sum()), starts=int(onsets.sum()), upgrades=int(upgrades.sum()),
        repeat_notifications=int((emitted > 0).sum())-int(after >= 0),
        repeat_strong_notifications=int((emitted == 2).sum())-int(after_strong >= 0),
        grade_first_any=before, notification_first_any=after, grade_first_strong=before_strong,
        notification_first_strong=after_strong, grade_first_timely=before_timely, notification_first_timely=after_timely,
        grade_first_strong_timely=before_strong_timely, notification_first_strong_timely=after_strong_timely)


def aggregate(rows):
    names = ('notification_count', 'light_notifications', 'strong_notifications', 'starts', 'upgrades',
             'repeat_notifications', 'repeat_strong_notifications')
    return dict(streams=len(rows), active_streams=sum(r['notification_first_any'] >= 0 for r in rows),
        timely_streams=sum(r['notification_first_timely'] >= 0 for r in rows),
        timely_strong_streams=sum(r['notification_first_strong_timely'] >= 0 for r in rows),
        **{name:sum(r[name] for r in rows) for name in names},
        first_any_changed=sum(r['grade_first_any'] != r['notification_first_any'] for r in rows),
        first_strong_changed=sum(r['grade_first_strong'] != r['notification_first_strong'] for r in rows))


def run():
    began = time.monotonic()
    if (OUT/'PLAN.json').exists(): raise FileExistsError('Preserve prior episode attempt')
    inputs = {str((PARENT/name).relative_to(ROOT)):C.sha(PARENT/name) for name in
              ('PLAN.json', 'metrics.json', 'cal_grades.npz', 'validation_grades.npz')}
    C.save(OUT/'PLAN.json', dict(task='CNH_GRADED_EPISODE_MERGE_DEV_20261010',
        authorization='User 继续 fixed0/1gap per-query episode pilot', lane='EXPLORE consumed simulated Development',
        budget=dict(main_CPU_command_wall_seconds=90, audit_CPU_command_wall_seconds=90,
                    integration_CPU_command_wall_seconds=120, total_CPU_command_wall_seconds=300, GPU_seconds=0),
        goal='Reduce repeat notifications preserving first any and first strong timing and original detector grades',
        backend='TASK_NOT_GPU_SUITABLE:13-frame integer state machines and cached CSV counts',
        inputs_sha256=inputs, source_sha256=C.sha(Path(__file__)), seeds=list(G.SEEDS),
        grade_policies=GRADE_POLICIES, notification_policies=NOTIFY_POLICIES,
        state='Current grade only; zero quiet++, reset when quiet>gap; positive quiet0, new episode if peak0; emit current grade only if>episode peak. Peak never downgrades; no zero emission.',
        ids='Positive episode id retained across permitted silent frame, else0; retained id is notification memory, not obstacle occupancy',
        framewise='Counterfactual notify everypositive frame, each markedstart for accounting; not App baseline or real notification rate',
        timing='f3..13 timely/f3..15 full; first any and first strong conserved; first grade1 after an earlierstrong may be suppressed, no all-firstlight equality claim',
        decision_check='Eachcontactquery384 per full split; jointcontact768/purepass256/clear512. Same first timings are construction constraints, not safety proof; counts of repeated same-query prompts are not counts of distinct obstacles. Fixed0/1 no outcome-selected hold sweep.',
        stop='Complete fixed54 cells/165888query rows/324query summaries; no longer gap, classification changes or App/device dispatch',
        adjustable_scope='Implementation repairs and state/prefix/fixture checks only, grade/threshold/population fixed',
        limits='13-frame independent clip reset; cross-clip and variable frame-rate untested; no long-term reminder refresh/same-object guarantee; no held occupancy or free inference'))
    try:
        for path, digest in inputs.items():
            if C.sha(ROOT/path) != digest: raise ValueError(f'Input drift: {path}')
        streams, summary, metrics = [], [], {}
        for split in ('cal', 'validation'):
            with np.load(PARENT/f'{split}_grades.npz', allow_pickle=False) as a:
                grades = dict(zip(a['keys'].tolist(), a['grades']))
                scene_ids, category = a['scene_ids'], a['category']
            saved, saved_ids, keys = [], [], []
            for seed in G.SEEDS:
                for grade_policy in GRADE_POLICIES:
                    grade = grades[f'{seed}/{grade_policy}']
                    controls = {}
                    for notify_policy in NOTIFY_POLICIES:
                        if time.monotonic()-began >= 90: raise TimeoutError('Main90s cap')
                        emitted = np.zeros_like(grade); ids = np.zeros_like(grade)
                        onsets = np.zeros_like(grade); upgrades = np.zeros_like(grade); group = []
                        for n, scene in enumerate(scene_ids):
                            for k in range(4):
                                for q, height in enumerate(('HEAD', 'BODY')):
                                    e, i, o, u = replay(grade[n, k, :, q], notify_policy)
                                    emitted[n, k, :, q], ids[n, k, :, q], onsets[n, k, :, q], upgrades[n, k, :, q] = e, i, o, u
                                    row = dict(split=split, seed=seed, grade_policy=grade_policy, notify_policy=notify_policy,
                                        scene=int(scene), replica=k, height=height, category=category[n, q],
                                        **describe_stream(grade[n, k, :, q], e, i, o, u))
                                    assert row['grade_first_any'] == row['notification_first_any']
                                    assert row['grade_first_strong'] == row['notification_first_strong']
                                    streams.append(row); group.append(row)
                        assert np.all(emitted <= grade) and not np.any(emitted[grade == 0])
                        if notify_policy == 'episode1':
                            assert np.all((emitted == 1) <= (controls['episode0'] == 1))
                            assert np.all((emitted == 2) <= (controls['episode0'] == 2))
                        controls[notify_policy] = emitted
                        for q, height in enumerate(('HEAD', 'BODY')):
                            for cat in ('contact', 'pass', 'clear'):
                                selected = [r for r in group if r['height'] == height and r['category'] == cat]
                                summary.append(dict(split=split, seed=seed, grade_policy=grade_policy,
                                    notify_policy=notify_policy, height=height, category=cat, **aggregate(selected)))
                        masks = dict(contact=(category == 'contact').any(-1),
                            pass_=(category == 'pass').any(-1) & ~(category == 'contact').any(-1),
                            clear=(category == 'clear').all(-1))
                        counts = {}
                        for name, mask in masks.items():
                            selected, joint = emitted[mask], emitted[mask].max(-1)
                            counts[name.rstrip('_')] = dict(clip_denominator=int(mask.sum())*4,
                                query_notifications=int((selected > 0).sum()), query_light=int((selected == 1).sum()), query_strong=int((selected == 2).sum()),
                                query_starts=int(onsets[mask].sum()), query_upgrades=int(upgrades[mask].sum()),
                                joint_notifications=int((joint > 0).sum()), joint_light=int((joint == 1).sum()), joint_strong=int((joint == 2).sum()),
                                joint_clips=int((joint > 0).any(-1).sum()))
                        key = f'{seed}/{grade_policy}/{notify_policy}'
                        metrics[f'{split}/{key}'] = dict(counts=counts,
                            paired_any_vs_grade=G.paired_timing(grade > 0, emitted > 0, category),
                            paired_strong_vs_grade=G.paired_timing(grade == 2, emitted == 2, category))
                        keys.append(key); saved.append(emitted); saved_ids.append(ids)
            np.savez_compressed(OUT/f'{split}_notifications.npz', keys=np.array(keys), notifications=np.array(saved),
                episode_ids=np.array(saved_ids), scene_ids=scene_ids, category=category)
        G.write_csv(OUT/'streams.csv', streams); G.write_csv(OUT/'summary.csv', summary)
        C.save(OUT/'metrics.json', metrics)
        C.save(OUT/'receipt.json', dict(status='COMPLETE', seconds=time.monotonic()-began, streams=len(streams),
            summaries=len(summary), cells=len(metrics), source_sha256=C.sha(Path(__file__)), GPU_seconds=0,
            fit=0, prediction=0, new_raw=0, detector_grade_changes=0))
        print(f'COMPLETE: {len(metrics)} cells, {len(streams)} streams, {len(summary)} groups, {time.monotonic()-began:.3f}s')
    except BaseException as error:
        C.save(OUT/f'failure_{time.time_ns()}.json', dict(error=repr(error), seconds=time.monotonic()-began)); raise


if __name__ == '__main__': run()
