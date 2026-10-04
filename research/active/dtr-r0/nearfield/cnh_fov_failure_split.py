"""Consumed Development geometry diagnostic; no observation/model rendering.

Only evaluator geometry and sealed scores are read. No photon simulation or
threshold fitting is performed. All K replicas share the same ray geometry.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
from datetime import datetime, timezone

import numpy as np
import cnh_proposal_attribution_scenes as S
import cnh_location_reference_evaluate as L

ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / 'artifacts.local/work/cnh-readout-pilot2-20261005'
REF = ROOT / 'artifacts.local/work/cnh-location-reference-20261004'
OUT = ROOT / 'artifacts.local/work/cnh-fov-failure-split-20261005'
ARMS = ('M3', 'VD', 'D', 'R_any')
STATE_NAMES = ('OUT_OF_FOV', 'OCCLUDED', 'VISIBLE')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'PLAN.json').exists():
        raise FileExistsError('Resume existing frozen plan with --stage run')
    r = read(REF/'result.json')
    save(out/'PLAN.json', dict(task='CNH_FOV_FAILURE_SPLIT_20261005',
        created_utc=datetime.now(timezone.utc).isoformat(), scope='EXPLORE consumed Development',
        source_commit='113b2966', units=[s['unit'] for s in r['scenes']],
        expected_fov_out_scenes=11, expected_fov_in_scenes=37, branches=[0, 1], K=4,
        geometry_frames=list(range(16)), scored_frames=list(range(3, 16)),
        deadline='Same .9m timely cutoff. Include sampled geometry frames with front_range>=.9, including history0..2. No interpolated ray exposure.',
        states='VISIBLE first-hit object_id=0; OCCLUDED no first-hit target but target-only hit; OUT_OF_FOV no target-only subray hit. Partial occlusion retains VISIBLE.',
        classes=dict(A='No visible predeadline frame and more OUT_OF_FOV than OCCLUDED frames',
            B='No visible predeadline frame and at least as many OCCLUDED as OUT_OF_FOV frames (ties disclosed)',
            C='At least one visible predeadline frame, VD not timely',
            D='At least one visible predeadline frame, VD timely'),
        medians='Equal sequence summaries: median number visible frames; last-visible range among sequences with predeadline visibility (missing otherwise); median per-sequence median first-hit target rays/16384 over all predeadline frames including zeros. Also report visible-frames-only ray fraction.',
        interpretation='A+B>44 -> COVERAGE_PRIORITY; C>44 -> EXPOSURE_STRENGTH_PRIORITY (visibility duration/closer exposure, no readout restart); otherwise MIXED. These describe the full88 sequence cohort; also report misses denominator62.',
        smoke='Replay all-object first hit for first2 FOV_IN scenes, both shallow branches, all16 frames; zero ID and visible differences required before proceeding',
        thresholds=r['thresholds'], no_training=True, no_observation_render=True,
        time_budget_seconds=3600, over_budget='Finish FOV_OUT only, report FOV_IN NOT_RUN plus estimate; no new tasks',
        source_sha256={str(p): sha(p) for p in (Path(__file__), Path(S.__file__),
            Path(L.__file__), REF/'result.json', REF/'episode_ledger.npz', REF/'scores/all_scores.npz')}))
    print('PLAN frozen', sha(out/'PLAN.json'), flush=True)


def stats(rows):
    def median(key):
        x = [r[key] for r in rows if r[key] is not None]
        return None if not x else float(np.median(x))
    return dict(n=len(rows), unique_scenes=len(set(r['unit'] for r in rows)),
        unique_scene_branches=len(set((r['unit'], r['branch']) for r in rows)),
        last_visible_range_n=sum(r['last_visible_range_m'] is not None for r in rows),
        median_last_visible_range_m=median('last_visible_range_m'),
        median_visible_frames=median('visible_frames_predeadline'),
        median_visible_ray_fraction=median('median_ray_fraction_predeadline'),
        median_nonzero_visible_ray_fraction=median('median_ray_fraction_visible_frames'),
        timely={a:sum(r['timely'][a] for r in rows) for a in ARMS},
        teacher_only_timely=sum(r['timely']['D'] and not r['timely']['VD']
                              and not r['timely']['R_any'] for r in rows),
        no_visible_ties=sum(r['no_visible_state_tie'] for r in rows))


def run(out):
    started = time.monotonic()
    plan = read(out/'PLAN.json')
    if (out/'result.json').exists():
        raise FileExistsError('Completed result immutable')
    for path, expected in plan['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Frozen source changed: '+path)
    reference = read(REF/'result.json')
    sources = dict(plan['source_sha256'])
    # Reconstruct only existing frozen scores; this does not run a network.
    scores, scenes, _, _, _, covered, baseline = L.load_baselines()
    with np.load(REF/'scores/all_scores.npz', allow_pickle=False) as z:
        scores['R_any'] = z['R_any'].copy()
        assert z['units'].tolist() == plan['units']
    with np.load(REF/'episode_ledger.npz', allow_pickle=False) as z:
        events = {}
        for arm in ARMS:
            event = L.events(scores[arm], plan['thresholds'][arm],
                             np.array([s['front_range_m'] for s in scenes]), covered)
            for key in ('alarm', 'timely', 'stopped', 'first_index', 'first_range'):
                np.testing.assert_array_equal(event[key], z[arm+'_'+key])
            events[arm] = event
    out_ids = [s['unit'] for s in scenes if not s['fov_in']]
    in_ids = [s['unit'] for s in scenes if s['fov_in']]
    assert (len(out_ids), len(in_ids)) == (11, 37)
    directions, _ = S.angular_rays(16)
    shape = directions.shape[:-1]
    assert int(np.prod(shape)) == 8*8*256
    smoke, unit_geometry, records = [], [], []
    source_hashes = reference['provenance']['baseline_input_sha256']

    def inputs(unit):
        tp = OLD/'truth/evaluation'/f'unit{unit}.json'
        op = OLD/'observations/evaluation'/f'unit{unit}.npz'
        ep = OLD/'templates/evaluation'/f'unit{unit}.npz'
        for p in (tp, op, ep):
            digest = sha(p)
            if source_hashes[str(p)] != digest:
                raise ValueError('Sealed geometry changed: '+str(p))
            sources[str(p)] = digest
        truth = read(tp)
        with np.load(op, allow_pickle=False) as z:
            poses = z['sensor'].copy()
        with np.load(ep, allow_pickle=False) as z:
            ids = z['object_id'][:2].copy()
        assert ids.shape == (2,16,8,8,256) and poses.shape == (16,4,4)
        return truth, poses, ids

    for unit in in_ids[:2]:
        truth, poses, ids = inputs(unit)
        differences = 0
        for branch in range(2):
            for frame, pose in enumerate(poses):
                replay = S.raycast_boxes(pose[:3,3], directions@pose[:3,:3].T,
                                        truth['boxes'][branch])['object_id']
                differences += int(np.count_nonzero(replay != ids[branch,frame]))
        assert differences == 0, (unit, differences)
        smoke.append(dict(unit=unit, checked_subrays=2*16*16384,
                          object_id_differences=differences, visible_differences=0))
    print('SMOKE PASS first2 FOV_IN all-object first-hit replay', flush=True)
    order = out_ids + in_ids
    scene_index = {s['unit']:i for i,s in enumerate(scenes)}
    for unit in order:
        if unit in in_ids and time.monotonic()-started >= plan['time_budget_seconds']:
            break
        truth, poses, ids = inputs(unit)
        ranges = np.array(truth['front_range_m'], float)
        pre = ranges >= .9
        first_counts = (ids == 0).sum(axis=(2,3,4))
        target_counts, blocked_counts = [], []
        for branch in range(2):
            target_masks = np.stack([S.raycast_boxes(p[:3,3], directions@p[:3,:3].T,
                                        [truth['boxes'][branch][0]])['object_id'] == 0 for p in poses])
            assert not np.any((ids[branch] == 0) & ~target_masks)
            target_counts.append(target_masks.sum(axis=(1,2,3)))
            blocked_counts.append((target_masks & (ids[branch] != 0)).sum(axis=(1,2,3)))
        target_counts = np.stack(target_counts)
        state = np.where(first_counts > 0, 2, np.where(target_counts > 0, 1, 0))
        vf = float((first_counts[:,3:] > 0).mean())
        i = scene_index[unit]
        assert vf == scenes[i]['visibility_fraction']
        unit_geometry.append(dict(unit=unit, fov_in=scenes[i]['fov_in'],
            target_visibility_fraction=vf, ranges=ranges.tolist(), predeadline=pre.tolist(),
            first_hit_target_counts=first_counts.tolist(), target_only_counts=target_counts.tolist(),
            blocked_target_subray_counts=np.stack(blocked_counts).tolist(),
            state=state.tolist()))
        for branch in range(2):
            visible = (state[branch] == 2) & pre
            nvisible = int(visible.sum())
            counts = np.bincount(state[branch,pre], minlength=3)
            for k in range(4):
                timely = {a: bool(events[a]['timely'][i,branch,k]) for a in ARMS}
                group = ('D' if timely['VD'] else 'C') if nvisible else ('A' if counts[0]>counts[1] else 'B')
                records.append(dict(unit=unit, branch=branch, intrusion_cm=branch+1, replica=k,
                    fov='FOV_IN' if scenes[i]['fov_in'] else 'FOV_OUT', category=group,
                    height='HEAD' if truth['group']==0 else 'BODY', context=truth['context'],
                    deadline_last_sample_frame=int(np.where(pre)[0][-1]),
                    predeadline_frames=int(pre.sum()), visible_frames_predeadline=nvisible,
                    state_frames_predeadline={name:int(counts[j]) for j,name in enumerate(STATE_NAMES)},
                    no_visible_state_tie=bool(not nvisible and counts[0]==counts[1]),
                    last_visible_range_m=None if not nvisible else float(ranges[np.where(visible)[0][-1]]),
                    median_ray_fraction_predeadline=float(np.median(first_counts[branch,pre]/16384)),
                    median_ray_fraction_visible_frames=None if not nvisible else float(np.median(first_counts[branch,visible]/16384)),
                    timely=timely, first_alarm_range_m={a:None if not events[a]['stopped'][i,branch,k]
                        else float(events[a]['first_range'][i,branch,k]) for a in ARMS}))
        print('GEOMETRY',unit,len(unit_geometry),'elapsed',round(time.monotonic()-started,2),flush=True)
    summary = {}
    for fov, denominator in (('FOV_OUT',88), ('FOV_IN',296)):
        rows = [r for r in records if r['fov']==fov]
        summary[fov] = dict(expected_n=denominator, status='COMPLETE' if len(rows)==denominator else 'NOT_RUN_OR_PARTIAL',
                            total=stats(rows), classes={c:stats([r for r in rows if r['category']==c]) for c in 'ABCD'})
    counts = {c:summary['FOV_OUT']['classes'][c]['n'] for c in 'ABCD'}
    assert sum(counts.values()) == 88
    branch = 'COVERAGE_PRIORITY' if counts['A']+counts['B']>44 else 'EXPOSURE_STRENGTH_PRIORITY' if counts['C']>44 else 'MIXED'
    for fov in summary:
        if summary[fov]['status']=='COMPLETE':
            for a in ARMS:
                expected = reference['sequences']['cells'][fov][a]['branches']['inside1_2cm']['timely']['stops']
                assert summary[fov]['total']['timely'][a] == expected
    # Preserve sealed source files; repeat hashes once near delivery.
    assert all(sha(p)==digest for p,digest in sources.items())
    elapsed=time.monotonic()-started
    result = dict(status='COMPLETE', branch=branch, summary=summary, smoke=smoke,
        score_event_parity='PASS all4 arms vs sealed episode ledger', baseline_parity=baseline['status'],
        geometry_replicas='K4 events share geometry;88 sequences are22 scene-branches from11 independent scenes',
        thresholds=plan['thresholds'], deadline=.9, elapsed_seconds=elapsed,
        no_training=True, no_observation_render=True, inputs_unchanged=True,
        provenance=dict(plan_sha256=sha(out/'PLAN.json'), input_sha256=sources))
    save(out/'geometry.json',unit_geometry)
    save(out/'sequences.json',records)
    save(out/'result.json',result)
    print('RESULT',branch,json.dumps(counts),'seconds',round(elapsed,2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('plan','run'),required=True)
    p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    (freeze if a.stage=='plan' else run)(a.out)
