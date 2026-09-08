"""Equal-budget allocation and shared quartet schedule; no capture or model run."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

FAMILIES = ('crossbar', 'cabinet', 'oblique_rod', 'hanging_sign')
RELATIONS = ('CLEAR', 'BODY_ONLY', 'HEAD_ONLY', 'BOTH')


def allocate(train_regions, concentrated_regions):
    if len(train_regions) != 4 or len(set(train_regions)) != 4:
        raise ValueError('Four distinct TRAIN regions required')
    if len(concentrated_regions) != 2 or len(set(concentrated_regions)) != 2 or not set(concentrated_regions) <= set(train_regions):
        raise ValueError('Two distinct concentrated TRAIN regions required')
    arms = {}
    for arm, regions in [('concentrated', concentrated_regions), ('distributed', train_regions)]:
        arms[arm] = [dict(template_index=i, template_id=f'geometry_{i:02d}', family=FAMILIES[i % 4],
            region_id=regions[(i // 4) % len(regions)],
            frame_indices=list(range(i*4, i*4+4)), relations=list(RELATIONS)) for i in range(64)]
    return dict(status='ALLOCATION_ONLY_NOT_SOURCE_ADMISSION', arms=arms,
        unresolved=['native geometry templates', 'final camera poses', 'route and background isolation'],
        source_rule='Region order and concentrated subset must be chosen without model outcomes')


def schedule(steps=2000, seed=17):
    rng = np.random.default_rng(seed)
    original = rng.integers(0, 1198, size=(steps, 20), dtype=np.int64)
    quartets = np.stack([rng.choice(64, size=3, replace=False) for _ in range(steps)])
    frames = (quartets[:, :, None]*4+np.arange(4)).reshape(steps, 12)
    return dict(original_indices=original, quartet_indices=quartets, new_frame_indices=frames)


def exposure(plan, draws):
    frames = draws['new_frame_indices'].reshape(-1)
    relation_counts = np.bincount(frames % 4, minlength=4)
    return dict(original_draws=int(draws['original_indices'].size), new_draws=len(frames),
        new_BODY_positive_draws=int(relation_counts[1]+relation_counts[3]),
        new_HEAD_positive_draws=int(relation_counts[2]+relation_counts[3]),
        relation_draws=dict(zip(RELATIONS, map(int, relation_counts))),
        per_new_frame=np.bincount(frames, minlength=256).tolist(),
        arms={a: dict(region_draws=dict(Counter(rows[int(i)//4]['region_id'] for i in frames)),
            family_draws=dict(Counter(rows[int(i)//4]['family'] for i in frames))) for a, rows in plan['arms'].items()})


def validate_final_records(plan, arms):
    """Validate admitted capture metadata, not self-declared source isolation.

    Source visibility admission and native label extraction remain independent
    prerequisites; this check cannot manufacture either from intended labels.
    """
    if set(arms) != set(plan['arms']):
        raise ValueError('Both matched arms required')
    by_arm = {}
    for arm, rows in arms.items():
        if len(rows) != 256 or [r['frame_index'] for r in rows] != list(range(256)):
            raise ValueError('Exactly256 ordered frames required')
        grouped = {}
        for r in rows:
            i = r['frame_index']; expected = plan['arms'][arm][i//4]
            if r.get('source_role') != 'TRAIN_ONLY' or any(r[k] != expected[k] for k in ('template_id', 'family', 'region_id')):
                raise ValueError('Source allocation or TRAIN role mismatch')
            if r['relation'] != RELATIONS[i % 4] or r.get('native_near') != [[0,0],[1,0],[0,1],[1,1]][i % 4]:
                raise ValueError('Native quartet relation mismatch or UNKNOWN')
            if any(not isinstance(r.get(k), str) or len(r[k]) != 64 for k in ('geometry_template_sha256', 'camera_context_sha256', 'native_label_sha256')):
                raise ValueError('Missing captured template/context/label hashes')
            grouped.setdefault(i//4, []).append(r)
        for rs in grouped.values():
            if len({r['camera_context_sha256'] for r in rs}) != 1 or len({r['geometry_template_sha256'] for r in rs}) != 1:
                raise ValueError('Quartet camera/context or shared geometry changed')
        by_arm[arm] = rows
    if [r['geometry_template_sha256'] for r in by_arm['concentrated']] != [r['geometry_template_sha256'] for r in by_arm['distributed']]:
        raise ValueError('Geometry templates differ between arms')
    return dict(status='PASS', scope='Allocation, captured relation and template correspondence only; separate visibility admission required')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-regions', nargs=4, required=True)
    p.add_argument('--concentrated-regions', nargs=2, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = (Path(__file__).resolve().parents[4]/'artifacts.local').resolve()
    if not a.output.resolve().is_relative_to(root) or a.output.resolve() == root:
        raise ValueError('Canonical task artifact output required')
    plan = allocate(a.train_regions, a.concentrated_regions); draws = schedule()
    a.output.mkdir(parents=True, exist_ok=False)
    np.savez(a.output/'schedule.npz', **draws)
    plan['exposure'] = exposure(plan, draws)
    plan['schedule_sha256'] = hashlib.sha256((a.output/'schedule.npz').read_bytes()).hexdigest()
    plan['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (a.output/'allocation.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=plan['status'], original_draws=40000, new_draws=24000,
        BODY_positive_new=12000, HEAD_positive_new=12000, optimizer_steps_executed=0)))
