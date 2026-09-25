"""Track A v1.3 geometry pilot (fast lane). Reuses the v1.2 planner and checks.

Changes from v1.2, decided by the user on 2026-09-26:
- object sizes: 75% realistic (w 4-14 cm, d 2-6 cm, h 10-40/34 cm), 25% tiny
  v1.2 stubs as a small-target stress stratum; wide bars unchanged;
- an infeasible config redraws its trajectory (<=3 replans, all logged);
- a unit that still cannot finish 32 configs is INCOMPLETE and excluded, other
  units continue (integrity checks G0/G1 stay hard, G2 quotas report-only);
- calib/audit configs also carry 10 Hz labels (23 frames; even frames = 5 Hz).
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from cnh_track_a_generate import BOXES, head_path, ry, check_unit, check_coordinates, save
from cnh_track_a_geometry import box_mesh, cylinder_mesh, signed_margin, clip_triangles
from cnh_track_a_v12_generate import (schedule, plan_centres, boundary_centres, candidate_check,
                                      fingerprint, evaluate_candidate)

FAMILY = 'cnh-track-a-v13-20260926'
PURPOSES = ('layout', 'trajectory', 'size_placement', 'material')
REPLANS = 3
TINY_FRACTION = .25
SPLIT_COUNTS = (6, 2, 4)  # train, calib, audit units; pilot default


def split_of(unit):
    train, calib, _ = SPLIT_COUNTS
    return 'train' if unit < train else 'calib' if unit < train+calib else 'audit'


def purpose_seed(unit, config, purpose, candidate=0):
    if purpose not in PURPOSES:
        raise ValueError('Unknown seed purpose')
    message = f'{FAMILY}|{unit}|{config}|{purpose}|{candidate}'
    return int(hashlib.sha256(message.encode()).hexdigest()[:16], 16)


def trajectory(unit, config, height, replan):
    """5 Hz path (12 poses, v1.2 layout) plus 10 Hz path (23 poses, even = 5 Hz)."""
    seed = purpose_seed(unit, config, 'trajectory', replan)
    rng = np.random.default_rng(seed)
    angles = head_path(rng)
    speed = float(rng.uniform(.8, 1.4))
    t10 = np.arange(23) / 2
    angles10 = np.stack([np.interp(t10, np.arange(12), angles[:, k]) for k in range(3)], 1)

    def poses(angle_rows, times):
        out = np.repeat(np.eye(4)[None], len(times), axis=0)
        out[:, :3, :3] = np.array([ry(a[2]) for a in angle_rows])
        out[:, :3, 3] = [[0, -height, (t-7)*.2*speed] for t in times]
        return out

    return dict(seed=seed, replan=replan, speed=speed, angles=angles.tolist(),
                world_from_Q=poses(angles, np.arange(12)).tolist(),
                angles_10hz=angles10.tolist(), world_from_Q_10hz=poses(angles10, t10).tolist())


def make_object(split, rng, category, wide):
    """Split-specific surface family (asset isolation kept); size class recorded."""
    if wide:
        size_class = 'wide'
        r = rng.uniform(.008, .024)
        half = np.array([.48, (.14 if category == 'HEAD' else .11)/2, r])
    elif rng.random() < TINY_FRACTION:
        size_class = 'tiny'
        r = rng.uniform(.008, .024)
        half = np.array([r, (.14 if category == 'HEAD' else .11)/2, r])
    else:
        size_class = 'realistic'
        half = np.array([rng.uniform(.04, .14)/2,
                         rng.uniform(.10, .40 if category == 'HEAD' else .34)/2,
                         rng.uniform(.02, .06)/2])
    if split == 'train':
        mesh = box_mesh(-half, half)
        family = 'rectangular-prism'
    elif split == 'calib':
        if wide:
            mesh = cylinder_mesh(np.zeros(3), half[2], 2*half[0], axis=0, sides=64)
        else:
            mesh = cylinder_mesh(np.zeros(3), max(half[0], half[2]), 2*half[1], axis=1, sides=64)
        family = 'cylinder64'
    else:
        d = half[2]
        mesh = np.concatenate([box_mesh([-half[0], -half[1], z-d/3], [half[0], half[1], z+d/3]) for z in (-d, d)])
        family = 'paired-plate-composite'
    return dict(triangles=mesh, family=family, category=category, center=[0., 0., 0.],
                radius=float(half[2]), wide=bool(wide), suspended=category == 'HEAD', boundary=False,
                size_class=size_class, extent=(2*half).tolist())


def object_templates(split, patterns, rng, material_rng):
    result = []
    for group, pattern in enumerate(patterns):
        states = {0: [], 1: [1], 2: [2], 3: [None], 4: [3], 5: [4], 6: [1, 4]}[pattern]
        for state in states:
            obj = make_object(split, rng, 'HEAD' if group == 0 else 'BODY', wide=state is None)
            obj.update(id=len(result)+1, group=group, target_state=state, rho=float(material_rng.uniform(.1, .9)))
            result.append(obj)
    return result


def labels_for_poses(objects, poses):
    """Exact per-frame margins/labels/contributors for arbitrary world_from_Q poses."""
    margins, labels, contributors = [], [], []
    for pose in np.asarray(poses):
        local = [(np.asarray(o['triangles_world'])-pose[:3, 3])@pose[:3, :3] for o in objects]
        row = np.array([[signed_margin(mesh, *box) for box in BOXES] for mesh in local])
        margins.append(row)
        labels.append((row.max(0) >= -1e-10).astype(int))
        contributors.append([[o['id'] for o, m in zip(objects, row[:, q]) if m >= -1e-10] for q in range(6)])
    margins = np.asarray(margins)
    boundary = (np.abs(margins) < .05).any((1, 2))
    return margins, np.asarray(labels), contributors, boundary


WITH_10HZ = True


def run(output, units=range(12)):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + 3*3600
    done, incomplete = [], []
    attempts = rejects = 0
    buckets = {}
    split_configs = {s: [] for s in ('train', 'calib', 'audit')}
    save(output/'manifest.json', dict(schema='cnh.track-a.geometry.v1.3', seed_family=FAMILY,
         source_sha256={n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                        for n in ('cnh_track_a_v13_generate.py', 'cnh_track_a_v12_generate.py',
                                  'cnh_track_a_generate.py', 'cnh_track_a_geometry.py')},
         units=list(units), configs_per_unit=32, frames_5hz=12, frames_10hz=23 if WITH_10HZ else 0, replans=REPLANS,
         tiny_fraction=TINY_FRACTION, split_counts=list(SPLIT_COUNTS)))

    def finish(status, **extra):
        result = dict(status=status, completed_units=[u['unit'] for u in done], incomplete_units=incomplete,
                      attempts=attempts, rejections=rejects, rejection_rate=rejects/max(1, attempts),
                      wall_s=time.monotonic()-started, units=done, **extra)
        save(output/'result.json', result)
        return result

    for u in units:
        split = split_of(u)
        layout_seed = purpose_seed(u, -1, 'layout')
        rng = np.random.default_rng(layout_seed)
        height = float(rng.uniform(1.45, 1.75))
        width = float(rng.uniform(4, 6))
        configs, logs, pending = [], [], []
        unit_start = time.monotonic()
        failed_at = None
        for ci in range(32):
            accepted = None
            for replan in range(REPLANS+1):
                path = trajectory(u, ci, height, replan)
                for attempt in range(32):
                    if time.monotonic() > deadline:
                        return finish('STOP_WALL_BUDGET', unit=u, config=ci)
                    attempts += 1
                    candidate = replan*32 + attempt
                    seeds = dict(layout=layout_seed, trajectory=path['seed'],
                                 size_placement=purpose_seed(u, ci, 'size_placement', candidate),
                                 material=purpose_seed(u, ci, 'material', candidate))
                    srng = np.random.default_rng(seeds['size_placement'])
                    mrng = np.random.default_rng(seeds['material'])
                    templates = object_templates(split, schedule(u)[ci], srng, mrng)
                    if ci < 28:
                        centres, planning = plan_centres(templates, path, (.8, 1.6, 2.4)[ci % 3], ci % 14 >= 7)
                        boundary_plan = None
                    else:
                        centres, boundary_plan = boundary_centres(templates, path, ci, srng)
                        planning = dict(boundary=boundary_plan)
                    reason = 'solver_no_incumbent'
                    if centres is not None:
                        c = evaluate_candidate(u, ci, split, height, width, path, templates, centres)
                        good, reason = candidate_check(c, ci, boundary_plan)
                        fp = key = None
                        if good:
                            fp = fingerprint(c)
                            key = hashlib.sha256(json.dumps(sorted(fp), separators=(',', ':')).encode()).hexdigest()
                            if fp and any(fp == old for old in buckets.get(key, [])):
                                good, reason = False, 'global_exact_voxel_duplicate'
                        if good:
                            c.update(seeds=seeds, candidate=candidate, replan=replan, planning=planning,
                                     fingerprint=sorted(fp), planned_multi=bool(ci < 28 and ci % 14 >= 7),
                                     boundary_plan=boundary_plan, planned_patterns=list(schedule(u)[ci]),
                                     speed=path['speed'], angles_10hz=path['angles_10hz'],
                                     world_from_Q_10hz=path['world_from_Q_10hz'],
                                     size_classes={str(o['id']): o['size_class'] for o in templates},
                                     extents={str(o['id']): o['extent'] for o in templates})
                            accepted = c
                            if fp:
                                pending.append((key, fp))
                    logs.append(dict(config=ci, replan=replan, candidate=candidate, accepted=accepted is not None,
                                     reason=reason, solver=planning if ci < 28 else None))
                    if accepted is not None:
                        break
                    rejects += 1
                if accepted is not None:
                    break
            if accepted is None:
                failed_at = ci
                break
            configs.append(accepted)
        save(output/f'candidates-unit{u:02d}.json', logs)
        if failed_at is not None:
            incomplete.append(dict(unit=u, split=split, config=failed_at, reason='no feasible plan after replans',
                                   completed_configs=len(configs)))
            save(output/f'unit{u:02d}-INCOMPLETE.json', dict(unit=u, split=split, configs=configs))
            print(json.dumps(dict(unit=u, status='INCOMPLETE', config=failed_at)), flush=True)
            continue
        for key, fp in pending:
            buckets.setdefault(key, []).append(fp)
        if WITH_10HZ and split in ('calib', 'audit'):
            for c in configs:
                objs = [dict(o, triangles_world=np.asarray(o['triangles_world'])) for o in c['objects']]
                m, lab, contrib, bnd = labels_for_poses(objs, c['world_from_Q_10hz'])
                c.update(margins_10hz=m.tolist(), labels_10hz=lab.tolist(), contributors_10hz=contrib,
                         boundary_10hz=bnd.tolist(), main_10hz=((np.arange(23) >= 6) & ~bnd).tolist())
        checks = check_unit(configs)
        g0 = check_coordinates(configs)
        meta = dict(unit=u, split=split, shape=('straight', 'L', 'T', 'open')[u % 4], height=height, width=width,
                    checks=checks, G0=g0, wall_s=time.monotonic()-unit_start, layout_seed=layout_seed,
                    replans_used=int(sum(c['replan'] > 0 for c in configs)))
        save(output/f'unit{u:02d}.json', dict(meta, configs=configs))
        done.append(meta)
        split_configs[split].extend(dict(c, config=u*32+c['config']) for c in configs)
        print(json.dumps(dict(unit=u, G0=g0['pass_gate'], G2=checks['pass_gate'], failures=checks['failures'],
                              replans=meta['replans_used'], wall_s=round(meta['wall_s'], 1))), flush=True)
    split_checks = {s: check_unit(cs) if cs else None for s, cs in split_configs.items()}
    for check in split_checks.values():
        if check and check['eligible_combinations'] < 20:
            check['failures'].append('split_eligible_combinations_lt20')
            check['pass_gate'] = False
    g0 = bool(done) and all(u['G0']['pass_gate'] for u in done)
    return finish('GEOMETRY_READY' if g0 else 'STOP_G0_FAIL', G0=dict(pass_gate=g0),
                  G1=dict(pass_gate=True, note='global exact 1 cm dedup at acceptance'),
                  G2=dict(report_only=True, units_pass=[u['unit'] for u in done if u['checks']['pass_gate']],
                          split_checks=split_checks))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--units', type=int, nargs='*', default=list(range(12)))
    parser.add_argument('--family', default=FAMILY)
    parser.add_argument('--split-counts', type=int, nargs=3, default=list(SPLIT_COUNTS))
    parser.add_argument('--fast-margin', action='store_true', help='certified AABB bound before exact LP')
    parser.add_argument('--no-10hz', action='store_true', help='skip 10 Hz labels (5 Hz-only cohorts)')
    args = parser.parse_args()
    FAMILY, SPLIT_COUNTS, WITH_10HZ = args.family, tuple(args.split_counts), not args.no_10hz
    if args.fast_margin:
        import cnh_track_a_v12_generate as v12
        from cnh_track_a_fastmargin import fast_margin, STATS
        v12.signed_margin = fast_margin
        signed_margin = fast_margin
    print(run(args.output, args.units)['status'])
    if args.fast_margin:
        print(json.dumps(dict(margin_stats=STATS)))
