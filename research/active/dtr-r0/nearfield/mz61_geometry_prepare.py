"""Prepare one 4096-frame geometry-disjoint candidate; no capture or labels."""
import argparse
from collections import Counter
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from mz48_prepare import read, sha, write, rotation, world


FAMILIES = ['shallow_awning', 'sign_panel', 'square_grille', 'variable_rod']
RELATIONS = ['HEAD_ONLY', 'BODY_ONLY', 'BOTH', 'VISIBLE_NONINTRUDING']
ROLES = ['TRAIN_CANDIDATE', 'CALIBRATION', 'HELDOUT_GEOMETRY']


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def euler(matrix):
    pitch = np.arcsin(np.clip(matrix[2, 0], -1, 1))
    assert abs(np.cos(pitch)) > 1e-6
    out = dict(pitch=float(np.rad2deg(pitch)),
               yaw=float(np.rad2deg(np.arctan2(matrix[1, 0], matrix[0, 0]))),
               roll=float(np.rad2deg(np.arctan2(-matrix[2, 1], matrix[2, 2]))))
    np.testing.assert_allclose(rotation(out), matrix, atol=1e-12, rtol=0)
    return out


def geometry(family, relation, distance, k, assets, rod_material):
    # Marginals occur in every split; held-out JOINT recipes never occur in fit.
    wi, di, ti = k % 4, (k // 4) % 2, k // 8
    fold = (wi + di + ti) % 4
    role = ROLES[0 if fold < 2 else fold - 1]
    front = (1.04 + .04 * ((k // 4) % 3)) if distance == 'near' else (1.93 + .11 * ((k // 4) % 4))
    ratio = [.35, .45, .55, .65][wi]
    width = ratio * front  # Both ranges share the same angular-width target.
    depth = [.055, .14][di]
    heights = {'shallow_awning': [.06, .11], 'sign_panel': [.10, .16],
               'square_grille': [.12, .16], 'variable_rod': [.025, .065]}
    height = heights[family][di] + (.005 if family == 'variable_rod' else .01) * (wi % 2)
    delta = dict(pitch=[-4., 2., -2., 4.][ti], yaw=[-5., -1., 2., 5.][ti],
                 roll=[-8., -3., 3., 8.][ti])
    lateral = [-.06, -.02, .02, .06][(wi + di + ti) % 4]
    body_z = 1.19 + .008 * ((wi + 2 * di) % 3 - 1)
    head_z = 1.62 + .01 * ((2 * wi + di) % 3 - 1)
    centers = {'BODY_ONLY': [body_z], 'HEAD_ONLY': [head_z],
               'BOTH': [body_z, head_z], 'VISIBLE_NONINTRUDING': [2.08 + .01 * (wi - 1.5)]}[relation]
    if family == 'variable_rod':
        corners = np.array(list(itertools.product(*zip(-np.array([depth, width, height]) / 2,
                                                       np.array([depth, width, height]) / 2))))
        base = np.eye(3)
        prototype = dict(size_m=[depth, width, height], material_asset=rod_material)
    else:
        asset = assets[family]
        span = np.array(asset['bounds_max_m']) - asset['bounds_min_m']
        desired = [height, width, depth] if family == 'square_grille' else [width, depth, height]
        scale = np.array(desired) / span
        assert np.isfinite(scale).all() and (scale > 0).all()
        corners = np.array(list(itertools.product(*zip(asset['bounds_min_m'], asset['bounds_max_m'])))) * scale
        base = rotation(dict(pitch=90. if family == 'square_grille' else 0.,
                             yaw=0. if family == 'square_grille' else 90., roll=0.))
        prototype = dict(mesh_asset=asset['asset'] + '.' + asset['asset'].split('/')[-1],
                         scale=scale.tolist(), placement='actor_origin')
    local_rotation = euler(rotation(delta) @ base)
    points = corners @ rotation(local_rotation).T
    lo, hi = points.min(0), points.max(0)
    assert hi[0] - lo[0] < .36
    targets, bounds, projections = [], [], []
    focal = 320 / np.tan(np.deg2rad(50.))
    for j, center_z in enumerate(centers):
        origin = np.array([front + j * .025, lateral, center_z]) - [lo[0], (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2]
        low, high = lo + origin, hi + origin
        if distance == 'near':
            assert low[0] > .18 and high[0] < 1.59
        else:
            assert low[0] > 1.73 and high[0] < 3.10
        if relation == 'VISIBLE_NONINTRUDING':
            assert low[2] > 1.85
        elif center_z < 1.4:
            assert low[2] >= .65 and high[2] < 1.4
        else:
            assert low[2] > 1.4 and high[2] <= 1.85
        corners_camera = points + origin
        pixels = np.stack([319.5 + focal * corners_camera[:, 1] / corners_camera[:, 0],
                           179.5 - focal * (corners_camera[:, 2] - 1.7) / corners_camera[:, 0]], 1)
        assert pixels.min(0)[0] >= 0 and pixels.max(0)[0] < 640
        assert pixels.min(0)[1] >= 0 and pixels.max(0)[1] < 360
        target = dict(prototype, name=family + '_' + str(j), target_part=True,
                      center_m=origin.tolist(), rotation_deg=local_rotation.copy())
        targets.append(target)
        bounds.append(dict(name=target['name'], camera_aligned_min_m=low.tolist(), camera_aligned_max_m=high.tolist()))
        projections.append(dict(width_px=float(np.ptp(pixels[:, 0])), height_px=float(np.ptp(pixels[:, 1]))))
    description = dict(family=family, recipe=k, joint_factors=[wi, di, ti], role=role,
                       desired_dimensions_m=[depth, width, height], front_anchor_m=front,
                       target_width_over_front=ratio, delta_rotation_deg=delta, lateral_m=lateral,
                       targets_camera_coordinates=targets)
    return targets, bounds, projections, description


def prepare(root, output):
    assert not output.exists()
    work = root / 'artifacts.local/work'; old = work / 'mz55-diverse-mesh-source-20260911'
    inputs = {}
    def bound(path, expected=None):
        digest = sha(path); assert expected is None or digest == expected
        inputs[str(path)] = digest; return read(path)
    prior = bound(old / 'spec-v1/manifest.json', '97a13ecb7d4cc9cf58c6ae7062bb70d699b913788332a14e29360ba2f059cd43')
    for name in ('mz61_geometry_prepare.py', 'MZ61_GEOMETRY_SOURCE_DESIGN_20260911.md', 'mz48_prepare.py'):
        p = Path(__file__).with_name(name); inputs[str(p)] = sha(p)
    storage = bound(old / 'storage-summary.json')
    specs = {}
    for shard in prior['shards']:
        if shard['canary']:
            spec = bound(old / 'spec-v1/shards' / (shard['shard_id'] + '.json'), shard['sha256'])
            specs[spec['cases'][0]['region_id']] = spec
    original = bound(work / 'mz42-rich-objects-20260910/spec-v2/spec.json', 'cc76970fe3f358c4cffca24d423cb1d81a73720d609a4e7dc21b8f3067cf7600')
    template = next(c for c in original['cases'] if c['condition']['family'] == 'oblique_rod' and c['variant_id'] == 'HEAD_ONLY')
    material = next(o['material_asset'] for o in template['objects'] if o['target_part'])
    sites = sorted(prior['sites'], key=lambda s: s['site_id'])
    cases, configurations = [], []
    for fi, family in enumerate(FAMILIES):
        for ri, relation in enumerate(RELATIONS):
            for di, distance in enumerate(('near', 'far')):
                canary_k = (6 * fi + 4 * ri + 9 * di) % 32
                audit_k = (canary_k + 13) % 32
                for k in range(32):
                    targets, bounds, projections, description = geometry(family, relation, distance, k, prior['assets'], material)
                    # Hash actual camera-relative target geometry only: a split name
                    # or recipe identifier must not manufacture disjointness.
                    geometry_id = fingerprint([{q: v for q, v in target.items()
                                                if q not in ('name', 'target_part')}
                                               for target in targets])
                    configurations.append(dict(geometry_id=geometry_id, family=family, relation=relation, range=distance,
                                               projections=projections, **{q: description[q] for q in description if q != 'family'}))
                    first_site = (k + fi + ri + di) % 8
                    for replica, si in enumerate((first_site, (first_site + 4) % 8)):
                        site = sites[si]
                        reference = next(c for c in specs[site['region_id']]['cases'] if c['site_id'] == site['site_id'])
                        local_targets = copy.deepcopy(targets)
                        for obj in local_targets:
                            obj['center_m'] = world(site['camera'], site['floor_z_m'], obj['center_m'])
                            obj['rotation_deg']['yaw'] += site['camera']['yaw']
                        supports = []
                        front = description['front_anchor_m']
                        for obj in (o for o in template['objects'] if not o['target_part']):
                            support = copy.deepcopy(obj)
                            sign = -1 if obj['name'].endswith('_-1') else 1
                            x = front + .10 + (1.50 if obj['name'].startswith('rear_') else .75 if obj['name'].startswith('side_rail') else 0.)
                            z = 1.50 if obj['name'].startswith('clamp') else obj['center_m'][2] - template['floor_z_m']
                            assert .80 - obj['size_m'][1] / 2 > .28
                            support['center_m'] = world(site['camera'], site['floor_z_m'], [x, sign * .80, z])
                            support['rotation_deg']['yaw'] = site['camera']['yaw']; supports.append(support)
                        assert len(supports) == 12
                        pair = f'mz61-{site["site_id"]}-{family}-{relation}-{distance}-g{k:02d}'
                        expected = {'HEAD_ONLY': [0, 1], 'BODY_ONLY': [1, 0], 'BOTH': [1, 1], 'VISIBLE_NONINTRUDING': [0, 0]}[relation]
                        events = [0] * 4; events[di], events[2 + di] = expected
                        for supported in (False, True):
                            context = 'supported' if supported else 'unsupported'
                            case = {q: copy.deepcopy(reference[q]) for q in ('camera', 'wearer', 'floor_z_m', 'site_id', 'region_id')}
                            case.update(name=pair + '-' + context, group_id=pair, pair_id=pair, variant_id=relation,
                                        declared_range=distance, source_role='DEV_ONLY', training_role=description['role'],
                                        split_unit='geometry_recipe', geometry_id=geometry_id, geometry_recipe_id=f'{family}-g{k:02d}',
                                        geometry_joint_factors=description['joint_factors'], site_replica=replica,
                                        floor_check=False, probe_native_floor=True, profile_id=k, support_context=context,
                                        expected_events=events, expected_collapsed_relation=expected, placement_bounds=copy.deepcopy(bounds),
                                        canary=k == canary_k and replica == 0,
                                        native_audit_sample=replica == 0 and (k == canary_k or (fi == 0 and k == audit_k)),
                                        objects=copy.deepcopy(local_targets) + (copy.deepcopy(supports) if supported else []),
                                        condition=dict(family=family, desired_relation=relation, condition_id=pair + '-' + context))
                            cases.append(case)
    assert len(cases) == len({c['name'] for c in cases}) == 4096
    assert Counter(c['training_role'] for c in cases) == dict(TRAIN_CANDIDATE=2048, CALIBRATION=1024, HELDOUT_GEOMETRY=1024)
    assert sum(c['canary'] for c in cases) == 64 and sum(c['native_audit_sample'] for c in cases) == 80
    hashes = {role: {c['geometry_id'] for c in cases if c['training_role'] == role} for role in ROLES}
    assert len({c['geometry_id'] for c in cases}) == 1024
    for a, b in itertools.combinations(ROLES, 2): assert hashes[a].isdisjoint(hashes[b])
    for a, b in zip(cases[::2], cases[1::2]):
        assert a['pair_id'] == b['pair_id'] and a['geometry_id'] == b['geometry_id'] and a['training_role'] == b['training_role']
        assert a['objects'] == [o for o in b['objects'] if o['target_part']]
    # Marginal factors are shared; the coarse joint recipes themselves are disjoint.
    splits = {role: {tuple(c['geometry_joint_factors']) for c in cases if c['training_role'] == role} for role in ROLES}
    assert [len(splits[r]) for r in ROLES] == [16, 8, 8]
    for a, b in itertools.combinations(ROLES, 2): assert splits[a].isdisjoint(splits[b])
    for role in ROLES:
        for axis, size in enumerate((4, 2, 4)): assert {v[axis] for v in splits[role]} == set(range(size))
    overlap = []
    for family, role in itertools.product(FAMILIES, ROLES):
        values = {d: [c['projections'][0]['width_px'] for c in configurations if c['family'] == family and c['role'] == role and c['range'] == d] for d in ('near', 'far')}
        lo = max(min(v) for v in values.values()); hi = min(max(v) for v in values.values())
        assert hi > lo
        fraction = sum(lo <= x <= hi for v in values.values() for x in v) / sum(map(len, values.values()))
        assert fraction >= .75
        overlap.append(dict(family=family, role=role, projected_bbox_width_px={d: [min(v), max(v)] for d, v in values.items()}, shared_interval_px=[lo, hi], fraction_in_shared_interval=fraction))
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for region in specs:
        subset = [c for c in cases if c['region_id'] == region and c['canary']]
        assert subset
        name = 'canary-' + region
        spec = {q: copy.deepcopy(v) for q, v in specs[region].items() if q not in ('cases', 'provenance')}
        spec.update(schema='mz61-geometry-candidate-v1', cases=subset, purpose='CANDIDATE_ONLY_NOT_REGISTERED', provenance=dict(inputs=inputs, fixed_budget=4096))
        p = output / 'shards' / (name + '.json'); write(p, spec)
        rows.append(dict(shard_id=name, canary=True, frames=len(subset), path=str(p), sha256=sha(p)))
    for site in sites:
        subset = [c for c in cases if c['site_id'] == site['site_id'] and not c['canary']]
        name = 'main-' + site['site_id']
        spec = {q: copy.deepcopy(v) for q, v in specs[site['region_id']].items() if q not in ('cases', 'provenance')}
        spec.update(schema='mz61-geometry-candidate-v1', cases=subset, purpose='CANDIDATE_ONLY_NOT_REGISTERED', provenance=dict(inputs=inputs, fixed_budget=4096))
        p = output / 'shards' / (name + '.json'); write(p, spec)
        rows.append(dict(shard_id=name, canary=False, frames=len(subset), path=str(p), sha256=sha(p)))
    write(output / 'geometry-configurations.json', dict(configurations=configurations))
    manifest = dict(status='CANDIDATE_NOT_REGISTERED_NOT_CAPTURED', frames=4096, canary_frames=64, main_frames=4032,
                    native_audit_frames=80, unique_camera_geometry_configurations=1024, independent_geometry_recipes_per_family=32,
                    roles=dict(Counter(c['training_role'] for c in cases)), sites=sites, families=FAMILIES, shards=rows,
                    inputs=inputs, split_rule='(width_level + depth_level + tilt_level)%4:0/1TRAIN,2CAL,3HELD; all relations/ranges/site replicas/support pairs grouped',
                    static_checks=dict(cross_split_geometry_fingerprint_collisions=0, coarse_joint_recipe_collisions=0,
                                       shared_factor_marginals=True, configured_AABB_intent=True, full_RGB_projected_bounds_inside=True,
                                       near_far_projected_width_overlap=overlap),
                    real_native_label_novelty='UNMEASURED: geometry fingerprints and bounding-box projections are not native masks or evidence of generalization',
                    actual_truth_policy='Keep every native>=3pixel query bit, including unintended near+far bits; preserve source failures and UNKNOWN',
                    calibration_policy='Geometry role only; no model outputs or thresholds used in source design. Any later cutoff fit uses only predeclared CALIBRATION geometry.',
                    estimated_from_mz55=dict(compact_source_bytes=storage['source_archive_bytes_total'] * 4096 // 2560,
                                              raw_capture_logical_bytes=storage['capture_raw_bytes_total'] * 4096 // 2560,
                                              worker_only_pipeline_seconds=4096 / storage['worker']['main_frames_per_pipeline_second'],
                                              balanced_two_host_idealized_seconds=4096 / (storage['worker']['main_frames_per_pipeline_second'] + storage['primary']['frames_per_pipeline_second']),
                                              limitation='Extrapolation only; canary admission, startup and host availability add time; RGB dominates bytes'),
                    predictor_inputs=['Original RGB', 'Original45degree ToF ranges/valid only'],
                    evaluator_only=['native depth', 'fullframe/angular labels', 'geometry/site/role metadata'],
                    new_captures=0, model_inference_frames=0, training_steps=0)
    write(output / 'manifest.json', manifest)
    write(output / 'receipt.json', dict(status='PASS_STATIC_CANDIDATE_ONLY', inputs=inputs,
                                        outputs={str(p.relative_to(output)): sha(p) for p in output.rglob('*.json')},
                                        new_captures=0, training_steps=0, model_inference_frames=0))
    print(json.dumps(dict(status=manifest['status'], manifest_sha256=sha(output / 'manifest.json'),
                         roles=manifest['roles'], shards=[(r['shard_id'], r['frames']) for r in rows]), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); prepare(a.root.resolve(), a.output.resolve())
