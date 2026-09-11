"""MZ72 executable physical source specification; CPU geometry, never capture.

The fixed MZ71 row identities and roles survive. Main objects retain scale 1;
angular controls transform target vertices about the optical centre. Bounds
place objects and mounts, not native labels or assertions of surface visibility.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from mz48_prepare import read, sha, rotation, world
from mz67_geometry_spec import euler


TASK = 'mz72-physical-source-20260911'
SCHEMA = 'mz72-physical-candidate-v1'
MANIFEST_SCHEMA = 'mz72-physical-manifest-v1'
ROLES = ('TRAIN_CANDIDATE', 'CALIBRATION', 'HELDOUT_GEOMETRY')
QUERIES = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')
DRAFT_SHA = 'b2e03374fd7ed6c32248b3777f68347fee9581ecb0734e31f5c1fb31040fc61d'
PRIOR_SHA = '851482eba0c6bc6c201897ef090e2b29d223542be94fef30b7735e24ab10b99a'
OPTICAL_CENTRE = np.array([0., 0., 1.7])
CORRIDOR_HALF_WIDTH = .28


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8', newline='\n')


def fingerprint(objects):
    """Hash only physical camera-relative parameters, independent of IDs/roles."""
    keys = ('mesh_asset', 'scale', 'placement', 'size_m', 'material_asset', 'center_m', 'rotation_deg')
    geometry = [{k: obj[k] for k in keys if k in obj} for obj in objects]
    entries = sorted(json.dumps(o, sort_keys=True, separators=(',', ':'), allow_nan=False) for o in geometry)
    return hashlib.sha256(json.dumps(entries, separators=(',', ':')).encode()).hexdigest()


def object_points(obj, assets):
    if 'mesh_asset' in obj:
        asset = next(a for a in assets.values() if obj['mesh_asset'] == a['object_path'])
        lo, hi = np.array(asset['local_min_m']), np.array(asset['local_max_m'])
        corners = np.array(list(itertools.product(*zip(lo, hi)))) * obj['scale']
    else:
        half = np.array(obj['size_m']) / 2
        corners = np.array(list(itertools.product(*zip(-half, half))))
    return corners @ rotation(obj['rotation_deg']).T + obj['center_m']


def bounds(obj, assets):
    p = object_points(obj, assets)
    return p.min(0), p.max(0)


def envelope(objects, assets):
    p = np.concatenate([object_points(o, assets) for o in objects])
    return p.min(0), p.max(0)


def box(name, center, size, material, parent='ground', optional=False):
    assert min(size) > 0
    return dict(name=name, center_m=list(map(float, center)), size_m=list(map(float, size)),
                rotation_deg=dict(pitch=0., yaw=0., roll=0.), material_asset=material,
                support_parent=parent, target_part=False, mandatory_mount=not optional,
                optional_context=optional, physical_role='context' if optional else 'mount')


def mesh(name, asset, angle, center=(0., 0., 0.), target=True):
    return dict(name=name, mesh_asset=asset['object_path'], scale=[1., 1., 1.],
                placement='actor_origin', center_m=list(map(float, center)), rotation_deg=angle,
                target_part=target, mandatory_mount=False, optional_context=False,
                physical_role='target' if target else 'occluder')


def translate(objects, vector):
    for obj in objects:
        obj['center_m'] = (np.array(obj['center_m']) + vector).tolist()


def resolved_recipe(recipe):
    out = copy.deepcopy(recipe)
    # Approved pre-freeze correction: the second physical six-recipe block in
    # the draft repeated the first. 4 degrees and 12 cm affect actual geometry.
    block = int(recipe['regime'] == 'ACTUAL_SIZE' and recipe['recipe_id'] >= 6)
    out['yaw_offset_deg'] += 4. * block
    for key in ('near_interaction_x_m', 'far_interaction_x_m'):
        out[key] += .12 * block
    out['physical_revision'] = dict(yaw_added_deg=4. * block, forward_added_m=.12 * block)
    return out


def target_geometry(row, recipe, assets):
    family, focus = row['family'], row['focus']
    angular = row['regime'] == 'ANGULAR_CONTROL'
    distance = 'far' if angular else row['distance']
    front = recipe[distance + '_interaction_x_m']
    sign, lateral = recipe['lateral_sign'], recipe['lateral_offset_m']
    edge = {'edge_yaw_minus': -20., 'edge_yaw_plus': 20.}.get(row['view'], 0.)
    yaw = recipe['yaw_offset_deg'] + edge
    if family in ('wooden_slats', 'landing_frame'):
        base = dict(pitch=0., yaw=90., roll=0.)
    elif family == 'compound_duct':
        base = dict(pitch=0., yaw=0., roll=90.)
    else:
        base = dict(pitch=75. if focus == 'BODY_PATH' else 0.,
                    yaw=180. if focus == 'BODY_PATH' else 0., roll=0.)
    angle = euler(rotation(dict(pitch=0., yaw=yaw, roll=0.)) @ rotation(base))
    asset = assets[family]
    prototype = mesh(family + '_0', asset, angle)
    lo, hi = bounds(prototype, assets)
    targets = []
    if family == 'wooden_slats':
        heights = {'HEAD_PATH': [1.70], 'BODY_PATH': [.65, .95],
                   'BOTH_PATH': [.65, .95, 1.70], 'VISIBLE_OFF_PATH': [1.70]}[focus]
        gap = [.10, .18, .26][recipe['recipe_id'] % 3]
        # Two 0.91 m slats cannot have centres +/-0.18 m AND a positive gap.
        # Place their inner envelope ends at +/- gap/2 instead, retaining 1x.
        for z in heights:
            for side in (-1, 1):
                obj = copy.deepcopy(prototype); obj['name'] = f'{family}_{len(targets)}'
                y = (-gap / 2 - hi[1]) if side < 0 else (gap / 2 - lo[1])
                obj['center_m'] = [float(front - lo[0]), float(y + lateral), float(z - (lo[2] + hi[2]) / 2)]
                targets.append(obj)
    else:
        obj = prototype
        if family == 'landing_frame':
            # Ground-connected unit: align its side envelope with the path for
            # the HEAD approach; lower/flight approaches retain all extra bits.
            low_z = 0.
            if focus == 'HEAD_PATH':
                intrusion = .15
                y = sign * (CORRIDOR_HALF_WIDTH - intrusion) - (lo[1] if sign > 0 else hi[1])
            else:
                y = lateral - (lo[1] + hi[1]) / 2
            origin = [front - lo[0], y, low_z - lo[2]]
        elif family == 'compound_duct':
            low_z = {'HEAD_PATH': 1.50, 'BODY_PATH': .02, 'BOTH_PATH': .60, 'VISIBLE_OFF_PATH': .02}[focus]
            origin = [front - lo[0], lateral - (lo[1] + hi[1]) / 2, low_z - lo[2]]
        else:
            # Tree x is its root/actor reference, not its potentially empty
            # crown envelope. A crown may extend behind or beyond the image.
            y = sign * (1.10 + lateral) if focus == 'HEAD_PATH' else (sign * .10 if focus == 'BOTH_PATH' else lateral)
            origin = [front, y, -lo[2]]
        obj['center_m'] = list(map(float, origin)); targets.append(obj)
    if focus == 'VISIBLE_OFF_PATH':
        a, b = envelope(targets, assets)
        # Extra clearance admits real platform/feet and preserves the optical
        # mapping in controls. Near targets are never independently recentered.
        inner = 2.20 if angular else .95
        shift = sign * inner - (a[1] if sign > 0 else b[1])
        translate(targets, np.array([0., shift, 0.]))
    far_targets = copy.deepcopy(targets) if angular else None
    if angular and row['distance'] == 'near':
        for obj in targets:
            obj['center_m'] = (OPTICAL_CENTRE + .5 * (np.array(obj['center_m']) - OPTICAL_CENTRE)).tolist()
            obj['scale'] = [.5] * 3
    return targets, far_targets


def mount_geometry(targets, row, assets, material):
    """Grounded objects stay grounded; overhead objects receive actual steel."""
    lo, hi = envelope(targets, assets)
    mounts = []
    if row['family'] in ('landing_frame', 'birch_branches'):
        if lo[2] > 1e-8:
            # A separately flagged half-scale optical control needs a real
            # raised plinth. Its hazard bits are part of native truth.
            size = hi - lo
            mounts.append(box('angular_raised_plinth', [(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, lo[2]/2],
                              [size[0]+.08, size[1]+.08, lo[2]], material))
        return mounts
    if row['family'] == 'wooden_slats':
        for side in (-1, 1):
            subset = targets[::2] if side < 0 else targets[1::2]
            a, b = envelope(subset, assets)
            x = (a[0] + b[0]) / 2
            y = a[1] + .045 if side < 0 else b[1] - .045
            top = max(bounds(obj, assets)[0][2] for obj in subset)
            post = f'rack_post_{side}'
            mounts.append(box(f'rack_foot_{side}', [x, y, .03], [.38, .25, .06], material))
            mounts.append(box(post, [x, y, top/2], [.07, .07, top], material, f'rack_foot_{side}'))
            for j, obj in enumerate(subset):
                bottom = bounds(obj, assets)[0][2]
                mounts.append(box(f'rack_bracket_{side}_{j}', [x, y, bottom-.025],
                                  [b[0]-a[0]+.08, .16, .05], material, post))
    else:
        # Two transverse cradle rails and four grounded side legs. Rails are
        # real obstacles too; no oracle deletes their native event contribution.
        rail_top = float(lo[2])
        rail_h = min(.05, rail_top)
        for j, fraction in enumerate((.2, .8)):
            x = lo[0] + fraction * (hi[0] - lo[0])
            mounts.append(box(f'cradle_rail_{j}', [x, (lo[1]+hi[1])/2, rail_top-rail_h/2],
                              [.08, hi[1]-lo[1]+.10, rail_h], material, f'cradle_leg_{j}_-1'))
            for side in (-1, 1):
                y = (lo[1] + .02) if side < 0 else (hi[1] - .02)
                h = max(rail_top - rail_h, .01)
                mounts.append(box(f'cradle_leg_{j}_{side}', [x, y, h/2], [.08, .08, h], material))
    return mounts


def occluder_geometry(row, assets, material):
    if not row['view'].startswith('partial_occlusion'):
        return []
    side = -1 if row['view'].endswith('left') else 1
    off = row['focus'] == 'VISIBLE_OFF_PATH'
    # In OFF_PATH, rotate the full slat along x and move its whole support
    # outside the corridor; the original y=.48, yaw=90 slat crossed the path.
    obj = mesh('fixed_foreground_slat', assets['wooden_slats'],
               dict(pitch=0., yaw=0. if off else 90., roll=0.), target=False)
    lo, hi = bounds(obj, assets)
    center = np.array([.75, side * (.80 if off else .48), 1.60])
    obj['center_m'] = (center - (lo + hi) / 2).tolist()
    a, b = bounds(obj, assets)
    y = (a[1]+.04) if side < 0 else (b[1]-.04)
    x = float(center[0]); top = a[2]
    return [obj, box('occluder_foot', [x, y, .03], [.32, .20, .06], material),
            box('occluder_post', [x, y, top/2], [.06, .06, top], material, 'occluder_foot'),
            box('occluder_bracket', [x, y, top-.02], [.20, .12, .04], material, 'occluder_post')]


def optional_geometry(template, front):
    out = []
    for obj in (o for o in template['objects'] if not o['target_part']):
        o = copy.deepcopy(obj)
        sign = -1 if o['name'].endswith('_-1') else 1
        x = front + .10 + (1.50 if o['name'].startswith('rear_') else .75 if o['name'].startswith('side_rail') else 0.)
        z = 1.50 if o['name'].startswith('clamp') else o['center_m'][2] - template['floor_z_m']
        o.update(center_m=[x, sign * .80, z], rotation_deg=dict(pitch=0., yaw=0., roll=0.),
                 target_part=False, mandatory_mount=False, optional_context=True, physical_role='context')
        out.append(o)
    assert len(out) == 12
    return out


def required_bits(row):
    bits = [False] * 4
    d = int(row['distance'] == 'far')
    if row['focus'] in ('BODY_PATH', 'BOTH_PATH'): bits[d] = True
    if row['focus'] in ('HEAD_PATH', 'BOTH_PATH'): bits[2+d] = True
    return bits


def validate(cases, rows, local_scenes, assets, angular_errors):
    assert len(cases) == len({c['name'] for c in cases}) == 4096
    assert [c['name'] for c in cases] == [r['frame_id'] for r in rows]
    assert Counter(c['training_role'] for c in cases) == dict(TRAIN_CANDIDATE=2048, CALIBRATION=1024, HELDOUT_GEOMETRY=1024)
    assert Counter(c['regime'] for c in cases) == dict(ACTUAL_SIZE=3072, ANGULAR_CONTROL=1024)
    assert sum(c['canary'] for c in cases) == 64 and sum(c['native_audit_sample'] for c in cases) == 80
    assert all(c['canary'] == r['is_canary'] and c['native_audit_sample'] == r['native_audit'] for c, r in zip(cases, rows))
    assert len({c['pair_id'] for c in cases}) == 2048
    for site in {c['site_id'] for c in cases}:
        assert Counter(c['training_role'] for c in cases if c['site_id'] == site) == dict(TRAIN_CANDIDATE=256, CALIBRATION=128, HELDOUT_GEOMETRY=128)
    angular_groups = defaultdict(list)
    for c in cases:
        if c['angular_pair_id'] is not None: angular_groups[c['angular_pair_id']].append(c)
    assert len(angular_groups) == 512
    for pair in angular_groups.values():
        assert len(pair) == 2 and {c['declared_range'] for c in pair} == {'near', 'far'}
        for key in ('training_role', 'recipe_group', 'site_id', 'support_context'):
            assert pair[0][key] == pair[1][key]
    for a, b in zip(cases[::2], cases[1::2]):
        assert a['pair_id'] == b['pair_id'] and a['support_context'] == 'unsupported' and b['support_context'] == 'supported'
        assert a['objects'] == [o for o in b['objects'] if not o['optional_context']]
        for key in ('geometry_id', 'scene_geometry_id', 'training_role', 'recipe_group', 'placement_bounds'):
            assert a[key] == b[key]
    groups = defaultdict(set)
    for c in cases: groups[c['recipe_group']].add(c['training_role'])
    assert len(groups) == 64 and all(len(v) == 1 for v in groups.values())
    collisions = {}
    for key in ('geometry_id', 'scene_geometry_id'):
        role_sets = {r: {c[key] for c in cases if c['training_role'] == r} for r in ROLES}
        collisions[key] = {a + '__' + b: len(role_sets[a] & role_sets[b]) for a, b in itertools.combinations(ROLES, 2)}
        assert not any(collisions[key].values()), (key, collisions[key])
    off_min, all_min_z, min_x = [], [], []
    for c, objects in zip(cases, local_scenes):
        assert len({o['name'] for o in objects}) == len(objects)
        for obj in objects:
            a, b = bounds(obj, assets)
            assert np.isfinite(object_points(obj, assets)).all()
            assert a[2] >= -1e-9, (c['name'], obj['name'], a.tolist())
            all_min_z.append(float(a[2])); min_x.append(float(a[0]))
            assert not (obj['mandatory_mount'] and obj['optional_context'])
            if c['regime'] == 'ACTUAL_SIZE' and obj['target_part']: assert obj['scale'] == [1.] * 3
            if obj['optional_context'] or c['off_path_expected_zero']:
                clear = min(abs(a[1]), abs(b[1])) if a[1] * b[1] > 0 else 0.
                assert clear > CORRIDOR_HALF_WIDTH, (c['name'], obj['name'], a.tolist(), b.tolist())
                if c['off_path_expected_zero']: off_min.append(float(clear))
    assert len(angular_errors) == 512 and max(angular_errors) < 1e-12
    return dict(frames=4096, intact_support_pairs=2048, recipes=64,
                unique_actual_target_geometries=len({c['geometry_id'] for c in cases}),
                unique_nonoptional_scenes=len({c['scene_geometry_id'] for c in cases}),
                cross_role_geometry_collisions=collisions, mandatory_context_identity=True,
                optional_context_outside_corridor=True, off_path_min_lateral_clearance_m=min(off_min),
                minimum_configured_z_m=min(all_min_z), minimum_configured_x_m=min(min_x),
                angular_pair_count=512, angular_target_vertex_max_error_m=max(angular_errors),
                full_AABB_in_view_required=False, native_visibility_verified=False,
                configured_bounds_are_not_native_truth=True)


def prepare(root, output, evidence):
    started = time.perf_counter()
    artifact_root = (root / 'artifacts.local').resolve()
    task = artifact_root / 'work' / TASK
    assert output == task / 'spec-v1' and evidence == task / 'spec-preparation-v1'
    assert not output.exists() and not (evidence / 'receipt.json').exists()
    work = artifact_root / 'work'; draft = work / 'mz71-missing-state-calibration-20260911/next-source-design-v1'
    inputs = {}
    def bound(path, expected=None):
        digest = sha(path); assert expected is None or digest == expected, str(path)
        inputs[str(path)] = digest
        return read(path) if path.suffix == '.json' else None
    draft_receipt = bound(draft / 'receipt.json', DRAFT_SHA)
    def draft_file(name):
        path = draft / name
        value = bound(path, draft_receipt['outputs'][name])
        return value if value is not None else [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
    config = draft_file('source-config-draft.json'); recipes = draft_file('recipes.json')
    rows = draft_file('rows-draft.jsonl'); dependencies = draft_file('local-dependencies.json')
    draft_file('native-and-asset-evidence.json')
    prior_task = work / 'mz67-topology-source-20260911'
    prior = bound(prior_task / 'spec-v1/manifest.json', PRIOR_SHA)
    specs = {}
    for shard in prior['shards']:
        if shard['canary']:
            p = prior_task / 'spec-v1/shards' / (shard['shard_id'] + '.json')
            spec = bound(p, shard['sha256']); specs[spec['cases'][0]['region_id']] = spec
    original = bound(work / 'mz42-rich-objects-20260910/spec-v2/spec.json',
                     'cc76970fe3f358c4cffca24d423cb1d81a73720d609a4e7dc21b8f3067cf7600')
    template = next(c for c in original['cases'] if c['condition']['family'] == 'oblique_rod' and c['variant_id'] == 'HEAD_ONLY')
    material = next(o['material_asset'] for o in template['objects'] if o['target_part'])
    for name in ('mz72_physical_spec.py', 'mz48_prepare.py', 'mz67_geometry_spec.py'):
        bound(Path(__file__).with_name(name))
    packages = {p['package']: p for p in dependencies['packages']}
    assets = {}
    for family, data in config['asset_refs'].items():
        local = packages[data['package']]
        p = Path(local['path']); bound(p, data['sha256'])
        assets[family] = dict(asset=data['package'], object_path=data['object_path'], asset_file=str(p),
                              asset_sha256=data['sha256'], local_min_m=data['local_min_m'],
                              local_max_m=data['local_max_m'], local_size_m=data['local_size_m'],
                              bounds_authority=data['authority'], native_inventory_instances=data['native_inventory_instances'])
    materials = {}
    for package in sorted({o['material_asset'] for o in template['objects']}):
        p = artifact_root / 'unreal/CitySample/Content' / (package.removeprefix('/Game/') + '.uasset')
        bound(p); materials[package] = dict(asset_file=str(p), sha256=sha(p))
    recipe_lookup = {(r['family'], r['recipe_id']): resolved_recipe(r) for r in recipes}
    sites = prior['sites']; site_lookup = {s['site_id']: s for s in sites}
    cases, configurations, scenes, angular_errors = [], [], [], []
    local_cache = {}
    for row in rows:
        key = (row['family'], row['recipe_id'], row['condition_index'])
        recipe = recipe_lookup[key[:2]]
        if key not in local_cache:
            targets, angular_far = target_geometry(row, recipe, assets)
            mandatory = targets + mount_geometry(targets, row, assets, material) + occluder_geometry(row, assets, material)
            optional = optional_geometry(template, recipe[row['distance'] + '_interaction_x_m'])
            target_sha, scene_sha = fingerprint(targets), fingerprint(mandatory)
            placement = []
            for obj in mandatory:
                lo, hi = bounds(obj, assets)
                placement.append(dict(name=obj['name'], camera_aligned_min_m=lo.tolist(), camera_aligned_max_m=hi.tolist()))
            local_cache[key] = (mandatory, optional, target_sha, scene_sha, placement)
            configurations.append(dict(family=row['family'], recipe_id=row['recipe_id'], recipe_group=row['recipe_group'],
                                       training_role=row['role'], condition_index=row['condition_index'], regime=row['regime'],
                                       placement_focus=row['focus'], declared_range=row['distance'], view=row['view'],
                                       geometry_id=target_sha, scene_geometry_id=scene_sha,
                                       targets_camera_coordinates=targets, resolved_recipe=recipe))
        mandatory, optional, target_sha, scene_sha, placement = local_cache[key]
        site = site_lookup[row['site_id']]
        # Complete sites are authoritative; prior canaries need not include
        # all eight sites. This is the original floor-aligned MZ48 wearer.
        reference = dict(camera=copy.deepcopy(site['camera']),
                         wearer={**site['camera'], 'z': site['floor_z_m']},
                         floor_z_m=site['floor_z_m'], site_id=site['site_id'], region_id=site['region_id'])
        for old_case in specs[row['region_id']]['cases']:
            if old_case['site_id'] == row['site_id']:
                assert all(reference[k] == old_case[k] for k in reference)
                break
        local = copy.deepcopy(mandatory + (optional if row['support_context'] == 'supported' else []))
        scenes.append(copy.deepcopy(local))
        if row['regime'] == 'ANGULAR_CONTROL' and row['distance'] == 'near':
            targets = [o for o in local if o['target_part']]
            far_targets = target_geometry({**row, 'distance': 'far'}, recipe, assets)[0]
            errors = [float(np.max(np.abs(object_points(n, assets) - (OPTICAL_CENTRE + .5 * (object_points(f, assets) - OPTICAL_CENTRE)))))
                      for n, f in zip(targets, far_targets)]
            angular_errors.append(max(errors))
        for obj in local:
            obj['center_m'] = world(site['camera'], site['floor_z_m'], obj['center_m'])
            obj['rotation_deg']['yaw'] += site['camera']['yaw']
        bits = required_bits(row)
        case = {k: copy.deepcopy(reference[k]) for k in ('camera', 'wearer', 'floor_z_m', 'site_id', 'region_id')}
        case.update(name=row['frame_id'], frame_id=row['frame_id'], global_index=row['index'],
                    pair_id=row['pair_id'], group_id=row['pair_id'], variant_id=row['focus'],
                    placement_focus=row['focus'], required_positive_bits=bits, expected_events=bits.copy(),
                    expected_collapsed_relation=[row['focus'] in ('BODY_PATH', 'BOTH_PATH'), row['focus'] in ('HEAD_PATH', 'BOTH_PATH')],
                    off_path_expected_zero=row['focus'] == 'VISIBLE_OFF_PATH',
                    declared_range=row['distance'], source_role='DEV_ONLY', training_role=row['role'],
                    split_unit='recipe_group', recipe_id=row['recipe_id'], recipe_group=row['recipe_group'],
                    geometry_recipe_id=row['recipe_group'], regime=row['regime'], angular_pair_id=row['angular_pair_id'],
                    geometry_id=target_sha, scene_geometry_id=scene_sha, site_replica=row['site_replica'],
                    view=row['view'], profile_id=row['condition_index'], floor_check=False, probe_native_floor=True,
                    support_context=row['support_context'], canary=row['is_canary'], native_audit_sample=row['native_audit'],
                    placement_bounds=copy.deepcopy(placement), objects=local,
                    condition=dict(family=row['family'], desired_relation=row['focus'], condition_id=row['frame_id']))
        cases.append(case)
    checks = validate(cases, rows, scenes, assets, angular_errors)
    revisions = [
        'Actual recipe IDs 6..11 receive +4 degree yaw and +0.12 m near/far forward translation; this corrects exact cross-role duplication using effective physical factors.',
        'Wood slat inner envelope ends define the 0.10/0.18/0.26 m gap; incompatible draft centres +/-0.18 m are replaced without resizing meshes.',
        'All actual mounts exist in both contexts. Half-scale grounded controls have a visible 0.85 m plinth. Optional inherited twelve-object scaffold is context only.',
        'OFF_PATH targets and real mounts clear the corridor; its foreground slat is along x outside the corridor instead of crossing the path.',
        'Whole trees use root x and measured-envelope grounding; upright crown bounds may be behind the camera. The fallen 75-degree tree is turned 180 degrees so its length extends away from the camera. Other units use front-envelope x. No complete AABB visibility or single-range containment requirement.'
    ]
    output.mkdir(parents=True, exist_ok=False)
    shards = []
    def emit(shard_id, subset, region, canary):
        spec = {k: copy.deepcopy(v) for k, v in specs[region].items() if k not in ('cases', 'provenance')}
        spec.update(schema=SCHEMA, cases=subset, purpose='SOURCE_SPEC_ONLY_NOT_CAPTURED',
                    scope='Controlled physical-size assets and explicitly separate optical-size controls; consumed sites',
                    provenance=dict(inputs=inputs, fixed_budget=4096, draft_row_identity_preserved=True))
        path = output / 'shards' / (shard_id + '.json'); write(path, spec)
        shards.append(dict(shard_id=shard_id, canary=canary, frames=len(subset), path=str(path), sha256=sha(path)))
    for region in sorted(specs):
        emit('canary-' + region, [c for c in cases if c['region_id'] == region and c['canary']], region, True)
    for site in sites:
        emit('main-' + site['site_id'], [c for c in cases if c['site_id'] == site['site_id'] and not c['canary']], site['region_id'], False)
    write(output / 'geometry-configurations.json', dict(configurations=configurations))
    write(output / 'resolved-recipes.json', list(recipe_lookup.values()))
    write(output / 'fixed-audit-ids.json', dict(canary=[c['name'] for c in cases if c['canary']],
                                               extra_native_audit=[c['name'] for c in cases if c['native_audit_sample'] and not c['canary']]))
    manifest = dict(schema=MANIFEST_SCHEMA, status='SPEC_READY_NOT_CAPTURED', task=TASK,
                    frames=4096, canary_frames=64, main_frames=4032, native_audit_frames=80,
                    roles=dict(Counter(c['training_role'] for c in cases)), regimes=dict(Counter(c['regime'] for c in cases)),
                    sites=sites, families=config['families'], assets=assets, materials=materials, shards=shards, inputs=inputs,
                    source_pairs=2048, independent_geometry_recipes_per_family=16,
                    unique_camera_geometry_configurations=checks['unique_actual_target_geometries'],
                    static_checks=checks, physical_revisions=revisions,
                    query_order=QUERIES, split_rule=config['split_rule'],
                    actual_truth_policy='Every native query bit remains; required_positive_bits is only a required subset. OFF_PATH expects zero. UNKNOWN is never negative.',
                    source_admission_policy='All 64 fixed canaries require source health, focus subset, physical visual review and native audit before main; no frame replacement or threshold selection.',
                    native_surface_and_material_visibility='UNMEASURED for these placements. Native catalogue/AABB is placement evidence only; Birch_h is not verified leaf transparency, wind or flexible geometry.',
                    predictor_inputs=['Original RGB', 'Original 64-zone ranges/valid with fixed calibration'],
                    evaluator_only=['native depth/world_support', 'truth/known', 'role/site/family/recipe/focus', 'all geometry metadata'],
                    new_captures=0, training_steps=0, model_inference_frames=0)
    write(output / 'manifest.json', manifest)
    outputs = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*.json'))}
    receipt = dict(status='PASS_CPU_SPEC_ONLY', seconds=time.perf_counter()-started, inputs=inputs,
                   outputs=outputs, manifest_sha256=sha(output / 'manifest.json'), checks=checks,
                   physical_revisions=revisions, code_sha256=sha(Path(__file__)),
                   new_captures=0, training_steps=0, model_inference_frames=0, raw_depth_reads=0,
                   renderer_visibility_checks=0, new_registrations=0)
    write(output / 'receipt.json', receipt)
    write(evidence / 'receipt.json', {**receipt, 'spec_receipt': dict(path=str(output/'receipt.json'), sha256=sha(output/'receipt.json'))})
    report = ('MZ72 executable spec prepared; CPU checks only.\n\n'
              f'4096 fixed rows, 2048 pairs; roles {manifest["roles"]}; 3072 scale-1 main rows and 1024 separate angular controls. '
              f'All original 64 canary and 80 audit IDs remain. Manifest SHA256: {receipt["manifest_sha256"]}.\n\n'
              + '\n'.join('- '+r for r in revisions) + '\n\n'
              f'Actual target configurations: {checks["unique_actual_target_geometries"]}; cross-role exact target/mandatory scene collisions: 0. '
              f'512 near/far optical pairs, vertex identity error <= {checks["angular_target_vertex_max_error_m"]:.3g} m. '
              'These are exact parameter-grouping checks, not native-mask novelty, natural-scene independence or a sensing impossibility result.\n\n'
              'Canary risks: unknown real Birch_h branch height/density; stair opening geometry; cradle contact with the actual duct surface; '
              'tree crown around the camera; foreground occlusion and additional mount/native hazards. Bounds do not establish any of these. '
              'All 64 must pass source/focus/native/physical visual admission before the remaining 4032 frames.\n')
    (evidence / 'REPORT.md').write_text(report, encoding='utf-8', newline='\n')
    print(json.dumps(dict(status=receipt['status'], manifest_sha256=receipt['manifest_sha256'],
                          seconds=receipt['seconds'], shards=[(s['shard_id'], s['frames']) for s in shards], checks=checks), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.output.resolve(), args.evidence.resolve())
