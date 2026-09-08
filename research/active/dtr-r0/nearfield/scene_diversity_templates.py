"""Shared64-quartet local geometry library; never an admitted capture source."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path

import contextual_collection1000 as source
from contextual_geometry import intrusion_metrics, target_contact
from contextual_sampling import FAMILIES, RELATIONS
from contextual_scene import MATERIALS

SEED = 20260920
AUTHORITY = 'LOCAL_GEOMETRY_ONLY_NOT_CAPTURE_SOURCE'
LOCAL_FRAME = 'CAMERA_XY_ZERO_WEARER_FLOOR_Z_ZERO_FORWARD_PLUS_X'
CONTEXT_FIELDS = ('family', 'distance_m', 'lateral_offset_m', 'eye_height_m',
    'camera_pitch_deg', 'tilt_degrees', 'thickness_m', 'common_height_jitter_m',
    'lighting_profile', 'route_lateral_delta_m', 'occluder')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _geometry(cases):
    # Intentionally includes materials and within-quartet appearance settings.
    # World/role/group names are excluded: the two arms share this exact payload.
    return [{key: case[key] for key in ('variant_id', 'camera', 'wearer', 'floor_z_m',
        'objects', 'sun_intensity_scale', 'skylight_intensity_scale')} for case in cases]


def _context(case):
    return dict(camera=case['camera'], wearer=case['wearer'], floor_z_m=case['floor_z_m'],
        conditions={k: case['condition'][k] for k in CONTEXT_FIELDS},
        objects=[dict(**{k: obj.get(k) for k in ('name', 'material_asset', 'support_parent',
            'target_part', 'rotation_deg')}, center_xy_m=obj['center_m'][:2],
            size_xy_m=obj['size_m'][:2], fixed_size_z_m=None if obj['name'].startswith('suspension_') else obj['size_m'][2])
            for obj in case['objects']],
        sun_intensity_scale=case['sun_intensity_scale'], skylight_intensity_scale=case['skylight_intensity_scale'])


def _backboard(case):
    if case['condition']['family'] not in ('cabinet', 'hanging_sign'):
        return
    case['objects'].append(dict(name='coverage_grounded_backboard',
        center_m=[-75.5, 15.08, 1.95], size_m=[2., .20, 3.5],
        material_asset=MATERIALS['metal'], support_parent='ground', target_part=False))
    for obj in case['objects']:
        if obj.get('support_parent') == 'existing_facade':
            obj['support_parent'] = 'coverage_grounded_backboard'
    case['scene_context'] += '; facade attachment replaced by grounded freestanding backboard'


def validate_quartet(quartet):
    """Validate local cuboids/context only. No native or visibility claim."""
    cases = quartet['cases']
    require(len(cases) == 4 and [c['variant_id'] for c in cases] == list(RELATIONS),
        'Require exactly the four ordered relations')
    require(quartet['authority'] == AUTHORITY and quartet['coordinate_frame'] == LOCAL_FRAME,
        'Expected a local geometry quartet, not a placed capture source')
    common = _context(cases[0])
    for case in cases:
        require(case['camera']['x'] == case['camera']['y'] == case['wearer']['z'] == case['floor_z_m'] == 0.,
            'Local camera XY and floor/wearer Z must be zero')
        require(_context(case) == common, 'Quartet context changed')
        require(case['condition']['family'] == quartet['family'], 'Family mismatch')
        names = {obj['name'] for obj in case['objects']}
        require(len(names) == len(case['objects']), 'Duplicate assembly names')
        require(all(obj['support_parent'] in names | {'ground'} for obj in case['objects']),
            'Assembly still depends on an external facade')
        contact = target_contact(case['objects'], case['wearer'])
        require(contact['relation'] == case['variant_id'] == case['condition']['desired_relation'],
            'Intended relation does not match local cuboid contact')
        require(contact == case['geometric_contact'], 'Stale local contact record')
        require(intrusion_metrics(case['objects'], case['wearer']) == case['geometric_intrusion'],
            'Stale local intrusion record')
    require(digest(_geometry(cases)) == quartet['shared_geometry_sha256'], 'Local geometry hash mismatch')
    require(digest(common) == quartet['shared_context_sha256'], 'Local context hash mismatch')
    return True


def generate(template):
    """Reuse exact first256 cases of collection(seed20260920), 64 quartets.

    The source seed is restored even on error. Input template and existing
    generator files are not modified. No map/capture admission is inherited.
    """
    before = source.SEED
    try:
        source.SEED = SEED
        generated = source.collection(template)
    finally:
        source.SEED = before
    selected = copy.deepcopy(generated['cases'][:256])
    quartets = []
    for unit in range(64):
        cases = selected[unit*4:(unit+1)*4]
        parent = cases[0]['group_id']
        group = f'scene-diversity-seed{SEED}-unit{unit:02d}'
        for case in cases:
            _backboard(case)
            camera = case['camera']
            origin = [camera['x'], camera['y'], case['floor_z_m']]
            require(camera['yaw'] == case['wearer']['yaw'] == 0., 'Collection source must face+X')
            require(case['wearer']['z'] == origin[2], 'Wearer is not on source floor')
            for key in ('camera', 'wearer'):
                for k, axis in enumerate(('x', 'y', 'z')):
                    case[key][axis] -= origin[k]
            for obj in case['objects']:
                obj['center_m'] = [v-origin[k] for k, v in enumerate(obj['center_m'])]
            case['floor_z_m'] = 0.
            case['source_lineage'] = dict(collection_group_id=parent, collection_case_name=case['name'],
                original_camera_xy_floor_xyz_m=origin, source_seed=SEED)
            case.update(name=group+'-'+case['variant_id'], group_id=group, template_unit_id=group,
                source_site_id='LOCAL_TEMPLATE_NO_WORLD_ID', source_role='UNASSIGNED_LOCAL_TEMPLATE',
                geometry_frame=LOCAL_FRAME, authority=AUTHORITY, native_labels_status='NOT_RUN')
            case['condition'].update(condition_id=case['name'], counterfactual_parent_id=group,
                source_site_id='LOCAL_TEMPLATE_NO_WORLD_ID', split_group_id='LOCAL_TEMPLATE_NO_WORLD_ID')
            case['geometric_contact'] = target_contact(case['objects'], case['wearer'])
            case['geometric_intrusion'] = intrusion_metrics(case['objects'], case['wearer'])
        quartet = dict(unit_id=group, family=cases[0]['condition']['family'], cases=cases,
            authority=AUTHORITY, coordinate_frame=LOCAL_FRAME, native_labels_status='NOT_RUN',
            shared_geometry_sha256=digest(_geometry(cases)), shared_context_sha256=digest(_context(cases[0])))
        validate_quartet(quartet)
        quartets.append(quartet)
    family_counts = Counter(q['family'] for q in quartets)
    relation_counts = Counter(c['variant_id'] for q in quartets for c in q['cases'])
    require(family_counts == dict.fromkeys(FAMILIES, 16), 'Expected16 quartets per family')
    require(relation_counts == dict.fromkeys(RELATIONS, 64), 'Expected64 cases per relation')
    return dict(schema='scene-diversity-local-templates-v1', authority=AUTHORITY,
        coordinate_frame=LOCAL_FRAME, native_labels_status='NOT_RUN', capture_source=False,
        seed=SEED, selection='contextual_collection1000.collection first256 cases; no model/outcome selection',
        suite_contract=dict(quartets=64, cases=256, family_quartets=dict(family_counts), relation_cases=dict(relation_counts)),
        shared_geometry_sha256=digest([q['shared_geometry_sha256'] for q in quartets]), quartets=quartets,
        source_sha256={name: file_sha(Path(__file__).with_name(name)) for name in (
            'scene_diversity_templates.py', 'contextual_collection1000.py', 'contextual_scene.py',
            'contextual_geometry.py', 'contextual_sampling.py', 'contact_retina_spec.py')},
        scope='Both experimental arms must reuse the same local geometry hashes; new world placements require native scene/visibility acceptance. No native labels or training admission supplied.')


def place_quartet(quartet, camera_xy_m, floor_z_m, yaw_deg, placement_id=None):
    """Rigid world placement of all four cases. +Z floor shift applies to ALL Z.

    UE Rotator yaw is the outer world-Z rotation: composing an outer yaw adds
    to each original yaw, preserving original pitch and roll (e.g. slanted rods).
    Analytic contact remains local; target_contact does not support world yaw.
    """
    validate_quartet(quartet)
    require(isinstance(camera_xy_m, (tuple, list)) and len(camera_xy_m) == 2, 'camera_xy_m must have two coordinates')
    x, y = map(float, camera_xy_m)
    floor, yaw = float(floor_z_m), float(yaw_deg)
    require(all(math.isfinite(v) for v in (x, y, floor, yaw)), 'Placement must be finite')
    co, si = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    result = copy.deepcopy(quartet)
    for case in result['cases']:
        for key in ('camera', 'wearer'):
            pose = case[key]
            px, py = pose['x'], pose['y']
            pose.update(x=x+co*px-si*py, y=y+si*px+co*py, z=floor+pose['z'], yaw=pose.get('yaw', 0.)+yaw)
        for obj in case['objects']:
            px, py, pz = obj['center_m']
            obj['center_m'] = [x+co*px-si*py, y+si*px+co*py, floor+pz]
            rotation = obj.setdefault('rotation_deg', {})
            for axis in ('pitch', 'yaw', 'roll'):
                rotation.setdefault(axis, 0.)
            rotation['yaw'] = rotation.get('yaw', 0.)+yaw
        case.update(floor_z_m=floor, geometry_frame='ANALYTIC_CONTACT_RECORDED_IN_LOCAL_FRAME_BEFORE_RIGID_PLACEMENT',
            native_labels_status='REQUIRES_NATIVE_WORLD_ACCEPTANCE', local_geometry_sha256=quartet['shared_geometry_sha256'])
        if placement_id is not None:
            case['group_id'] = str(placement_id)+'/'+quartet['unit_id']
            case['name'] = case['group_id']+'-'+case['variant_id']
            case['condition'].update(condition_id=case['name'], counterfactual_parent_id=case['group_id'])
    result.update(coordinate_frame='WORLD_PLACED_LOCAL_GEOMETRY_NOT_ADMITTED_CAPTURE_SOURCE',
        native_labels_status='REQUIRES_NATIVE_WORLD_ACCEPTANCE', capture_source=False,
        placement=dict(camera_xy_m=[x, y], floor_z_m=floor, yaw_deg=yaw, placement_id=placement_id),
        local_contact_authority='Original local+X cuboid calculation only; NOT world native depth/visibility labels')
    validate_placement(result, quartet)
    return result


def inverse_placement(placed):
    """Return inverse-transformed cases for numeric verification, not admission."""
    p = placed['placement']
    x, y = p['camera_xy_m']; floor = p['floor_z_m']; yaw = p['yaw_deg']
    co, si = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cases = copy.deepcopy(placed['cases'])
    for case in cases:
        for key in ('camera', 'wearer'):
            pose = case[key]; dx, dy = pose['x']-x, pose['y']-y
            pose.update(x=co*dx+si*dy, y=-si*dx+co*dy, z=pose['z']-floor, yaw=pose['yaw']-yaw)
        for obj in case['objects']:
            dx, dy, dz = obj['center_m'][0]-x, obj['center_m'][1]-y, obj['center_m'][2]-floor
            obj['center_m'] = [co*dx+si*dy, -si*dx+co*dy, dz]
            obj['rotation_deg']['yaw'] -= yaw
        case['floor_z_m'] -= floor
    return cases


def validate_placement(placed, local):
    """Bind actual placed coordinates/materials to the claimed shared local hash."""
    validate_quartet(local)
    require(placed['shared_geometry_sha256'] == local['shared_geometry_sha256'], 'Placement local hash mismatch')
    require(len(placed['cases']) == 4, 'Placement lost a relation')
    recovered = inverse_placement(placed)
    close = lambda a, b: math.isclose(float(a), float(b), rel_tol=0., abs_tol=1e-9)
    for case, raw in zip(recovered, local['cases']):
        require(case['variant_id'] == raw['variant_id'], 'Placement relation order changed')
        require(close(case['floor_z_m'], 0.), 'Placement floor transform mismatch')
        for key in ('camera', 'wearer'):
            require(all(close(case[key].get(k, 0.), raw[key].get(k, 0.)) for k in ('x', 'y', 'z', 'pitch', 'yaw', 'roll')),
                'Placement pose does not invert to shared local geometry')
        require(len(case['objects']) == len(raw['objects']), 'Placement assembly changed')
        for obj, original in zip(case['objects'], raw['objects']):
            require(all(obj[k] == original[k] for k in ('name', 'size_m', 'material_asset', 'support_parent', 'target_part')),
                'Placement assembly identity/material changed')
            require(all(close(a, b) for a, b in zip(obj['center_m'], original['center_m'])), 'Placement center transform mismatch')
            require(all(close(obj.get('rotation_deg', {}).get(k, 0.), original.get('rotation_deg', {}).get(k, 0.))
                for k in ('pitch', 'yaw', 'roll')), 'Placement rotation transform mismatch')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    artifacts = (root/'artifacts.local').resolve(strict=True)
    out = args.output.resolve()
    require(out.is_relative_to(artifacts) and out != artifacts and not out.exists(), 'Fresh ignored artifacts.local output required')
    template_bytes = args.template.read_bytes()
    library = generate(json.loads(template_bytes.decode('utf-8-sig')))
    library['template_input'] = dict(path=str(args.template.resolve()), sha256=hashlib.sha256(template_bytes).hexdigest())
    out.mkdir(parents=True)
    path = out/'templates.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(library, stream, indent=2, allow_nan=False)
    require(args.template.read_bytes() == template_bytes, 'Input template changed')
    print(json.dumps(dict(authority=AUTHORITY, native_labels_status='NOT_RUN', quartets=64, cases=256,
        output=str(path), sha256=file_sha(path), shared_geometry_sha256=library['shared_geometry_sha256'])))


if __name__ == '__main__':
    main()
