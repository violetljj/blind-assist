"""Fixed same-world coverage pilot; source roles precede all model access."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import contextual_collection1000 as source
from contextual_geometry import target_contact, intrusion_metrics
from contextual_scene import MATERIALS

SITES = {'train': (32., 29.5, 0.), 'eval': (67., 53.5, 180.)}


def generate(template, role):
    seed = 20260918 + (role == 'eval')
    before = source.SEED
    try:
        source.SEED = seed
        result = source.collection(template)
    finally:
        source.SEED = before
    cases = result['cases'][:64]
    x, y, yaw = SITES[role]
    angle = math.radians(yaw)
    co, si = math.cos(angle), math.sin(angle)
    for i, case in enumerate(cases):
        unit = i // 4
        # Sixteen stations in an 8x2 layout; families rotate between stations.
        # Each station retains one family across its four relation variants.
        offset_x = (2. if role == 'train' else 1.5) * (unit % 8)
        offset_y = float(unit // 8)
        cx, cy = case['camera']['x'], case['camera']['y']
        objects = case['objects']
        if case['condition']['family'] in ('cabinet', 'hanging_sign'):
            # Replace missing facade attachment by a visibly grounded freestanding
            # backboard. It is outside the swept aisle and fixed within the quartet.
            objects.append(dict(name='coverage_grounded_backboard',
                center_m=[-75.5, 15.08, 1.95], size_m=[2.0, .20, 3.5],
                material_asset=MATERIALS['metal'], support_parent='ground', target_part=False))
            for obj in objects:
                if obj.get('support_parent') == 'existing_facade':
                    obj['support_parent'] = 'coverage_grounded_backboard'
        # Exact contact remains calculated in the route-local frame. The rendered
        # scene receives the same rigid transform, including oblique rod rotations.
        contact = target_contact(objects, case['wearer'])
        assert contact['relation'] == case['condition']['desired_relation']
        case['geometric_contact'] = contact
        case['geometric_intrusion'] = intrusion_metrics(objects, case['wearer'])
        case['geometry_frame'] = 'PRE_RIGID_TRANSFORM_ROUTE_LOCAL_X; native depth validates world visibility'
        def xy(px, py):
            dx, dy = px - cx, py - cy
            return x + offset_x + co*dx-si*dy, y + offset_y + si*dx+co*dy
        for key in ('camera', 'wearer'):
            pose = case[key]
            pose['x'], pose['y'] = xy(pose['x'], pose['y'])
            pose['yaw'] = yaw
        for obj in objects:
            obj['center_m'][:2] = xy(*obj['center_m'][:2])
            obj.setdefault('rotation_deg', dict(pitch=0., roll=0.))['yaw'] = yaw
        gid = f'coverage64-{role}-seed{seed}-unit{unit:02d}'
        case.update(name=f'{gid}-{case["variant_id"]}', group_id=gid, parent_group_id=gid,
                    source_site_id=f'plaza-{role}-{x:g}-{y:g}', source_role=role.upper()+'_ONLY',
                    scene_context=case['scene_context']+'; freestanding plaza maintenance assembly')
        case['condition'].update(condition_id=case['name'], source_site_id=case['source_site_id'],
                                 split_group_id=case['source_site_id'], counterfactual_parent_id=gid)
    result.update(schema='city-coverage64-v1', cases=cases, seed=seed,
        source_role=role.upper()+'_ONLY', purpose='ADDITIVE_COVERAGE64_NO_MODEL_ACCESS',
        scope='New same-Street200V7 regional observations; not independent world; old TRAIN/DEV/plaza untouched',
        suite_contract=dict(frames=64, units=16, relations_per_unit=4, families=4,
                            negative_intervention='Raise fixture; retain assembly and grounded supports'),
        coverage_site=dict(camera_center_xy_m=[x,y], yaw_deg=yaw, station_grid=dict(columns=8, rows=2, dx_m=2. if role=='train' else 1.5, dy_m=1.),
                           floor_z_m=.2, appearance_admission='PENDING_VISUAL_AND_NATIVE_CANARY'))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    template = json.loads(args.template.read_text(encoding='utf-8-sig'))
    args.output.mkdir(parents=True, exist_ok=True)
    both = {role: generate(template, role) for role in SITES}
    for role, spec in both.items():
        spec['provenance'] = dict(generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            template_sha256=hashlib.sha256(args.template.read_bytes()).hexdigest(),
            TRAIN_EVAL_min_camera_distance_m=min(math.hypot(a['camera']['x']-b['camera']['x'], a['camera']['y']-b['camera']['y'])
                for a in both['train']['cases'] for b in both['eval']['cases']))
        with (args.output/f'spec-{role}-v1.json').open('x') as f:
            json.dump(spec,f,indent=2)
    canary = copy.deepcopy(both['train'])
    canary['cases'] = [copy.deepcopy(both[r]['cases'][i]) for r in SITES for i in (0,21,42,63)]
    for role in SITES:
        floor = copy.deepcopy(both[role]['cases'][0])
        floor.update(name=f'floor-{role}', objects=[], floor_check=True, group_id=f'floor-{role}')
        floor['camera']['pitch']=-8.
        canary['cases'].append(floor)
    canary.update(source_role='ENGINEERING_CANARY', purpose='VISUAL_GEOMETRY_ADMISSION_ONLY')
    with (args.output/'spec-canary-v1.json').open('x') as f:
        json.dump(canary,f,indent=2)
