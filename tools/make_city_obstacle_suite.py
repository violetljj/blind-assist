"""Make a fixed City obstacle acquisition spec from a verified saved map."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = REPO / 'artifacts.local/nearfield/city-pcg-20260908/street200-spec-distance.json'
KEY_VIEWS = ('west_sidewalk', 'intersection', 'building_entrance', 'greenway_link', 'plaza', 'east_return')
ASSETS = {
    'railing': '/Game/Prop/Kit_Railing_B/Mesh/SM_Railing_B_railing_N01',
    'sign_pole': '/Game/Prop/Kit_PoleSign_A/Mesh/SM_PoleSign_A',
    'trashcan': '/Game/Prop/Kit_Trashcan_A/Mesh/SM_Trashcan_A_01',
}
COMPLEX_ASSETS = {
    'bicycle': '/Game/Prop/Kit_Bicycle_A/Mesh/SM_Bicycle_A_01',
    'scaffold_frame': '/Game/Prop/Kit_Scaffolding_RR/Mesh/SM_Scaffolding_metal_N1',
    'barricade': '/Game/Prop/Kit_Barricade_A/Mesh/SM_Barricade_A',
    'stone_table': '/Game/Prop/Kit_Umbella_StoneTable_A/Mesh/SM_StoneTable_Square_A',
    'bench': '/Game/Prop/Kit_bench_RR/Mesh/SM_park_bench_N01',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def under_artifacts(path):
    path = Path(path).resolve()
    root = (REPO / 'artifacts.local').resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Inputs and outputs must be strictly under artifacts.local')
    return path


def suite(template):
    """Six overview views and eighteen grouped observations at one west pose."""
    by_name = {case['name']: case for case in template['cases']}
    views = []
    for name in KEY_VIEWS:
        case = copy.deepcopy(by_name[name])
        if case.get('objects'):
            raise ValueError('Key view must not contain injected objects: ' + name)
        case.update(group_id='key_views', group_type='SCENE_VIEWS', variant_id=name)
        views.append(case)
    west = by_name['west_sidewalk']
    camera, floor = copy.deepcopy(west['camera']), float(west['floor_z_m'])
    if float(camera['yaw']) != 0 or float(camera['roll']) != 0:
        raise ValueError('Fixed west-camera suite requires zero yaw and roll')
    x, y = float(camera['x']), float(camera['y'])
    baseline = len(views)

    def add(group, kind, variant, objects, **extra):
        case = dict(name=f'{group}__{variant}', camera=copy.deepcopy(camera), floor_z_m=floor,
                    floor_check=not objects, objects=objects, group_id=group, group_type=kind,
                    variant_id=variant, baseline_index=baseline)
        case.update(extra)
        views.append(case)

    def cube(name, forward, side, height, size):
        return dict(name=name, center_m=[x+forward, y+side, floor+height], size_m=size)

    add('shared_clear', 'EMPTY_BASELINE', 'clear', [])
    for side, variant in ((0., 'center'), (1., 'lateral')):
        add('body_box', 'SHAPE_PLACEMENT_PAIR', variant,
            [cube('body_box', 2., side, 1.05, [.35,.7,.3])], analytic_control=True)
        add('head_bar', 'SHAPE_PLACEMENT_PAIR', variant,
            [cube('head_bar', 2., side, 1.64, [.12,.9,.12])], analytic_control=True)
        add('composite_u', 'SHAPE_PLACEMENT_PAIR', variant, [
            cube('left_leg', 2., side-.3, .75, [.05,.05,1.5]),
            cube('right_leg', 2., side+.3, .75, [.05,.05,1.5]),
            cube('top_bar', 2., side, 1.5, [.05,.7,.05])])
        for name, asset in ASSETS.items():
            # The railing pivot is at an end; its verified local Y bounds are [-4.501, 0] m.
            # Center the mesh for the first variant and move its full span outside the corridor for the second.
            asset_side=(2.25056+(3.2 if side else 0.)) if name=='railing' else side
            add('real_'+name, 'REAL_ASSET_PLACEMENT_PAIR', variant, [dict(name=name,
                mesh_asset=asset, center_m=[x+2., y+asset_side, floor], scale=1.,
                rotation_deg=dict(pitch=0., yaw=0., roll=0.))])
    for gap, state in ((.8,'DANGER'), (3.,'WARNING'), (7.,'OBSERVE')):
        # Body front extent .18 m plus half wall thickness .10 m gives the requested gap.
        add('same_height_wall', 'DISTANCE_LADDER', f'gap_{gap:g}m',
            [cube('wall', gap+.28, 0., 1.5, [.2,2.,3.])],
            expected_distance_states=[state,state], expected_geometry='CONTROLLED_CUBE_DISTANCE_ONLY',
            nominal_body_front_gap_m=gap)
    for side, variant in ((-2., 'left'), (2., 'right')):
        add('side_wall', 'LATERAL_WALL_CONTROLS', variant,
            [cube('side_wall', 1., side, 1.5, [3.,.15,3.])],
            expected_distance_states=['NO_VISIBLE_SUPPORT','NO_VISIBLE_SUPPORT'],
            expected_geometry='CONTROLLED_CUBE_DISTANCE_ONLY')
    assert len(views) == 24 and all(case['camera'] == camera for case in views[6:])
    result = copy.deepcopy(template)
    result.update(schema='city-obstacle-suite-v1', seed=42, cases=views,
        suite_contract=dict(key_views=6, grouped_west_frames=18,
            sampling='STATIC_SETTLED_POSES_NOT_MOTION_OR_TRAINING',
            object_semantics='Real mesh center_m is actor origin, scale is dimensionless; no predicted real-asset labels',
            model_contract='RGB/calibration only; group names, object geometry and expectations remain evaluator-side spec metadata'))
    return result


def complex_suite(template):
    """Two existing scene poses, real-shape pairs and table/seating combinations."""
    by_name = {case['name']: case for case in template['cases']}
    cases = []
    for scene in ('west_sidewalk', 'plaza'):
        base = copy.deepcopy(by_name[scene])
        camera, floor = base['camera'], base['floor_z_m']
        if camera['yaw'] != 0 or camera['roll'] != 0:
            raise ValueError('Bounds-aligned suite requires forward world X')
        baseline = len(cases)
        base.update(group_id=scene+'_clear', variant_id='clear', group_type='EMPTY_BASELINE')
        cases.append(base)

        def prop(name, lateral=False, forward=2., side=None):
            return dict(name=name, mesh_asset=COMPLEX_ASSETS[name], scale=1.,
                center_m=[camera['x']+forward, camera['y']+(.8 if lateral else 0.) if side is None else camera['y']+side, floor],
                placement='bounds_front_right_floor' if lateral else 'bounds_front_center_floor',
                rotation_deg=dict(pitch=0., yaw=90. if name=='bicycle' else 0., roll=0.))

        def add(name, variant, objects):
            case = copy.deepcopy(base)
            case.update(name=scene+'__'+name+'__'+variant, objects=objects, floor_check=False,
                group_id=scene+'__'+name, group_type='COMPLEX_REAL_PLACEMENT_PAIR',
                variant_id=variant, baseline_index=baseline)
            cases.append(case)

        for name in COMPLEX_ASSETS:
            for lateral, variant in ((False,'center'), (True,'right_clearance')):
                add(name, variant, [prop(name,lateral)])
        # Two separate real objects, with a controllable aisle between them.
        add('table_and_seating', 'center_table', [prop('stone_table'), prop('bench',True,forward=3.)])
        add('table_and_seating', 'right_clearance', [prop('stone_table',True), prop('bench',True,forward=5.)])
    assert len(cases) == 26
    result = copy.deepcopy(template)
    result.update(schema='city-complex-obstacle-suite-v1', seed=42, cases=cases,
        suite_contract=dict(scene_poses=2, frames=26, assets=list(COMPLEX_ASSETS),
            placement='World bounds anchor placement only; native visible surfaces determine labels',
            sampling='STATIC_SETTLED_POSES_NOT_MOTION_OR_TRAINING',
            model_contract='RGB/calibration only; grouping and geometry remain evaluator-side'))
    return result


def make(build, output, template=DEFAULT_TEMPLATE, complex_structures=False):
    build, output, template = map(under_artifacts, (build, output, template))
    if output.exists():
        raise FileExistsError('Refuse overwrite suite output')
    completion_path = build / 'completion.json' if build.is_dir() else build
    completion = read(completion_path)
    if completion.get('status') != 'PASS' or completion.get('source_unchanged') is not True:
        raise ValueError('Build must pass and preserve sources')
    map_file = under_artifacts(completion['map_file'])
    if map_file.suffix.lower() != '.umap' or sha(map_file) != completion['map_sha256']:
        raise ValueError('Saved map differs from build completion')
    parts = map_file.parts
    content_index = next((i for i, part in enumerate(parts) if part.lower() == 'content'), None)
    if content_index is None:
        raise ValueError('Map must be in project Content')
    content = Path(*parts[:content_index+1])
    for asset in (COMPLEX_ASSETS if complex_structures else ASSETS).values():
        if not (content / (asset[len('/Game/'):] + '.uasset')).is_file():
            raise FileNotFoundError('Required existing prop missing: ' + asset)
    result = (complex_suite if complex_structures else suite)(read(template))
    result.update(map_file=str(map_file), map_asset='/Game/'+map_file.relative_to(content).with_suffix('').as_posix(),
                  map_sha256=sha(map_file), provenance=dict(build_completion=str(completion_path),
                  build_completion_sha256=sha(completion_path), template=str(template), template_sha256=sha(template),
                  generator_sha256=sha(Path(__file__))))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return dict(status='PASS', output=str(output), frames=len(result['cases']), spec_sha256=sha(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--template', type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument('--complex-structures', action='store_true')
    args = parser.parse_args()
    print(json.dumps(make(args.build, args.output, args.template, args.complex_structures)))
