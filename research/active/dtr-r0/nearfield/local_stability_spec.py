"""Fixed composite shapes, size strata and two posed approach schedules."""
from collections import Counter, defaultdict
import copy
from pathlib import Path

import numpy as np

from core_transfer_spec import MAP_SHA, PROFILE
from query_occupancy_data import camera_bounds, geometric_labels, write, sha
from query_occupancy_spec import relative_signature

FRAMES, CLIPS, DT_S = 576, 48, .2
SEED = 202609231
TRAJECTORIES = {
    'approach_return': (4.1,3.5,3.15,2.9,2.55,2.,1.45,1.05,1.8,2.65,3.2,3.9),
    'approach_dwell_return': (4.1,3.5,3.15,2.9,2.55,2.55,2.55,2.55,2.55,2.9,3.2,3.9)}
FAMILIES = (
    ('body_protruding_plane','BODY','L_plate',(.12,.34,.30),(.42,1.04,.84),1.25),
    ('body_suspended_solid','BODY','depth_step',(.22,.34,.28),(.62,.82,.60),1.25),
    ('head_hanging_plane','HEAD','inverted_L_plate',(.09,.34,.22),(.27,.88,.52),1.80),
    ('head_horizontal','HEAD','T_bar',(.09,.48,.06),(.27,1.16,.20),1.78))


def parts(shape, size):
    """Two actual axis-aligned cubes; never fill a composite's bounding-box gaps."""
    d,w,h = size
    if shape == 'L_plate':
        return [((0,.35*w,0),(d,.30*w,h)),((0,0,-.375*h),(d,w,.25*h))]
    if shape == 'inverted_L_plate':
        return [((0,.35*w,0),(d,.30*w,h)),((0,0,.375*h),(d,w,.25*h))]
    if shape == 'depth_step':
        return [((-.175*d,-.22*w,-.15*h),(.65*d,.56*w,.70*h)),
                ((.175*d,.22*w,0),(.65*d,.56*w,h))]
    if shape == 'T_bar':
        return [((0,0,.30*h),(d,w,.40*h)),((0,0,0),(d,.30*w,h))]
    raise ValueError(shape)


def planned_bounds(case):
    objects = [dict(render_bounds_center_m=o['center_m'],
                    render_bounds_extent_m=[s/2 for s in o['size_m']]) for o in case['objects']]
    return camera_bounds(objects,case['camera'])


def specification():
    cases, clips, groups = [], [], []
    for fi,(kind,layer,shape,small,large,height) in enumerate(FAMILIES):
        for si,(level,size) in enumerate((('small',small),('large',large))):
            group = f'local_stability_{kind}_{level}'
            side = -1 if (fi+si)%2 else 1
            world_x, camera_y = 32.+fi*.4+si*.11, (-.035 if fi%2 else .035)
            common = dict(base_group_id=group,type_id=kind,layer=layer,shape=shape,
                          size_level=level,split='evaluation')
            groups.append(dict(**common,size_m=list(size)))
            for trajectory, fronts in TRAJECTORIES.items():
                for relation, overlap in (('INSIDE',.14),('BOUNDARY',.012),('OUTSIDE',-.10)):
                    clip = f'{group}_{trajectory}_{relation.lower()}'
                    lateral = side*(.3+size[1]/2-overlap)
                    objects = []
                    for k,(offset,extent) in enumerate(parts(shape,size)):
                        objects.append(dict(name='target_'+('a','b')[k],kind='cube',
                            center_m=[round(world_x+offset[0],9),round(camera_y+lateral+side*offset[1],9),round(height+offset[2],9)],
                            size_m=[round(v,9) for v in extent],
                            material='/Game/StreetLab/Materials/'+('Sage','Cream','Terracotta','Charcoal')[fi]))
                    objects.append(dict(name='background',kind='cube',center_m=[world_x+4.4,camera_y+.1,2.12],
                        size_m=[.24,6.6,4.4],material='/Game/StreetLab/Materials/'+('Brick','Wood','Plaster','Cream')[fi]))
                    front_offset = min(o['center_m'][0]-o['size_m'][0]/2 for o in objects[:2])
                    meta = dict(**common,trajectory=trajectory,layout_relation=relation,clip_id=clip)
                    clips.append(dict(**meta,frames=12))
                    for i,z in enumerate(fronts):
                        cases.append(dict(**meta,name=f'{clip}_{i:02d}',frame_in_clip=i,time_s=round(i*DT_S,6),
                            target_names=['target_a','target_b'],phase=('dwell' if trajectory=='approach_dwell_return' and 4<=i<=8
                                else 'approach' if i<=7 else 'depart'),nominal_front_m=z,
                            sensor_noise_key=f'{group}_{relation.lower()}_t{i:02d}',
                            camera=dict(x=round(front_offset-z,9),y=camera_y,z=1.82,pitch=0.,yaw=0.,roll=0.),
                            objects=copy.deepcopy(objects)))
    return dict(schema='local-stability-source-v1',seed=SEED,frames=FRAMES,frames_per_clip=12,
        dt_s=DT_S,groups=groups,clips=clips,cases=cases,expected_map_sha256=MAP_SHA,profile=copy.deepcopy(PROFILE),
        map='/Game/StreetLab/WillowSampleV1',sampling='POSED_QUASI_STATIC_NOT_PHYSICAL_CONTINUOUS_MOTION',
        scope='Eight composite geometries, two sizes per family; paired sampled approach/dwell schedules. '
              'Same cube renderer, scene and sensor law; not natural shapes or independent scenes.',
        noise='Common identity across trajectory profiles at corresponding indices, not guaranteed equal returns; '
              'dwell indices have distinct draws. No temporal averaging or repeated-fit selection.')


def check_spec(spec, old_specs=()):
    assert spec == specification(), 'Fixed source changed'
    assert len(spec['cases']) == FRAMES and len(spec['clips']) == CLIPS and len(spec['groups']) == 8
    assert len({c['name'] for c in spec['cases']}) == FRAMES
    grouped = defaultdict(list)
    truth_counts = Counter()
    for c in spec['cases']:
        grouped[c['clip_id']].append(c)
        assert len(c['objects']) == 3 and {o['kind'] for o in c['objects']} == {'cube'}
        assert c['time_s'] == c['frame_in_clip']*.2 or abs(c['time_s']-c['frame_in_clip']*.2)<1e-9
        assert all(c['camera'][a] == 0 for a in ('pitch','yaw','roll'))
        labels, _ = geometric_labels(planned_bounds(c))
        truth = bool((labels[[1,4]]<6).any())
        assert truth == (c['layout_relation'] != 'OUTSIDE' and c['frame_in_clip'] in range(3,10)), c['name']
        truth_counts[c['trajectory']] += truth
        front = min(lo[2] for lo,hi in planned_bounds(c)[:2])
        assert abs(front-c['nominal_front_m']) < 1e-8
    for clip in grouped.values():
        assert len(clip) == 12 and [c['frame_in_clip'] for c in clip] == list(range(12))
        assert all(c['objects'] == clip[0]['objects'] for c in clip)
    assert truth_counts == {t:112 for t in TRAJECTORIES}
    signatures = {relative_signature(c) for c in spec['cases']}
    for old in old_specs:
        assert not signatures & {relative_signature(c) for c in old['cases']}, 'Exact old layout reused'
    return dict(status='PASS',frames=FRAMES,clips=CLIPS,groups=8,positive_frames=224,negative_frames=352,
        positive_events=32,truth_by_trajectory=dict(truth_counts),old_specs_compared=len(old_specs),
        label_geometry='Union of each component cube, not composite enclosing AABB',
        no_old_exact_relative_geometry=True,unique_relative_geometries=len(signatures))


def freeze(plan):
    from local_transfer_spec import specification as old_transfer
    from query_occupancy_spec import specification as old_query
    from data_coverage_spec import specification as old_coverage
    from launch_local_stability import MANDATORY_CODE,MANDATORY_INPUTS,verify_protocol
    repo = Path(__file__).resolve().parents[4]
    assert plan.resolve().is_relative_to((repo/'artifacts.local').resolve()) and not plan.exists()
    spec = specification(); check = check_spec(spec,[old_query(),old_coverage(),old_transfer()])
    write(plan/'spec.json',spec);write(plan/'source-check.json',check)
    write(plan/'protocol.json',dict(schema='local-stability-capture-protocol-v1',frames=576,clips=48,
        capture_timeout_s=900,spec_sha256=sha(plan/'spec.json'),source_check_sha256=sha(plan/'source-check.json'),
        code_hashes={n:sha(Path(__file__).parent/n) for n in sorted(MANDATORY_CODE)},
        input_hashes={n:sha(repo/n) for n in sorted(MANDATORY_INPUTS)}))
    verify_protocol(plan/'protocol.json',plan/'spec.json',repo)
    return check
