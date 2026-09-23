"""Primitive geometry fixtures for CNH unit tests, not benchmark source assets.

UE uses world X forward, Y right, Z up, metres. Geometry used here only checks
source eligibility; final labels must authenticate actual exported render meshes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random


ENVIRONMENTS = ('sidewalk', 'intersection', 'plaza', 'alley')
FAMILIES = ('head', 'rod', 'body', 'wall', 'low')
COUNTS = {'train': (54, 45, 27, 27, 27), 'dev': (12, 10, 6, 6, 6),
          'test': (64, 64, 12, 12, 12)}
CLIPS = ('centre', 'boundary', 'outside', 'removed')
RIG = dict(width=640, height=360, hfov_deg=100., baseline_m=.06,
           rgb_hz=10., tof_hfov_deg=45., tof_vfov_deg=45.,
           camera_coordinates='X_right_Y_down_Z_forward', world_coordinates='X_forward_Y_right_Z_up')


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def camera_coordinates(point, camera):
    p,y,r = (math.radians(camera[k]) for k in ('pitch','yaw','roll'))
    cp,sp,cy,sy,cr,sr = math.cos(p),math.sin(p),math.cos(y),math.sin(y),math.cos(r),math.sin(r)
    f=(cp*cy,cp*sy,sp)
    right=(sr*sp*cy-cr*sy,sr*sp*sy+cr*cy,-sr*cp)
    up=(-cr*sp*cy-sr*sy,-cr*sp*sy+sr*cy,cr*cp)
    delta=[point[i]-camera[k] for i,k in enumerate(('x','y','z'))]
    return [sum(a*b for a,b in zip(delta,right)), -sum(a*b for a,b in zip(delta,up)),
            sum(a*b for a,b in zip(delta,f))]


def pose(layout, clip, t):
    p = layout['trajectory']
    # A floor on travelled distance prevents the rig entering the target.
    travel=min(p['speed_mps']*t, p['start_m']-.5)
    phase=2*math.pi*p['frequency_hz']*t+p['phase_rad']
    return dict(x=-p['start_m']+travel, y=layout['path_y'][clip],
                z=p['height_m']+p['bounce_m']*math.sin(phase),
                pitch=p['pitch_deg'], yaw=p['yaw_amplitude_deg']*math.sin(phase), roll=0.)


def object_row(index, name, centre, size, albedo, rho, kind='cube', family='background', rotation=None):
    return dict(id=index, name=name, **{'class':family}, kind=kind,
                center_m=list(centre), size_m=list(size),
                rotation=rotation or dict(pitch=0.,yaw=0.,roll=0.),
                albedo=albedo, nir_rho=rho)


def make_layout(split, family, environment, index, seed):
    rng=random.Random(seed)
    rejected=[]
    for attempt in range(20):
        v=rng.uniform(.8,1.4)
        trajectory=dict(speed_mps=v,start_m=rng.uniform(3.8,min(4.5,1+3.9*v)),
                        height_m=rng.uniform(1.5,1.75),pitch_deg=rng.uniform(-15,5),
                        yaw_amplitude_deg=rng.uniform(0,5),bounce_m=rng.uniform(.02,.04),
                        frequency_hz=rng.uniform(1.4,2.2),phase_rad=rng.uniform(0,2*math.pi))
        colour=[rng.uniform(.08,.9) for _ in range(3)]
        rho=rng.uniform(.05,.15) if rng.random()<.2 else rng.uniform(.15,.8)
        if family=='head':
            thick=rng.uniform(.02,.30)
            size=[rng.uniform(.02,.30),rng.uniform(.18,.75),thick]
            centre=[0.,0.,rng.uniform(1.4,2.)]
        elif family=='rod':
            d=rng.uniform(.02,.15); size=[d,d,rng.uniform(1.7,2.2)]
            centre=[0.,0.,size[2]/2]
        elif family=='body':
            size=[rng.uniform(.15,.5),rng.uniform(.15,.6),rng.uniform(.35,1.)]
            centre=[0.,0.,rng.uniform(.6,1.1)]
        elif family=='wall':
            size=[rng.uniform(.08,.2),rng.uniform(.8,1.8),rng.uniform(1.8,2.6)]
            centre=[0.,0.,size[2]/2]
        else:
            size=[rng.uniform(.15,.6),rng.uniform(.15,.6),rng.uniform(.05,.5)]
            centre=[0.,0.,size[2]/2]
        target=object_row(1,'target',centre,size,colour,rho,'cylinder' if family=='rod' else 'cube',family)
        side=rng.choice((-1.,1.))
        net_boundary=rng.uniform(-.02,.02) if index%2==0 else rng.uniform(-.12,.12)
        path_y=dict(centre=0.,removed=0.,boundary=side*(.3+size[1]/2+net_boundary),
                    outside=side*(.3+size[1]/2+rng.uniform(.06,.30)))
        row=dict(layout_id=f'{split}_{environment}_{family}_{index:03d}', split=split,
                 family=family,environment=environment,seed=seed,trajectory=trajectory,
                 path_y=path_y,target=target)
        # Points strictly inside the target establish a conservative positive
        # duration; these are not final mesh-contact labels.
        points=[centre]
        if family in ('rod','wall'):
            points += [[0.,0.,z] for z in (.8,1.,1.2,1.4,1.6,1.8)
                       if centre[2]-size[2]/2 < z < centre[2]+size[2]/2]
        positive=[]
        for i in range(40):
            cam=pose(row,'centre',i*.1)
            xyz=[camera_coordinates(q,cam) for q in points]
            positive.append(any(abs(x)<=.3 and -.2<=y<=.9 and .3<=z<=3. for x,y,z in xyz))
        closest=min(camera_coordinates(q,pose(row,'centre',3.9))[2] for q in points)
        if family=='low' or (sum(positive)>=20 and closest<=1.):
            row['source_geometric_check']=dict(attempt=attempt,prior_rejections=rejected,
                conservative_positive_samples=sum(positive),final_target_point_axial_m=closest,
                authority='DECLARED_INTERIOR_POINT_CHECK_NOT_FINAL_MESH_LABELS')
            break
        rejected.append(dict(attempt=attempt,positive_samples=sum(positive),closest_axial_m=closest))
    else:
        raise RuntimeError(f'20 geometric generation attempts exhausted: {split}/{environment}/{family}/{index}')
    # Composed assets vary by layout; Engine primitive classes are explicitly shared.
    tint=[rng.uniform(.12,.8) for _ in range(3)]
    ground=object_row(100,'ground',(-1.,0.,-.10),(30.,20.,.2),tint,rng.uniform(.15,.7))
    backgrounds=[ground]
    back_distance=rng.choice((1.,2.,4.,6.,10.,15.,None))
    if back_distance is not None:
        backgrounds.append(object_row(101,'rear_wall',(back_distance,0.,1.8),(.2,12.,3.6),
                                      [rng.uniform(.1,.8) for _ in range(3)],rng.uniform(.15,.8)))
    if environment=='alley':
        width=rng.uniform(1.4,2.4)
        for sign,ident in ((-1,102),(1,103)):
            backgrounds.append(object_row(ident,'side_wall',(-1.,sign*width,1.6),(24.,.15,3.2),
                                          [rng.uniform(.12,.8) for _ in range(3)],rng.uniform(.15,.8)))
    elif environment=='sidewalk':
        backgrounds.append(object_row(105,'kerb',(-1.,-2.,.15),(24.,.2,.3),tint,.3))
    elif environment=='intersection':
        # Representative open junction fixture only, not a realistic street source.
        for sign,ident in ((-1,106),(1,107)):
            backgrounds.append(object_row(ident,'junction_kerb',(-3.,sign*3.,.15),
                                          (.2,2.,.3),tint,.3))
    row['backgrounds']=backgrounds
    row['illumination']=dict(mode=rng.choice(('daylight','dusk')),intensity=rng.uniform(.7,1.3))
    row['asset_bundle_id']=digest(dict(target=target,backgrounds=backgrounds,illumination=row['illumination']))
    row['shared_source_primitives']=['/Engine/BasicShapes/Cube','/Engine/BasicShapes/Cylinder']
    return row


def layouts(seed=20260924):
    result=[]
    # Rotate environment allocation within each family while exactly preserving
    # each split's environment quotas and family counts.
    ordinal=0
    for split,counts in COUNTS.items():
        environment_cursor=0
        for family,count in zip(FAMILIES,counts):
            for i in range(count):
                environment=ENVIRONMENTS[environment_cursor%4]; environment_cursor+=1
                result.append(make_layout(split,family,environment,i,seed+ordinal*1009))
                ordinal+=1
    assert len(result)==384
    assert len({r['asset_bundle_id'] for r in result})==384
    return result


def frames(selected):
    result=[]
    for layout in selected:
        for kind in CLIPS:
            clip_id=layout['layout_id']+'_'+kind
            objects=layout['backgrounds']+([] if kind=='removed' else [layout['target']])
            for i in range(40):
                result.append(dict(id=clip_id+f'_{i:02d}',layout_id=layout['layout_id'],
                    split=layout['split'],family=layout['family'],environment=layout['environment'],
                    clip_id=clip_id,clip_kind=kind,time_s=round(i*.1,6),frame_in_clip=i,
                    camera=pose(layout,kind,i*.1),objects=objects,illumination=layout['illumination']))
    return result


def pilot_layouts(all_layouts):
    # 20 train layouts, four environments by five; distribution 6/5/3/3/3.
    target=dict(zip(FAMILIES,(6,5,3,3,3)))
    env_remaining={k:5 for k in ENVIRONMENTS}
    selected=[]
    for family in FAMILIES:
        pool=[x for x in all_layouts if x['split']=='train' and x['family']==family]
        for _ in range(target[family]):
            possible=[x for x in pool if x not in selected and env_remaining[x['environment']]>0]
            chosen=max(possible,key=lambda x:(env_remaining[x['environment']],-x['seed']))
            selected.append(chosen);env_remaining[chosen['environment']]-=1
    assert len(selected)==20 and not any(env_remaining.values())
    return selected


def materialize(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    manifest=output/'source-manifest.json'
    if manifest.exists():
        raise FileExistsError('Do not overwrite frozen source; inspect its identity first')
    all_layouts=layouts(); pilot=pilot_layouts(all_layouts)
    write(output/'layouts.json',dict(schema='cnh-route-layouts-v1',rig=RIG,layouts=all_layouts))
    pilot_frames=frames(pilot)
    common=dict(schema='cnh-route-capture-v1',rig=RIG,source_role='SENSOR_UNIT_TEST_ONLY',
                scene_layer='PRIMITIVE_UNIT_TEST', benchmark_eligible=False,
                shared_primitives_disclosure='Procedural composed assets are disjoint; Engine primitive mesh classes are shared')
    write(output/'pilot-spec.json',dict(common,frames=pilot_frames))
    # Canary is an engineering subset, not extra scientific cases or test access.
    canary=[f for f in pilot_frames if f['layout_id'] in {pilot[0]['layout_id'],pilot[-1]['layout_id']}
            and f['clip_kind']=='centre' and f['frame_in_clip']==20]
    write(output/'canary-spec.json',dict(common,frames=canary))
    write(manifest,dict(schema='cnh-route-source-manifest-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=sha(__file__),plan='CNH_ROUTE_COMPARISON_PLAN_20260924.md',
        plan_sha256=sha(Path(__file__).with_name('CNH_ROUTE_COMPARISON_PLAN_20260924.md')),
        source_role='SENSOR_UNIT_TEST_ONLY',benchmark_eligible=False,
        expected_layouts=384,expected_time_samples=61440,pilot_layout_ids=[x['layout_id'] for x in pilot],
        split_layout_counts=dict(Counter(x['split'] for x in all_layouts)),
        split_environment_counts=dict(Counter(x['split']+'/'+x['environment'] for x in all_layouts)),
        split_family_counts=dict(Counter(x['split']+'/'+x['family'] for x in all_layouts)),
        no_model_or_sensor_outcomes_used=True,
        files={p.name:sha(p) for p in (output/'layouts.json',output/'pilot-spec.json',output/'canary-spec.json')}))
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True,type=Path)
    print(materialize(p.parse_args().output))
