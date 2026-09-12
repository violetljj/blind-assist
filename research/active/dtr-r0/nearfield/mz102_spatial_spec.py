"""Fresh paired posed scene replication; no measurement/outcome-driven placement."""
import argparse
import json
import math
from pathlib import Path
import random

RIG=dict(width=640,height=360,hfov_deg=70.,baseline_m=.10,tof_hfov_deg=45.)
CASES=('head_bar','small_head','overhead_clear','thin_left','thin_right',
       'lateral_pole','wall','open_corridor','cross_hit','cross_miss','receding','occluded_thin')


def obj(name,center,size,seed):
    return dict(name=name,kind='cube',center_m=list(center),size_m=list(size),texture_seed=seed)


def specification(seed=102013):
    rng=random.Random(seed);frames=[];parameters={}
    for case in CASES:
        parameters[case]=dict(offset=rng.uniform(-.035,.035),distance=rng.uniform(3.60,3.88),
            speed=rng.uniform(.64,.82),phase=rng.uniform(-.09,.09),
            thickness=rng.uniform(.85,1.15),texture_seed=rng.randrange(100000,999999),
            head_height=rng.uniform(1.60,1.72))
    for case in CASES:
        p=parameters[case];d=p['offset'];s=p['thickness'];seed_obj=p['texture_seed']
        for appearance in ('textured','flat'):
            episode=f'{case}_{appearance}'
            for j in range(12):
                t=j*.25;phase_t=t+p['phase'];wearer_x=p['speed']*phase_t
                objects=[];yaw=0.;pitch=-2.5+.5*math.sin(phase_t*1.9)
                if case=='head_bar':
                    objects=[obj('head_bar',[p['distance'],d,p['head_height']],[.10*s,.80*s,.10*s],seed_obj)]
                    yaw=10.*math.sin(phase_t*2.3);pitch+=3.*math.sin(phase_t*1.6)
                elif case=='small_head':
                    objects=[obj('small_head',[p['distance'],d,p['head_height']],[.12,.14,.10],seed_obj)]
                elif case=='overhead_clear':
                    objects=[obj('overhead',[p['distance'],d,2.13],[.11*s,.78*s,.10*s],seed_obj)]
                elif case in ('thin_left','thin_right','lateral_pole','occluded_thin'):
                    lateral={'thin_left':-.11,'thin_right':.11,'lateral_pole':.57,'occluded_thin':.11}[case]+d
                    objects=[obj('pole',[p['distance'],lateral,1.26],[.043*s,.043*s,1.30],seed_obj)]
                    if case=='occluded_thin':
                        # A real foreground short obstacle masks part of the rod.
                        # Its own BODY occupancy is retained in evaluator geometry.
                        objects.append(obj('foreground_occluder',[p['distance']-.95,lateral,.95],[.12,.32,.55],seed_obj+1))
                elif case=='wall':
                    objects=[obj('wall',[p['distance'],d,1.52],[.13*s,2.5,3.04],seed_obj)]
                elif case=='open_corridor':
                    gap=1.35+d
                    objects=[obj('left_wall',[3.1,-gap,1.5],[7.2,.14,3.],seed_obj),
                             obj('right_wall',[3.1,gap,1.5],[7.2,.14,3.],seed_obj+1)]
                elif case in ('cross_hit','cross_miss'):
                    wearer_x=0.
                    lateral=(-1.55+1.12*phase_t if case=='cross_hit' else .72+.14*phase_t)+d
                    objects=[obj('crossing',[2.32+d,lateral,1.26],[.17*s,.19*s,1.18],seed_obj)]
                elif case=='receding':
                    wearer_x=0.
                    objects=[obj('receding',[1.02+(1.10+p['speed']*.12)*phase_t,d,1.27],[.19*s,.32*s,1.16],seed_obj)]
                camera=dict(x=wearer_x,y=0.,z=1.7,yaw=yaw,pitch=pitch,roll=0.)
                frames.append(dict(id=f'{episode}_{j:02d}',episode=episode,family=case,appearance=appearance,
                    time_s=t,camera=camera,body_origin_m=[wearer_x,0.,0.],objects=objects))
    return dict(schema='mz102-fresh-paired-spatial-scene-v1',seed=seed,rig=RIG,frames=frames,
        background=dict(name='background',center_m=[12.,0.,1.6],size_m=[.12,18.,9.],
                        texture_grid=[29,15],texture_seed=1029001),
        floor=dict(center_m=[4.,0.,-.05],size_m=[24.,20.,.1]),
        sampling='POSED_4_HZ_SEQUENCE_NOT_WALLCLOCK_SENSOR',
        label_scope='CURRENT_FORWARD_BODY_HEAD_CORRIDOR_TASK_ACTOR_AABB_NOT_FUTURE_COLLISION',
        controlled_pose='KNOWN_CAMERA_BODY_POSE_NOT_ESTIMATED_IMU',
        appearance_pairs='SAME_GEOMETRY_NOT_INDEPENDENT_EPISODE_FAMILIES',
        replication_scope='FRESH_SEED_CONTROLLED_SCENE_REPLICATION_NOT_NEW_FAMILY_GENERALIZATION',
        occlusion_scope='FOREGROUND_OCCLUDER_IS_REAL_TASK_GEOMETRY_INCLUDED_IN_GT',
        source_parameters=parameters)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[4]
    assert a.output.resolve().is_relative_to((root/'artifacts.local').resolve())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(specification(),f,indent=2,allow_nan=False)
