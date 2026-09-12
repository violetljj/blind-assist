"""Fresh bounded, posed UE scene design; actor geometry is evaluator-only."""
import argparse
import json
from pathlib import Path
import numpy as np
from mz101_spatial import RIG

CASES=('head_bar','overhead_clear','thin_left','thin_right','lateral_pole','wall',
       'open_corridor','cross_hit','cross_miss','receding','occluded_head','head_turn')


def obj(name, center, size):
    return dict(name=name,kind='cube',center_m=list(center),size_m=list(size))


def specification():
    rng=np.random.default_rng(101013)
    frames=[]
    # Sub-zone phase is fixed independently of observed ToF hits. Pair textures
    # uses identical poses/actors, not an extra independent scenario replicate.
    offsets={case:float(rng.uniform(-.025,.025)) for case in CASES}
    for case in CASES:
        for appearance in ('textured','flat'):
            episode=f'{case}_{appearance}'
            for j in range(12):
                t=j*.25;wearer_x=.70*t;d=offsets[case]
                objects=[];yaw=0.;pitch=-3.
                if case in ('head_bar','occluded_head','head_turn'):
                    objects=[obj('head_bar',[3.7,d,1.68],[.10,.85,.10])]
                    if case=='occluded_head':
                        objects.append(obj('lateral_occluder',[2.6,.52,1.6],[.12,.60,.8]))
                    if case=='head_turn':
                        yaw=12*np.sin(t*2.1);pitch=-3+4*np.sin(t*1.7)
                elif case=='overhead_clear':
                    objects=[obj('overhead',[3.7,d,2.10],[.10,.85,.10])]
                elif case in ('thin_left','thin_right','lateral_pole'):
                    y={'thin_left':-.10,'thin_right':.10,'lateral_pole':.53}[case]+d
                    objects=[obj('pole',[3.7,y,1.25],[.045,.045,1.25])]
                elif case=='wall':
                    objects=[obj('wall',[3.7,0,1.5],[.12,2.4,3.])]
                elif case=='open_corridor':
                    objects=[obj('left_wall',[3.,-1.3,1.5],[7.,.15,3.]),obj('right_wall',[3.,1.3,1.5],[7.,.15,3.])]
                elif case in ('cross_hit','cross_miss'):
                    wearer_x=0.
                    y=-1.4+t if case=='cross_hit' else .65+.10*t
                    objects=[obj('crossing',[2.3,y,1.25],[.18,.18,1.2])]
                elif case=='receding':
                    wearer_x=0.
                    objects=[obj('receding',[1.1+1.1*t,d,1.25],[.18,.35,1.2])]
                camera=dict(x=wearer_x,y=0.,z=1.7,yaw=float(yaw),pitch=float(pitch),roll=0.)
                frames.append(dict(id=f'{episode}_{j:02d}',episode=episode,family=case,appearance=appearance,
                    time_s=t,camera=camera,body_origin_m=[wearer_x,0.,0.],objects=objects))
    return dict(schema='mz101-spatial-scene-v1',seed=101013,rig=RIG,frames=frames,
        sampling='POSED_4_HZ_SEQUENCE_NOT_WALLCLOCK_SENSOR',
        label_scope='CURRENT_FORWARD_BODY_HEAD_CORRIDOR_TASK_ACTOR_AABB_NOT_FUTURE_COLLISION',
        controlled_pose='KNOWN_CAMERA_BODY_POSE_NOT_ESTIMATED_IMU',
        appearance_pairs='SAME_GEOMETRY_NOT_INDEPENDENT_EPISODE_FAMILIES')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[4]
    assert a.output.resolve().is_relative_to((root/'artifacts.local').resolve())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf-8') as f:json.dump(specification(),f,indent=2,allow_nan=False)
