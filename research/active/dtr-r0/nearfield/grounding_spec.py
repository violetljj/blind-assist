"""Fresh static bar/box factorial interventions; no G8 source reuse or fitting."""
import argparse
import json
from pathlib import Path
import random
from contact_retina_spec import MAP_SHA, BODY_BOXES

VARIANTS = ('both', 'bar_only', 'box_only', 'neither')


def specification(preflight=False, settling=8):
    rng = random.Random(90907)
    clips, cases, samples = [], [], []
    for group in range(16):
        bx, by = 26+rng.uniform(-.25,.25), rng.uniform(-.6,.6)
        distance = rng.uniform(1.9,2.7)
        material = '/Game/StreetLab/Materials/'+('Bronze','Charcoal','Limestone')[group%3]
        bar = dict(name='bar',kind='cube',center_m=[bx+distance,by, rng.uniform(1.65,1.84)],
                   size_m=[.10,rng.uniform(.7,1.5),rng.uniform(.05,.10)],material=material)
        height = rng.uniform(.85,1.15)
        box = dict(name='box',kind='cube',center_m=[bx+distance+.15,by,.12+height/2],
                   size_m=[.45,rng.uniform(.7,1.1),height],material=material)
        if group not in (range(2) if preflight else range(4,16)):
            continue
        for setting in ((8,2) if preflight else (settling,)):
            order=VARIANTS if preflight else VARIANTS[group%4:]+VARIANTS[:group%4]
            for variant in order:
                objects = [o for o in (bar,box) if variant=='both' or variant==o['name']+'_only']
                gid=f'g{group:03d}'+(f'_settle{setting}' if preflight else '')
                cid=gid+'_'+variant
                camera=dict(x=bx,y=by,z=1.82,pitch=-5.,yaw=0.,roll=0.)
                wearer=dict(x=bx,y=by,z=.12,pitch=0.,yaw=0.,roll=0.)
                clips.append(dict(clip_id=cid,group_id=gid,split='test',variant=variant,objects=objects,
                                  preflight=preflight))
                indices=[]
                for j in range(3):
                    indices.append(len(cases))
                    cases.append(dict(name=cid+f'_f{j}',clip_id=cid,frame_in_clip=j,time_s=j*.2,
                                      camera=camera,wearer=wearer,objects=objects,settling_frames=setting))
                samples.append(dict(sample_id=cid+'_t2',clip_id=cid,group_id=gid,split='test',frame_indices=indices))
    return dict(schema='nf-g9-grounding-spec-v1',map_sha256=MAP_SHA,clips=clips,cases=cases,samples=samples,
                labels={'targets':{}},body_boxes=BODY_BOXES,query_range_m=3.,preflight=preflight,
                sampling='THREE_STATIC_SETTLED_POSES_SIMULATED_5HZ_NOT_MOTION_TEST',
                scope='VISIBLE_TASK_OBJECT_QUERY_ONLY_NOT_ALL_SCENE_FREE_SPACE')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--preflight',action='store_true');p.add_argument('--settling',type=int,default=8)
    a=p.parse_args();assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(specification(a.preflight,a.settling),indent=2),encoding='utf-8')
