"""G10 fresh complete quartets with bounded view and geometry diversity."""
import argparse
import json
from pathlib import Path
import random
from contact_retina_spec import MAP_SHA, BODY_BOXES
from grounding_spec import VARIANTS


def specification(preflight=False):
    rng=random.Random(10080907);clips=[];cases=[];samples=[]
    count=8 if preflight else 216
    for i in range(count):
        split='test' if preflight or i>=184 else 'val' if i>=160 else 'train'
        stratum='diverse' if preflight or split=='train' else 'narrow' if (i-160 if split=='val' else i-184)%2==0 else 'diverse'
        if stratum=='narrow':
            bx,by=26+rng.uniform(-.4,.4),rng.uniform(-.6,.6)
            pitch=-5.;yaw=0.;offset=0.;barwidth=rng.uniform(.7,1.5);boxwidth=rng.uniform(.65,1.2)
        else:
            anchors=(22.,26.,30.) if split=='train' else (23.,27.,31.) if split=='val' else (24.,28.,32.)
            if preflight: anchors=(22.,23.,24.,26.,27.,28.,30.,32.)
            bx=anchors[i%len(anchors)]+rng.uniform(-.2,.2);by=rng.uniform(-1.1,1.1)
            pitch=rng.uniform(-12.,3.);yaw=rng.uniform(-7.,7.);offset=rng.uniform(-.18,.18)
            barwidth=rng.uniform(.6,1.65);boxwidth=rng.uniform(.55,1.3)
        distance=rng.uniform(1.7,2.85)
        material='/Game/StreetLab/Materials/'+('Bronze','Charcoal','Limestone')[i%3]
        bar=dict(name='bar',kind='cube',center_m=[bx+distance,by+offset,rng.uniform(1.63,1.87)],
                 size_m=[.10,barwidth,rng.uniform(.05,.10)],material=material)
        height=rng.uniform(.85,1.18)
        box=dict(name='box',kind='cube',center_m=[bx+distance+.15,by-offset,.12+height/2],
                 size_m=[rng.uniform(.35,.6),boxwidth,height],material=material)
        group=f'g{4000+i if preflight else 2000+i}'
        camera=dict(x=bx,y=by,z=1.82,pitch=pitch,yaw=yaw,roll=0.)
        wearer=dict(x=bx,y=by,z=.12,pitch=0.,yaw=0.,roll=0.)
        for variant in VARIANTS[i%4:]+VARIANTS[:i%4]:
            objects=[o for o in (bar,box) if variant=='both' or variant==o['name']+'_only']
            cid=group+'_'+variant;idx=len(cases)
            clips.append(dict(clip_id=cid,group_id=group,split=split,variant=variant,stratum=stratum,objects=objects,preflight=preflight))
            cases.append(dict(name=cid+'_f0',clip_id=cid,frame_in_clip=0,time_s=0.,camera=camera,wearer=wearer,objects=objects,settling_frames=26))
            samples.append(dict(sample_id=cid+'_t0',clip_id=cid,group_id=group,split=split,frame_indices=[idx]))
    return dict(schema='nf-g10-diversity-spec-v1',map_sha256=MAP_SHA,clips=clips,cases=cases,samples=samples,
                labels={'targets':{}},body_boxes=BODY_BOXES,query_range_m=3.,preflight=preflight,conversion_mode='single_access',
                sampling='ONE_CURRENT_RGB_EXPORT_AFTER_27_RENDER_UPDATES_NOT_VIDEO',
                scope='VISIBLE_TASK_OBJECT_QUERY_ONLY_NOT_ALL_SCENE_FREE_SPACE')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(specification(a.preflight),indent=2),encoding='utf-8')
