"""G9-B independent geometry groups; one current-frame export per factorial cell."""
import argparse
import json
from pathlib import Path
import random
from contact_retina_spec import MAP_SHA, BODY_BOXES
from grounding_spec import VARIANTS

def specification(preflight=False, conversion_mode='legacy'):
    rng=random.Random(100907);clips=[];cases=[];samples=[]
    for i in range(68):
        bx,by=26+rng.uniform(-.4,.4),rng.uniform(-.6,.6)
        distance=rng.uniform(1.7,2.9);material='/Game/StreetLab/Materials/'+('Bronze','Charcoal','Limestone')[i%3]
        bar=dict(name='bar',kind='cube',center_m=[bx+distance,by,rng.uniform(1.63,1.87)],
                 size_m=[.10,rng.uniform(.7,1.5),rng.uniform(.05,.10)],material=material)
        h=rng.uniform(.85,1.18)
        box=dict(name='box',kind='cube',center_m=[bx+distance+.15,by,.12+h/2],
                 size_m=[rng.uniform(.35,.6),rng.uniform(.65,1.2),h],material=material)
        if (i>=64)!=preflight:continue
        group=f'g{1000+i}';split='test' if preflight or i>=52 else 'val' if i>=40 else 'train'
        camera=dict(x=bx,y=by,z=1.82,pitch=-5.,yaw=0.,roll=0.)
        wearer=dict(x=bx,y=by,z=.12,pitch=0.,yaw=0.,roll=0.)
        order=VARIANTS[i%4:]+VARIANTS[:i%4]
        for variant in order:
            objects=[o for o in (bar,box) if variant=='both' or variant==o['name']+'_only']
            cid=group+'_'+variant;idx=len(cases)
            clips.append(dict(clip_id=cid,group_id=group,split=split,variant=variant,objects=objects,preflight=preflight))
            cases.append(dict(name=cid+'_f0',clip_id=cid,frame_in_clip=0,time_s=0.,camera=camera,wearer=wearer,
                              objects=objects,settling_frames=26))
            samples.append(dict(sample_id=cid+'_t0',clip_id=cid,group_id=group,split=split,frame_indices=[idx]))
    return dict(schema='nf-g9b-factorial-spec-v1',map_sha256=MAP_SHA,clips=clips,cases=cases,samples=samples,
                labels={'targets':{}},body_boxes=BODY_BOXES,query_range_m=3.,preflight=preflight,
                conversion_mode=conversion_mode,sampling='ONE_CURRENT_RGB_EXPORT_AFTER_27_RENDER_UPDATES_NOT_VIDEO',
                scope='VISIBLE_TASK_OBJECT_QUERY_ONLY_NOT_ALL_SCENE_FREE_SPACE')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight',action='store_true')
    p.add_argument('--conversion-mode',choices=['legacy','single_access'],default='legacy');a=p.parse_args()
    assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(specification(a.preflight,a.conversion_mode),indent=2),encoding='utf-8')
