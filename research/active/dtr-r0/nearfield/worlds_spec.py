"""G14-A two procedural layout families; same-shape height counterfactuals."""
import argparse
import json
from pathlib import Path
import random
from contact_retina_spec import MAP_SHA, BODY_BOXES


def specification():
    rng=random.Random(14080908); cases=[]; clips=[]; samples=[]
    for world_id,y in [('sidewalk',0.),('corridor',100.)]:
        for layout in (1401,1402):
            for g in range(4):
                family='bar' if g<2 else 'box'
                group=f'w{world_id}_l{layout}_g{g}'
                distance=rng.uniform(1.85,2.55); offset=rng.uniform(-.08,.08)
                size=[.12,rng.uniform(.75,1.1),.09] if family=='bar' else [.38,rng.uniform(.42,.58),.30]
                material='/Game/StreetLab/Materials/'+('Bronze' if g%2 else 'Charcoal')
                # Coordinates are above wearer floor .12, not object categories.
                relations=[('BODY',1.05),('HEAD',1.64),('NEITHER',2.35)]
                for relation,z in relations[g%3:]+relations[:g%3]:
                    cid=group+'_'+relation;idx=len(cases)
                    objects=[dict(name='target',kind='cube',center_m=[1000.+distance,y+offset,.12+z],size_m=size,material=material)]
                    clip=dict(clip_id=cid,group_id=group,split='test',world_id=world_id,layout_id=layout,asset_family=family,relation=relation,objects=objects)
                    clips.append(clip)
                    cases.append(dict(clip,name=cid+'_f0',frame_in_clip=0,time_s=0.,camera=dict(x=1000.,y=y,z=1.82,pitch=-5.,yaw=0.,roll=0.),wearer=dict(x=1000.,y=y,z=.12,pitch=0.,yaw=0.,roll=0.),settling_frames=26))
                    samples.append(dict(sample_id=cid+'_t0',clip_id=cid,group_id=group,split='test',frame_indices=[idx]))
    return dict(schema='nf-g14-worlds-spec-v1',map_sha256=MAP_SHA,body_boxes=BODY_BOXES,query_range_m=3.,clips=clips,cases=cases,samples=samples,labels={'targets':{}},sampling='ONE_CURRENT_RGB_AFTER_SETTLING_NOT_VIDEO',scope='ALL_VISIBLE_SCENE_SURFACES_IN_BODY_QUERY_NOT_HIDDEN_COLLISION',world_scope='TWO_PROCEDURAL_LAYOUT_FAMILIES_IN_FROZEN_WILLOW_CONTAINER_NOT_TWO_INDEPENDENT_MAP_ASSETS',seed=14080908)


def scene_preview():
    """Two target-free views for look development, never a model benchmark."""
    spec=specification()
    selected=[next(c for c in spec['cases'] if c['world_id']==world and c['relation']=='NEITHER')
              for world in ('sidewalk','corridor')]
    ids={c['clip_id'] for c in selected}
    spec['cases']=selected
    spec['clips']=[c for c in spec['clips'] if c['clip_id'] in ids]
    spec['samples']=[s for s in spec['samples'] if s['clip_id'] in ids]
    for index,sample in enumerate(spec['samples']):
        sample['frame_indices']=[index]
    for case in spec['cases']+spec['clips']:
        case['objects']=[]
    spec['appearance_preview']=True
    spec['purpose']='SCENE_LOOK_DEVELOPMENT_NO_CONTROLLED_TARGETS_NO_MODEL_SCORING'
    return spec


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scene-preview',action='store_true');a=p.parse_args()
    assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(scene_preview() if a.scene_preview else specification(),indent=2)+'\n',encoding='utf-8')
