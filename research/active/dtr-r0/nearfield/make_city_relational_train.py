"""Fresh matched fixture quartets at three supported TRAIN-only frontages."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import contextual_collection1000 as original
from contextual_geometry import target_contact, intrusion_metrics

SITES=(('TRAIN-north-x-76',-76.,False),('TRAIN-north-x-35',-35.,False),('TRAIN-south-x-55',-55.,True))
SEED=20260909


def generate(template, dev):
    result=copy.deepcopy(template);cases=[]
    original_seed=original.SEED
    try:
        for site_index,(site,x,mirror) in enumerate(SITES):
            original.SEED=SEED+site_index
            fresh=original.collection(template)['cases'][:128]
            for i,c in enumerate(fresh):
                group=f'{site}:seed{SEED+site_index}:quartet_{i//4:03d}'
                c.update(name=f'{group}:{c["variant_id"]}',group_id=group,parent_group_id=group,
                         source_site_id=site,source_role='TRAIN_ONLY')
                for key in ('camera','wearer'):
                    c[key]['x']+=x+76
                    if mirror:
                        c[key]['y']=-c[key]['y'];c[key]['yaw']=-c[key]['yaw'];c[key]['roll']=-c[key]['roll']
                for obj in c['objects']:
                    obj['center_m'][0]+=x+76
                    if mirror:
                        obj['center_m'][1]=-obj['center_m'][1]
                        if 'rotation_deg' in obj:
                            obj['rotation_deg']['roll']=-obj['rotation_deg'].get('roll',0)
                            obj['rotation_deg']['yaw']=-obj['rotation_deg'].get('yaw',0)
                c['condition'].update(condition_id=c['name'],source_site_id=site,split_group_id=site,counterfactual_parent_id=group,
                    lighting_profile=('daylight_080','daylight_100','daylight_120')[(i//4)%3])
                if mirror:
                    c['condition']['lateral_offset_m']=-c['condition']['lateral_offset_m'];c['condition']['tilt_degrees']=-c['condition']['tilt_degrees']
                c['sun_intensity_scale']=(.8,1.,1.2)[(i//4)%3]
                c['skylight_intensity_scale']=(.8,1.,1.2)[(i//4)%3]
                c['geometric_contact']=target_contact(c['objects'],c['wearer']);c['geometric_intrusion']=intrusion_metrics(c['objects'],c['wearer'])
                if c['geometric_contact']['relation']!=c['condition']['desired_relation']:raise ValueError('Geometry changed relation')
                cases.append(c)
    finally:original.SEED=original_seed
    dev_ids={c['group_id'] for c in dev['cases']}|{c['parent_group_id'] for c in dev['cases']}
    if dev_ids & {c['parent_group_id'] for c in cases}:raise ValueError('DEV parent reuse')
    def dist(a,b):return math.hypot(a['camera']['x']-b['camera']['x'],a['camera']['y']-b['camera']['y'])
    distances={site:min(dist(c,d) for c in cases if c['source_site_id']==site for d in dev['cases']) for site,_,_ in SITES}
    inter={f'{a}/{b}':min(dist(c,d) for c in cases if c['source_site_id']==a for d in cases if d['source_site_id']==b) for i,(a,_,_) in enumerate(SITES) for b,_,_ in SITES[i+1:]}
    if min(distances.values())<75 or min(inter.values())<30:raise ValueError('Insufficient source separation')
    signatures={json.dumps(dict(camera=c['camera'],objects=c['objects']),sort_keys=True) for c in cases}
    if len(signatures)!=384:raise ValueError('Duplicate conditions')
    result.update(schema='city-relational-train-384-v1',cases=cases,source_role='TRAIN_ONLY',seed=SEED,
        suite_contract=dict(frames=384,sites=3,quartets_per_site=32,family_quartets_per_site=8,geometry_states_per_site=32),
        provenance=dict(fresh_site_seeds=[SEED+i for i in range(3)],parent_identity_overlap_DEV=False,
            minimum_DEV_camera_distance_by_site_m=distances,minimum_intersite_camera_distance_m=inter,
            source_evidence='Street200V7 north building row centered(-55,30), south-left(-55,-30), each90m; fixtures at corresponding +/-15m frontage',
            roles_frozen_before_model_inference=True,lights='Existing per-case sun/skylight intensity multipliers .8/1/1.2; identical within quartet'),
        scope='Three same-map TRAIN source regions; fresh procedural groups, not independent worlds. Additive384 source; original750 retained by caller.',
        purpose='TRAIN_ONLY_RELATIONAL_NATIVE_SOURCE_NO_MODEL_INFERENCE')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('template','dev','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();raw=a.template.read_bytes();devraw=a.dev.read_bytes()
    result=generate(json.loads(raw.decode('utf-8-sig')),json.loads(devraw.decode('utf-8-sig')))
    result['provenance'].update(template_sha256=hashlib.sha256(raw).hexdigest(),dev_spec_sha256=hashlib.sha256(devraw).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
