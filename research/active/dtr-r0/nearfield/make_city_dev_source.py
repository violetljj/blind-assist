"""Freeze 32 translated/mirrored fixture quartets as distant-region DEV only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from collections import Counter

from contextual_geometry import target_contact, intrusion_metrics


def generate(parent, parent_sha):
    cases=copy.deepcopy(parent['cases'][:128])
    if len(cases)!=128 or len({c['group_id'] for c in cases})!=32:
        raise ValueError('Earliest 32 complete quartets required')
    site='DEV-frontage-x54-y-15'
    for c in cases:
        original_group=c['group_id'];original_name=c['name']
        c.update(name='dev_'+original_name,group_id=site+':'+original_group.split(':')[-1],source_site_id=site,
                 source_role='DEV_ONLY',parent_group_id=original_group,parent_case_name=original_name)
        for key in ('camera','wearer'):
            if c[key]['yaw']!=0 or c[key].get('roll',0)!=0:raise ValueError('Only original straight +X source supported')
            c[key]['x']+=130.;c[key]['y']=-c[key]['y']
        for obj in c['objects']:
            obj['center_m'][0]+=130.;obj['center_m'][1]=-obj['center_m'][1]
            if 'rotation_deg' in obj:
                if obj['rotation_deg'].get('pitch',0)!=0 or obj['rotation_deg'].get('yaw',0)!=0:raise ValueError('Unsupported solid rotation')
                obj['rotation_deg']['roll']=-obj['rotation_deg'].get('roll',0)
        c['condition'].update(source_site_id=site,split_group_id=site,counterfactual_parent_id=c['group_id'],
                               condition_id=c['name'],lateral_offset_m=-c['condition']['lateral_offset_m'],
                               tilt_degrees=-c['condition']['tilt_degrees'])
        c['geometric_contact']=target_contact(c['objects'],c['wearer'])
        c['geometric_intrusion']=intrusion_metrics(c['objects'],c['wearer'])
        if c['geometric_contact']['relation']!=c['condition']['desired_relation']:
            raise ValueError('Mirror changed relation')
    counts=Counter(c['condition']['family'] for c in cases)
    if sorted(counts.values())!=[32]*4:raise ValueError('Require eight quartets per family')
    result=copy.deepcopy(parent)
    result.update(schema='city-distant-region-dev-128-v1',cases=cases,source_role='DEV_ONLY',
        purpose='DEV_NATIVE_COVERAGE_BEFORE_ANY_MODEL_INFERENCE',
        scope='Same Street200V7, distant southeast frontage; not an independent world or fresh TEST',
        suite_contract=dict(frames=128,quartets=32,family_frame_counts=dict(counts),source_site_id=site,
                            intended_geometry_counts=dict(Counter(c['geometric_contact']['relation'] for c in cases)),
                            labels='Actual all-scene native visible support; no desired-state override'),
        dev_provenance=dict(parent_sha256=parent_sha,parent_groups=list(dict.fromkeys(c['parent_group_id'] for c in cases)),
                            selection='Earliest32 deterministic contextual1000 quartets before model inference',
                            transform='x +=130; y=-y for all objects/camera/wearer; solid roll=-roll; yaw0 unchanged',
                            site_evidence='Street200 build-3 southeast building row centered (55,-30), length90m; frontage y=-15; original proposed (54,+15) has no northeast row',
                            admission='Canary RGB/native attachment and floor first; actual DEV per-head >=48positive and >=48negative; HEAD positive in each family'))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();data=a.parent.read_bytes();result=generate(json.loads(data.decode('utf-8-sig')),hashlib.sha256(data).hexdigest())
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
