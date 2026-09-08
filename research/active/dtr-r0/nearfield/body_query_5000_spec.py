"""Versioned source-only scout and 20-frame site blocks for expanded collection."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
from contextual_geometry import target_contact

FAMILIES=('crossbar','cabinet','oblique_rod','hanging_sign')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def base_spec(region):
    spec=read(region['template_spec_file'])
    for key in ('source_floor_grid','native_targets','route'):spec.pop(key,None)
    spec.update(cases=[],map_file=region['primary_map_file'],schema='body-query-5000-source-v1',
        export_native_inventory=True,inventory_indices=[0],export_hlod_membership=False,
        export_dependencies=False,pair_export_mode='native_async',export_appearance=False,
        scope='Expanded synthetic Development; shared assets/background possible; no model scoring or protected TEST',
        source_role=region['split'].upper()+'_ONLY')
    assert sha(spec['map_file'])==spec['map_sha256']
    return spec

def scout_spec(region):
    spec=base_spec(region)
    for site in region['primary_sites']+region['reserve_sites']:
        x,y=site['camera_xy_m'];floor=site['floor_z_m']
        spec['cases'].append(dict(name=site['site_id'],site_id=site['site_id'],region_id=region['region_id'],
            source_role='ENGINEERING_SCOUT',objects=[],floor_z_m=floor,floor_check=False,
            probe_native_floor=True,camera=dict(x=x,y=y,z=floor+1.7,pitch=0.,roll=0.,yaw=site['yaw_deg'])))
    spec['purpose']='EMPTY_SITE_ENGINEERING_PROBES_NOT_5000_DATASET'
    return spec

def fixture_catalog(fixtures):
    cases=[c for c in fixtures['cases'] if c['source_role']=='TRAIN_ONLY']
    catalog={f:[] for f in FAMILIES}
    for gid in dict.fromkeys(c['group_id'] for c in cases):
        group=[c for c in cases if c['group_id']==gid]
        catalog[group[0]['condition']['family']].append(group)
    assert all(len(v)>=12 for v in catalog.values())
    return catalog

def site_cases(catalog,region,site,rank):
    result=[];yaw=site['yaw_deg'];co,si=math.cos(math.radians(yaw)),math.sin(math.radians(yaw))
    for family_index,family in enumerate(FAMILIES):
        selected=catalog[family][(rank+3*family_index)%len(catalog[family])]
        near=(rank+family_index)%2==0
        rng=random.Random(20260909+rank*101+family_index)
        distance=rng.uniform(.85,1.35) if near else rng.uniform(2.,2.6)
        gid=f"bq5000-{site['site_id']}-{family}"
        for raw in selected:
            c=copy.deepcopy(raw);x,y,floor=c['camera']['x'],c['camera']['y'],c['floor_z_m']
            assert c['camera']['yaw']==c['wearer']['yaw']==0
            delta=distance-c['condition']['distance_m']
            for key in ('camera','wearer'):
                p=c[key];p.update(x=p['x']-x,y=p['y']-y,z=p['z']-floor)
            c['camera'].update(z=1.7,pitch=0.,roll=0.)
            for o in c['objects']:
                px,py,pz=o['center_m'];o['center_m']=[px-x+delta,py-y,pz-floor]
                for axis in ('pitch','yaw','roll'):o.setdefault('rotation_deg',{}).setdefault(axis,0.)
            relation=target_contact(c['objects'],c['wearer'])['relation']
            assert relation==c['condition']['desired_relation'],(gid,c['variant_id'],relation)
            geometry_hash=hashlib.sha256(json.dumps(c['objects'],sort_keys=True).encode()).hexdigest()
            for key in ('camera','wearer'):
                p=c[key];px,py=p['x'],p['y']
                p.update(x=site['camera_xy_m'][0]+co*px-si*py,y=site['camera_xy_m'][1]+si*px+co*py,
                         z=p['z']+site['floor_z_m'],yaw=yaw)
            for o in c['objects']:
                px,py,pz=o['center_m'];o['center_m']=[site['camera_xy_m'][0]+co*px-si*py,
                    site['camera_xy_m'][1]+si*px+co*py,pz+site['floor_z_m']];o['rotation_deg']['yaw']+=yaw
            c.update(name=gid+'-'+c['variant_id'],group_id=gid,parent_group_id=gid,site_id=site['site_id'],
                region_id=region['region_id'],source_site_id=site['site_id'],source_role=region['split'].upper()+'_ONLY',
                floor_z_m=site['floor_z_m'],floor_check=False,probe_native_floor=True,
                declared_range='near' if near else 'far',local_geometry_sha256=geometry_hash,
                original_fixture_group_id=raw['group_id'],geometry_frame='Local analytic intent then rigid world placement')
            c['condition'].update(condition_id=c['name'],counterfactual_parent_id=gid,source_site_id=site['site_id'],
                split_group_id=region['region_id'],distance_m=distance,eye_height_m=1.7,camera_pitch_deg=0.)
            c['geometric_contact']=dict(relation=relation,authority='LOCAL_INTENT_NOT_NATIVE_LABEL')
            c.pop('geometric_intrusion',None);result.append(c)
    assert len(result)==20
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['scout','full'])
    p.add_argument('--sites',type=Path,required=True);p.add_argument('--fixtures',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=(Path(__file__).resolve().parents[4]/'artifacts.local').resolve()
    assert a.output.resolve().is_relative_to(root) and not a.output.exists()
    source=read(a.sites);a.output.mkdir(parents=True)
    catalog=fixture_catalog(read(a.fixtures)) if a.mode=='full' else None
    outputs=[]
    for region in source['regions']:
        spec=scout_spec(region) if a.mode=='scout' else base_spec(region)
        if a.mode=='full':
            assert len(region['accepted_sites'])==25
            for rank,site in enumerate(region['accepted_sites']):spec['cases']+=site_cases(catalog,region,site,rank)
            assert len(spec['cases'])==500
            spec['purpose']='500_FRAME_REGION_CHUNK_OF_5000_SOURCE_ONLY'
        spec['provenance']=dict(generator_sha256=sha(__file__),sites_sha256=sha(a.sites),
            fixtures_sha256=sha(a.fixtures) if a.fixtures else None)
        path=a.output/(region['region_id']+'.json');path.write_text(json.dumps(spec,indent=2))
        outputs.append(dict(region_id=region['region_id'],spec=str(path.resolve()),sha256=sha(path),frames=len(spec['cases'])))
    (a.output/'manifest.json').write_text(json.dumps(dict(mode=a.mode,outputs=outputs,
        frames=sum(v['frames'] for v in outputs),status='SPEC_ONLY_NOT_CAPTURED'),indent=2))
    print(json.dumps(outputs))

if __name__=='__main__':main()
