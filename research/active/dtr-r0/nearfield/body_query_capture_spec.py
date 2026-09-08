"""Fixed optical calibration, grounded contextual BODY/HEAD Development source."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import contextual_collection1000 as source
from contextual_geometry import target_contact, intrusion_metrics
from contextual_scene import MATERIALS

SITES = {'train': (-90.,12.,0.,48,2.), 'dev': (54.,-13.5,180.,8,1.5),
         'eval': (67.,53.5,180.,8,1.5)}

def generate(template,role):
    before=source.SEED
    seed=20260928+list(SITES).index(role)
    try:
        source.SEED=seed
        result=source.collection(template)
    finally:
        source.SEED=before
    x,y,yaw,units,spacing=SITES[role]
    originals=result['cases'][:units*4]
    expanded=[]
    for unit in range(units):
        quartet=originals[unit*4:unit*4+4]
        for c in quartet:c['_unit']=unit
        expanded.extend(quartet)
        if quartet[0]['condition']['family']=='crossbar':
            for variant in ('LOW','ABOVE','LATERAL_OUT','FAR_OUT'):
                c=copy.deepcopy(quartet[1]);c['variant_id']=variant
                if variant in ('LOW','ABOVE'):
                    height=.45 if variant=='LOW' else 2.65
                    for obj in c['objects']:
                        if obj['name'].startswith('clamp_') or obj['name']=='adjustable_cross_member':
                            obj['center_m'][2]=c['floor_z_m']+height
                    c['effective_target_height_m']=height;c['condition']['height_m']=height
                else:
                    axis=1 if variant=='LATERAL_OUT' else 0
                    for obj in c['objects']:obj['center_m'][axis]+=3. if axis else 4.
                c['condition']['desired_relation']='CLEAR'
                c['condition']['control_variant']=variant
                expanded.append(c)
    result['cases']=expanded
    co,si=math.cos(math.radians(yaw)),math.sin(math.radians(yaw))
    for i,c in enumerate(result['cases']):
        unit=c.pop('_unit')
        # Extra TRAIN groups reuse admitted camera locations with new fixtures.
        ox,oy=x+spacing*(unit%8),y+(.3 if role=='train' else 1.)*((unit//8)%4)
        cx,cy=c['camera']['x'],c['camera']['y']
        c['camera'].update(z=c['floor_z_m']+1.70,pitch=0.)
        c['condition'].update(eye_height_m=1.70,camera_pitch_deg=0.)
        if c['condition']['family'] in ('cabinet','hanging_sign'):
            c['objects'].append(dict(name='query_grounded_backboard',center_m=[-75.5,15.08,1.95],
                size_m=[2.,.20,3.5],material_asset=MATERIALS['metal'],support_parent='ground',target_part=False))
            for obj in c['objects']:
                if obj.get('support_parent')=='existing_facade':obj['support_parent']='query_grounded_backboard'
        c['geometric_contact']=target_contact(c['objects'],c['wearer'])
        c['geometric_intrusion']=intrusion_metrics(c['objects'],c['wearer'])
        assert c['geometric_contact']['relation']==c['condition']['desired_relation']
        def xy(px,py):
            dx,dy=px-cx,py-cy
            return ox+co*dx-si*dy,oy+si*dx+co*dy
        for key in ('camera','wearer'):
            p=c[key];p['x'],p['y']=xy(p['x'],p['y']);p['yaw']=yaw
        for obj in c['objects']:
            obj['center_m'][:2]=xy(*obj['center_m'][:2])
            obj.setdefault('rotation_deg',dict(pitch=0.,roll=0.))['yaw']=yaw
        gid=f'body-query-{role}-s{seed}-u{unit:02d}'
        sid=f'plaza-{role}-x{ox:g}-y{oy:g}'
        c.update(name=gid+'-'+c['variant_id'],group_id=gid,parent_group_id=gid,source_role=role.upper()+'_ONLY',
            source_site_id=sid,geometry_frame='PRE_RIGID_TRANSFORM_ROUTE_LOCAL_X; native depth authoritative')
        c['condition'].update(condition_id=c['name'],source_site_id=sid,split_group_id=sid,counterfactual_parent_id=gid)
    result.update(schema='body-query-fixed-camera-source-v1',seed=seed,source_role=role.upper()+'_ONLY',
        purpose='NEW_MATCHED_BODY_HEAD_QUERY_DEVELOPMENT',
        scope='Same-world plaza Development; old source regions consumed; not fresh-world or protected test',
        suite_contract=dict(frames=len(result['cases']),groups=units,relations=['CLEAR','BODY_ONLY','HEAD_ONLY','BOTH'],
            crossbar_extra_controls=['LOW','ABOVE','LATERAL_OUT','FAR_OUT'],
            fixed_calibration=dict(eye_height_m=1.70,pitch_deg=0.,roll_deg=0.,width=640,height=360,hfov_deg=100.),
            body_boxes='UNCHANGED',query_range_m=3.,visible_positive='at least3 native pixels globally per band'))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--template',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();template=json.loads(a.template.read_text(encoding='utf-8-sig'));a.output.mkdir(parents=True,exist_ok=False)
    roles={r:generate(template,r) for r in SITES}
    full=copy.deepcopy(roles['train']);full['cases']=[c for s in roles.values() for c in s['cases']]
    full['source_role']='MIXED_EXPLICIT_ROLES';full['suite_contract']['frames']=len(full['cases'])
    full['suite_contract']['role_frames']={r:len(s['cases']) for r,s in roles.items()}
    full['provenance']=dict(generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),template_sha256=hashlib.sha256(a.template.read_bytes()).hexdigest())
    canary=copy.deepcopy(full);canary['cases']=[copy.deepcopy(s['cases'][i]) for s in roles.values() for i in (0,1,2,3,4,5,6,7,9,14,19)]
    for role,s in roles.items():
        c=copy.deepcopy(s['cases'][0]);c.update(name='floor-'+role,objects=[],floor_check=True,group_id='floor-'+role)
        canary['cases'].append(c)
    canary['source_role']='ENGINEERING_CANARY';canary['suite_contract']['frames']=len(canary['cases'])
    for name,s in [('all',full),('canary',canary)]:
        (a.output/f'spec-{name}-v1.json').write_text(json.dumps(s,indent=2))
    print(json.dumps(dict(frames=len(full['cases']),canary=len(canary['cases']),roles=full['suite_contract']['role_frames'])))

if __name__=='__main__':main()
