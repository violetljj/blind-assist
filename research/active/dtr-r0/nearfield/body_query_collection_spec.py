"""Small fixed-calibration BigCity counterfactual collection, no model selection."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from contextual_geometry import target_contact


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(fixtures, scout, pose, role):
    originals = [c for c in fixtures['cases'] if c['source_role']=='TRAIN_ONLY']
    groups = list(dict.fromkeys(c['group_id'] for c in originals))[:8]
    out = copy.deepcopy(scout)
    for key in ('source_floor_grid', 'native_targets', 'route'):
        out.pop(key, None)
    out.update(schema='body-query-new-scenes-pilot-v1', cases=[], export_hlod_membership=False,
        export_native_inventory=True, inventory_indices=[0], pair_export_mode='native_async',
        source_role=role.upper()+'_ONLY', purpose='BODY_QUERY_COUNTERFACTUAL_SOURCE_PILOT',
        scope='Synthetic BigCity Development. New regional placements; shared assets/background possible. '
              'No strict visibility isolation or protected-test claim. No model inference or training.',
        fixed_calibration=dict(eye_height_m=1.7,pitch=0.,roll=0.,width=640,height=360,hfov=100.),
        suite_contract=dict(frames=40,groups=8,range_centers_m=[1.,2.4],
            primary_relations=['CLEAR','BODY_ONLY','HEAD_ONLY','BOTH'],
            crossbar_controls=['LOW','ABOVE','LATERAL_OUT','FAR_OUT']))
    yaw=float(pose['yaw']); co,si=math.cos(math.radians(yaw)),math.sin(math.radians(yaw))
    for unit, group in enumerate(groups):
        selected=[c for c in originals if c['group_id']==group]
        distance=1. if unit<4 else 2.4
        gid=f"bq-scenes-20260909-{pose['region_id']}-u{unit:02d}"
        for raw in selected:
            c=copy.deepcopy(raw)
            assert abs(c['camera']['yaw'])<1e-8 and abs(c['wearer']['yaw'])<1e-8
            x,y,floor=c['camera']['x'],c['camera']['y'],c['floor_z_m']
            delta=distance-c['condition']['distance_m']
            for key in ('camera','wearer'):
                p=c[key];p.update(x=p['x']-x,y=p['y']-y,z=p['z']-floor)
            c['camera'].update(z=1.7,pitch=0.,roll=0.)
            for obj in c['objects']:
                px,py,pz=obj['center_m'];obj['center_m']=[px-x+delta,py-y,pz-floor]
                for axis in ('pitch','yaw','roll'):obj.setdefault('rotation_deg',{}).setdefault(axis,0.)
            c['floor_z_m']=0.
            relation=target_contact(c['objects'],c['wearer'])['relation']
            assert relation==c['condition']['desired_relation'], (group,c['variant_id'],relation)
            local_hash=hashlib.sha256(json.dumps(c['objects'],sort_keys=True).encode()).hexdigest()
            for key in ('camera','wearer'):
                p=c[key];px,py=p['x'],p['y']
                p.update(x=pose['x']+co*px-si*py,y=pose['y']+si*px+co*py,
                         z=p['z']+pose['floor_z_m'],yaw=yaw)
            for obj in c['objects']:
                px,py,pz=obj['center_m']
                obj['center_m']=[pose['x']+co*px-si*py,pose['y']+si*px+co*py,pz+pose['floor_z_m']]
                obj['rotation_deg']['yaw']+=yaw
            c.update(name=gid+'-'+c['variant_id'],group_id=gid,parent_group_id=gid,
                source_site_id=pose['region_id'],region_id=pose['region_id'],source_role=role.upper()+'_ONLY',
                declared_range='near' if unit<4 else 'far',floor_z_m=pose['floor_z_m'],
                probe_native_floor=True,floor_check=False,local_geometry_sha256=local_hash,
                geometry_frame='Analytic check in local +X before rigid placement; native depth authoritative')
            c['condition'].update(condition_id=c['name'],counterfactual_parent_id=gid,
                source_site_id=pose['region_id'],split_group_id=pose['region_id'],distance_m=distance,
                eye_height_m=1.7,camera_pitch_deg=0.)
            c['geometric_contact']=dict(relation=relation,authority='LOCAL_ASSEMBLY_INTENT_ONLY')
            c.pop('geometric_intrusion',None)
            out['cases'].append(c)
    assert len(out['cases'])==40
    return out


def main():
    p=argparse.ArgumentParser()
    for key in ('fixtures','placements','output'):p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args();root=(Path(__file__).resolve().parents[4]/'artifacts.local').resolve()
    assert a.output.resolve().is_relative_to(root) and not a.output.exists()
    fixtures=json.loads(a.fixtures.read_text(encoding='utf-8-sig'))
    placements=json.loads(a.placements.read_text(encoding='utf-8-sig'))
    assert [v['role'] for v in placements]==['train','dev','eval']
    assert len({v['region_id'] for v in placements})==3
    a.output.mkdir(parents=True)
    outputs=[]
    for pose in placements:
        scout=json.loads(Path(pose['scout_spec']).read_text(encoding='utf-8-sig'))
        scout['map_file']=pose['map_file']
        assert sha(scout['map_file'])==scout['map_sha256'], 'Primary map differs from scouted source'
        spec=generate(fixtures,scout,pose,pose['role'])
        spec['provenance']=dict(fixtures_sha256=sha(a.fixtures),placements_sha256=sha(a.placements),
                                scout_sha256=sha(pose['scout_spec']),generator_sha256=sha(__file__))
        path=a.output/(pose['role']+'.json');path.write_text(json.dumps(spec,indent=2))
        outputs.append(dict(role=pose['role'],region=pose['region_id'],frames=40,spec=str(path),sha256=sha(path)))
    (a.output/'manifest.json').write_text(json.dumps(dict(status='SPEC_FROZEN_BEFORE_CAPTURE',
        frames=120,outputs=outputs,model_runs=0),indent=2))
    print(json.dumps(outputs))


if __name__=='__main__':main()
