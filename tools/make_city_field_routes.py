"""Build native City field reconnaissance without admitting unverified routes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def make_scout(plan, template):
    spec = copy.deepcopy(template)
    spec.pop('native_targets', None)
    spec.pop('inventory_indices', None)
    spec.update(export_native_inventory=True, cases=[], selection='Geometry-only field reconnaissance; no model inference or training.')
    bounds = plan['world_partition_region_m']
    spec['world_partition_region_m'] = bounds
    for region in plan['regions']:
        x, y = region['scout_center_xy_m']
        for view, z, cy, yaw, pitch in [('overview', 80., y, 0., -90.), ('sidewalk', 2.43, -1., 0., -5.), ('opposite_frontage', 2.43, -16., 180., -5.)]:
            spec['cases'].append(dict(name=region['region_id']+'_'+view,
                camera=dict(x=x,y=cy,z=z,yaw=yaw,pitch=pitch,roll=0.),
                objects=[],probe_native_floor=view!='overview',region_id=region['region_id'],
                scout_view=view,route_admission='PENDING_GEOMETRY_AND_VISUAL_REVIEW'))
    # Ground inventories, not elevated camera radius neighborhoods.
    spec['inventory_indices'] = [i for i,c in enumerate(spec['cases']) if c['scout_view']!='overview']
    return spec

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=ROOT/'experiments/city-field/native-field-v1.json')
    parser.add_argument('--template',type=Path,default=ROOT/'experiments/city-field/capture-template-v1.json')
    parser.add_argument('--project',type=Path,help='Resolve relative map_file against this CitySample project')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('scout','ground-preview'),default='scout')
    args=parser.parse_args()
    out=args.output.resolve()
    assert out.is_relative_to((ROOT/'artifacts.local').resolve())
    plan=json.loads(args.plan.read_bytes());template=json.loads(args.template.read_bytes())
    if args.project and not Path(template['map_file']).is_absolute():
        template['map_file']=str((args.project.resolve().parent/template['map_file']).resolve())
    spec=make_scout(plan,template)
    if args.mode=='ground-preview':
        spec['cases']=[]
        spec.pop('inventory_indices',None)
        for route in plan['routes']:
            for index in route.get('preview_indices',[0,len(route['waypoints'])//2,len(route['waypoints'])-1]):
                spec['cases'].append(dict(name=route['route_id']+'_'+str(index),
                    camera=copy.deepcopy(route['waypoints'][index].get('camera',route['waypoints'][index])),objects=copy.deepcopy(route.get('objects',[])),probe_native_floor=True,
                    route_id=route['route_id'],waypoint_index=index))
        spec['selection']='Native route ground admission preview; no fixture or model conclusions.'
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [(args.mode+'-spec.json',spec),(args.mode+'-protocol.json',dict(schema='city-field-scout-v1',frames=len(spec['cases']),fits=0,
            plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),template_sha256=hashlib.sha256(args.template.read_bytes()).hexdigest(),
            purpose='Locate native streets/intersections and candidate passage surfaces; scout poses are not admitted routes.',
            stop='Capture only the declared poses and inspect; do not infer narrow passage or clear space from a missing obstacle.'))]:
        with (out/name).open('x',encoding='utf-8',newline='\n') as f: json.dump(data,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='SCOUT_SPEC_READY',frames=len(spec['cases']),output=str(out))))

if __name__=='__main__': main()
