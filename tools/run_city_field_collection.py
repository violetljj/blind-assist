"""Compile and execute region-batched City field acquisition; never train models."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts.local'
KINDS = ('clear', 'thin_pole', 'body_protrusion', 'head_bar', 'suspended_sign')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def compile_region(plan, template, region_id):
    from city_field_fixtures import fixture_objects
    import math
    routes = [r for r in plan['routes'] if r['region_id'] == region_id]
    if not routes:
        raise ValueError('No admitted routes for region ' + region_id)
    result = copy.deepcopy(template)
    for key in ('native_targets', 'route', 'selection', 'suite_contract'):
        result.pop(key, None)
    result.update(schema='city-field-capture-v1', cases=[], native_targets=[],
        export_controlled_targets=True, export_native_inventory=True,
        export_appearance=False, inventory_indices=[],
        collection_scope='Static settled route samples, not continuous video; original city with declared controlled assemblies')
    region = next(r for r in plan['regions'] if r['region_id'] == region_id)
    bounds = region['bounds_xy_m']
    result['world_partition_region_m'] = dict(min=[bounds[0]-100,bounds[1]-100,-10],max=[bounds[2]+100,bounds[3]+100,200])
    target_ids = set()
    for route in routes:
        rid = route['route_id']
        if route.get('status') not in ('ADMITTED', 'READY', 'VERIFIED'):
            raise ValueError('Route has not passed scene review: ' + rid)
        waypoints = route['waypoints']
        if len(waypoints) < 3:
            raise ValueError('Route requires start/middle/end samples')
        common = dict(route_id=rid, region_id=region_id, split=region['split'],scene_type=route['scene_type'])
        base_objects = copy.deepcopy(route.get('objects', []))
        result['inventory_indices'].append(len(result['cases']))
        for j, waypoint in enumerate(waypoints):
            result['cases'].append(dict(**common,name=f'{rid}_path_{j:03d}',
                camera=waypoint['camera'],objects=base_objects,probe_native_floor=True,
                variant_id='route_baseline',variant='route_baseline',pair_id=None,
                route_distance_m=waypoint.get('route_distance_m'),floor_z_m=waypoint['floor_z_m']))
        mid = waypoints[len(waypoints)//2]
        camera = route.get('fixture_camera',mid['camera'])
        fixture_floor = route.get('fixture_floor_z_m',mid['floor_z_m'])
        yaw = camera['yaw']
        angle = math.radians(yaw)
        anchor = route.get('fixture_anchor_m', [camera['x']+2*math.cos(angle),camera['y']+2*math.sin(angle),fixture_floor])
        for kind in KINDS:
            objects = (copy.deepcopy(route['fixture_variants'][kind]) if 'fixture_variants' in route
                       else fixture_objects(kind, anchor, yaw, rid))
            case = dict(**common,name=f'{rid}_fixture_{kind}',camera=camera,
                objects=base_objects+objects,probe_native_floor=True,floor_z_m=fixture_floor,
                pair_id=f'{rid}_fixture',variant_id=kind,variant=kind,
                fixture_scope='Controlled supported assembly; clear removes hazard parts, not native city geometry')
            result['cases'].append(case)
            for obj in objects:
                if obj.get('target_part') and obj['instance_id'] not in target_ids:
                    tid=obj['instance_id']; target_ids.add(tid)
                    result['native_targets'].append(dict(target_id=tid,instance_id=tid,
                        category=kind,source_kind='CONTROLLED_ASSEMBLY',route_id=rid,
                        position_m=obj['center_m'],size_m=obj.get('size_m'),rotation_deg=obj.get('rotation_deg'),
                        support_parent=obj.get('support_parent')))
    for case in result['cases']:
        route_targets = {t['target_id'] for t in result['native_targets'] if t['route_id'] == case['route_id']}
        active = {o['instance_id'] for o in case['objects'] if o.get('target_part')}
        case['active_target_ids'] = sorted(route_targets & active)
        case['absent_target_ids'] = sorted(route_targets - active)
        case['controlled_target_presence'] = {
            tid: 'PRESENT' if tid in active else 'NOT_PRESENT' for tid in sorted(route_targets)}
    return result


def run(args):
    from city_field_contract import validate_plan, ingest_native_bundle, validate_bundles
    plan,template=read(args.plan),read(args.template)
    unknown_regions=set(args.region or [])-{r['region_id'] for r in plan['regions']}
    if unknown_regions:
        raise ValueError('Unknown regions: '+','.join(sorted(unknown_regions)))
    if args.project and not Path(template['map_file']).is_absolute():
        template['map_file']=str((args.project.resolve().parent/template['map_file']).resolve())
    out=args.output.resolve()
    if not out.is_relative_to(ARTIFACTS.resolve()) or out==ARTIFACTS.resolve() or out.exists():
        raise ValueError('Fresh output strictly inside canonical artifacts required')
    check=validate_plan(plan)
    if check['status']=='FAIL':
        raise ValueError('Invalid collection plan: '+json.dumps(check))
    out.mkdir(parents=True)
    write(out/'plan-validation.json',check)
    write(out/'plan.json',plan)
    specs={}
    for region in plan['regions']:
        if args.region and region['region_id'] not in args.region: continue
        spec=compile_region(plan,template,region['region_id'])
        path=out/(region['region_id']+'-spec.json');write(path,spec);specs[region['region_id']]=path
    if args.compile_only:
        write(out/'receipt.json',dict(status='COMPILED',regions=list(specs),frames=sum(len(read(p)['cases']) for p in specs.values())))
        return
    if not all((args.project,args.plugin,args.engine)):
        raise ValueError('Execution requires project/plugin/engine')
    bundles=[]
    try:
        for region,path in specs.items():
            if os.name == 'nt':
                live = subprocess.check_output(['pwsh','-NoProfile','-Command',
                    'Get-Process -Name UnrealEditor,UnrealEditor-Cmd -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id; exit 0'],text=True).strip()
                if live:
                    raise RuntimeError('Unreal is already running; leave existing work untouched. PIDs: '+live)
            capture=out/region/'capture'; labels=out/region/'labels'
            subprocess.run([sys.executable,str(ROOT/'tools/run_city_pcg_capture.py'),
                '--project',str(args.project),'--plugin',str(args.plugin),'--engine',str(args.engine),
                '--spec',str(path),'--output',str(capture),'--timeout',str(args.timeout)],check=True,cwd=ROOT)
            subprocess.run([sys.executable,str(ROOT/'tools/verify_city_native_targets.py'),
                '--capture',str(capture),'--output',str(labels),'--raycheck',str(capture/'evaluator/target-raycheck.json')],check=True,cwd=ROOT)
            for route in [r for r in plan['routes'] if r['region_id']==region]:
                bundle=ingest_native_bundle(plan,capture,route['route_id'],labels/'native-route-labels.json')
                write(out/region/(route['route_id']+'-bundle.json'),bundle);bundles.append(bundle)
            write(out/'progress.json',dict(completed_regions=list(dict.fromkeys(b['region_id'] for b in bundles)),current=region))
        validation=validate_bundles(plan,bundles,require_coverage=not bool(args.region) and plan.get('require_all_splits',True))
        write(out/'collection-validation.json',validation)
        from export_city_field_ready import export_collection
        from report_city_field_collection import build as report_collection
        exported=export_collection(out,out/'ready')
        report_collection(out)
        if not exported['accepted']:
            raise RuntimeError('No usable frames after filtering; inspect ready/excluded.json')
        write(out/'receipt.json',dict(status=validation['status'],bundles=len(bundles),model_training=False,
            plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),completed=True,
            ready_export=exported['status'],accepted_frames=exported['accepted'],excluded_frames=exported['excluded']))
    except BaseException as error:
        write(out/'receipt.json',dict(status='INCOMPLETE',error=repr(error),bundles_completed=len(bundles),
            recovery='Completed immutable region outputs remain available; do not overwrite or silently recapture them.'))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('plan','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--template',type=Path,default=ROOT/'experiments/city-field/capture-template-v1.json')
    for name in ('project','plugin','engine'): p.add_argument('--'+name,type=Path)
    p.add_argument('--region',action='append')
    p.add_argument('--timeout',type=float,default=1800)
    p.add_argument('--compile-only',action='store_true')
    run(p.parse_args())
