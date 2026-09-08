"""Compile fixed paired interventions and score verified binary counterfactual pairs."""
import copy
from collections import Counter,defaultdict
import math

from city_field_fixtures import fixture_objects

CONDITIONS=('NONE','BODY_ONLY','HEAD_ONLY','BOTH')


def interventions(anchor,yaw,prefix):
    # Every carrier and clamp is identical even when the corresponding hazard is absent.
    full={}
    for kind in ('clear','body_protrusion','head_bar'):
        for obj in fixture_objects(kind,anchor,yaw,prefix):full[obj['instance_id']]=obj
    common=[o for o in full.values() if not o.get('target_part')]
    body=[o for o in full.values() if o['instance_id'].endswith('__body_arm')]
    head=[o for o in full.values() if o['instance_id'].endswith('__head_crossbar')]
    # Inner clamp faces are at +/-0.78m. Butt-joint instead of penetrating them.
    # This avoids collision identity ambiguity at the bar ends without changing labels.
    for obj in head:obj['size_m'][1]=1.56
    return {c:copy.deepcopy(common+(body if c in ('BODY_ONLY','BOTH') else [])+
                           (head if c in ('HEAD_ONLY','BOTH') else [])) for c in CONDITIONS}


def compile_route(route,perturbations):
    camera=route['camera'];yaw=camera['yaw'];angle=math.radians(yaw)
    anchor=route.get('anchor_m',[camera['x']+2*math.cos(angle),camera['y']+2*math.sin(angle),route['floor_z_m']])
    variants=interventions(anchor,yaw,route['route_id']);cases=[]
    for j,perturbation in enumerate(perturbations):
        cam=dict(camera,x=camera['x']-math.sin(angle)*perturbation['right_m'],
                 y=camera['y']+math.cos(angle)*perturbation['right_m'],yaw=yaw+perturbation['yaw_delta_deg'])
        group=f"{route['route_id']}_p{j}"
        for condition,objects in variants.items():
            cases.append(dict(name=group+'_'+condition,camera=copy.deepcopy(cam),
                floor_z_m=route['floor_z_m'],probe_native_floor=True,
                objects=copy.deepcopy(route.get('context_objects',[]))+copy.deepcopy(objects),
                route_id=route['route_id'],region_id=route['region_id'],split=route['split'],scene_type=route['scene_type'],
                pair_id=group,condition=condition,variant=condition,
                expected_near=[int(condition in ('BODY_ONLY','BOTH')),int(condition in ('HEAD_ONLY','BOTH'))],
                expected_near_authority='PLACEMENT_INTENT_ONLY_VERIFY_NATIVE_GEOMETRY_BEFORE_SCORING'))
    target_ids=sorted({o['instance_id'] for c in cases for o in c['objects'] if o.get('target_part')})
    for case in cases:
        active=[o['instance_id'] for o in case['objects'] if o.get('target_part')]
        case.update(active_target_ids=active,absent_target_ids=sorted(set(target_ids)-set(active)))
    return cases


def compile_design(protocol,source,template,canary=False):
    routes=source['routes']
    if len({r['route_id'] for r in routes})!=len(routes):raise ValueError('Duplicate route identity')
    if canary:
        if any(r['split']!='train' for r in routes):raise ValueError('Canary must use TRAIN source only')
    else:
        if source.get('status')!='ADMITTED' or source.get('visible_background_isolation')!='PASS':
            raise ValueError('Physical visible-background isolation is not admitted')
        if not source.get('map_asset') or source['map_asset']!=template.get('map_asset'):
            raise ValueError('Source/capture map identity mismatch')
        regions=source['regions']
        if len({r['region_id'] for r in regions})!=len(regions):raise ValueError('Duplicate region identity')
        for i,region in enumerate(regions):
            box=region['bounds_xy_m']
            if len(box)!=4 or not all(math.isfinite(v) for v in box) or box[0]>=box[2] or box[1]>=box[3]:raise ValueError('Invalid region bounds')
            for other in regions[:i]:
                b=other['bounds_xy_m']
                if min(box[2],b[2])>max(box[0],b[0]) and min(box[3],b[3])>max(box[1],b[1]):raise ValueError('Overlapping region blocks')
        if Counter(r['split'] for r in regions)!=Counter(protocol['region_splits']):raise ValueError('Split allocation differs')
        by_region=defaultdict(list)
        for r in routes:by_region[r['region_id']].append(r)
        if set(by_region)!={r['region_id'] for r in regions}:raise ValueError('Region coverage mismatch')
        for region in regions:
            rs=by_region[region['region_id']]
            if sorted(r['scene_type'] for r in rs)!=sorted(protocol['scene_types']) or any(r['split']!=region['split'] for r in rs):
                raise ValueError('Every region requires exactly three matching route types')
            box=region['bounds_xy_m']
            for r in rs:
                for c in compile_route(r,protocol['camera_perturbations']):
                    cam=c['camera']
                    if not(box[0]<=cam['x']<=box[2] and box[1]<=cam['y']<=box[3]):
                        raise ValueError('Perturbed camera leaves region')
        visible=defaultdict(set)
        for r in regions:
            ids=r.get('visible_background_instance_ids')
            if not ids:raise ValueError('Missing visible background identity evidence')
            visible[r['split']].update(ids)
        for a,b in [('train','dev'),('train','test'),('dev','test')]:
            if visible[a]&visible[b]:raise ValueError('Cross-split background instance leak')
    spec=copy.deepcopy(template);spec.update(cases=[],native_targets=[],export_controlled_targets=True,
        export_native_inventory=True,inventory_indices=[],export_appearance=False,
        schema='city-crossregion-capture-v1',protocol_id=protocol['protocol_id'])
    for route in routes:
        if route.get('status')!='ADMITTED':raise ValueError('Unreviewed source route')
        spec['inventory_indices'].append(len(spec['cases']))
        spec['cases'].extend(compile_route(route,protocol['camera_perturbations']))
    if not canary and len(spec['cases'])!=protocol['planned_frames']:raise ValueError('Frame budget differs')
    declarations={}
    for case in spec['cases']:
        for obj in case['objects']:
            if obj.get('target_part'):
                tid=obj['instance_id'];declarations[tid]=dict(target_id=tid,instance_id=tid,
                    position_m=obj['center_m'],size_m=obj['size_m'],route_id=case['route_id'],
                    category='HEAD' if tid.endswith('__head_crossbar') else 'BODY',support_parent=obj['support_parent'])
    spec['native_targets']=list(declarations.values());return spec


def causal_flip(rows):
    if len({r['split'] for r in rows})>1:raise ValueError('Score one split at a time')
    groups=defaultdict(dict)
    for row in rows:
        key=(row['region_id'],row['pair_id']);condition=row['condition']
        if condition not in CONDITIONS or condition in groups[key]:raise ValueError('Duplicate/unknown condition')
        groups[key][condition]=row
    scores={h:dict(eligible_pairs=0,correct_pairs=0,probability_delta_sum=0.) for h in ('BODY','HEAD')}
    excluded=[]
    for key,group in groups.items():
        if set(group)!=set(CONDITIONS) or any(r.get('eligible') is not True for r in group.values()):
            excluded.append(key);continue
        if len({r['split'] for r in group.values()})!=1:raise ValueError('Pair crosses splits')
        if not all(r.get('context_hash') for r in group.values()) or len({r['context_hash'] for r in group.values()})!=1:
            raise ValueError('Counterfactual camera/background/light context mismatch')
        for r in group.values():
            if len(r['truth'])!=2 or len(r['prediction'])!=2 or any(v not in (0,1) for v in r['truth']+r['prediction']):
                raise ValueError('Verified binary truth/prediction required')
            if len(r['probability'])!=2 or any(not math.isfinite(v) or not 0<=v<=1 for v in r['probability']):
                raise ValueError('Finite probabilities required')
        expected={'NONE':[0,0],'BODY_ONLY':[1,0],'HEAD_ONLY':[0,1],'BOTH':[1,1]}
        if any(r['truth']!=expected[c] for c,r in group.items()):
            excluded.append(key);continue
        for h,name,comparisons in [(0,'BODY',[('NONE','BODY_ONLY'),('HEAD_ONLY','BOTH')]),
                                    (1,'HEAD',[('NONE','HEAD_ONLY'),('BODY_ONLY','BOTH')])]:
            for low,high in comparisons:
                a,b=group[low],group[high]
                if a['truth'][h]!=0 or b['truth'][h]!=1:continue
                s=scores[name];s['eligible_pairs']+=1
                s['correct_pairs']+=int(a['prediction'][h]==0 and b['prediction'][h]==1)
                s['probability_delta_sum']+=b['probability'][h]-a['probability'][h]
    for s in scores.values():
        n=s['eligible_pairs'];s['accuracy']=s['correct_pairs']/n if n else None
        s['mean_probability_delta']=s.pop('probability_delta_sum')/n if n else None
    return dict(scores=scores,excluded_quartets=len(excluded),excluded_ids=excluded,
                scope='Paired intervention responsiveness; not unique proof of geometric reasoning')


if __name__=='__main__':
    import argparse
    import json
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    compile_parser=sub.add_parser('compile')
    compile_parser.add_argument('--protocol',type=Path,default=Path('experiments/city-field/crossregion-v1.json'))
    compile_parser.add_argument('--source',type=Path,required=True)
    compile_parser.add_argument('--template',type=Path,required=True)
    compile_parser.add_argument('--canary',action='store_true')
    compile_parser.add_argument('--output',type=Path,required=True)
    score_parser=sub.add_parser('score')
    score_parser.add_argument('--rows',type=Path,required=True)
    score_parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists() or not out.is_relative_to((Path(__file__).resolve().parents[1]/'artifacts.local').resolve()):
        raise ValueError('Fresh canonical output required')
    def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
    if args.command=='compile':result=compile_design(load(args.protocol),load(args.source),load(args.template),args.canary)
    else:
        rows=load(args.rows);result=causal_flip(rows)
        result['by_region']={r:causal_flip([x for x in rows if x['region_id']==r]) for r in sorted({x['region_id'] for x in rows})}
        result['region_macro_accuracy']={h:(sum(values)/len(values) if values and all(v is not None for v in values) else None) for h in ('BODY','HEAD')
            for values in [[r['scores'][h]['accuracy'] for r in result['by_region'].values()]]}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8',newline='\n')
    print(out)
