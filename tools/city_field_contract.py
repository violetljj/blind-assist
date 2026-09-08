"""CPU-only City collection-field contract and native capture ingestion.

Plan schema ``city-collection-field-v1``:
  required_scene_types_per_region: sidewalk, intersection, narrow_passage
  separation: min_region_gap_m, min_route_gap_cross_split_m, adjacent_frame_gap_m
  regions: [{region_id, split, bounds_xy_m:[xmin,ymin,xmax,ymax]}]
  routes: [{route_id, region_id, split, scene_type, waypoints:[{camera:{x,y,z,
           yaw,pitch,roll}, pair_id?, variant?}], instance_ids:[], pair_groups:[]}]
Splits are train/dev/test. Exact target instance IDs, not shared background IDs,
are split-isolated. Each split must contain every required scene type for READY.
Missing coverage/waypoints is REVIEW, permitting a CANDIDATE scouting plan.

``ingest_native_bundle(plan,capture,route_id,labels=None,assignments=None)`` reads
only current native capture artifacts. Optional assignments is a list aligned to
dataset frames, or an original-sample-index map, containing pair_id/variant;
otherwise these are read from case top-level fields, case.field or waypoint.
Multi-route capture cases are filtered by route_id, preserving sample_index.
Pairs require clear plus an obstacle variant; named hazard variants are allowed.
Controlled identities must be listed in route.instance_ids; inactive declared
targets are NOT_PRESENT only when explicitly absent or in a clear fixture case.
This is fixture absence, not whole-scene free space. It returns a serializable
``city-field-bundle-v1`` inventory; it does not write, alter labels or run models.
Each frame records RGB/depth hashes, same-index pair-export synchronization,
static calibration contract, commanded pose and optional engine RGB/depth poses,
per-case floor-probe diagnostic, requested-asset readiness, declared exact target
identity/relative geometry, and UNKNOWN-aware label counts. Native depth is axial
metres, PNG is RGB, poses are UE world metres/degrees, +X forward/+Y right/+Z up.

``validate_plan``, ``validate_bundle`` and ``validate_bundles`` return structured
issues without silently dropping inputs. FAIL=contradictory contract/leakage;
INCOMPLETE=missing frame/asset/sync; REVIEW=missing readiness/labels or UNKNOWN;
PASS=complete admissible inventory, not universal geometric truth. Collection
coverage is checked only with validate_bundles(...,require_coverage=True).
"""
import hashlib
import json
import math
from pathlib import Path

SPLITS = ('train', 'dev', 'test')
SCENE_TYPES = ('sidewalk', 'intersection', 'narrow_passage')
POSE_KEYS = ('x', 'y', 'z', 'yaw', 'pitch', 'roll')
FLOOR_POLICY = dict(review_deviation_m=.02,severe_deviation_m=.5,probe_xy_tolerance_m=.05)


def limits(plan):
    """Canonical field keys; accept earlier prototype spellings for callers."""
    raw=plan.get('separation',{})
    return dict(minimum_region_gap_m=raw.get('min_region_gap_m',raw.get('minimum_region_gap_m')),
        minimum_route_gap_m=raw.get('min_route_gap_cross_split_m',raw.get('minimum_route_gap_m')),
        adjacent_frame_gap_m=raw.get('adjacent_frame_gap_m'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def identity(value):
    """A declared stable string or exact actor/component/instance tuple."""
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict) and all(k in value for k in ('actor_path', 'component_path', 'instance_index')):
        return json.dumps([value['actor_path'], value['component_path'], value['instance_index']], separators=(',', ':'))
    raise ValueError('Exact target identity requires stable ID or actor/component/instance')


def issue(issues, status, code, where, detail):
    issues.append(dict(status=status, code=code, where=str(where), detail=detail))


def finish(issues, **data):
    status = next((s for s in ('FAIL', 'INCOMPLETE', 'REVIEW') if any(i['status'] == s for i in issues)), 'PASS')
    return dict(status=status, issues=issues, **data)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def pose(value):
    return isinstance(value, dict) and all(finite(value.get(k)) for k in POSE_KEYS)


def distance(a, b):
    return math.hypot(a['x'] - b['x'], a['y'] - b['y'])


def box_gap(a, b):
    return math.hypot(max(a[0]-b[2], b[0]-a[2], 0), max(a[1]-b[3], b[1]-a[3], 0))


def inside(camera, bounds):
    return bounds[0] <= camera['x'] <= bounds[2] and bounds[1] <= camera['y'] <= bounds[3]


def pose_delta(a,b):
    return dict(translation_m=math.sqrt(sum((a[k]-b[k])**2 for k in ('x','y','z'))),
        rotation_degrees=max(abs((a[k]-b[k]+180)%360-180) for k in ('yaw','pitch','roll')))


def capture_pose_evidence(commanded,reading,required=False):
    """Prefer engine RGB pose; do not call a commanded fallback measured."""
    rgb=reading.get('rgb') if reading else None
    depth=reading.get('depth') if reading else None
    result=dict(status='NOMINAL_ONLY',required=required,commanded=commanded,
        rgb_actual=rgb,depth_actual=depth,authority='COMMANDED_SPEC_NOT_INDEPENDENTLY_MEASURED')
    if not reading: return commanded,result
    if not pose(rgb) or not pose(depth):
        result.update(status='ANOMALY',reason='Malformed/missing engine capture pose')
        return rgb if pose(rgb) else commanded,result
    pair=pose_delta(rgb,depth)
    mismatch=pair['translation_m']>.001+1e-9 or pair['rotation_degrees']>.01+1e-9
    command=pose_delta(rgb,commanded) if pose(commanded) else None
    result.update(status='ANOMALY' if mismatch else 'MEASURED',pair_delta=pair,
        command_delta=command,authority=reading.get('authority','ENGINE_TRANSFORMS_UNSPECIFIED_AUTHORITY'),
        command_mismatch=command is not None and (command['translation_m']>.001+1e-9 or command['rotation_degrees']>.01+1e-9))
    if mismatch: result['reason']='RGB/depth poses differ beyond 1mm or 0.01degree'
    return rgb,result


def floor_evidence(camera,nominal,probe,policy=None):
    """Collision floor diagnostic; never move the camera or rewrite labels.

    Default >2cm is REVIEW, >=0.5m mismatch or camera at/below the measured
    surface is ANOMALY. A collision hit is not a universal walkable-floor claim.
    Missing case nominal is reported, never reconstructed from camera height.
    """
    policy=dict(FLOOR_POLICY,**(policy or {}))
    if not all(finite(v) and v>0 for v in policy.values()) or policy['severe_deviation_m']<=policy['review_deviation_m']:
        raise ValueError('Invalid floor diagnostic policy')
    result=dict(status='UNKNOWN',nominal_floor_z_m=nominal if finite(nominal) else None,
        measured_floor_z_m=None,deviation_m=None,absolute_deviation_m=None,probe=probe or {},policy=policy,
        authority='NATIVE_DOWNWARD_COLLISION_HIT_NOT_UNIVERSAL_WALKABLE_FLOOR')
    point=(probe or {}).get('point_m')
    if (probe or {}).get('hit') is not True or not isinstance(point,list) or len(point)!=3 or not all(finite(v) for v in point) or not pose(camera):
        result['reason']='Missing/unreliable collision hit or camera pose';return result
    result.update(measured_floor_z_m=point[2],probe_xy_distance_m=math.hypot(point[0]-camera['x'],point[1]-camera['y']),
        camera_clearance_m=camera['z']-point[2])
    if finite(nominal):
        result.update(deviation_m=point[2]-nominal,absolute_deviation_m=abs(point[2]-nominal))
    if result['probe_xy_distance_m']>policy['probe_xy_tolerance_m']:
        result['reason']='Collision point not beneath camera within declared tolerance';return result
    if result['camera_clearance_m']<=1e-6:
        result.update(status='ANOMALY',reason='Camera at/below measured collision surface');return result
    if not finite(nominal):
        result['reason']='No per-case nominal floor declaration; measured hit retained';return result
    deviation=point[2]-nominal
    result.update(deviation_m=deviation,absolute_deviation_m=abs(deviation))
    if abs(deviation)>=policy['severe_deviation_m']:
        result.update(status='ANOMALY',reason='Severe nominal/measured floor mismatch')
    elif abs(deviation)>policy['review_deviation_m']+1e-9:
        result.update(status='REVIEW',reason='Floor deviation exceeds review tolerance; no automatic height correction')
    else: result.update(status='CONSISTENT')
    return result


def validate_plan(plan):
    issues, regions, routes, owners = [], {}, {}, {}
    if plan.get('schema') != 'city-collection-field-v1':
        issue(issues,'FAIL','PLAN_SCHEMA','plan','Expected city-collection-field-v1')
    required = plan.get('required_scene_types_per_region', plan.get('required_scene_types', []))
    if set(required) != set(SCENE_TYPES) or len(required) != 3:
        issue(issues,'FAIL','SCENE_TYPES','plan','Declare exactly sidewalk/intersection/narrow_passage')
    separation = limits(plan)
    for key in ('minimum_region_gap_m','minimum_route_gap_m','adjacent_frame_gap_m'):
        if not finite(separation.get(key)) or separation[key] <= 0:
            issue(issues,'FAIL','SEPARATION_DECLARATION',key,'Require positive finite metres; no implicit separation')
    for region in plan.get('regions', []):
        rid = region.get('region_id')
        bounds = region.get('bounds_xy_m', [])
        if not rid or rid in regions or region.get('split') not in SPLITS:
            issue(issues,'FAIL','REGION_ID_SPLIT',rid,'Unique region and valid split required')
            continue
        if not isinstance(bounds,list) or len(bounds)!=4 or not all(finite(v) for v in bounds) or not (bounds[0]<bounds[2] and bounds[1]<bounds[3]):
            issue(issues,'FAIL','REGION_BOUNDS',rid,'Require nonempty measured XY rectangle')
            continue
        regions[rid] = region
    coverage = {s:set() for s in SPLITS}
    for route in plan.get('routes', []):
        rid, split = route.get('route_id'), route.get('split')
        region = regions.get(route.get('region_id'))
        if not rid or rid in routes or split not in SPLITS or region is None or region['split'] != split:
            issue(issues,'FAIL','ROUTE_ID_REGION_SPLIT',rid,'Unique route with matching region/split required')
            continue
        if route.get('scene_type') not in SCENE_TYPES:
            issue(issues,'FAIL','ROUTE_SCENE_TYPE',rid,'Unknown scene type')
        else:
            coverage[split].add(route['scene_type'])
        routes[rid] = route
        points = route.get('waypoints', [])
        if not points:
            issue(issues,'REVIEW','ROUTE_NOT_SCOUTED',rid,'No measured waypoints yet')
        for i, point in enumerate(points):
            camera = point.get('camera')
            if not pose(camera):
                issue(issues,'FAIL','WAYPOINT_POSE',f'{rid}/{i}','Require finite six-axis camera pose')
            elif not inside(camera, region['bounds_xy_m']):
                issue(issues,'FAIL','WAYPOINT_OUTSIDE_REGION',f'{rid}/{i}','Camera lies outside declared region bounds')
        for kind, values in (('instance', route.get('instance_ids', [])), ('pair', route.get('pair_groups', []))):
            for value in values:
                try:
                    token = identity(value)
                except ValueError as error:
                    issue(issues,'FAIL','TARGET_IDENTITY',rid,str(error)); continue
                key = (kind,token)
                if key in owners and owners[key] != split:
                    issue(issues,'FAIL','CROSS_SPLIT_'+kind.upper(),rid,'Target instance or paired assignment shared across splits')
                owners[key] = split
    for split in SPLITS:
        missing = sorted(set(SCENE_TYPES)-coverage[split])
        if missing:
            issue(issues,'REVIEW','SPLIT_SCENE_COVERAGE',split,'Missing scene types: '+','.join(missing))
    for rid in regions:
        missing=set(SCENE_TYPES)-{r['scene_type'] for r in routes.values() if r['region_id']==rid}
        if missing:
            issue(issues,'REVIEW','REGION_SCENE_COVERAGE',rid,'Missing scene types: '+','.join(sorted(missing)))
    region_list = list(regions.values())
    for i,a in enumerate(region_list):
        for b in region_list[i+1:]:
            minimum = separation.get('minimum_region_gap_m')
            if a['split'] != b['split'] and finite(minimum) and box_gap(a['bounds_xy_m'],b['bounds_xy_m']) < minimum:
                issue(issues,'FAIL','REGION_GAP',a['region_id']+'/'+b['region_id'],'Measured region bounds violate declared separation')
    route_list = list(routes.values())
    for i,a in enumerate(route_list):
        for b in route_list[i+1:]:
            if a['split'] == b['split']: continue
            ap = [p['camera'] for p in a.get('waypoints',[]) if pose(p.get('camera'))]
            bp = [p['camera'] for p in b.get('waypoints',[]) if pose(p.get('camera'))]
            minimum = separation.get('minimum_route_gap_m')
            if ap and bp and finite(minimum) and min(distance(x,y) for x in ap for y in bp) < minimum:
                issue(issues,'FAIL','ROUTE_GAP',a['route_id']+'/'+b['route_id'],'Measured camera waypoints violate declared cross-split route gap')
    result = finish(issues,coverage={s:sorted(v) for s,v in coverage.items()},region_count=len(regions),route_count=len(routes))
    result['readiness'] = 'READY' if result['status']=='PASS' else 'CANDIDATE'
    return result


def validate_bundles(plan, bundles, require_coverage=False):
    issues = list(validate_plan(plan)['issues'])
    routes = {r['route_id']:r for r in plan.get('routes',[]) if r.get('route_id')}
    regions = {r['region_id']:r for r in plan.get('regions',[]) if r.get('region_id')}
    frames, hashes, pairs, instances, sample_ids = [], {}, {}, {}, set()
    coverage = {s:set() for s in SPLITS}
    for bundle in bundles:
        bid = bundle.get('bundle_id','unknown')
        r = routes.get(bundle.get('route_id'))
        if bundle.get('schema')!='city-field-bundle-v1' or r is None:
            issue(issues,'FAIL','BUNDLE_SCHEMA_ROUTE',bid,'Unknown bundle schema/route'); continue
        split = bundle.get('split')
        if split != r['split'] or bundle.get('region_id') != r['region_id'] or bundle.get('scene_type') != r['scene_type']:
            issue(issues,'FAIL','BUNDLE_ASSIGNMENT',bid,'Bundle must match route split, region and scene type'); continue
        coverage[split].add(r['scene_type'])
        source = bundle.get('source', {})
        if source.get('source_unchanged') is not True:
            issue(issues,'FAIL','SOURCE_CHANGED_OR_UNVERIFIED',bid,'Capture must attest unchanged source')
        if source.get('status') != 'PASS':
            issue(issues,'INCOMPLETE','CAPTURE_INCOMPLETE',bid,'Capture receipt did not PASS')
        if source.get('load_status') != 'READY':
            issue(issues,'REVIEW','SOURCE_LOAD_UNVERIFIED',bid,'Requested source load not READY; not complete-world proof')
        actual = bundle.get('frames',[])
        if not actual or len(actual) != bundle.get('expected_frame_count'):
            issue(issues,'INCOMPLETE','MISSING_FRAME',bid,'Expected frame count differs from inventory')
        for frame in actual:
            where = f"{bid}/{frame.get('sample_index')}"
            uid=(bid,frame.get('sample_index'))
            if uid in sample_ids:
                issue(issues,'FAIL','DUPLICATE_FRAME_ID',where,'Repeated sample index')
            sample_ids.add(uid)
            for kind in ('rgb','depth'):
                asset=frame.get(kind,{})
                if asset.get('status')!='PRESENT' or not asset.get('sha256'):
                    issue(issues,'INCOMPLETE','MISSING_'+kind.upper(),where,'Missing asset or digest')
                if asset.get('status')=='PRESENT' and asset.get('valid') is False:
                    issue(issues,'FAIL','INVALID_'+kind.upper(),where,'Invalid decoded dimensions/values')
                if asset.get('path'):
                    path=Path(asset['path'])
                    if not path.is_file():
                        issue(issues,'INCOMPLETE','MISSING_'+kind.upper(),where,'Inventoried payload no longer exists')
                    elif asset.get('sha256') and sha(path)!=asset['sha256']:
                        issue(issues,'FAIL','CHANGED_'+kind.upper(),where,'Payload bytes changed after ingestion')
            if frame.get('synchronization',{}).get('status')!='PAIRED':
                issue(issues,'INCOMPLETE','UNSYNCHRONIZED_RGB_DEPTH',where,'No matching completed export pair')
            cal=frame.get('calibration',{})
            if not (isinstance(cal.get('width'),int) and cal['width']>0 and isinstance(cal.get('height'),int) and cal['height']>0
                    and finite(cal.get('horizontal_fov_degrees')) and 0<cal['horizontal_fov_degrees']<180
                    and finite(cal.get('depth_max_m')) and cal['depth_max_m']>0):
                issue(issues,'FAIL','CALIBRATION',where,'Missing/invalid shared calibration')
            camera=frame.get('camera')
            if not pose(camera):
                issue(issues,'FAIL','POSE',where,'Missing/invalid actual camera pose')
            else:
                region=regions.get(r['region_id'],{})
                if 'bounds_xy_m' in region and not inside(camera,region['bounds_xy_m']):
                    issue(issues,'FAIL','FRAME_OUTSIDE_REGION',where,'Actual pose violates declared region')
                frames.append((split,r['route_id'],where,camera))
            pe=frame.get('pose_evidence',{})
            if pe.get('status')=='ANOMALY':
                issue(issues,'FAIL','CAPTURE_POSE_ANOMALY',where,pe.get('reason','RGB/depth pose mismatch'))
            elif pe.get('required') and pe.get('status')!='MEASURED':
                issue(issues,'REVIEW','POSE_NOMINAL_ONLY',where,'Commanded pose fallback; engine pose not measured')
            if pe.get('command_mismatch'):
                issue(issues,'REVIEW','COMMAND_ACTUAL_POSE_MISMATCH',where,'Actual RGB pose differs from command; actual retained')
            floor=frame.get('floor_probe',{})
            if floor.get('status')=='ANOMALY':
                issue(issues,'FAIL','FLOOR_ANOMALY',where,floor.get('reason','Severe floor geometry mismatch'))
            elif floor and floor.get('status')!='CONSISTENT':
                issue(issues,'REVIEW','FLOOR_REVIEW',where,floor.get('reason','Floor evidence incomplete'))
            if frame.get('source_load_status')!='READY':
                issue(issues,'REVIEW','FRAME_SOURCE_LOAD',where,'Per-view requested-asset readiness missing or not READY')
            label=frame.get('labels',{})
            assignment=frame.get('assignment',{})
            if any(assignment.get(k,r[k])!=r[k] for k in ('route_id','region_id','split','scene_type')):
                issue(issues,'FAIL','FRAME_ASSIGNMENT',where,'Per-case assignment differs from bundle route')
            values=label.get('near')
            if values is not None and (not isinstance(values,list) or len(values)!=2 or any(v not in (-1,0,1) for v in values)):
                issue(issues,'FAIL','LABEL_VALUES',where,'Near labels must retain -1/0/1')
            if label.get('status')=='KNOWN' and (values is None or -1 in values or any(label.get('support_unknown_cells',[]))):
                issue(issues,'FAIL','UNKNOWN_ERASURE',where,'KNOWN status contradicts missing/UNKNOWN supervision')
            if label.get('status')=='ANOMALY':
                issue(issues,'FAIL','LABEL_ANOMALY',where,label.get('reason','Invalid label payload'))
            elif label.get('status')!='KNOWN':
                issue(issues,'REVIEW','LABEL_UNKNOWN',where,'Missing/UNKNOWN labels preserved; not negative evidence')
            digest=frame.get('rgb',{}).get('sha256')
            if digest:
                if digest in hashes and hashes[digest]!=split:
                    issue(issues,'FAIL','CROSS_SPLIT_DUPLICATE_RGB',where,'Identical RGB across splits')
                hashes[digest]=split
            pair=frame.get('pair_id')
            if pair:
                if pair not in r.get('pair_groups',[]):
                    issue(issues,'FAIL','UNDECLARED_PAIR',where,'Pair must be declared on assigned route')
                if pair in pairs and pairs[pair]['split']!=split:
                    issue(issues,'FAIL','CROSS_SPLIT_PAIR',where,'Clear/obstacle pair crosses split')
                entry=pairs.setdefault(pair,dict(split=split,variants=set()))
                entry['variants'].add(frame.get('variant'))
            declared={identity(v) for v in r.get('instance_ids',[])}
            for target in frame.get('targets',[]):
                try: token=identity(target.get('identity'))
                except ValueError as error:
                    issue(issues,'FAIL','TARGET_IDENTITY',where,str(error)); continue
                if token not in declared:
                    issue(issues,'FAIL','UNDECLARED_TARGET',where,'Exact target is not assigned to this route')
                if token in instances and instances[token]!=split:
                    issue(issues,'FAIL','CROSS_SPLIT_INSTANCE',where,'Exact target instance shared across splits')
                instances[token]=split
                if target.get('label_status')=='NOT_PRESENT' and target.get('source_kind')=='CONTROLLED_SYNTHETIC_FIXTURE':
                    # Configured fixture absence is not whole-scene free space.
                    continue
                geometry=target.get('relative_geometry',{})
                if target.get('source_kind')=='CONTROLLED_SYNTHETIC_FIXTURE' and not target.get('runtime_binding'):
                    issue(issues,'REVIEW','CONTROLLED_BINDING_UNKNOWN',where,'Configured synthetic identity lacks measured runtime binding')
                if not all(finite(geometry.get(k)) for k in ('forward_m','right_m','height_from_camera_m','horizontal_distance_m')):
                    issue(issues,'REVIEW','TARGET_GEOMETRY_UNKNOWN',where,'Missing relative geometry must remain UNKNOWN')
                if target.get('label_status')!='EVALUABLE':
                    issue(issues,'REVIEW','TARGET_LABEL_UNKNOWN',where,'Target missing/unreliable geometry excluded from target negatives/misses')
    for i,(split,route_id,where,camera) in enumerate(frames):
        for other_split,other_route,other_where,other_camera in frames[i+1:]:
            if split==other_split: continue
            gap=distance(camera,other_camera)
            separation=limits(plan)
            adjacent=separation.get('adjacent_frame_gap_m')
            minimum=separation.get('minimum_route_gap_m')
            if finite(adjacent) and gap<=adjacent:
                issue(issues,'FAIL','CROSS_SPLIT_ADJACENT_FRAMES',where+'/'+other_where,f'Actual XY camera gap {gap:.6f}m')
            elif finite(minimum) and gap<minimum:
                issue(issues,'FAIL','ACTUAL_ROUTE_GAP',where+'/'+other_where,f'Actual XY camera gap {gap:.6f}m below route separation')
    for pair,entry in pairs.items():
        if 'clear' not in entry['variants'] or not (entry['variants']-{None,'clear'}):
            issue(issues,'INCOMPLETE','PAIR_VARIANTS',pair,'Declared pair needs clear and at least one obstacle variant in same split')
    if require_coverage:
        for split in SPLITS:
            if coverage[split]!=set(SCENE_TYPES):
                issue(issues,'INCOMPLETE','COLLECTION_COVERAGE',split,'Each split needs all three scene types')
    return finish(issues,bundle_count=len(bundles),frame_count=len(sample_ids),
        coverage={s:sorted(v) for s,v in coverage.items()},authority='Collection integrity only; UNKNOWN is not free space')


def validate_bundle(plan,bundle):
    return validate_bundles(plan,[bundle])


def _asset(path,kind,cal):
    record=dict(path=str(path),status='MISSING',valid=False)
    if not path.is_file(): return record
    record.update(status='PRESENT',sha256=sha(path))
    try:
        import numpy as np
        if kind=='rgb':
            from PIL import Image
            with Image.open(path) as im:
                record.update(shape=[im.height,im.width,3],valid=im.size==(cal['width'],cal['height']))
        else:
            value=np.load(path,mmap_mode='r',allow_pickle=False)
            record.update(shape=list(value.shape),dtype=str(value.dtype),
                valid=value.shape==(cal['height'],cal['width']) and value.dtype==np.dtype('<f4') and
                bool(np.isfinite(value).all()) and bool(((value>=0)&(value<cal['depth_max_m'])).all()))
    except (OSError,ValueError,KeyError) as error:
        record.update(valid=False,error=str(error))
    return record


def ingest_native_bundle(plan,capture,route_id,labels=None,assignments=None):
    """Read capture/spec/receipt and optional native labels; never infer truth."""
    import numpy as np
    capture=Path(capture).resolve(strict=True)
    route=next(r for r in plan['routes'] if r['route_id']==route_id)
    dataset=read(capture/'model/dataset.json'); spec=read(capture/'source/spec.json'); receipt=read(capture/'receipt.json')
    if assignments is not None and not isinstance(assignments,dict) and len(assignments)!=len(dataset['frames']):
        raise ValueError('Assignments must align exactly with dataset frames')
    cal=dataset['calibration']
    cal=dict(cal,depth_convention='AXIAL_METRES',pose_convention='UE_WORLD_METRES_DEGREES_X_FORWARD_Y_RIGHT_Z_UP',
        provenance='STATIC_CAPTURE_CONFIGURATION_NOT_INDEPENDENT_CAMERA_CALIBRATION')
    exports=receipt.get('pair_exports',{})
    export_ids={r['sample_index'] for r in exports.get('rows',[]) if r.get('mode') in ('gpu_async','gpu_sync','sync','cpu')}
    profile=exports.get('profile',{})
    ready={r['index']:r.get('status','UNKNOWN') for r in receipt.get('view_readiness',[])}
    floor_probes={r.get('case'):r for r in receipt.get('native_floor_probes',[])}
    capture_poses={r.get('sample_index'):r for r in receipt.get('capture_poses',[])}
    label_meta={}; near=support=None; target_arrays={}; label_error=None
    if labels is not None:
        labels=Path(labels); labels=labels/'native-route-labels.json' if labels.is_dir() else labels
        try:
            label_meta=read(labels)
            ids=[f['sample_index'] for f in dataset['frames']]
            if label_meta.get('schema')!='city-native-route-labels-v1' or label_meta.get('sample_indices')!=ids:
                raise ValueError('Label schema/order mismatch')
            provenance=label_meta.get('provenance',{})
            if provenance.get('spec_sha256')!=sha(capture/'source/spec.json') or provenance.get('receipt_sha256')!=sha(capture/'receipt.json'):
                raise ValueError('Labels belong to different capture')
            def array(relative):
                path=(labels.parent/relative).resolve(strict=True)
                if not path.is_relative_to(labels.parent.resolve()): raise ValueError('Label path escape')
                return np.load(path,mmap_mode='r',allow_pickle=False)
            near,support=array(label_meta['near']),array(label_meta['support'])
            if near.shape!=(len(ids),2) or support.shape not in ((len(ids),2,cal['height'],cal['width']),(len(ids),2,18,32)):
                raise ValueError('Native label dimensions mismatch')
            if not np.isin(near,[-1,0,1]).all() or not np.isin(support,[-1,0,1]).all():
                raise ValueError('Invalid labels; expected -1 UNKNOWN/0/1')
            for target in label_meta.get('targets',[]):
                value=array(target['masks'])
                if value.shape!=support.shape or not np.isin(value,[-1,0,1]).all():
                    raise ValueError('Target label dimensions/values mismatch')
                target_arrays[target['target_id']]=(target,value)
        except (OSError,ValueError,KeyError) as error: label_error=str(error)
    has_route_assignments=any(c.get('route_id') or c.get('field',{}).get('route_id') for c in spec['cases'])
    selected_case_ids={i for i,c in enumerate(spec['cases']) if not has_route_assignments or c.get('route_id',c.get('field',{}).get('route_id'))==route_id}
    result=dict(schema='city-field-bundle-v1',bundle_id=capture.name+'/'+route_id,route_id=route_id,region_id=route['region_id'],
        split=route['split'],scene_type=route['scene_type'],expected_frame_count=len(selected_case_ids),
        capture=str(capture),source=dict(status=receipt.get('status'),
            source_unchanged=receipt.get('source_unchanged') is True and receipt.get('spec_sha256')==sha(capture/'source/spec.json'),
            load_status=receipt.get('readiness',{}).get('status','UNKNOWN'),
            spec_sha256=sha(capture/'source/spec.json'),receipt_sha256=sha(capture/'receipt.json')),
        label_manifest_sha256=sha(labels) if labels is not None and labels.is_file() else None,frames=[])
    for position,frame in enumerate(dataset['frames']):
        sid=frame['sample_index']; case=spec['cases'][sid] if isinstance(sid,int) and 0<=sid<len(spec['cases']) else {}
        if sid not in selected_case_ids: continue
        commanded=case.get('camera',{})
        camera,pose_info=capture_pose_evidence(commanded,capture_poses.get(sid),bool(spec.get('export_controlled_targets')))
        if assignments is not None:
            assignment=assignments.get(sid,assignments.get(str(sid),{})) if isinstance(assignments,dict) else assignments[position]
        else:
            assignment=dict(route.get('waypoints',[])[position] if position<len(route.get('waypoints',[])) else {})
            assignment.update(case.get('field',{}));assignment.update({k:case[k] for k in ('route_id','region_id','split','scene_type','pair_id','variant','variant_id') if k in case})
        rgb_path=(capture/'model'/frame['rgb_path']).resolve()
        if not rgb_path.is_relative_to((capture/'model').resolve()): raise ValueError('Capture RGB path escape')
        row=dict(sample_index=sid,camera=camera,commanded_camera=commanded,depth_camera=pose_info.get('depth_actual'),
            pose_evidence=pose_info,calibration=cal,
            floor_probe=floor_evidence(camera,case.get('floor_z_m'),floor_probes.get(case.get('name')),plan.get('floor_check')),
            rgb=_asset(rgb_path,'rgb',cal),depth=_asset(capture/f'evaluator/native/{sid:04d}.npy','depth',cal),
            synchronization=dict(status='PAIRED' if sid in export_ids and profile.get('completed')==receipt.get('frame_count') and profile.get('failed',0)==0 and profile.get('pending',0)==0 else 'UNKNOWN',
                authority='Same native capture pair export/sample index; static settled view, no sensor timestamp claim'),
            source_load_status=ready.get(sid,'UNKNOWN'),pair_id=assignment.get('pair_id'),variant=assignment.get('variant',assignment.get('variant_id')),targets=[])
        row['assignment']={k:assignment.get(k,route.get(k)) for k in ('route_id','region_id','split','scene_type')}
        if label_error: row['labels']=dict(status='ANOMALY',reason=label_error)
        elif near is None: row['labels']=dict(status='UNKNOWN',reason='Labels not supplied')
        else:
            row['labels']=dict(status='UNKNOWN' if (near[position]==-1).any() or (support[position]==-1).any() else 'KNOWN',
                near=near[position].tolist(),support_positive_cells=(support[position]==1).sum(axis=(-2,-1)).tolist(),
                support_unknown_cells=(support[position]==-1).sum(axis=(-2,-1)).tolist())
            if any(near[position,h]==1 and not (support[position,h]==1).any() for h in range(2)):
                row['labels'].update(status='ANOMALY',reason='Positive near label without any positive support')
        controlled=bool(spec.get('export_controlled_targets'))
        bindings=receipt.get('controlled_target_bindings',[])
        if isinstance(bindings,dict): binding_rows=bindings.get(str(sid),bindings.get(sid,[]))
        else: binding_rows=next((b.get('targets',[]) for b in bindings if b.get('sample_index')==sid),[])
        if isinstance(binding_rows,dict): binding_rows=binding_rows.get('targets',[])
        active={o.get('instance_id'):o for o in case.get('objects',[]) if o.get('target_part')}
        for target in spec.get('native_targets',[]):
            stable=target.get('instance_id'); obj=active.get(stable,{})
            if controlled and stable not in {identity(v) for v in route.get('instance_ids',[])}: continue
            if controlled and not obj:
                tid=target.get('target_id')
                absent=tid in case.get('absent_target_ids',[]) or row['variant']=='clear'
                row['targets'].append(dict(target_id=tid,identity=stable,source_kind='CONTROLLED_SYNTHETIC_FIXTURE',
                    configured_target=target,label_status='NOT_PRESENT' if absent else 'UNKNOWN',
                    relative_geometry={},absence_scope='Configured fixture only; no whole-scene clear claim' if absent else None))
                continue
            binding=next((b for b in binding_rows if b.get('target_id')==target.get('target_id') or
                stable is not None and b.get('instance_id')==stable),{})
            relative={}; point=binding.get('position_m',obj.get('center_m',target.get('position_m')))
            if pose(camera) and isinstance(point,list) and len(point)==3 and all(finite(v) for v in point):
                dx,dy=point[0]-camera['x'],point[1]-camera['y']; yaw=math.radians(camera['yaw'])
                relative=dict(forward_m=dx*math.cos(yaw)+dy*math.sin(yaw),right_m=-dx*math.sin(yaw)+dy*math.cos(yaw),
                    height_from_camera_m=point[2]-camera['z'],horizontal_distance_m=math.hypot(dx,dy))
            tid=target.get('target_id'); entry,value=target_arrays.get(tid,({},None))
            reliable=entry.get('status')=='EVALUABLE' and bool(entry.get('authority')) and value is not None and bool((value[position]==1).any())
            row['targets'].append(dict(target_id=tid,
                identity=stable if controlled else {k:target[k] for k in ('actor_path','component_path','instance_index') if k in target},
                source_kind='CONTROLLED_SYNTHETIC_FIXTURE' if controlled else 'DECLARED_NATIVE_INSTANCE',
                runtime_binding=binding if controlled else None,
                configured_transform=obj if controlled else None,
                relative_geometry=relative,label_status='EVALUABLE' if reliable else 'UNKNOWN'))
        result['frames'].append(row)
    return result
