"""Development observations and separate physical-surface targets; no clip labels.

Six closed camera-frame corridor boxes use exact triangle/AABB separating axes.
Negatives require authenticated loaded-world bounds and query coverage; missing
physical geometry overlapping a query is UNKNOWN (-1), never a negative.
Uniform rho=.5/incidence=1 are uncalibrated forward-proxy assumptions.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
from cnh_street_development_baseline import load_depth, sample_depth
from cnh_route_sensor import synthesize_response, derive_readout, H3

QUERY_NAMES = ['left_HEAD','left_BODY','centre_HEAD','centre_BODY','right_HEAD','right_BODY']
BOXES = np.array([[[x-.3,y[0],.3],[x+.3,y[1],3.]] for x in (-.3,0.,.3)
                  for y in ((-.2,.42),(.42,.9))],dtype=float)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def triangle_box_hits(triangles, low, high):
    """Closed 13-axis triangle/AABB SAT, including face/edge crossing contacts."""
    triangles=np.asarray(triangles,dtype=float).reshape(-1,3,3)
    if not np.isfinite(triangles).all():raise ValueError('Nonfinite physical triangles')
    low=np.asarray(low);high=np.asarray(high)
    selected=np.all(triangles.max(1)>=low,axis=1)&np.all(triangles.min(1)<=high,axis=1)
    candidates=triangles[selected]-(low+high)/2
    if not len(candidates):return False
    half=(high-low)/2
    edges=np.roll(candidates,-1,axis=1)-candidates
    axes=[np.cross(edges[:,0],edges[:,1])]
    axes.extend(np.cross(edges[:,edge],axis) for edge in range(3) for axis in np.eye(3))
    survives=np.ones(len(candidates),dtype=bool)
    for axis in axes:
        projection=np.einsum('tvc,tc->tv',candidates,axis)
        radius=np.abs(axis)@half
        survives &= (projection.min(1)<=radius+1e-10)&(projection.max(1)>=-radius-1e-10)
    return bool(survives.any())


def world_query_bounds(matrix):
    result=[]
    for low,high in BOXES:
        corners=np.asarray(list(itertools.product(*zip(low,high))))@matrix[:3,:3].T+matrix[:3,3]
        result.append((corners.min(0),corners.max(0),corners))
    return result


def physical_labels(native, inserted, matrix, complete, probe_center, probe_radius, missing_bounds=()):
    """Positive witnesses survive missing coverage; absence needs known coverage."""
    matrix=np.asarray(matrix,dtype=float)
    if matrix.shape!=(4,4) or not np.isfinite(matrix).all() or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-6):
        raise ValueError('Rigid camera transform required')
    triangles=np.concatenate([native,inserted],axis=0)
    local=(triangles-matrix[:3,3])@matrix[:3,:3]
    labels=[];reasons=[]
    for (low,high),(world_low,world_high,corners) in zip(BOXES,world_query_bounds(matrix)):
        if triangle_box_hits(local,low,high):
            labels.append(1);reasons.append('EXPORTED_PHYSICAL_TRIANGLE_CONTACT');continue
        covered=complete and np.all(np.linalg.norm(corners-probe_center,axis=1)<=probe_radius+1e-8)
        uncertain=any(np.all(np.asarray(b)>=world_low)&np.all(np.asarray(a)<=world_high) for a,b in missing_bounds)
        labels.append(0 if covered and not uncertain else -1)
        reasons.append('LOADED_WORLD_COVERED_NO_SURFACE_CONTACT' if covered and not uncertain else 'UNKNOWN_PHYSICAL_COVERAGE')
    return np.asarray(labels,dtype=np.int8),reasons


def frame_identity(manifest_hash,row,camera_hash,depth_hash):
    payload=json.dumps([manifest_hash,row['id'],camera_hash,depth_hash],separators=(',',':')).encode()
    identity=hashlib.sha256(payload).hexdigest()
    return identity,int(identity[:16],16)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def materialize(capture,output,sensor_backend='numpy',device='cpu'):
    from cnh_route_scene_depth_audit import world_triangles
    from cnh_route_insert_geometry_audit import world_triangles as inserted_triangles
    capture=Path(capture).resolve(strict=True);output=Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Fresh or runtime-precreated empty output directory required')
    if output.is_relative_to(capture):raise ValueError('Keep outputs separate from source capture')
    manifest=read(capture/'raw-manifest.json'); spec=read(capture/'source/spec.json'); transport=read(capture/'format-receipt.json')
    rows=manifest['frames'];layout_ids=[r['layout_id'] for r in spec['layouts']]
    if spec.get('scope')!='STREET_DEVELOPMENT_PILOT_NOT_BENCHMARK' or not rows:
        raise ValueError('Explicit nonempty Development capture required')
    if transport.get('status')!='PASS_SEVEN_PASS_SOURCE_TRANSPORT_ONLY' or transport.get('frame_count')!=len(rows):
        raise ValueError('Completed transport frame count mismatch')
    identity=lambda r:(r['id'],r['layout_id'],r['clip_id'],r['pose_index'])
    if len({identity(r) for r in rows})!=len(rows) or [identity(r) for r in transport['frames']]!=[identity(r) for r in rows]:
        raise ValueError('Transport frame identity mismatch')
    selected=read(capture/'candidate-selection.json')['selected_layouts']
    expected={(l['layout_id'],c['id'],i) for l in selected if l for c in l['clips'] for i in range(len(c['poses']))}
    if len(expected)!=len(rows) or {(r['layout_id'],r['clip_id'],r['pose_index']) for r in rows}!=expected:
        raise ValueError('Full frozen selected-layout membership mismatch')
    if any(r.get('data_role')!='Development' or str(r.get('split','')).lower() in ('test','locked_test','blind') for r in rows):
        raise ValueError('Only Development frames supported')
    source=read(capture/'source-receipt.json');clearance=read(capture/'clearance.json')
    hidden_actors=set(source.get('hidden_native_hlod_actors',[]))
    hidden_components={r['component'] for r in clearance.get('excluded_primitives',[]) if r.get('reason')=='HIDDEN_OR_INVISIBLE_IN_FIXED_SETTLED_CAPTURE'}
    native={}; geometry_receipts=[]
    for index,lid in enumerate(layout_ids):
        folder=capture/'evaluator'/f'layout-{index:02d}';probe=read(folder/'scene-probe.json')
        probe=dict(probe,instances=[r for r in probe['instances'] if r['actor_path'] not in hidden_actors and r['component_path'] not in hidden_components])
        # No DepthSurfaceFilter: glass belongs to physical occupancy.
        triangles,_,_,missing=world_triangles(folder,probe)
        missing_ids={r['id'] for r in missing};bounds=[];unbounded=False
        for row in probe['instances']:
            if row['id'] in missing_ids:
                if 'bounds_min_m' in row and 'bounds_max_m' in row:bounds.append((row['bounds_min_m'],row['bounds_max_m']))
                else:unbounded=True
        # Probe exports are local; clearance inventories the entire loaded world.
        # A bounded but unexported native component must not silently disappear.
        exported_components={r['component_path']+':'+str(r.get('instance_index')) for r in probe['instances']
            if r['mesh'].get('status')=='EXPORTED_LOD0_NOT_RENDER_VERIFIED'}
        inventories=[c['manifest']['native_entities'] for c in clearance.get('candidates',[]) if c.get('layout_id')==lid]
        if not inventories:unbounded=True
        for entity in inventories[0] if inventories else []:
            if entity['id'] not in exported_components:
                if entity.get('bounds_conservative') is True and entity.get('deformation_bounded') is True:
                    bounds.append((entity['bounds_min_m'],entity['bounds_max_m']))
                else:unbounded=True
        complete=clearance.get('loaded_world_coverage_complete') is True and not unbounded
        native[lid]=(triangles,complete,np.asarray(probe['camera_m']),probe['radius_m'],bounds)
        geometry_receipts.append(dict(layout_id=lid,triangles=len(triangles),missing_exports=missing,
            loaded_world_bounds_complete=complete,probe_sha256=sha(folder/'scene-probe.json')))
    output.mkdir(parents=True, exist_ok=True)
    started=time.monotonic();manifest_hash=sha(capture/'raw-manifest.json');count=len(rows)
    histogram=np.empty((count,8,8,16),np.float32);ambient=np.empty((count,8,8),np.float32)
    distance=np.empty((count,8,8),np.float32);status=np.empty((count,8,8),np.uint8)
    labels=np.empty((count,6),np.int8);keys=[];metadata=[]
    try:
        for index,row in enumerate(rows):
            folder=(capture/row['folder']).resolve(strict=True)
            if not folder.is_relative_to(capture):raise ValueError('Frame folder escapes capture')
            camera=read(folder/'camera.json');insertions=read(folder/'inserted-geometry.json')
            depth,depth_path=load_depth(folder)
            camera_hash=sha(folder/'camera.json');depth_hash=sha(depth_path)
            key,seed=frame_identity(manifest_hash,row,camera_hash,depth_hash)
            radial,weights=sample_depth(depth,camera,16)
            response=synthesize_response(radial,.5,1.,weights,seed=seed,backend=sensor_backend,device=device)
            observed=derive_readout(response,H3)
            histogram[index]=observed['histogram'];ambient[index]=observed['ambient'];distance[index]=observed['distance_m'];status[index]=observed['status']
            chunks=[]
            for instance in insertions['instances']:
                if instance.get('hidden') is False:chunks.append(inserted_triangles(instance))
                elif instance.get('hidden') is not True:raise ValueError('Explicit inserted visibility required')
            inserted=np.concatenate(chunks) if chunks else np.empty((0,3,3))
            tri,complete,center,radius,bounds=native[row['layout_id']]
            labels[index],reasons=physical_labels(tri,inserted,np.asarray(camera['T_world_camera']),complete,center,radius,bounds)
            keys.append(key)
            metadata.append({k:row.get(k) for k in ('id','layout_id','clip_id','pose_index','machine_id','environment_category','physical_site_id','nominal_time_s')}|
                dict(frame_key=key,data_role='Development',seed=seed,camera_sha256=camera_hash,depth_sha256=depth_hash,
                    depth_valid_sha256=sha(folder/'depth_left_valid.npy'),inserted_geometry_sha256=sha(folder/'inserted-geometry.json'),label_reasons=reasons))
            if (index+1)%160==0:print(json.dumps(dict(materialized=index+1,total=count,wall_s=time.monotonic()-started)),flush=True)
        key_array=np.asarray(keys,dtype='U64')
        np.savez_compressed(output/'observations.npz',histogram=histogram,ambient=ambient,distance_m=distance,
            valid=status==5,status=status,frame_key=key_array,bin_centers_m=observed['bin_centers_m'].astype(np.float32))
        np.savez_compressed(output/'targets.npz',labels=labels,frame_key=key_array)
        result=dict(schema='cnh-street-e2e-materialize-v1',status='PASS_DEVELOPMENT_MATERIALIZATION_ONLY',
            count=count,query_names=QUERY_NAMES,query_boxes_camera_m=BOXES.tolist(),source_manifest_sha256=manifest_hash,
            frames=metadata,native_geometry=geometry_receipts,
            assumptions=dict(rho=.5,incidence_cos=1.,sensor='H3_UNCALIBRATED_STATIC_EXPOSURE_PROXY',samples_per_axis=16),
            label_authority='EXACT_CLOSED_TRIANGLE_BOX_EXPORTED_PHYSICAL_SURFACE_OCCUPANCY_LOADED_WORLD_ONLY',
            unknown=-1,independent_label_precision='NOT_ESTABLISHED',formal_eligible=False)
        (output/'manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        receipt=dict(status=result['status'],count=count,wall_s=time.monotonic()-started,sensor_backend=sensor_backend,device=device,
            label_backend='NUMPY_VECTORIZED_SAT_WITH_AABB_PREFILTER',
            positive_per_query=(labels==1).sum(0).tolist(),negative_per_query=(labels==0).sum(0).tolist(),unknown_per_query=(labels==-1).sum(0).tolist(),
            code_sha256=sha(__file__),outputs={name:sha(output/name) for name in ('observations.npz','targets.npz','manifest.json')})
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        return receipt
    except Exception as error:
        (output/'failure.json').write_text(json.dumps(dict(status='FAIL',error=str(error),completed_frames=len(keys))),encoding='utf-8')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--sensor-backend',choices=('numpy','torch'),default='numpy');parser.add_argument('--device',default='cpu')
    args=parser.parse_args();print(json.dumps(materialize(args.capture,args.output,args.sensor_backend,args.device),indent=2))
