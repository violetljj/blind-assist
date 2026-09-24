"""Frozen bounded source-depth geometry checks; no energy/label/source admission."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

PROTOCOL=dict(schema='cnh-source-depth-grid-v1',width=640,height=360,stride=2,offset=1,
    tof_fov_deg=45.,radial_limit_m=5.,zone_sizes=[8,4],radial_p95_m=.03,radial_max_m=.075,
    per_zone_missing_fraction=.02,per_zone_extra_fraction=.02,
    angular_weight='TANGENT_PLANE_SOLID_ANGLE_JACOBIAN_ALL_RAYS_DENOMINATOR',
    thin_layer_rule='EACH_NATIVE_VISIBLE_INSERTED_ID_FRAME_HAS_SAME_OWNER_HIT_WITHIN_75MM_RADIAL',
    convergence='NOT_RUN',label_precision='NOT_RUN',provenance='NOT_RUN')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ray_grid(camera):
    if (camera['height'],camera['width'])!=(360,640):
        raise ValueError('Frozen image size differs')
    k=np.asarray(camera['K'],dtype=float)
    yy,xx=np.meshgrid(np.arange(1,360,2),np.arange(1,640,2),indexing='ij')
    x=(xx.ravel()-k[0,2])/k[0,0];y=(yy.ravel()-k[1,2])/k[1,1]
    edge=np.tan(np.deg2rad(22.5));selected=(np.abs(x)<=edge)&(np.abs(y)<=edge)
    x=x[selected];y=y[selected]
    rays=np.stack([x,y,np.ones_like(x)],axis=-1)
    weights=np.power(1+x*x+y*y,-1.5)*(2/k[0,0])*(2/k[1,1])
    zone8=np.minimum(7,np.floor((y+edge)/(2*edge)*8).astype(int))*8+np.minimum(7,np.floor((x+edge)/(2*edge)*8).astype(int))
    return yy.ravel()[selected],xx.ravel()[selected],rays,weights,zone8


def zone_coverage(observed,predicted,norm,weights,zone8):
    native=np.isfinite(observed)&(observed>0)&(observed*norm<=5)
    mesh=np.isfinite(predicted)&(predicted>0)&(predicted*norm<=5)
    rows=[]
    for size in (8,4):
        zones=zone8 if size==8 else (zone8//8//2)*4+(zone8%8//2)
        for zone in range(size*size):
            selected=zones==zone;den=float(weights[selected].sum())
            missing=float(weights[selected&native&~mesh].sum());extra=float(weights[selected&mesh&~native].sum())
            rows.append(dict(size=size,zone=zone,rays=int(selected.sum()),angular_weight_sr=den,
                missing_weight_sr=missing,extra_weight_sr=extra,
                missing_fraction=missing/den if den else None,extra_fraction=extra/den if den else None,
                status=('NOT_RUN' if not den else 'PASS' if missing/den<=.02 and extra/den<=.02 else 'FAIL')))
    return rows,native,mesh


def statistics(values):
    return dict(count=int(len(values)),p95_m=float(np.quantile(values,.95)) if len(values) else None,
        maximum_m=float(np.max(values)) if len(values) else None,
        median_m=float(np.median(values)) if len(values) else None)


def distance_gate(values):
    metrics=statistics(values)
    return dict(metrics,status='NOT_RUN' if not len(values) else 'PASS' if metrics['p95_m']<=.03 and metrics['maximum_m']<=.075 else 'FAIL')


def thin_layer(full_visible_pixels,sampled_visible,native,predicted,norm,owner_matches=None):
    mask=np.asarray(sampled_visible,dtype=bool)
    if not full_visible_pixels:
        return dict(status='NOT_APPLICABLE',full_visible_pixels=0,sampled_visible_pixels=0,retained_near_pixels=0)
    if not mask.any():
        return dict(status='NOT_RUN',full_visible_pixels=int(full_visible_pixels),sampled_visible_pixels=0,retained_near_pixels=0)
    close=mask&np.isfinite(predicted)&(predicted>0)&(predicted*norm<=5)&(np.abs(predicted.astype(float)-native.astype(float))*norm<=.075)
    if owner_matches is not None:
        close &= np.asarray(owner_matches,dtype=bool)
    return dict(status='PASS' if close.any() else 'FAIL',full_visible_pixels=int(full_visible_pixels),
        sampled_visible_pixels=int(mask.sum()),retained_near_pixels=int(close.sum()))


def aggregate_status(statuses):
    return 'FAIL' if 'FAIL' in statuses else ('NOT_RUN' if 'NOT_RUN' in statuses or not statuses else 'PASS')


def audit(capture, diagnostic=None, scene_depth_materials=False, development_sample=False):
    import torch
    from PIL import Image
    from cnh_route_scene_depth_audit import world_triangles,trace_cuda
    from cnh_route_insert_geometry_audit import world_triangles as inserted_triangles
    from cnh_route_depth_surface import DepthSurfaceFilter
    surface_filter=DepthSurfaceFilter() if scene_depth_materials else None
    journal=os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    if not journal or json.loads(Path(journal).read_text(encoding='utf-8-sig')).get('state')!='running':
        raise RuntimeError('Governed research-ue execution required')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    capture=Path(capture).resolve(strict=True)
    if json.loads((capture/'format-receipt.json').read_text()).get('status')!='PASS_SEVEN_PASS_SOURCE_TRANSPORT_ONLY':
        raise ValueError('Successful 16-frame source transport required')
    manifest=json.loads((capture/'raw-manifest.json').read_text());spec=json.loads((capture/'source/spec.json').read_text(encoding='utf-8-sig'))
    frame_rows=manifest['frames'];layout_ids=[r['layout_id'] for r in spec['layouts']]
    sample_policy=None
    if development_sample:
        if spec.get('scope')!='STREET_DEVELOPMENT_PILOT_NOT_BENCHMARK':
            raise ValueError('Explicit Development capture required for sampling')
        selected=[l for l in json.loads((capture/'candidate-selection.json').read_text())['selected_layouts'] if l is not None]
        expected={(l['layout_id'],c['id'],i) for l in selected for c in l['clips'] for i in range(len(c['poses']))}
        if len(frame_rows)!=len(expected) or {(r['layout_id'],r['clip_id'],r['pose_index']) for r in frame_rows}!=expected:
            raise ValueError('Development transport does not match selected layouts')
        # Fixed before first batch outcomes. The original 8m export around the
        # starting camera covers the <=5m rays after a 1.9m translation; it does
        # not authenticate the 3.9m endpoint. No endpoint PASS is inferred.
        sample_policy=dict(name='DEVELOPMENT_FIXED_START_MIDPOINT_V1',pairs=[['centre',0],['boundary',19],['removed',0]],
            limitation='Three frames per layout only; original 8m native export does not certify terminal pose geometry')
        frame_rows=[r for r in frame_rows if [r['clip_id'],r['pose_index']] in sample_policy['pairs']]
    else:
        expected={(layout,clip,pose) for layout in layout_ids for clip in ('centre','boundary','outside','removed') for pose in (0,1)}
        if len(set(layout_ids))!=2 or len(frame_rows)!=16 or {(r['layout_id'],r['clip_id'],r['pose_index']) for r in frame_rows}!=expected:
            raise ValueError('Exactly two fixed layouts and 16 endpoint frames required')
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    native_geometry={};native_surface_ids={};native_surfaces={};export_reports={};background_errors={layout:[] for layout in layout_ids};frames=[]
    source_receipt=json.loads((capture/'source-receipt.json').read_text())
    hidden_actors=set(source_receipt.get('hidden_native_hlod_actors',[]))
    clearance=json.loads((capture/'clearance.json').read_text())
    hidden_components={r['component'] for r in clearance.get('excluded_primitives',[]) if r.get('reason')=='HIDDEN_OR_INVISIBLE_IN_FIXED_SETTLED_CAPTURE'}
    for index,layout in enumerate(layout_ids):
        folder=capture/'evaluator'/f'layout-{index:02d}'
        probe_path=folder/'scene-probe.json';probe=json.loads(probe_path.read_text())
        original_count=len(probe['instances'])
        probe=dict(probe,instances=[r for r in probe['instances'] if r['actor_path'] not in hidden_actors and r['component_path'] not in hidden_components])
        triangles,surface_ids,_,missing=world_triangles(folder,probe,section_filter=surface_filter)
        native_geometry[layout]=triangles
        native_surface_ids[layout]=surface_ids
        native_surfaces[layout]=[r for r in probe['instances'] if r['mesh'].get('status')=='EXPORTED_LOD0_NOT_RENDER_VERIFIED']
        export_reports[layout]=dict(probe_sha256=digest(probe_path),exported_triangles=len(triangles),authenticated_hidden_instances_excluded=original_count-len(probe['instances']),
            missing_exports=missing,unsupported_primitives=len(probe.get('unsupported_primitives',[])))
    for row in frame_rows:
        folder=(capture/row['folder']).resolve(strict=True)
        if not folder.is_relative_to(capture):raise ValueError('Frame path escapes capture')
        camera=json.loads((folder/'camera.json').read_text());geometry=json.loads((folder/'inserted-geometry.json').read_text())
        from cnh_street_development_baseline import load_depth
        depth,depth_path=load_depth(folder) if development_sample else (np.load(folder/'depth_left.transport.npy',allow_pickle=False),folder/'depth_left.transport.npy')
        with Image.open(folder/'instance_left.png') as image: ids=np.asarray(image).copy()
        if depth.shape!=(360,640) or depth.dtype!=np.dtype('<f4') or ids.shape!=depth.shape:raise ValueError('Native depth/ID schema differs')
        yy,xx,optical,weights,zones=ray_grid(camera);norm=np.linalg.norm(optical,axis=-1)
        matrix=np.asarray(camera['T_world_camera'],dtype=float);origin=matrix[:3,3]
        chunks=[native_geometry[row['layout_id']]];owner_chunks=[np.zeros(len(chunks[0]),dtype=np.uint16)]
        surface_chunks=[native_surface_ids[row['layout_id']]]
        for instance in geometry['instances']:
            if instance.get('hidden') is False:
                chunks.append(inserted_triangles(instance));owner_chunks.append(np.full(len(chunks[-1]),instance['inserted_id'],dtype=np.uint16))
                surface_chunks.append(np.full(len(chunks[-1]),-int(instance['inserted_id']),dtype=np.int64))
            elif instance.get('hidden') is not True:raise ValueError('Explicit hidden flag required')
        triangles=np.concatenate(chunks);owners=np.concatenate(owner_chunks);surface_ids=np.concatenate(surface_chunks)
        # Conservative triangle-AABB/sphere filter cannot remove a <=5m hit.
        low=triangles.min(axis=1);high=triangles.max(axis=1)
        separation=np.maximum(np.maximum(low-origin,origin-high),0)
        in_range=(separation*separation).sum(axis=-1)<=25
        triangles=triangles[in_range];owners=owners[in_range]
        predicted,winner=trace_cuda(origin,optical@matrix[:3,:3].T,triangles)
        predicted_owner=np.zeros(len(predicted),dtype=np.uint16)
        has_winner=winner>=0
        predicted_owner[has_winner]=owners[winner[has_winner]]
        if diagnostic is not None:
            predicted_surface=np.full(len(predicted),-65535,dtype=np.int64)
            predicted_surface[has_winner]=surface_ids[in_range][winner[has_winner]]
            diagnostic(row=row,folder=folder,camera=camera,yy=yy,xx=xx,optical=optical,
                weights=weights,zones=zones,predicted=predicted,observed=depth[yy,xx],
                sampled_ids=ids[yy,xx],predicted_owner=predicted_owner,
                predicted_surface=predicted_surface,surfaces=native_surfaces[row['layout_id']])
        observed=depth[yy,xx];sampled_ids=ids[yy,xx]
        coverage,native_hit,mesh_hit=zone_coverage(observed,predicted,norm,weights,zones)
        common=native_hit&mesh_hit;background=common&(sampled_ids==0)
        radial=np.abs(predicted[background]-observed[background])*norm[background]
        background_errors[row['layout_id']].extend(radial.tolist())
        all_y,all_x=np.indices(depth.shape);k=np.asarray(camera['K']);edge=np.tan(np.deg2rad(22.5))
        full_x=(all_x-k[0,2])/k[0,0];full_y=(all_y-k[1,2])/k[1,1]
        full_norm=np.sqrt(1+full_x**2+full_y**2)
        full_near=(np.abs(full_x)<=edge)&(np.abs(full_y)<=edge)&np.isfinite(depth)&(depth>0)&(depth*full_norm<=5)
        layers={str(identifier):thin_layer(int(((ids==identifier)&full_near).sum()),
            (sampled_ids==identifier)&native_hit,observed,predicted,norm,predicted_owner==identifier) for identifier in (1,254)}
        frames.append(dict(id=row['id'],layout_id=row['layout_id'],clip_id=row['clip_id'],pose_index=row['pose_index'],
            traced_triangles=len(triangles),selected_rays=len(observed),common_hits=int(common.sum()),
            background_common_hits=int(background.sum()),native_near_hits=int(native_hit.sum()),predicted_near_hits=int(mesh_hit.sum()),
            unknown_id_sampled_pixels=int((sampled_ids==65535).sum()),
            native_only_hits=int((native_hit&~mesh_hit).sum()),predicted_only_hits=int((mesh_hit&~native_hit).sum()),
            background_radial_abs_error=statistics(radial),all_common_radial_abs_error=statistics(np.abs(predicted[common]-observed[common])*norm[common]),
            all_common_axial_abs_error=statistics(np.abs(predicted[common]-observed[common])),
            coverage_status=aggregate_status([r['status'] for r in coverage]),zones=coverage,thin_layers=layers,
            input_sha256={name:digest(folder/name) for name in ('camera.json','inserted-geometry.json',depth_path.name,'instance_left.png')}))
    layouts=[]
    for layout in layout_ids:
        selected=[r for r in frames if r['layout_id']==layout]
        distance=distance_gate(np.asarray(background_errors[layout]))
        coverage=aggregate_status([r['coverage_status'] for r in selected])
        layer_statuses=[l['status'] for r in selected for l in r['thin_layers'].values() if l['status']!='NOT_APPLICABLE']
        thin=aggregate_status(layer_statuses)
        layouts.append(dict(layout_id=layout,background_radial_gate=distance,coverage_gate=coverage,
            sampled_inserted_near_layer_gate=thin,tested_gates_status=aggregate_status([distance['status'],coverage,thin])))
    return dict(status=aggregate_status([l['tested_gates_status'] for l in layouts]),
        authority='NUMERICAL_GEOMETRY_AND_SAMPLED_COVERAGE_GATES_ONLY',benchmark_eligible=False,formal_admission='NOT_RUN',
        sampling=sample_policy,capture_manifest_sha256=digest(capture/'raw-manifest.json'),
        protocol=PROTOCOL,protocol_sha256=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest(),
        energy_convergence='NOT_RUN',label_precision='NOT_RUN',provenance='NOT_RUN',
        thin_layer_scope='VISIBLE_INSERTED_IDENTITIES_AT_FIXED_GRID_ONLY_NOT_ALL_SUBPIXEL_LAYERS',
        backend='CUDA',device=torch.cuda.get_device_name(),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        source_exports=export_reports,layouts=layouts,frames=frames,capture=str(capture),
        depth_surface_policy=surface_filter.receipt() if surface_filter else None,
        auditor_sha256=digest(__file__),spec_sha256=digest(capture/'source/spec.json'),
        limitations='Exported static LOD geometry only; dynamic or unsupported surfaces can cause measured misses. No energy-density convergence test or label/provenance admission.',
        wall_s=time.monotonic()-started)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--development-sample',action='store_true')
    args=parser.parse_args();root=(Path(__file__).resolve().parents[4]/'artifacts.local').resolve();output=args.output.resolve()
    if output.exists() or not output.is_relative_to(root) or output==root:raise ValueError('Fresh artifact output required')
    result=audit(args.capture,development_sample=args.development_sample)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
