"""Run the bounded source collector; rejected prefilters remain terminal evidence."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import cnh_route_insert_launch as common
from cnh_route_source_capture import validate_insertions
from cnh_route_source_compare_adapter import validate_spec as validate_source, is_development

HERE=Path(__file__).resolve().parent


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_rgb_replay_spec(old_capture):
    """Prepare a new immutable output recipe from a completed Development alley run."""
    old=Path(old_capture).resolve()
    source=old/'source/spec.json';format_path=old/'format-receipt.json';manifest_path=old/'raw-manifest.json'
    spec=json.loads(source.read_text(encoding='utf-8-sig'))
    receipt=json.loads(format_path.read_text())
    manifest=json.loads(manifest_path.read_text())
    if (spec.get('scene_layer')!='ALLEY_DEVELOPMENT_PILOT_NOT_BENCHMARK' or
            spec.get('alley_manifest',{}).get('proposed_split') not in ('train','dev') or
            len(spec.get('layouts',[]))!=1 or receipt.get('status')!='PASS_SEVEN_PASS_SOURCE_TRANSPORT_ONLY' or
            len(receipt.get('frames',[]))!=160 or len(manifest.get('frames',[]))!=160):
        raise ValueError('Completed 160-frame train/dev alley source required')
    replay=copy.deepcopy(spec)
    replay.update(capture_mode='ALLEY_RGB_ONLY_REPLAY_V1',
        rgb_exposure_policy='ALLEY_SINGLE_PROBE_SHADOW_FIXED_EV_V2',rgb_probe_ev100=-2.,
        rgb_insert_material_policy='ALLEY_DERIVED_MFPD_OFF_V1',
        rgb_replay_source=dict(capture_root=str(old),source_spec_sha256=_sha(source),
            format_receipt_sha256=_sha(format_path),raw_manifest_sha256=_sha(manifest_path)))
    replay['disclosure']=list(replay.get('disclosure',[]))+[
        'RGB-only Development replay; old depth, ToF, geometry labels and receipts stay immutable.',
        'Each output frame is bound to its old camera and depth hashes by source frame identity.']
    return replay


def relocate_rgb_replay_spec(spec, project, engine, family_receipt):
    """Rebind identical hashed map dependencies on a different capture host."""
    spec=copy.deepcopy(spec)
    root=Path(project).resolve()/'Content'
    engine_root=Path(engine).resolve()/'Engine/Content'
    manifest=spec['alley_manifest']
    for entry in manifest['files']:
        original=entry['path'].replace('\\','/')
        if '/Content/' not in original:
            raise ValueError('Cannot rebind non-Content dependency')
        local=(engine_root if '/Engine/Content/' in original else root)/original.split('/Content/',1)[1]
        if not local.is_file() or _sha(local)!=entry['sha256']:
            raise ValueError('Rebound dependency hash differs: '+str(local))
        entry['path']=str(local)
    local_map=root/(spec['map_asset'].removeprefix('/Game/')+'.umap')
    if _sha(local_map)!=spec['map_sha256']:
        raise ValueError('Rebound alley map hash differs')
    spec['map_file']=str(local_map)
    family=Path(family_receipt).resolve()
    if _sha(family)!=manifest['family_receipt']['sha256']:
        raise ValueError('Rebound family receipt hash differs')
    manifest['family_receipt']['path']=str(family)
    return spec


def verify_rgb_replay_source(spec, old_root):
    source=spec['rgb_replay_source']
    old=Path(old_root).resolve()
    if old!=Path(source['capture_root']).resolve():
        raise ValueError('RGB replay source path differs from frozen spec')
    for name,key in [('source/spec.json','source_spec_sha256'),
                     ('format-receipt.json','format_receipt_sha256'),
                     ('raw-manifest.json','raw_manifest_sha256')]:
        if _sha(old/name)!=source[key]:
            raise ValueError('Historical replay source changed: '+name)
    original=json.loads((old/'source/spec.json').read_text(encoding='utf-8-sig'))
    for key in ('map_asset','map_sha256','layouts','assets','native_material_policy',
                'render_recipe','transport_policy'):
        if spec.get(key)!=original.get(key):
            raise ValueError('RGB replay changed source geometry or capture identity: '+key)
    def manifest_identity(value):
        result=copy.deepcopy(value)
        result['files']=[dict(sha256=row['sha256'],relative=row['path'].replace('\\','/').split('/Content/',1)[1])
                         for row in result['files']]
        result['family_receipt']={'sha256':result['family_receipt']['sha256']}
        return result
    if manifest_identity(spec['alley_manifest'])!=manifest_identity(original['alley_manifest']):
        raise ValueError('RGB replay changed scene dependency identities')
    receipt=json.loads((old/'format-receipt.json').read_text())
    if receipt.get('status')!='PASS_SEVEN_PASS_SOURCE_TRANSPORT_ONLY':
        raise ValueError('Old seven-pass capture receipt is incomplete')
    rows={row['id']:row for row in receipt['frames']}
    if len(rows)!=len(receipt['frames']):
        raise ValueError('Old frame IDs are not unique')
    return rows


def validate(spec):
    validate_source(spec)
    validate_insertions(spec)
    if spec.get('capture_mode') in ('ALLEY_RGB_ONLY_REPLAY_V1','ALLEY_MFPD_DEPTH_DIAGNOSTIC_V1'):
        verify_rgb_replay_source(spec,spec['rgb_replay_source']['capture_root'])


def verify_layout_exposure_receipts(spec, engine, manifest):
    if spec.get('rgb_exposure_policy') != 'ALLEY_SINGLE_PROBE_SHADOW_FIXED_EV_V2':
        return {}
    decisions=engine.get('rgb_exposure_by_layout',{})
    rows=manifest['frames']
    if (not rows or set(decisions)!={row['layout_id'] for row in rows} or
            manifest.get('rgb_exposure_by_layout')!=decisions):
        raise ValueError('One complete RGB exposure probe required per collected layout')
    for row in rows:
        decision=decisions[row['layout_id']]
        if (row.get('rgb_exposure')!=decision or
                decision.get('policy')!=spec['rgb_exposure_policy'] or
                decision.get('probe_pose_index')!=0 or
                decision.get('probe_clip_id')!='centre' or
                decision.get('physical_site_id')!=row['physical_site_id']):
            raise ValueError('Frame RGB exposure differs from its fixed layout probe')
    return decisions


def finalize_rgb_replay(out, spec, engine, manifest):
    import struct
    from PIL import Image
    out=Path(out).resolve()
    old=Path(spec['rgb_replay_source']['capture_root']).resolve()
    if out==old or out.is_relative_to(old):
        raise ValueError('New RGB output overlaps historical capture')
    old_rows=verify_rgb_replay_source(spec,old)
    exposure=verify_layout_exposure_receipts(spec,engine,manifest)
    material_receipts={}
    for asset in spec['assets']:
        path=out/f'derived-{asset["id"]}.json'
        derived=json.loads(path.read_text())
        intervention=derived.get('mfpd_intervention',{})
        if (derived.get('source_mesh')!=asset['mesh_asset'] or
                not derived.get('source_configuration_unchanged') or
                intervention.get('parameter')!='Enable MFPD' or
                intervention.get('derived_switch_after') is not False or
                intervention.get('source_switch_after') is not True or
                intervention.get('source_used_textures_unchanged') is not True or
                not intervention.get('removed_editor_used_textures') or
                intervention.get('added_editor_used_textures')):
            raise ValueError('RGB replay derived MFPD-off material receipt incomplete')
        material_receipts[str(asset['id'])]=dict(path=str(path),sha256=_sha(path),
            source_mesh=derived['source_mesh'],source_material=intervention['source_material'],
            derived_material=intervention['derived_material'],
            source_editor_used_textures=intervention['source_editor_used_textures_after'],
            derived_editor_used_textures=intervention['derived_editor_used_textures_after'],
            removed_editor_used_textures=intervention['removed_editor_used_textures'])
    rows=manifest['frames']
    if len(rows)!=len(old_rows) or {r['id'] for r in rows}!=set(old_rows):
        raise ValueError('RGB replay frame identity differs from old capture')
    results=[]
    rgb_usability=[]
    for row in rows:
        prior=old_rows[row['id']]
        for key in ('layout_id','physical_site_id','clip_id','pose_index','target_hidden'):
            if row[key]!=prior[key]:
                raise ValueError('RGB replay changed frame identity: '+key)
        folder=(out/row['folder']).resolve()
        old_folder=(old/prior['folder']).resolve()
        if not folder.is_relative_to(out) or not old_folder.is_relative_to(old):
            raise ValueError('RGB replay frame path escaped its capture')
        if _sha(folder/'camera.json')!=prior['hashes']['camera.json']:
            raise ValueError('RGB replay camera differs from original pose')
        new_geometry=json.loads((folder/'inserted-geometry.json').read_text())['instances']
        old_geometry=json.loads((old_folder/'inserted-geometry.json').read_text())['instances']
        if len(new_geometry)!=len(old_geometry) or len(new_geometry)!=2:
            raise ValueError('RGB replay inserted instance count differs')
        for new_instance,old_instance in zip(new_geometry,old_geometry):
            for key in ('inserted_id','hidden','source_asset','sections','actual_translation_m',
                        'actual_rotation_quaternion','actual_scale','desired_bounds_center_m'):
                if new_instance[key]!=old_instance[key]:
                    raise ValueError('RGB replay inserted geometry differs: '+key)
        retained={}
        for name in ('depth_left.exr','depth_right.exr','depth_left_valid.npy','depth_right_valid.npy'):
            path=old_folder/name
            if path.is_file():
                digest=_sha(path)
                if digest!=prior['hashes'][name]:
                    raise ValueError('Old depth/ToF source hash changed: '+name)
                retained[name]=dict(sha256=digest,verified=True)
            elif name.startswith('depth_left'):
                raise ValueError('Primary old depth source is unavailable: '+name)
            else:
                retained[name]=dict(sha256=prior['hashes'][name],verified=False,
                    reason='NOT_PRESENT_IN_COMPACT_LOCAL_SOURCE')
        rgb={}
        for side in ('left','right'):
            path=folder/(side+'.png')
            with path.open('rb') as stream:
                header=stream.read(24)
            if (len(header)!=24 or header[:8]!=b'\x89PNG\r\n\x1a\n' or
                    header[12:16]!=b'IHDR' or struct.unpack('>II',header[16:24])!=(640,360)):
                raise ValueError('RGB replay PNG dimensions differ')
            rgb[path.name]=_sha(path)
            with Image.open(path) as im:
                im.load()
                if im.size!=(640,360):
                    raise ValueError('RGB replay decoded PNG dimensions differ')
                pixels=im.convert('RGB').load()
                def levels(x0,x1,y0,y1):
                    values=[]
                    for y in range(int(360*y0),int(360*y1),4):
                        for x in range(int(640*x0),int(640*x1),4):
                            red,green,blue=pixels[x,y]
                            values.append((.2126*red+.7152*green+.0722*blue)/255.)
                    return sorted(values)
                centre=levels(.15,.85,.30,.85)
                near=levels(.421875,.671875,.50,.9583333333)
            quantile=lambda values,fraction: values[int(fraction*(len(values)-1))]
            p10,p60,p90,p99=(quantile(centre,fraction) for fraction in (.10,.60,.90,.99))
            near_p30,near_p60=(quantile(near,fraction) for fraction in (.30,.60))
            rgb_usability.append(dict(frame_id=row['id'],side=side,
                roi_luma_p10=p10,roi_luma_p60=p60,roi_luma_p90=p90,
                roi_luma_p99=p99,roi_contrast_p90_minus_p10=p90-p10,
                nearfield_luma_p30=near_p30,nearfield_luma_p60=near_p60,
                passed=.12<=p60<=.85 and p90-p10>=.08 and
                       near_p30>=.10 and near_p60>=.14 and p99<=.98))
        results.append(dict(frame_id=row['id'],layout_id=row['layout_id'],clip_id=row['clip_id'],
            pose_index=row['pose_index'],old_camera_sha256=prior['hashes']['camera.json'],
            old_depth_sha256=retained,new_rgb_sha256=rgb,
            old_frame_receipt_sha256=spec['rgb_replay_source']['format_receipt_sha256']))
    failed=[item for item in rgb_usability if not item['passed']]
    sample_indices=sorted({0,len(rows)//2,len(rows)-1})
    visual_samples=[dict(frame_id=rows[i]['id'],side=side,
        path=str((out/rows[i]['folder']/(side+'.png')).resolve()),
        sha256=results[i]['new_rgb_sha256'][side+'.png'])
        for i in sample_indices for side in ('left','right')]
    usability=dict(status='FAIL_UNUSABLE_RGB' if failed else 'NUMERIC_PASS_VISUAL_REVIEW_REQUIRED',
        roi_fraction=dict(centre=dict(x=[.15,.85],y=[.30,.85]),
                          nearfield=dict(x=[.421875,.671875],y=[.50,.9583333333])),stride_px=4,
        gate='EACH_EYE_CENTRE_P60_0.12_TO_0.85_CONTRAST_AT_LEAST_0.08_P99_AT_MOST_0.98_NEARFIELD_P30_AT_LEAST_0.10_P60_AT_LEAST_0.14',
        checked_image_count=len(rgb_usability),failed_image_count=len(failed),
        failed_images=failed,visual_review_status='PENDING',visual_samples=visual_samples,
        image_metrics=rgb_usability)
    receipt=dict(status='FAIL_UNUSABLE_RGB' if failed else 'RGB_NUMERIC_PASS_VISUAL_REVIEW_REQUIRED',
        data_role='Development',benchmark_eligible=False,frame_count=len(rows),
        rgb_exposure_by_layout=exposure,source_capture=str(old),frames=results,
        rgb_usability=usability,transport_status='PASS_RGB_ONLY_REPLAY_TRANSPORT',
        rgb_insert_material_policy=spec['rgb_insert_material_policy'],
        derived_insert_materials=material_receipts,
        geometry_labels='REUSE_OLD_BY_FRAME_ID_NO_REWRITE',
        depth_tof='REUSE_OLD_HASH_VERIFIED_NO_REWRITE')
    common.write(out/'format-receipt.json',receipt)
    return receipt


def finalize(out, parity_receipt_name='format-receipt.json'):
    spec=json.loads((Path(out)/'source/spec.json').read_text(encoding='utf-8-sig'))
    if spec.get('capture_mode')=='ALLEY_MFPD_DEPTH_DIAGNOSTIC_V1':
        import sys
        import numpy as np
        runtime=HERE.parents[3]/'artifacts.local/work/cnh-route-comparison-20260924/runtime'
        if not (runtime/'OpenEXR.cp311-win_amd64.pyd').is_file():
            raise ValueError('Pinned OpenEXR 3.5.0 runtime unavailable')
        sys.path.insert(0,str(runtime))
        import OpenEXR
        if OpenEXR.__version__!='3.5.0':
            raise ValueError('MFPD parity requires pinned OpenEXR 3.5.0')
        out=Path(out).resolve()
        engine=json.loads((out/'engine-receipt.json').read_text())
        manifest=json.loads((out/'raw-manifest.json').read_text())
        if (engine.get('status')!='PASS_MFPD_DEPTH_DIAGNOSTIC' or
                not engine.get('source_unchanged') or
                not engine.get('actor_release',{}).get('released') or
                len(manifest.get('frames',[]))!=1):
            raise ValueError('MFPD depth diagnostic did not complete one native frame')
        old=Path(spec['rgb_replay_source']['capture_root']).resolve()
        prior=verify_rgb_replay_source(spec,old)['frame-0000']
        new=manifest['frames'][0]
        if new['id']!='frame-0000':
            raise ValueError('MFPD depth diagnostic first pose differs')
        old_folder=old/prior['folder'];new_folder=out/new['folder']
        if _sha(new_folder/'camera.json')!=prior['hashes']['camera.json']:
            raise ValueError('MFPD depth diagnostic camera differs')
        old_geometry=json.loads((old_folder/'inserted-geometry.json').read_text())['instances']
        new_geometry=json.loads((new_folder/'inserted-geometry.json').read_text())['instances']
        keys=('inserted_id','hidden','source_asset','sections','actual_translation_m',
              'actual_rotation_quaternion','actual_scale','desired_bounds_center_m')
        geometry_equal=(len(new_geometry)==len(old_geometry)==2 and
                        all(all(a[key]==b[key] for key in keys)
                            for a,b in zip(new_geometry,old_geometry)))
        original_depth=old_folder/'depth_left.exr'
        if _sha(original_depth)!=prior['hashes']['depth_left.exr'] or _sha(old_folder/'depth_left_valid.npy')!=prior['hashes']['depth_left_valid.npy']:
            raise ValueError('Old depth source changed during MFPD diagnostic')
        previous=OpenEXR.File(str(original_depth)).channels()['Z'].pixels
        current=np.load(new_folder/'depth_left.transport.npy',allow_pickle=False)
        previous_valid=np.load(old_folder/'depth_left_valid.npy',allow_pickle=False)
        current_valid=np.isfinite(current)&(current>0)&(current<100)
        if previous.shape!=(360,640) or current.shape!=(360,640):
            raise ValueError('MFPD diagnostic depth dimensions differ')
        changed=int(np.count_nonzero(previous_valid!=current_valid))
        comparable=previous_valid&current_valid
        difference=np.abs(previous[comparable].astype(np.float64)-current[comparable].astype(np.float64))
        max_difference=float(difference.max()) if difference.size else float('inf')
        material={}
        for asset in spec['assets']:
            path=out/f'derived-{asset["id"]}.json'
            receipt=json.loads(path.read_text())
            material[str(asset['id'])]=dict(path=str(path),sha256=_sha(path),
                intervention=receipt.get('mfpd_intervention'))
        parity=(geometry_equal and changed==0 and max_difference<=.001 and
                all(row['intervention'] and row['intervention']['derived_switch_after'] is False
                    for row in material.values()))
        result=dict(status='PASS_MFPD_DEPTH_GEOMETRY_PARITY' if parity else 'FAIL_MFPD_DEPTH_GEOMETRY_PARITY',
            benchmark_eligible=False,data_role='Development',frames=1,
            depth_decoder='PINNED_OPENEXR_3.5.0_Z_CHANNEL',
            old_capture=str(old),new_capture=str(out),camera_sha256=prior['hashes']['camera.json'],
            old_depth_sha256=prior['hashes']['depth_left.exr'],
            new_depth_transport_sha256=_sha(new_folder/'depth_left.transport.npy'),
            old_geometry_sha256=prior['hashes']['inserted-geometry.json'],
            new_geometry_sha256=_sha(new_folder/'inserted-geometry.json'),
            inserted_geometry_equal=geometry_equal,valid_mask_changed_pixels=changed,
            comparable_pixel_count=int(np.count_nonzero(comparable)),
            max_absolute_depth_difference_m=max_difference,depth_tolerance_m=.001,
            derived_insert_materials=material)
        if parity_receipt_name!='format-receipt.json':
            failed=out/'format-receipt.json'
            result['supersedes_failed_decoder_receipt']=dict(path=str(failed),sha256=_sha(failed),
                reason='CV2_EXR_DECODER_RETURNED_ZERO_FOR_SINGLE_Z_CHANNEL')
        common.write(out/parity_receipt_name,result)
        return result
    if spec.get('capture_mode')=='ALLEY_RGB_EXPOSURE_DIAGNOSTIC_V1':
        import struct
        engine=json.loads((Path(out)/'engine-receipt.json').read_text())
        if (engine.get('status')!='PASS_RGB_EXPOSURE_DIAGNOSTIC' or not engine.get('source_unchanged') or
                not engine.get('actor_release',{}).get('released')):
            raise ValueError('RGB exposure diagnostic engine did not complete cleanly')
        candidates=[]
        names=[row['name'] for row in engine['rgb_exposure_diagnostic']]
        if not names or names[0]!='baseline' or len(names)!=len(set(names)):
            raise ValueError('Diagnostic exposure arms are missing or repeated')
        for name in names:
            path=Path(out)/'rgb-exposure-diagnostic'/(name+'.png')
            with path.open('rb') as stream:header=stream.read(24)
            if (len(header)!=24 or header[:8]!=b'\x89PNG\r\n\x1a\n' or
                    struct.unpack('>II',header[16:24])!=(640,360)):
                raise ValueError('Diagnostic PNG dimensions differ')
            candidates.append(dict(name=name,path=str(path),sha256=_sha(path)))
        result=dict(status='DIAGNOSTIC_TRANSPORT_PASS_RGB_USABILITY_UNDECIDED',
            benchmark_eligible=False,data_role='Development',frames=0,
            exposure=engine['rgb_exposure_diagnostic'],candidates=candidates)
        common.write(Path(out)/'format-receipt.json',result)
        return result
    if spec.get('capture_mode')=='ALLEY_RGB_ONLY_REPLAY_V1':
        engine=json.loads((Path(out)/'engine-receipt.json').read_text())
        if (engine.get('status')!='PASS_NATIVE_TRANSPORT' or not engine.get('source_unchanged') or
                not engine.get('actor_release',{}).get('released')):
            raise ValueError('RGB-only engine did not complete cleanly')
        manifest=json.loads((Path(out)/'raw-manifest.json').read_text())
        return finalize_rgb_replay(out,spec,engine,manifest)
    import numpy as np
    import OpenEXR
    from PIL import Image
    out=Path(out)
    engine=json.loads((out/'engine-receipt.json').read_text())
    if not engine.get('source_unchanged') or not engine.get('actor_release',{}).get('released'):
        raise ValueError('Native source integrity/release failed: '+str(engine.get('error')))
    if engine['status']=='CONTROL_PREFLIGHT_REJECTED':
        return dict(status='NOT_ADMITTED_CONTROL_PREFLIGHT',benchmark_eligible=False,frames=0,
            source_gate='NOT_RUN_NEAR_FAR_INSTANCE_SCOPE_UNRESOLVED',
            control_preflight=engine['control_preflight'],seven_pass='NOT_RUN',formal_pilot='NOT_ADMITTED')
    if engine['status']=='PREFILTER_REJECTED':
        return dict(status='NOT_ADMITTED_PREFILTER',benchmark_eligible=False,frames=0,
            source_gate='FAIL_PREFILTER_OR_UNRESOLVED',selection=json.loads((out/'candidate-selection.json').read_text()),
            seven_pass='NOT_RUN_PREFILTER_REJECTED',formal_pilot='NOT_ADMITTED')
    if engine['status']!='PASS_NATIVE_TRANSPORT':
        raise ValueError('Source transport failed: '+str(engine.get('error')))
    manifest=json.loads((out/'raw-manifest.json').read_text())
    rows=manifest['frames']; h,w=manifest['rig']['height'],manifest['rig']['width']
    layout_ids={r['layout_id'] for r in rows}
    spec=json.loads((out/'source/spec.json').read_text(encoding='utf-8-sig'))
    validate(spec)
    selected=spec['layouts']
    if is_development(spec):
        selected=[l for l in json.loads((out/'candidate-selection.json').read_text())['selected_layouts'] if l is not None]
    expected={(l['layout_id'],c['id'],i) for l in selected for c in l['clips'] for i in range(len(c['poses']))}
    if layout_ids!={l['layout_id'] for l in selected} or len(rows)!=len(expected) or {(r['layout_id'],r['clip_id'],r['pose_index']) for r in rows}!=expected:
        raise ValueError('Frames do not match the frozen source spec')
    exposure_receipts=verify_layout_exposure_receipts(spec,engine,manifest)
    reports=[]
    for row in rows:
        folder=(out/row['folder']).resolve()
        if not folder.is_relative_to(out.resolve()):
            raise ValueError('Frame path outside capture')
        depths={}
        for side in ('left','right'):
            with Image.open(folder/(side+'.png')) as rgb:
                rgb.load()
                if rgb.size!=(w,h): raise ValueError('RGB dimensions differ')
            depth=np.load(folder/f'depth_{side}.transport.npy',allow_pickle=False)
            if depth.dtype!=np.dtype('<f4') or depth.shape!=(h,w): raise ValueError('Depth schema differs')
            valid=np.isfinite(depth)&(depth>0)&(depth<100)
            depths[side]=(depth,valid)
            target=folder/f'depth_{side}.exr'
            if target.exists(): raise FileExistsError(target)
            OpenEXR.File({'compression':OpenEXR.ZIP_COMPRESSION,'type':OpenEXR.scanlineimage},
                {'Z':np.where(valid,depth,np.nan).astype(np.float32)}).write(str(target))
            np.save(folder/f'depth_{side}_valid.npy',valid,allow_pickle=False)
        for kind in ('normal','albedo'):
            values=np.load(folder/f'{kind}_left.transport.npy',allow_pickle=False)
            if values.dtype!=np.dtype('<f4') or values.shape!=(h,w,3): raise ValueError('Attribute schema differs')
            valid=depths['left'][1]&np.isfinite(values).all(-1)
            if kind=='normal': valid &= np.abs(np.linalg.norm(values,axis=-1)-1)<.04
            elif np.any((values[valid]<0)|(values[valid]>1.001)): raise ValueError('Albedo outside range')
            if valid.sum()<.9*depths['left'][1].sum(): raise ValueError('Attribute coverage below 90 percent')
            np.save(folder/f'{kind}_left.npy',values.astype(np.float16),allow_pickle=False)
            np.save(folder/f'{kind}_left_valid.npy',valid,allow_pickle=False)
        depth,valid=depths['left']
        ids=np.where(valid,0,65535).astype(np.uint16)
        owners=np.zeros((h,w),dtype=np.uint8); ambiguous=np.zeros((h,w),dtype=bool)
        for identifier in (1,254):
            isolated=np.load(folder/f'isolated_depth_{identifier}.transport.npy',allow_pickle=False)
            candidate=common.compose_instance_ids(depth,isolated,identifier)
            mask=candidate==identifier
            ambiguous |= candidate==65535
            owners[mask]+=1;ids[mask]=identifier
        ambiguous |= owners>1
        ids[ambiguous]=65535
        Image.fromarray(ids).save(folder/'instance_left.png')
        counts={str(i):int((ids==i).sum()) for i in (0,1,254,65535)}
        if row['target_hidden'] and counts['1']: raise ValueError('Removed target remains visible')
        reports.append(dict(row,instance_pixels=counts,overlapping_inserted_id_pixels=int((owners>1).sum()),
            hashes={p.name:common.file_hash(p) for p in folder.iterdir() if p.is_file()}))
    receipt=dict(status='PASS_SEVEN_PASS_SOURCE_TRANSPORT_ONLY',benchmark_eligible=False,
        source_gate='NOT_ADMITTED_REQUIRES_GEOMETRY_ECHO_LABEL_AND_PROVENANCE_GATES',
        formal_pilot='NOT_ADMITTED',data_role='Development',frame_count=len(rows),frames=reports,
        temporal_authority=manifest['temporal_authority'],rgb_exposure_by_layout=exposure_receipts)
    common.write(out/'format-receipt.json',receipt)
    return receipt


def launch(args):
    original_run=common.run_owned
    original_validate=common.validate_spec
    original_finalize=common.finalize
    def source_run(command,env,out,timeout):
        extras=('cnh_route_source_capture.py','cnh_route_source_compare_adapter.py','cnh_route_source_clearance.py',
                'cnh_route_native_clearance.py','cnh_route_scene_probe.py','cnh_route_source_launch.py',
                'cnh_route_street_static_background.py','cnh_route_city_lod0.py','cnh_city_nearfield_derived.py',
                'ue_attribute_export.py','ue_rgb_export.py','cnh_city_vehicle_mask.py',
                'cnh_city_instance_substitution.py')
        launch_path=out/'launch.json';receipt=json.loads(launch_path.read_text())
        for name in extras:
            dest=out/'source'/name;shutil.copy2(HERE/name,dest)
            receipt['source_hashes'][name]=common.file_hash(dest)
        command=[('-ExecCmds=py '+(out/'source/cnh_route_source_capture.py').as_posix())
                 if arg.startswith('-ExecCmds=py ') else arg for arg in command]
        env=dict(env,BA_CNH_SOURCE_SPEC=env['BA_CNH_INSERT_SPEC'],BA_CNH_SOURCE_OUTPUT=env['BA_CNH_INSERT_OUTPUT'])
        frozen_spec=json.loads((out/'source/spec.json').read_text(encoding='utf-8-sig'))
        receipt.update(command=command,scope=frozen_spec['scope'],machine_id=__import__('platform').node(),
            configuration_sha256=common.file_hash(out/'source/spec.json'))
        common.write(launch_path,receipt)
        stop=threading.Event();samples=[];errors=[]
        def sample_gpu():
            while not stop.is_set():
                try:
                    result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.total',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    if result.returncode: raise RuntimeError(result.stderr.strip())
                    samples.append(dict(monotonic_s=time.monotonic(),gpus_mib=[[int(v) for v in line.split(',')]
                        for line in result.stdout.strip().splitlines()]))
                except Exception as exc:
                    errors.append(str(exc));break
                stop.wait(1.)
        worker=threading.Thread(target=sample_gpu,daemon=True);worker.start()
        try:
            return original_run(command,env,out,timeout)
        finally:
            stop.set();worker.join(timeout=4.)
            common.write(out/'gpu-memory.json',dict(scope='WHOLE_GPU_OBSERVED_NOT_PROCESS_ALLOCATION',
                cadence_s=1.,samples=samples,errors=errors,worker_released=not worker.is_alive(),
                observed_peak_used_mib=max((g[0] for s in samples for g in s['gpus_mib']),default=None)))
    common.validate_spec=validate;common.finalize=finalize;common.run_owned=source_run
    try:
        return common.launch(args)
    finally:
        common.run_owned=original_run
        common.validate_spec=original_validate
        common.finalize=original_finalize


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('project','engine','plugin','spec','output','result'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--timeout',type=float,default=600)
    result=launch(p.parse_args())
    print(json.dumps({k:v for k,v in result.items() if k not in ('frames','control_preflight','selection')}))
