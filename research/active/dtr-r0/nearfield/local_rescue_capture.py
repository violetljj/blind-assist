"""Unreal-only576-frame local-rescue-fresh capture (48 clips, 12 posed frames each).

The launcher owns process cleanup. This script owns only newly spawned actors
and render targets; geometry/traces/native depth are written under evaluator/.
Missing/occluded target support is retained, never a source-selection filter.
"""
import array
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback

import unreal as u

OUT = Path(os.environ['BA_LOCAL_RESCUE_FRESH_OUTPUT'])
SPEC = Path(os.environ['BA_LOCAL_RESCUE_FRESH_SPEC'])
SCRIPT = Path(os.environ['BA_LOCAL_RESCUE_FRESH_SCRIPT'])
PROTOCOL_SHA = os.environ['BA_LOCAL_RESCUE_FRESH_PROTOCOL_SHA']
SPEC_SHA = os.environ['BA_LOCAL_RESCUE_FRESH_SPEC_SHA']
SCRIPT_SHA = os.environ['BA_LOCAL_RESCUE_FRESH_SCRIPT_SHA']
sys.path.insert(0,str(SCRIPT.parent))
from ue_capture_readiness import CaptureReadiness
READINESS_HELPER = SCRIPT.with_name('ue_capture_readiness.py')
EXPECTED = 'cf35e5c9df54cd0f781f09ea8105fe8ef6078ed0822d4e594d64216e79a254fb'
MAP = '/Game/StreetLab/WillowSampleV1'
MAP_FILE = Path(u.Paths.project_dir())/'Content/StreetLab/WillowSampleV1.umap'
api = u.get_editor_subsystem(u.EditorActorSubsystem)
levels = u.get_editor_subsystem(u.LevelEditorSubsystem)
editor = u.get_editor_subsystem(u.UnrealEditorSubsystem)
captures, objects, frames, geometry = [], [], [], []
world = None
stage = index = warm = 0
after = time.monotonic()+8
finished = False
started = time.monotonic()
report = dict(status='RUNNING', expected_map_sha256=EXPECTED, protocol_sha256=PROTOCOL_SHA,
              visibility_policy='RETAIN_ALL_FRAMES_REPORT_MISSING_OR_OCCLUDED_SUPPORT_SEPARATELY',
              script_sha256=hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
              readiness_helper_sha256=hashlib.sha256(READINESS_HELPER.read_bytes()).hexdigest())
readiness = CaptureReadiness(u,timeout=900)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def vector_m(vector):
    return [float(v)/100 for v in vector.to_tuple()]


def clear_objects():
    while objects:
        _, actor = objects.pop()
        assert api.destroy_actor(actor), 'Task actor destroy failed'


def component(source, fmt):
    actor = api.spawn_actor_from_class(u.SceneCapture2D, u.Vector(0,0,100000))
    captures.append(actor)
    actor.set_actor_label('BA local rescue fresh capture')
    c = actor.capture_component2d
    c.capture_every_frame = False; c.capture_on_movement = False
    c.always_persist_rendering_state = True; c.capture_source = source
    c.texture_target = u.RenderingLibrary.create_render_target2d(world,640,360,fmt)
    c.fov_angle = 100
    if source == u.SceneCaptureSource.SCS_FINAL_COLOR_LDR:
        c.texture_target.target_gamma = 2.2
        pp = max((a for a in api.get_all_level_actors() if isinstance(a,u.PostProcessVolume)), key=lambda a:a.priority)
        c.post_process_settings = pp.settings; c.post_process_blend_weight = 1.
    return actor


def prepare(case):
    if not frames or frames[-1]['clip_id'] != case['clip_id']:
        clear_objects()
        for obj in case['objects']:
            actor = api.spawn_actor_from_class(u.StaticMeshActor, u.Vector(*(v*100 for v in obj['center_m'])))
            objects.append((obj, actor))
            actor.set_actor_label('BA local rescue fresh '+case['clip_id']+' '+obj['name'])
            assert obj['kind'] == 'cube', 'Composite truth requires exact cube meshes'
            mesh_path = '/Engine/BasicShapes/Cube'
            mesh, material = u.load_asset(mesh_path), u.load_asset(obj['material'])
            assert mesh and material
            actor.static_mesh_component.set_static_mesh(mesh)
            actor.set_actor_scale3d(u.Vector(*obj['size_m']))
            actor.static_mesh_component.set_collision_profile_name('BlockAll')
            actor.static_mesh_component.set_material(0,material)
    pose = case['camera']
    for actor in (rgb,depth):
        actor.set_actor_location(u.Vector(*(pose[k]*100 for k in ('x','y','z'))),False,False)
        actor.set_actor_rotation(u.Rotator(pitch=pose['pitch'],yaw=pose['yaw'],roll=pose['roll']),False)


def authenticate(case):
    pose = case['camera']; origin = u.Vector(*(pose[k]*100 for k in ('x','y','z')))
    for actor in (rgb, depth):
        assert max(abs(a - pose[k]) for a, k in zip(vector_m(actor.get_actor_location()), ('x','y','z'))) <= .002
        rotation = actor.get_actor_rotation()
        assert all(abs((getattr(rotation, k) - pose[k] + 180) % 360 - 180) <= .002
                   for k in ('pitch','yaw','roll')), 'Native camera rotation differs from specification'
    rows = []
    for obj, actor in objects:
        center, extent = actor.get_actor_bounds(False)
        center_m, extent_m = vector_m(center), vector_m(extent)
        assert max(abs(a-b) for a,b in zip(center_m,obj['center_m'])) <= .002
        assert max(abs(2*a-b) for a,b in zip(extent_m,obj['size_m'])) <= .002
        end = center+(center-origin)*.1
        hit = u.SystemLibrary.line_trace_single(world,origin,end,u.TraceTypeQuery.TRACE_TYPE_QUERY1,
                                                True,[],u.DrawDebugTrace.NONE)
        fields = hit.to_tuple() if hit is not None else None
        hit_actor = fields[9] if fields and fields[0] else None
        same_actor = hit_actor == actor
        # Record visibility evidence without dropping a frame or substituting
        # geometry. Missing/occluded support is reported by the label audit.
        mesh = actor.static_mesh_component.get_editor_property('static_mesh')
        material = actor.static_mesh_component.get_material(0)
        rows.append(dict(name=obj['name'], actor_path=actor.get_path_name(),
            actor_label=actor.get_actor_label(), mesh_path=mesh.get_path_name(),
            material_path=material.get_path_name(), actual_location_m=vector_m(actor.get_actor_location()),
            actual_scale=list(actor.get_actor_scale3d().to_tuple()), actual_rotation=list(actor.get_actor_rotation().to_tuple()),
            render_bounds_center_m=center_m,render_bounds_extent_m=extent_m,
            trace=dict(blocking=bool(fields and fields[0]), hit_expected_actor=same_actor,
                       hit_actor_path=hit_actor.get_path_name() if hit_actor else None,
                       impact_point_m=vector_m(fields[5]) if fields and fields[0] else None)))
    floor_hit = u.SystemLibrary.line_trace_single(world,u.Vector(pose['x']*100,pose['y']*100,80),
        u.Vector(pose['x']*100,pose['y']*100,-100),u.TraceTypeQuery.TRACE_TYPE_QUERY1,True,[],u.DrawDebugTrace.NONE)
    assert floor_hit and floor_hit.to_tuple()[0], 'Native road trace missing'
    floor_m = float(floor_hit.to_tuple()[5].z)/100
    assert abs(floor_m-.12) <= .002 and abs(pose['z']-floor_m-1.7) <= .002
    return dict(id=case['name'],sample_index=index,clip_id=case['clip_id'],frame_in_clip=case['frame_in_clip'],
                nominal_time_s=case['time_s'],actual_capture_monotonic_s=time.monotonic()-started,
                declared_camera=pose,actual_camera_location_m=vector_m(rgb.get_actor_location()),
                actual_camera_rotation=list(rgb.get_actor_rotation().to_tuple()),road_world_z_m=floor_m,objects=rows)


def finish(error=None):
    global finished
    if finished:return
    finished = True
    report.update(status='FAIL' if error else 'PASS',frame_count=len(frames),wall_elapsed_s=time.monotonic()-started)
    report['readiness_terminal']=readiness.receipt()
    if error:report['error']=error
    try:
        report['map_sha256_after']=sha(MAP_FILE)
        report['source_unchanged']=report.get('map_sha256_before')==report['map_sha256_after']==EXPECTED
        if not report['source_unchanged']:report['status']='FAIL'
        clear_objects()
        while captures:
            actor=captures.pop()
            if actor.capture_component2d.texture_target:
                u.RenderingLibrary.release_render_target2d(actor.capture_component2d.texture_target)
            assert api.destroy_actor(actor), 'Capture actor destroy failed'
        report['task_actors_released']=not objects and not captures
        if report['status']=='PASS':
            assert len(frames)==576 and len(geometry)==576
    except Exception:
        report['cleanup_error']=traceback.format_exc();report['status']='FAIL'
    finally:
        write(OUT/'evaluator/geometry.json',geometry)
        write(OUT/'receipt.json',report)
        u.unregister_slate_post_tick_callback(handle)
        u.SystemLibrary.quit_editor()


def tick(delta):
    global after,stage,world,rgb,depth,cases,index,warm
    if finished or time.monotonic()<after:return
    after=float('inf')
    try:
        if stage==0:
            assert sha(SPEC)==SPEC_SHA and sha(SCRIPT)==SCRIPT_SHA, 'Frozen source changed after launch'
            report['map_sha256_before']=sha(MAP_FILE)
            assert report['map_sha256_before']==EXPECTED
            spec=json.loads(SPEC.read_text(encoding='utf-8-sig'));cases=spec['cases']
            assert spec['schema'] == 'local-rescue-fresh-source-v1'
            assert spec['expected_map_sha256']==EXPECTED and len(cases)==576
            assert all(len(c['objects']) == 3 and {o['name'] for o in c['objects']} ==
                       {'target_a', 'target_b', 'background'} and
                       all(o['kind'] == 'cube' for o in c['objects']) for c in cases)
            assert len({c['name'] for c in cases})==576 and len({c['clip_id'] for c in cases})==48
            for offset in range(0, 576, 12):
                clip = cases[offset:offset+12]
                assert len({c['clip_id'] for c in clip})==1
                assert [c['frame_in_clip'] for c in clip]==list(range(12))
                assert all(clip[i]['time_s'] < clip[i+1]['time_s'] for i in range(11))
                assert all(c['objects']==clip[0]['objects'] for c in clip), 'Static layout must remain fixed within a clip'
            report['spec_sha256']=sha(SPEC);write(OUT/'evaluator/spec.json',spec)
            assert levels.load_level(MAP)
            world=editor.get_editor_world();stage=1;after=time.monotonic()+30;return
        if stage==1:
            rgb=component(u.SceneCaptureSource.SCS_FINAL_COLOR_LDR,u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
            depth=component(u.SceneCaptureSource.SCS_SCENE_DEPTH,u.TextureRenderTargetFormat.RTF_RGBA32F);stage=2
        if stage==2:
            case=cases[index]
            prepare(case)
            readiness.begin(world,rgb.capture_component2d)
            stage=3;after=0;return
        if stage==3:
            if not readiness.poll():
                write(OUT/'progress.json',dict(status='READINESS',frames=index,expected=576,
                                              readiness=readiness.receipt()))
                after=time.monotonic()+.1;return
            report.setdefault('view_readiness',[]).append(dict(sample_index=index,**readiness.receipt()))
            warm=0;stage=4
        if stage==4:
            case=cases[index]
            rgb.capture_component2d.capture_scene();warm+=1
            if warm <= (32 if case['frame_in_clip']==0 else 16):after=0;return
            depth.capture_component2d.capture_scene()
            evidence=authenticate(case)
            evidence.update(readiness=readiness.receipt(),post_ready_rgb_render_calls=warm,
                            unchanged_warmup_ticks=32 if case['frame_in_clip']==0 else 16)
            rgb_folder=OUT/'observations/rgb';rgb_folder.mkdir(parents=True,exist_ok=True)
            u.RenderingLibrary.export_render_target(world,rgb.capture_component2d.texture_target,str(rgb_folder),f'{index:04d}.png')
            values=u.RenderingLibrary.read_render_target_raw(world,depth.capture_component2d.texture_target,normalize=False)
            assert len(values)==640*360
            native_folder=OUT/'evaluator/native';native_folder.mkdir(parents=True,exist_ok=True)
            native_path=native_folder/f'{index:04d}.npy'
            header=str({'descr':'<f4','fortran_order':False,'shape':(360,640)})
            header+=' '*((64-(10+len(header)+1)%64)%64)+'\n'
            with native_path.open('xb') as stream:
                stream.write(b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header.encode())
                array.array('f',(v.r/100 if math.isfinite(v.r) and 0<v.r<10000 else 0. for v in values)).tofile(stream)
            rgb_path=rgb_folder/f'{index:04d}.png';assert rgb_path.is_file()
            evidence.update(native_path=f'native/{index:04d}.npy',native_sha256=sha(native_path),rgb_sha256=sha(rgb_path))
            geometry.append(evidence)
            frames.append(dict(id=case['name'],sample_index=index,clip_id=case['clip_id'],
                frame_in_clip=case['frame_in_clip'],time_s=case['time_s'],rgb_path=f'rgb/{index:04d}.png',rgb_sha256=sha(rgb_path)))
            index+=1;warm=0;stage=2
            write(OUT/'progress.json',dict(status='RUNNING',frames=index,expected=576))
            if index==len(cases):
                fx=640/(2*math.tan(math.radians(50)))
                write(OUT/'observations/manifest.json',dict(schema='ba-local-rescue-fresh-rgb-v1',
                    calibration=dict(width=640,height=360,fx=fx,fy=fx,cx=319.5,cy=179.5,hfov_deg=100.,depth_axis='OPTICAL_Z_METRES'),
                    sampling='POSED_QUASI_STATIC_SAMPLED_TRAJECTORY_NOT_REAL_TIME',
                    authority='RGB_AND_FIXED_CAMERA_CALIBRATION_ONLY_NO_NATIVE_DEPTH_OR_OBJECT_GEOMETRY',frames=frames))
                finish()
            else:after=0
    except Exception:finish(traceback.format_exc())


handle=u.register_slate_post_tick_callback(tick)
