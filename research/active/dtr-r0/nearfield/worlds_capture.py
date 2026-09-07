"""G14-A procedural worlds; all-scene visible native-depth truth is generated after capture."""
import array
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
import traceback
import unreal as u

# UE removes __file__ after executing the script; resolve imports before callbacks.
if os.environ.get('BA_UE_DEPTH_EXPORT') or os.environ.get('BA_UE_RGB_EXPORT') or os.environ.get('BA_UE_SETTLING') or os.environ.get('BA_UE_PAIR_EXPORT'):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ue_depth_export import export_depth
    from ue_rgb_export import RgbExporter
    from ue_settling import settling_count
    from ue_pair_export import create_pair_exporter

import sys
sys.path.insert(0,str(Path(__file__).parent))
from worlds_scene import build_world, destroy_world
scene_actors=[]
scene_key=None
world_changed=False
beauty=None

pair_exporter = create_pair_exporter(u, os.environ['BA_UE_PAIR_EXPORT']) if os.environ.get('BA_UE_PAIR_EXPORT') else None

rgb_exporter = RgbExporter(u, os.environ.get('BA_UE_RGB_EXPORT', 'legacy')) if os.environ.get('BA_UE_RGB_EXPORT') and not pair_exporter else None

STARTUP = os.environ.get('BA_UE_STARTUP', 'fixed')
if STARTUP not in ('fixed', 'ready'):
    raise ValueError('Unknown startup policy: ' + STARTUP)
SESSION = globals().get('__ba_session__')
readiness = None
first_prepared = False
readiness_progress_at = 0
if STARTUP == 'ready':
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ue_capture_readiness import CaptureReadiness
    readiness = CaptureReadiness(u)

OUT = Path(os.environ['BA_NEARFIELD_OUTPUT'])
SPEC = Path(os.environ['BA_NEARFIELD_SPEC'])
EXPECTED = 'cf35e5c9df54cd0f781f09ea8105fe8ef6078ed0822d4e594d64216e79a254fb'
MAP = '/Game/StreetLab/WillowSampleV1'
MAP_FILE = Path(u.Paths.project_dir()) / 'Content/StreetLab/WillowSampleV1.umap'
api = u.get_editor_subsystem(u.EditorActorSubsystem)
levels = u.get_editor_subsystem(u.LevelEditorSubsystem)
editor = u.get_editor_subsystem(u.UnrealEditorSubsystem)
captures, objects, frames = [], [], []
world = None
stage = index = warm = 0
after = time.monotonic() + (8 if STARTUP == 'fixed' else 0)
finished = False
started = time.monotonic()
started_unix_s = time.time()
capture_times = []
profiles = []
frame_started = None
report = {'status': 'RUNNING', 'pid': os.getpid(), 'startup_policy': STARTUP, 'startup_settling_cadence': 'tick' if STARTUP == 'ready' else os.environ.get('BA_UE_CADENCE', 'tick'), 'map_reused': False, 'capture_time_semantics': 'GPU_PAIR_SUBMISSION' if pair_exporter else 'SYNCHRONOUS_READBACK_COMPLETE', 'settling_policy': os.environ.get('BA_UE_SETTLING', 'full'), 'cadence': os.environ.get('BA_UE_CADENCE', 'tick'), 'expected_map_sha256': EXPECTED,
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'scene_helper_sha256': {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('worlds_scene.py','worlds_materials.py')}}
report['started_unix_s'] = started_unix_s


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def clear_objects():
    while objects:
        actor = objects[-1]
        if not api.destroy_actor(actor):
            raise RuntimeError('Task object destruction failed')
        objects.pop()


def component(source, fmt, width=640, height=360):
    actor = api.spawn_actor_from_class(u.SceneCapture2D, u.Vector(0, 0, 100000))
    captures.append(actor)
    c = actor.capture_component2d
    c.capture_every_frame = False
    c.capture_on_movement = False
    c.always_persist_rendering_state = True
    c.capture_source = source
    c.texture_target = u.RenderingLibrary.create_render_target2d(world, width, height, fmt)
    c.fov_angle = 100
    if source == u.SceneCaptureSource.SCS_FINAL_COLOR_LDR:
        c.texture_target.target_gamma = 2.2
        c.set_editor_property('show_flag_settings', [u.EngineShowFlagsSetting(show_flag_name='TemporalAA',enabled=True),u.EngineShowFlagsSetting(show_flag_name='AntiAliasing',enabled=True)])
        pp = max((a for a in api.get_all_level_actors() if isinstance(a, u.PostProcessVolume)), key=lambda a: a.priority)
        c.post_process_settings = pp.settings
        c.post_process_blend_weight = 1.
    return actor


def prepare(case):
    global scene_actors, scene_key, world_changed
    key=(case['world_id'],case['layout_id'])
    world_changed=(key!=scene_key)
    if world_changed:
        clear_objects()
        destroy_world(api,scene_actors)
        start_build=time.perf_counter()
        scene_actors,metadata=build_world(u,api,*key)
        scene_key=key
        report.setdefault('world_builds',[]).append(dict(world_id=key[0],layout_id=key[1],elapsed_s=time.perf_counter()-start_build,metadata=metadata))
        settings=rgb.capture_component2d.post_process_settings
        exposure=9.5 if key[0]=='sidewalk' else 5.8
        # Borrow the existing CC0 overcast lighting setup, without saving Willow.
        hdr=u.load_asset('/Game/StreetLabV3/OvercastCourtyard')
        assert hdr, 'Missing reused overcast HDRI'
        for actor in api.get_all_level_actors():
            if isinstance(actor,u.DirectionalLight):
                actor.light_component.set_editor_property('intensity',0. if 'fill' in actor.get_actor_label().lower() else 2200.)
                actor.light_component.set_editor_property('light_source_angle',5.)
            elif isinstance(actor,u.SkyLight):
                c=actor.light_component
                c.set_editor_property('source_type',u.SkyLightSourceType.SLS_SPECIFIED_CUBEMAP)
                c.set_editor_property('real_time_capture',False)
                c.set_editor_property('cubemap',hdr)
                c.set_editor_property('intensity',350.)
                c.set_editor_property('lower_hemisphere_is_black',False)
                c.recapture_sky()
        for name,value in [('auto_exposure_min_brightness',exposure),('auto_exposure_max_brightness',exposure),('auto_exposure_bias',0.),('bloom_intensity',0.),('motion_blur_amount',0.),('vignette_intensity',0.),('lens_flare_intensity',0.),('film_grain_intensity',0.),('lumen_scene_lighting_quality',2.),('lumen_final_gather_quality',2.)]:
            settings.set_editor_property('override_'+name,True)
            settings.set_editor_property(name,value)
        settings.override_dynamic_global_illumination_method=True
        settings.dynamic_global_illumination_method=u.DynamicGlobalIlluminationMethod.LUMEN
        settings.override_reflection_method=True
        settings.reflection_method=u.ReflectionMethod.LUMEN
        rgb.capture_component2d.post_process_settings=settings
        if beauty:
            beauty.capture_component2d.post_process_settings=settings
        report['world_builds'][-1]['render_profile']=dict(name='RESEARCH_REALISTIC_STATIC_V2',resolution=[640,360],hfov=100.,fixed_exposure_ev100=exposure,Lumen=True,lumen_quality=2.,lighting='reused OvercastCourtyard HDRI; sky350; sun2200lux angle5deg',temporal_aa_showflag=True,motion_blur=0,bloom=0,vignette=0,lens_flare=0,film_grain=0,sensor_noise='NOT_SIMULATED',camera_claim='fixed wearable-view geometry, not calibrated real sensor')
    if not frames or frames[-1]['clip_id'] != case['clip_id']:
        clear_objects()
    for obj in (case.get('objects', []) if not objects and case.get('objects') else []):
        mesh_name = {'cube': 'Cube', 'cylinder': 'Cylinder'}[obj['kind']]
        actor = api.spawn_actor_from_class(u.StaticMeshActor, u.Vector(*(v * 100 for v in obj['center_m'])))
        objects.append(actor)
        actor.set_actor_label('Nearfield task ' + obj['name'])
        actor.static_mesh_component.set_static_mesh(u.load_asset('/Engine/BasicShapes/' + mesh_name))
        actor.set_actor_scale3d(u.Vector(*obj['size_m']))
        actor.static_mesh_component.set_collision_profile_name('BlockAll')
        if obj.get('material'):
            material = u.load_asset(obj['material'])
            assert material, obj['material']
            actor.static_mesh_component.set_material(0, material)
    pose = case['camera']
    for actor in ([rgb, depth, beauty] if beauty else [rgb, depth]):
        actor.set_actor_location(u.Vector(*(pose[k] * 100 for k in ('x', 'y', 'z'))), False, False)
        actor.set_actor_rotation(u.Rotator(pitch=pose['pitch'], yaw=pose['yaw'], roll=pose.get('roll', 0)), False)


def finish(error=None):
    global finished
    if finished:
        return
    finished = True
    report['status'] = 'FAIL' if error else 'PASS'
    if error:
        report['error'] = error
    if pair_exporter:
        try:
            report['pair_exports'] = pair_exporter.finish()
            report['exports_drained_monotonic_s'] = time.monotonic() - started
        except Exception:
            report['status'] = 'FAIL'
            report['pair_error'] = traceback.format_exc()
    if rgb_exporter:
        try:
            report['rgb_exports'] = rgb_exporter.finish()
            report['exports_drained_monotonic_s'] = time.monotonic() - started
        except Exception:
            report['status'] = 'FAIL'
            report['rgb_error'] = traceback.format_exc()
    report['map_sha256_after'] = hashlib.sha256(MAP_FILE.read_bytes()).hexdigest()
    report['source_unchanged'] = report.get('map_sha256_before') == report['map_sha256_after'] == EXPECTED
    if not report['source_unchanged']:
        report['status'] = 'FAIL'
    report['frame_count'] = len(frames)
    report['profiles'] = profiles
    report['wall_elapsed_s'] = time.monotonic() - started
    report['actual_capture_monotonic_s'] = capture_times
    try:
        clear_objects()
        destroy_world(api,scene_actors)
        while captures:
            actor = captures.pop()
            if actor.capture_component2d.texture_target:
                u.RenderingLibrary.release_render_target2d(actor.capture_component2d.texture_target)
            if not api.destroy_actor(actor):
                raise RuntimeError('Task capture destruction failed')
    except Exception:
        report['cleanup_error'] = traceback.format_exc()
        report['status'] = 'FAIL'
    finally:
        report['readiness'] = readiness.receipt() if readiness else {'status': 'FIXED_WAIT_REFERENCE'}
        try:
            u.unregister_slate_post_tick_callback(handle)
        except Exception:
            report['status'] = 'FAIL'
            report['callback_cleanup_error'] = traceback.format_exc()
        write(OUT / 'receipt.json', report)
        if SESSION is not None:
            SESSION['on_complete'](report)
        else:
            u.SystemLibrary.quit_editor()


def tick(delta):
    global after, stage, world, rgb, depth, beauty, cases, index, warm, spec, frame_started, first_prepared, readiness_progress_at
    if finished or time.monotonic() < after:
        return
    if (OUT / 'stop.request').exists():
        finish('Task owner requested acquisition stop; partial source retained')
        return
    after = float('inf')  # Loading can reenter Slate callbacks.
    try:
        if stage == 0:
            report['map_sha256_before'] = hashlib.sha256(MAP_FILE.read_bytes()).hexdigest()
            assert report['map_sha256_before'] == EXPECTED, 'Frozen Willow hash mismatch'
            spec = json.loads(SPEC.read_text(encoding='utf-8-sig'))
            cases = spec['cases']
            assert cases and len({c['name'] for c in cases}) == len(cases)
            write(OUT / 'evaluator/spec.json', spec)
            report['spec_sha256'] = hashlib.sha256(SPEC.read_bytes()).hexdigest()
            current_world = editor.get_editor_world()
            reuse = (SESSION is not None and SESSION.get('map_loaded') == MAP
                     and current_world is not None
                     and current_world.get_path_name().split('.')[0] == MAP)
            if not reuse:
                assert levels.load_level(MAP)
            world = editor.get_editor_world()
            assert world.get_path_name().split('.')[0] == MAP, 'Unexpected loaded world'
            report['map_reused'] = reuse
            if SESSION is not None:
                SESSION['map_loaded'] = MAP
            stage = 1
            after = time.monotonic() + (30 if STARTUP == 'fixed' else 0)
            return
        if stage == 1:
            u.SystemLibrary.execute_console_command(world,'r.AntiAliasingMethod 2')
            if os.environ.get('BA_UE_CADENCE') == 'burst':
                for key in levels.get_viewport_config_keys():
                    levels.editor_set_viewport_realtime(False, key)
            rgb = component(u.SceneCaptureSource.SCS_FINAL_COLOR_LDR, u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
            depth = component(u.SceneCaptureSource.SCS_SCENE_DEPTH, u.TextureRenderTargetFormat.RTF_RGBA32F)
            if spec.get('appearance_preview'):
                beauty=component(u.SceneCaptureSource.SCS_FINAL_COLOR_LDR,u.TextureRenderTargetFormat.RTF_RGBA8_SRGB,1280,720)
                report['appearance_preview']=dict(resolution=[1280,720],purpose='native RGB visual inspection only; no high-resolution labels or model scoring')
            if readiness:
                prepare(cases[0])
                first_prepared = True
                rgb.capture_component2d.capture_scene()
                depth.capture_component2d.capture_scene()
                readiness.begin(world, rgb.capture_component2d)
                stage = 3
                after = 0
                return
            stage = 2
        if stage == 3:
            if not readiness.poll():
                if time.monotonic() >= readiness_progress_at:
                    write(OUT / 'progress.json', dict(phase='READINESS', frames=0, readiness=readiness.receipt()))
                    readiness_progress_at = time.monotonic() + 1
                after = 0
                return
            report['readiness'] = readiness.receipt()
            stage = 2
        if stage == 2:
            if pair_exporter and not pair_exporter.ready():
                after = 0
                return
            if rgb_exporter and not rgb_exporter.ready():
                after = 0
                return
            # Bounded EXR transport: hold the same simulation state if host I/O lags.
            if os.environ.get('BA_UE_DEPTH_EXPORT') == 'exr' and index >= 8:
                if not (OUT / f'evaluator/native/{index-8:04d}.npy').is_file():
                    after = 0
                    return
            if warm == 0:
                if index == 0:
                    report['acquisition_started_monotonic_s'] = time.monotonic() - started
                frame_started = time.perf_counter()
                if index == 0 and first_prepared:
                    first_prepared = False
                else:
                    prepare(cases[index])
                prepare_done = time.perf_counter()
            settling = settling_count(cases[index], cases[index-1] if index else None, index, 50, 26, os.environ.get('BA_UE_SETTLING', 'full')) if os.environ.get('BA_UE_SETTLING') else (50 if index == 0 else cases[index].get('settling_frames',26))
            if world_changed:
                settling=max(settling,60)
            if os.environ.get('BA_UE_CADENCE') == 'burst' and not beauty and not world_changed and not (STARTUP == 'ready' and index == 0):
                # Initial ready-mode settling spans real engine ticks so GPU
                # feedback (including Nanite) can progress before publication.
                for _ in range(settling + 1):
                    rgb.capture_component2d.capture_scene()
                warm = settling + 1
            else:
                rgb.capture_component2d.capture_scene()
                if beauty:
                    beauty.capture_component2d.capture_scene()
                warm += 1
            if warm <= settling:
                after = time.monotonic()+.05 if beauty else 0
                return
            settle_done = time.perf_counter()
            depth.capture_component2d.capture_scene()
            depth_submit_done = time.perf_counter()
            folder = OUT / 'model/sample'
            folder.mkdir(parents=True, exist_ok=True)
            if beauty:
                appearance=OUT/'appearance'
                appearance.mkdir(exist_ok=True)
                u.RenderingLibrary.export_render_target(world,beauty.capture_component2d.texture_target,str(appearance),f'{index:04d}.png')
            if pair_exporter:
                result = pair_exporter.export(world, rgb.capture_component2d.texture_target, depth.capture_component2d.texture_target, folder / f'{index:04d}.png', OUT / f'evaluator/native/{index:04d}.npy', index)
                report.setdefault('depth_exports', []).append(result)
                png_done = readback_done = time.perf_counter()
            else:
                if rgb_exporter:
                    rgb_exporter.export(world, rgb.capture_component2d.texture_target, folder / f'{index:04d}.png', index)
                else:
                    u.RenderingLibrary.export_render_target(world, rgb.capture_component2d.texture_target, str(folder), f'{index:04d}.png')
                png_done = time.perf_counter()
                if os.environ.get('BA_UE_DEPTH_EXPORT'):
                    native_folder = OUT / 'evaluator/native'
                    result = export_depth(u, world, depth.capture_component2d.texture_target,
                                          native_folder / f'{index:04d}.npy',
                                          os.environ['BA_UE_DEPTH_EXPORT'], index)
                    report.setdefault('depth_exports', []).append(dict(sample_index=index, **result))
                    readback_done = time.perf_counter()
                else:
                    values = u.RenderingLibrary.read_render_target_raw(world, depth.capture_component2d.texture_target, normalize=False)
                    readback_done = time.perf_counter()
                    assert len(values) == 640 * 360
                    header = str({'descr': '<f4', 'fortran_order': False, 'shape': (360, 640)})
                    header += ' ' * ((64 - (10 + len(header) + 1) % 64) % 64) + '\n'
                    native_folder = OUT / 'evaluator/native'
                    native_folder.mkdir(parents=True, exist_ok=True)
                    with (native_folder / f'{index:04d}.npy').open('wb') as f:
                        f.write(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header.encode())
                        if spec.get('preflight'):
                            timings = {}
                            order = ('legacy','single_access') if index%2==0 else ('single_access','legacy')
                            outputs = {}
                            for mode in order:
                                tick_conversion = time.perf_counter()
                                if mode == 'legacy':
                                    outputs[mode] = array.array('f', (v.r / 100 if math.isfinite(v.r) and 0 < v.r < 10000 else 0. for v in values))
                                else:
                                    outputs[mode] = array.array('f', (r / 100 if math.isfinite(r) and 0 < r < 10000 else 0. for r in (v.r for v in values)))
                                timings[mode] = time.perf_counter()-tick_conversion
                            assert outputs['legacy'].tobytes() == outputs['single_access'].tobytes(), 'Native conversion byte mismatch'
                            report.setdefault('conversion_probe', []).append(dict(index=index,seconds=timings,bytes_equal=True))
                            outputs['single_access'].tofile(f)
                        elif spec.get('conversion_mode') == 'single_access':
                            array.array('f', (r / 100 if math.isfinite(r) and 0 < r < 10000 else 0. for r in (v.r for v in values))).tofile(f)
                        else:
                            array.array('f', (v.r / 100 if math.isfinite(v.r) and 0 < v.r < 10000 else 0. for v in values)).tofile(f)
            profiles.append(dict(sample_index=index,actual_settling_frames=settling,settling_frames=cases[index].get('settling_frames',8),prepare_and_settle_s=settle_done-frame_started,depth_submit_s=depth_submit_done-settle_done,png_export_s=png_done-depth_submit_done,native_readback_s=readback_done-png_done,native_marshal_write_s=time.perf_counter()-readback_done,total_s=time.perf_counter()-frame_started))
            capture_times.append(time.monotonic() - started)
            pose = dict(cases[index]['camera'])
            pose.setdefault('roll', 0.)
            frames.append({'sample_index': index, 'clip_id': cases[index]['clip_id'], 'frame_in_clip': cases[index]['frame_in_clip'],
                           'rgb_path': f'sample/{index:04d}.png',

                           'time_s': cases[index]['time_s']})
            index += 1
            write(OUT / 'progress.json', {'frames': index, 'total': len(cases), 'clip_id': cases[index-1]['clip_id'], 'wall_elapsed_s': time.monotonic()-started})
            warm = 0
            if index == len(cases):
                write(OUT / 'model/dataset.json', {
                    'schema': 'nf-g14-worlds-dataset-v1',
                    'calibration': {'width': 640, 'height': 360, 'horizontal_fov_degrees': 100., 'depth_max_m': 100., 'fixed_camera_height_m': 1.70, 'nominal_camera_pitch_degrees': -5.},
                    'sampling': spec['sampling'],
                    'pose_authority': 'NO_POSE_OR_SPEED_IN_MODEL_INPUT',
                    'camera_convention': 'UE_X_FORWARD_Y_RIGHT_Z_UP_METRES; horizontal_fov; principal_point=(319.5,179.5)',
                    'authority': 'RGB_FIXED_CALIBRATION_ONLY_NO_NATIVE_DEPTH_POSE_SPEED_OR_OBJECTS', 'frames': frames, 'samples': spec['samples']})
                write(OUT / 'evaluator/labels.json', spec['labels'])
                train_ids = {s['sample_id'] for s in spec['samples'] if s['split'] != 'test'}
                # Native visibility verifier materializes training labels after acquisition.
                finish()
            else:
                after = 0
    except Exception:
        finish(traceback.format_exc())


handle = u.register_slate_post_tick_callback(tick)
