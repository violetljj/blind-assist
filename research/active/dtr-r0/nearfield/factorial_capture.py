"""NF-G9-B grouped-intervention acquisition; settled poses with evaluator-only native depth."""
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
if os.environ.get('BA_UE_DEPTH_EXPORT'):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from ue_depth_export import export_depth

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
after = time.monotonic() + 8
finished = False
started = time.monotonic()
capture_times = []
profiles = []
frame_started = None
report = {'status': 'RUNNING', 'cadence': os.environ.get('BA_UE_CADENCE', 'tick'), 'expected_map_sha256': EXPECTED,
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def clear_objects():
    while objects:
        api.destroy_actor(objects.pop())


def component(source, fmt):
    actor = api.spawn_actor_from_class(u.SceneCapture2D, u.Vector(0, 0, 100000))
    captures.append(actor)
    c = actor.capture_component2d
    c.capture_every_frame = False
    c.capture_on_movement = False
    c.always_persist_rendering_state = True
    c.capture_source = source
    c.texture_target = u.RenderingLibrary.create_render_target2d(world, 640, 360, fmt)
    c.fov_angle = 100
    if source == u.SceneCaptureSource.SCS_FINAL_COLOR_LDR:
        c.texture_target.target_gamma = 2.2
        pp = max((a for a in api.get_all_level_actors() if isinstance(a, u.PostProcessVolume)), key=lambda a: a.priority)
        c.post_process_settings = pp.settings
        c.post_process_blend_weight = 1.
    return actor


def prepare(case):
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
    for actor in (rgb, depth):
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
        while captures:
            actor = captures.pop()
            if actor.capture_component2d.texture_target:
                u.RenderingLibrary.release_render_target2d(actor.capture_component2d.texture_target)
            api.destroy_actor(actor)
    except Exception:
        report['cleanup_error'] = traceback.format_exc()
        report['status'] = 'FAIL'
    finally:
        write(OUT / 'receipt.json', report)
        u.unregister_slate_post_tick_callback(handle)
        u.SystemLibrary.quit_editor()


def tick(delta):
    global after, stage, world, rgb, depth, cases, index, warm, spec, frame_started
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
            assert levels.load_level(MAP)
            world = editor.get_editor_world()
            stage = 1
            after = time.monotonic() + 30
            return
        if stage == 1:
            if os.environ.get('BA_UE_CADENCE') == 'burst':
                for key in levels.get_viewport_config_keys():
                    levels.editor_set_viewport_realtime(False, key)
            rgb = component(u.SceneCaptureSource.SCS_FINAL_COLOR_LDR, u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
            depth = component(u.SceneCaptureSource.SCS_SCENE_DEPTH, u.TextureRenderTargetFormat.RTF_RGBA32F)
            stage = 2
        if stage == 2:
            # Bounded EXR transport: hold the same simulation state if host I/O lags.
            if os.environ.get('BA_UE_DEPTH_EXPORT') == 'exr' and index >= 8:
                if not (OUT / f'evaluator/native/{index-8:04d}.npy').is_file():
                    after = 0
                    return
            if warm == 0:
                frame_started = time.perf_counter()
                prepare(cases[index])
                prepare_done = time.perf_counter()
            settling = (50 if index == 0 else cases[index].get('settling_frames',26))
            if os.environ.get('BA_UE_CADENCE') == 'burst':
                # Same number of view renders, without waiting for unused editor ticks.
                for _ in range(settling + 1):
                    rgb.capture_component2d.capture_scene()
                warm = settling + 1
            else:
                rgb.capture_component2d.capture_scene()
                warm += 1
            if warm <= settling:
                after = 0
                return
            settle_done = time.perf_counter()
            depth.capture_component2d.capture_scene()
            depth_submit_done = time.perf_counter()
            folder = OUT / 'model/sample'
            folder.mkdir(parents=True, exist_ok=True)
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
            profiles.append(dict(sample_index=index,settling_frames=cases[index].get('settling_frames',8),prepare_and_settle_s=settle_done-frame_started,depth_submit_s=depth_submit_done-settle_done,png_export_s=png_done-depth_submit_done,native_readback_s=readback_done-png_done,native_marshal_write_s=time.perf_counter()-readback_done,total_s=time.perf_counter()-frame_started))
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
                    'schema': 'nf-g9b-factorial-dataset-v1',
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
