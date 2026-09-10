"""MZ6 settled scheduled sequence capture; exact transforms evaluator-only."""
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

sys.path.insert(0, str(Path(__file__).parent))
from ue_capture_readiness import CaptureReadiness

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
frame_prepared = False
readiness = CaptureReadiness(u, timeout=900)
report = {'status': 'RUNNING', 'expected_map_sha256': EXPECTED,
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
    assert len(objects) == len(case['objects'])
    for actor, obj in zip(objects, case['objects']):
        actor.set_actor_location(u.Vector(*(v * 100 for v in obj['center_m'])), False, False)
        actor.set_actor_scale3d(u.Vector(*obj['size_m']))
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
    report['sampling'] = 'POSED_SETTLED_SAMPLES_NOMINAL_12HZ_NOT_WALLCLOCK_WALKING'
    report['frame_count'] = len(frames)
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
    global after, stage, world, rgb, depth, cases, index, warm, spec_calibration, frame_prepared
    if finished or time.monotonic() < after:
        return
    after = float('inf')  # Loading can reenter Slate callbacks.
    try:
        if stage == 0:
            report['map_sha256_before'] = hashlib.sha256(MAP_FILE.read_bytes()).hexdigest()
            assert report['map_sha256_before'] == EXPECTED, 'Frozen Willow hash mismatch'
            spec = json.loads(SPEC.read_text(encoding='utf-8-sig'))
            cases = spec['cases']
            spec_calibration = spec['calibration']
            assert len(cases) == 200
            assert cases and len({c['name'] for c in cases}) == len(cases)
            write(OUT / 'evaluator/spec.json', spec)
            report['spec_sha256'] = hashlib.sha256(SPEC.read_bytes()).hexdigest()
            assert levels.load_level(MAP)
            world = editor.get_editor_world()
            stage = 1
            after = time.monotonic() + 30
            return
        if stage == 1:
            rgb = component(u.SceneCaptureSource.SCS_FINAL_COLOR_LDR, u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
            depth = component(u.SceneCaptureSource.SCS_SCENE_DEPTH, u.TextureRenderTargetFormat.RTF_RGBA32F)
            stage = 2
        if stage == 2:
            if not frame_prepared:
                prepare(cases[index])
                rgb.capture_component2d.capture_scene()
                readiness.begin(world, rgb.capture_component2d)
                frame_prepared = True
                after = 0
                return
            if readiness.state != 'READY':
                if not readiness.poll():
                    write(OUT / 'progress.json', dict(phase='READINESS', index=index, readiness=readiness.receipt()))
                    after = time.monotonic() + .1
                    return
                report.setdefault('view_readiness', []).append(dict(index=index, **readiness.receipt()))
            rgb.capture_component2d.capture_scene()
            warm += 1
            if warm <= (32 if index == 0 else 16):
                after = 0
                return
            depth.capture_component2d.capture_scene()
            folder = OUT / 'model/sample'
            folder.mkdir(parents=True, exist_ok=True)
            u.RenderingLibrary.export_render_target(world, rgb.capture_component2d.texture_target, str(folder), f'{index:04d}.png')
            values = u.RenderingLibrary.read_render_target_raw(world, depth.capture_component2d.texture_target, normalize=False)
            assert len(values) == 640 * 360
            header = str({'descr': '<f4', 'fortran_order': False, 'shape': (360, 640)})
            header += ' ' * ((64 - (10 + len(header) + 1) % 64) % 64) + '\n'
            native_folder = OUT / 'evaluator/native'
            native_folder.mkdir(parents=True, exist_ok=True)
            with (native_folder / f'{index:04d}.npy').open('wb') as f:
                f.write(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header.encode())
                array.array('f', (v.r / 100 if math.isfinite(v.r) and 0 < v.r < 10000 else 0. for v in values)).tofile(f)
            capture_times.append(time.monotonic() - started)
            pose = dict(cases[index]['camera'])
            pose.setdefault('roll', 0.)
            frames.append({'sample_index': index, 'clip_id': cases[index]['clip_id'],
                           'rgb_path': f'sample/{index:04d}.png',
                           'nominal_time_s': cases[index]['nominal_time_s']})
            index += 1
            warm = 0
            frame_prepared = False
            if index == len(cases):
                write(OUT / 'model/sensor_manifest.json', {
                    'calibration': spec_calibration,
                    'frames': frames})
                finish()
            else:
                after = 0
    except Exception:
        finish(traceback.format_exc())


handle = u.register_slate_post_tick_callback(tick)
