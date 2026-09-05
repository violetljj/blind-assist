"""Transient UE depth calibration; run in a dedicated disposable editor process.

Set BA_UE_DEPTH_CALIBRATION_OUTPUT and BA_UE_DEPTH_CALIBRATION_OWNED_PROCESS=1.
Invoke with -ExecutePythonScript=<this file>; never saves a map or asset.
"""
import array
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

W, H, FOV = 640, 360, 100.0
TOLERANCE_M = .02
CASES = [(pitch, distance) for pitch in (0., -10.) for distance in (1., 3., 6.)]


def convert_depth(raw):
    """Identical red-channel conversion to capture_street_closed_loop.py."""
    return raw / 100 if math.isfinite(raw) and 0 < raw < 10000 else 0.0


def assess(raw_red, expected_m):
    if len(raw_red) != W * H:
        raise ValueError('empty_or_wrong_size_readback')
    samples = []
    for fy in (.15, .5, .85):
        for fx in (.15, .5, .85):
            x, y = round(fx * (W - 1)), round(fy * (H - 1))
            values = [convert_depth(raw_red[(y + dy) * W + x + dx])
                      for dy in range(-2, 3) for dx in range(-2, 3)]
            valid = all(v > 0 and math.isfinite(v) for v in values)
            error = max(abs(v - expected_m) for v in values)
            samples.append({'pixel': [x, y], 'valid': valid,
                            'min_m': min(values), 'max_m': max(values),
                            'max_absolute_error_m': error,
                            'passed': valid and error <= TOLERANCE_M})
    return {'passed': all(s['passed'] for s in samples), 'samples': samples,
            'expected_axial_m': expected_m,
            'max_absolute_error_m': max(s['max_absolute_error_m'] for s in samples)}


def run():
    import unreal as u
    if os.environ.get('BA_UE_DEPTH_CALIBRATION_OWNED_PROCESS') != '1':
        raise RuntimeError('Requires a dedicated disposable editor process')
    output = Path(os.environ['BA_UE_DEPTH_CALIBRATION_OUTPUT']).resolve()
    canonical = (Path(__file__).resolve().parents[4] / 'artifacts.local').resolve()
    if not output.is_relative_to(canonical):
        raise ValueError('Output must be under canonical artifacts.local')
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'authority': 'ENGINEERING_DEVELOPMENT_ONLY',
              'resolution': [W, H], 'horizontal_fov_degrees': FOV,
              'absolute_error_limit_m': TOLERANCE_M, 'cases': [],
              'depth_path': 'SCS_SCENE_DEPTH / RTF_RGBA32F / red divided by 100; invalid becomes zero',
              'geometry': 'Fronto-parallel cube front face at known camera-axis distance; no scene occluders',
              'rgb_depth_edge_alignment': {'status': 'NOT_RUN',
                  'reason': 'This probe isolates depth units and axial geometry; no validated RGB edge pattern'},
              'engine_version': u.SystemLibrary.get_engine_version(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    actors = []
    state = {'index': 0, 'phase': 'setup', 'after': time.monotonic() + 3, 'handle': None}
    api = u.get_editor_subsystem(u.EditorActorSubsystem)

    def finish(error=None):
        cleanup_errors = []
        try:
            if state['handle'] is not None:
                u.unregister_slate_post_tick_callback(state['handle'])
            for actor in reversed(actors):
                try:
                    if not api.destroy_actor(actor):
                        cleanup_errors.append('destroy_actor returned false')
                except Exception as exc:
                    cleanup_errors.append(repr(exc))
        finally:
            report['cleanup_errors'] = cleanup_errors
            report['status'] = ('PASS' if error is None and not cleanup_errors
                                and len(report['cases']) == 6
                                and all(c['passed'] for c in report['cases']) else 'FAIL')
            if error is not None:
                report['error'] = error
            with (output / 'result.json').open('x', encoding='utf-8') as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
            print('DEPTH_CALIBRATION ' + report['status'] + ' ' + str(output))
            u.SystemLibrary.quit_editor()

    try:
        # Discarding the loaded world is authorized only in this owned process.
        world = u.EditorLoadingAndSavingUtils.new_blank_map(False)
        if world is None:
            raise RuntimeError('Transient blank world creation failed')
        camera = api.spawn_actor_from_class(u.SceneCapture2D, u.Vector(0, 0, 0))
        actors.append(camera)
        capture = camera.capture_component2d
        capture.capture_every_frame = False
        capture.capture_on_movement = False
        capture.always_persist_rendering_state = True
        capture.fov_angle = FOV
        capture.capture_source = u.SceneCaptureSource.SCS_SCENE_DEPTH
        capture.texture_target = u.RenderingLibrary.create_render_target2d(
            world, W, H, u.TextureRenderTargetFormat.RTF_RGBA32F)
        plane = api.spawn_actor_from_class(u.StaticMeshActor, u.Vector(0, 0, 0))
        actors.append(plane)
        plane.static_mesh_component.set_static_mesh(u.load_asset('/Engine/BasicShapes/Cube'))
        plane.static_mesh_component.set_collision_profile_name('NoCollision')
        # Engine cube is 100 cm wide: thickness 2 cm; face spans 30 x 30 m.
        plane.set_actor_scale3d(u.Vector(.02, 30., 30.))

        def tick(delta):
            if time.monotonic() < state['after']:
                return
            try:
                pitch, distance = CASES[state['index']]
                if state['phase'] == 'setup':
                    rotation = u.Rotator(pitch=pitch, yaw=0., roll=0.)
                    camera.set_actor_rotation(rotation, False)
                    plane.set_actor_rotation(rotation, False)
                    forward = u.MathLibrary.get_forward_vector(rotation)
                    center_cm = distance * 100 + 1.0
                    plane.set_actor_location(u.Vector(forward.x * center_cm,
                        forward.y * center_cm, forward.z * center_cm), False, False)
                    capture.capture_scene()
                    state.update(phase='read', after=time.monotonic() + 2.)
                    return
                capture.capture_scene()
                raw = [v.r for v in u.RenderingLibrary.read_render_target_raw(
                    world, capture.texture_target, normalize=False)]
                assessment = assess(raw, distance)
                payload = array.array('f', (convert_depth(v) for v in raw))
                if sys.byteorder != 'little':
                    payload.byteswap()
                name = 'case_%02d.depth-f32le' % state['index']
                data = payload.tobytes()
                with (output / name).open('xb') as stream:
                    stream.write(data)
                assessment.update(pitch_degrees=pitch, depth_path=name,
                    depth_bytes=len(data), depth_sha256=hashlib.sha256(data).hexdigest())
                report['cases'].append(assessment)
                state['index'] += 1
                if state['index'] == len(CASES):
                    finish()
                else:
                    state.update(phase='setup', after=time.monotonic() + .1)
            except Exception:
                finish(traceback.format_exc())
        state['handle'] = u.register_slate_post_tick_callback(tick)
    except Exception:
        finish(traceback.format_exc())


if __name__ == '__main__':
    run()
