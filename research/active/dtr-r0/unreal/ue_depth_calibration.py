"""Transient UE depth calibration; run in a dedicated disposable editor process.

Set BA_UE_DEPTH_CALIBRATION_OUTPUT and BA_UE_DEPTH_CALIBRATION_OWNED_PROCESS=1.
Invoke with -ExecCmds="py <this file>" so Slate callbacks remain alive.
Never saves a map or asset.
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
import statistics

W, H, FOV = 640, 360, 100.0
TOLERANCE_M = .02
CASES = [(pitch, distance) for pitch in (0., -10.) for distance in (1., 3., 6.)]
EDGE_CASES = [(pitch, x, y) for pitch in (0., -10.)
              for x, y in ((.25, .3), (.5, .5), (.75, .7))]
MOVING_EDGE_CASES = [(-10., .5 + index * 10 / W, .5) for index in range(4)]
EDGE_LIMIT_PX = 2.0


def alignment_schedule(index):
    """Frozen geometry and capture policy; later moving frames match live no-warmup."""
    moving_index = index - len(EDGE_CASES)
    moving = moving_index >= 0
    pitch, x, y = MOVING_EDGE_CASES[moving_index] if moving else EDGE_CASES[index]
    return {'pitch_degrees': pitch, 'expected_bounds_px': [x*W-60,y*H-45,x*W+60,y*H+45],
            'sequence': 'moving' if moving else 'static',
            'sequence_index': moving_index if moving else index,
            'time_s': round(moving_index*.1, 5) if moving else None,
            'warmup_captures': 32 if not moving or moving_index == 0 else 0,
            'settle_delay_s': .1 if moving else 2.0,
            'rgb_captures_outside_warmup': 2,
            'geometry_held_between_rgb_depth': True}


def assess_alignment(rgb, depth, bounds, width=W, height=H):
    """Compare independently observed intensity/depth crossings, never inferred labels."""
    if len(rgb) != width * height or len(depth) != width * height:
        raise ValueError('empty_or_wrong_size_alignment')
    left, top, right, bottom = bounds
    checks = []
    for edge, expected, horizontal, increasing in (
            ('left', left, True, True), ('right', right, True, False),
            ('top', top, False, True), ('bottom', bottom, False, False)):
        for fraction in (.3, .5, .7):
            cross = round((top + fraction * (bottom-top)) if horizontal
                          else (left + fraction * (right-left)))
            def line(values):
                return ([values[cross * width + i] for i in range(width)] if horizontal
                        else [values[i * width + cross] for i in range(height)])
            intensity, metric = line(rgb), line(depth)
            center = round(expected)
            if center < 16 or center + 16 >= len(intensity):
                raise ValueError('edge_outside_supported_region')
            low, high = range(center-12, center-6), range(center+6, center+12)
            foreground, background = (high, low) if increasing else (low, high)
            foreground_rgb = statistics.median(intensity[i] for i in foreground)
            background_rgb = statistics.median(intensity[i] for i in background)
            if not all(math.isfinite(v) for v in intensity):
                raise ValueError('invalid_rgb')
            if foreground_rgb - background_rgb < .15:
                raise ValueError('missing_or_weak_rgb_edge')
            for indices, distance in ((foreground, 3.), (background, 6.)):
                if any(not math.isfinite(metric[i]) or abs(metric[i]-distance) > TOLERANCE_M
                       for i in indices):
                    raise ValueError('invalid_depth_edge_plateau')
            def crossing(values, threshold, rises):
                found = []
                for i in range(center-8, center+8):
                    a, b = values[i], values[i+1]
                    if not math.isfinite(a) or not math.isfinite(b):
                        raise ValueError('invalid_edge_pixel')
                    if (a < threshold <= b) if rises else (a > threshold >= b):
                        found.append(i + .5 + (threshold-a)/(b-a))
                if len(found) != 1:
                    raise ValueError('missing_or_ambiguous_edge')
                return found[0]
            rgb_edge = crossing(intensity, (foreground_rgb+background_rgb)/2, increasing)
            depth_edge = crossing(metric, 4.5, not increasing)
            displacement = abs(rgb_edge-depth_edge)
            geometry_error = abs(depth_edge-expected)
            checks.append({'edge': edge, 'scanline': cross, 'rgb_edge_px': rgb_edge,
                           'depth_edge_px': depth_edge, 'expected_edge_px': expected,
                           'displacement_px': displacement, 'depth_geometry_error_px': geometry_error,
                           'rgb_contrast': foreground_rgb-background_rgb,
                           'passed': displacement <= EDGE_LIMIT_PX and geometry_error <= EDGE_LIMIT_PX})
    return {'passed': all(c['passed'] for c in checks), 'checks': checks,
            'max_displacement_px': max(c['displacement_px'] for c in checks)}


def verify(output):
    """Write an exclusive PASS/FAIL receipt, including verification exceptions."""
    output = Path(output).resolve()
    destination = output / 'alignment-validation.json'
    if destination.exists():
        raise FileExistsError(destination)
    try:
        return _verify(output)
    except Exception as error:
        receipt = {'status': 'FAIL', 'authority': 'ENGINEERING_DEVELOPMENT_ONLY',
                   'reason': str(error), 'exception_type': type(error).__name__,
                   'validator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        if (output/'result.json').is_file():
            receipt['result_sha256'] = hashlib.sha256((output/'result.json').read_bytes()).hexdigest()
        with destination.open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
        raise


def _verify(output):
    """Offline independent PNG decode and serialized metric/edge re-evaluation."""
    from PIL import Image
    output = Path(output).resolve()
    destination = output / 'alignment-validation.json'
    if destination.exists():
        raise FileExistsError(destination)
    report_path = output / 'result.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if (report['status'] != 'CAPTURE_COMPLETE_PENDING_ALIGNMENT_VALIDATION'
            or report.get('cleanup_errors') or len(report['cases']) != 6
            or len(report['rgb_depth_edge_alignment']['cases']) != len(EDGE_CASES)+len(MOVING_EDGE_CASES)):
        raise ValueError('incomplete_native_calibration')
    def payload(row, prefix):
        path = output / row[prefix+'_path']
        if path.resolve().parent != output:
            raise ValueError('payload_path_escape')
        data = path.read_bytes()
        if len(data) != row[prefix+'_bytes'] or hashlib.sha256(data).hexdigest() != row[prefix+'_sha256']:
            raise ValueError('payload_hash_mismatch')
        return path, data
    def depth_values(row):
        _, data = payload(row, 'depth')
        if len(data) != W*H*4:
            raise ValueError('depth_payload_size')
        values = array.array('f'); values.frombytes(data)
        if sys.byteorder != 'little': values.byteswap()
        return values
    for row, (pitch, distance) in zip(report['cases'], CASES):
        if row['pitch_degrees'] != pitch or row['expected_axial_m'] != distance:
            raise ValueError('metric_case_identity')
        if not assess([v*100 for v in depth_values(row)], distance)['passed']:
            raise ValueError('serialized_metric_calibration_failed')
    results = []
    for index, row in enumerate(report['rgb_depth_edge_alignment']['cases']):
        schedule = alignment_schedule(index)
        bounds = schedule['expected_bounds_px']
        pitch = schedule['pitch_degrees']
        if row['case'] != index or any(row.get(key) != value for key, value in schedule.items()):
            raise ValueError('alignment_case_identity')
        path, _ = payload(row, 'rgb')
        with Image.open(path) as image:
            if image.format != 'PNG' or image.size != (W,H):
                raise ValueError('rgb_format_or_size')
            intensity = [sum(pixel)/(3*255.) for pixel in image.convert('RGB').getdata()]
        measured = assess_alignment(intensity, depth_values(row), bounds)
        if not measured['passed']:
            raise ValueError('rgb_depth_alignment_exceeds_two_pixels')
        results.append(dict(case=row['case'], **schedule, **measured))
    receipt = {'status': 'PASS', 'authority': 'ENGINEERING_DEVELOPMENT_ONLY',
               'independent_decoder': 'Pillow', 'metric_cases_verified': 6,
               'alignment_cases_verified': len(results), 'edge_scanlines_verified': 12*len(results),
               'static_alignment_status': 'PASS', 'static_cases_verified': len(EDGE_CASES),
               'moving_alignment_status': 'PASS', 'moving_cases_verified': len(MOVING_EDGE_CASES),
               'edge_limit_px': EDGE_LIMIT_PX, 'cases': results,
               'result_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
               'validator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with destination.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    return receipt


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
              'rgb_depth_edge_alignment': {'status': 'PENDING_INDEPENDENT_VALIDATION',
                  'cases': [], 'edge_limit_px': EDGE_LIMIT_PX},
              'engine_version': u.SystemLibrary.get_engine_version(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    actors = []
    material_paths = []
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
            for path in material_paths:
                try:
                    if not u.EditorAssetLibrary.delete_asset(path):
                        cleanup_errors.append('Temporary material deletion failed: '+path)
                except Exception as exc:
                    cleanup_errors.append(repr(exc))
        finally:
            report['cleanup_errors'] = cleanup_errors
            report['status'] = ('CAPTURE_COMPLETE_PENDING_ALIGNMENT_VALIDATION' if error is None and not cleanup_errors
                                and len(report['cases']) == 6
                                and len(report['rgb_depth_edge_alignment']['cases']) == len(EDGE_CASES)+len(MOVING_EDGE_CASES)
                                and all(c['passed'] for c in report['cases']) else 'FAIL')
            if error is not None:
                report['error'] = error
            with (output / 'result.json').open('x', encoding='utf-8') as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
            print('DEPTH_CALIBRATION ' + report['status'] + ' ' + str(output))
            u.SystemLibrary.quit_editor()

    try:
        original_world = u.get_editor_subsystem(u.UnrealEditorSubsystem).get_editor_world()
        report['inherited_map'] = original_world.get_path_name()
        config_path = Path(u.Paths.project_dir())/'Config/DefaultEngine.ini'
        report['project_config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        pp = max((a for a in api.get_all_level_actors() if isinstance(a, u.PostProcessVolume)
                  and a.get_editor_property('unbound')),
                 key=lambda a: a.get_editor_property('priority'), default=None)
        live_postprocess = pp.settings if pp else None
        report['inherited_postprocess_settings'] = live_postprocess.export_text() if live_postprocess else None
        report['inherited_unbound_postprocess'] = pp is not None
        report['calibration_target_emission'] = {'background': 0., 'foreground': 1024.,
            'reason': 'Unit emission was invisible under the unchanged live daylight exposure; target radiance changed, not camera settings or acceptance'}
        if live_postprocess is not None:
            report['inherited_exposure'] = {name: live_postprocess.get_editor_property(name)
                for name in ('auto_exposure_min_brightness', 'auto_exposure_max_brightness', 'auto_exposure_bias')}
        # pp.settings is an owner-backed struct view. Its owning actor is destroyed
        # by new_blank_map; never carry that view across world destruction.
        live_postprocess = None
        pp = None
        # Discarding the loaded world is authorized only in this owned process.
        world = u.EditorLoadingAndSavingUtils.new_blank_map(False)
        if world is None:
            raise RuntimeError('Transient blank world creation failed')
        if report['inherited_postprocess_settings'] is not None:
            live_postprocess = u.PostProcessSettings()
            live_postprocess.import_text(report['inherited_postprocess_settings'])
        u.log('CALIBRATION_STAGE owned_postprocess_restored')
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
        rgb_actor = api.spawn_actor_from_class(u.SceneCapture2D, u.Vector(0,0,0))
        actors.append(rgb_actor)
        rgb_capture = rgb_actor.capture_component2d
        rgb_capture.capture_every_frame = False
        rgb_capture.capture_on_movement = False
        rgb_capture.always_persist_rendering_state = True
        rgb_capture.fov_angle = FOV
        rgb_capture.capture_source = u.SceneCaptureSource.SCS_FINAL_COLOR_LDR
        rgb_capture.texture_target = u.RenderingLibrary.create_render_target2d(
            world, W, H, u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
        rgb_capture.texture_target.set_editor_property('target_gamma', 2.2)
        for component in (capture, rgb_capture):
            if live_postprocess is not None:
                component.post_process_settings = live_postprocess
                component.post_process_blend_weight = 1.0
            settings = component.post_process_settings
            settings.override_dynamic_global_illumination_method = True
            settings.dynamic_global_illumination_method = u.DynamicGlobalIlluminationMethod.LUMEN
            settings.override_reflection_method = True
            settings.reflection_method = u.ReflectionMethod.LUMEN
            component.post_process_settings = settings
        u.log('CALIBRATION_STAGE captures_configured')
        # Existing opaque unlit engine shader; no MaterialEditor graph mutation/compile.
        material_path = '/Engine/EngineDebugMaterials/LevelColorationUnlitMaterial'
        parent_material = u.load_asset(material_path)
        u.log('CALIBRATION_STAGE existing_material_loaded')
        if parent_material is None:
            raise RuntimeError('Opaque unlit calibration material missing')
        parameter_names = [str(name) for name in u.MaterialEditingLibrary.get_vector_parameter_names(parent_material)]
        if 'Color' not in parameter_names:
            raise RuntimeError('Expected Color vector parameter missing: '+str(parameter_names))
        report['calibration_material'] = {'parent': material_path, 'vector_parameters': parameter_names,
            'construction': 'Existing opaque unlit shader plus per-component dynamic instance; no material graph compile'}
        def set_color(component, value):
            instance = component.create_dynamic_material_instance(0, parent_material)
            if instance is None:
                raise RuntimeError('Dynamic material instance creation failed')
            instance.set_vector_parameter_value('Color', u.LinearColor(value,value,value,1.))
            return instance
        plane = api.spawn_actor_from_class(u.StaticMeshActor, u.Vector(0, 0, 0))
        actors.append(plane)
        plane.static_mesh_component.set_static_mesh(u.load_asset('/Engine/BasicShapes/Cube'))
        plane.static_mesh_component.set_collision_profile_name('NoCollision')
        # Engine cube is 100 cm wide: thickness 2 cm; face spans 30 x 30 m.
        plane.set_actor_scale3d(u.Vector(.02, 30., 30.))
        background_material = set_color(plane.static_mesh_component, 0.)
        target = api.spawn_actor_from_class(u.StaticMeshActor, u.Vector(-1000,0,0))
        actors.append(target)
        target.static_mesh_component.set_static_mesh(u.load_asset('/Engine/BasicShapes/Cube'))
        target.static_mesh_component.set_collision_profile_name('NoCollision')
        foreground_material = set_color(target.static_mesh_component, 1024.)
        u.log('CALIBRATION_STAGE targets_ready')

        def save_depth(raw, name):
            payload = array.array('f', (convert_depth(v) for v in raw))
            if sys.byteorder != 'little': payload.byteswap()
            data = payload.tobytes()
            with (output / name).open('xb') as stream: stream.write(data)
            return {'depth_path': name, 'depth_bytes': len(data),
                    'depth_sha256': hashlib.sha256(data).hexdigest()}

        def tick(delta):
            if time.monotonic() < state['after']:
                return
            try:
                edge_index = state['index'] - len(CASES)
                is_edge = edge_index >= 0
                if is_edge:
                    schedule = alignment_schedule(edge_index)
                    pitch = schedule['pitch_degrees']
                    left, top, right_px, bottom = schedule['expected_bounds_px']
                    cx, cy = (left+right_px)/(2*W), (top+bottom)/(2*H)
                    distance = 6.
                else:
                    pitch, distance = CASES[state['index']]
                if state['phase'] == 'setup':
                    rotation = u.Rotator(pitch=pitch, yaw=0., roll=0.)
                    camera.set_actor_rotation(rotation, False)
                    rgb_actor.set_actor_rotation(rotation, False)
                    plane.set_actor_rotation(rotation, False)
                    forward = u.MathLibrary.get_forward_vector(rotation)
                    center_cm = distance * 100 + 1.0
                    plane.set_actor_location(u.Vector(forward.x * center_cm,
                        forward.y * center_cm, forward.z * center_cm), False, False)
                    if is_edge:
                        focal = W/(2*math.tan(math.radians(FOV/2)))
                        right = u.MathLibrary.get_right_vector(rotation)
                        up = u.MathLibrary.get_up_vector(rotation)
                        lateral = (cx*W-W/2)*300/focal
                        vertical = (H/2-cy*H)*300/focal
                        target.set_actor_rotation(rotation, False)
                        target.set_actor_scale3d(u.Vector(.02, 120*3/focal, 90*3/focal))
                        target.set_actor_location(u.Vector(
                            forward.x*301+right.x*lateral+up.x*vertical,
                            forward.y*301+right.y*lateral+up.y*vertical,
                            forward.z*301+right.z*lateral+up.z*vertical), False, False)
                        state['warmups'] = schedule['warmup_captures']
                    if not is_edge or schedule['sequence'] != 'moving':
                        capture.capture_scene()
                    rgb_capture.capture_scene()
                    state.update(phase='warmup' if is_edge and state['warmups'] else 'read',
                                 after=time.monotonic() + (schedule['settle_delay_s'] if is_edge else 2.))
                    return
                if state['phase'] == 'warmup':
                    rgb_capture.capture_scene()
                    state['warmups'] -= 1
                    if state['warmups'] <= 0: state['phase'] = 'read'
                    return
                rgb_capture.capture_scene()
                capture.capture_scene()
                raw = [v.r for v in u.RenderingLibrary.read_render_target_raw(
                    world, capture.texture_target, normalize=False)]
                if is_edge:
                    name = 'edge_%02d.png' % edge_index
                    if (output/name).exists(): raise FileExistsError(name)
                    u.RenderingLibrary.export_render_target(world, rgb_capture.texture_target, str(output), name)
                    data = (output/name).read_bytes()
                    report['rgb_depth_edge_alignment']['cases'].append({
                        'case': edge_index, **schedule,
                        'rgb_path': name, 'rgb_bytes': len(data), 'rgb_sha256': hashlib.sha256(data).hexdigest(),
                        **save_depth(raw, 'edge_%02d.depth-f32le' % edge_index)})
                else:
                    assessment = assess(raw, distance)
                    assessment.update(pitch_degrees=pitch,
                        **save_depth(raw, 'case_%02d.depth-f32le' % state['index']))
                    report['cases'].append(assessment)
                state['index'] += 1
                if state['index'] == len(CASES) + len(EDGE_CASES) + len(MOVING_EDGE_CASES):
                    finish()
                else:
                    state.update(phase='setup', after=time.monotonic() + .1)
            except Exception:
                finish(traceback.format_exc())
        state['handle'] = u.register_slate_post_tick_callback(tick)
    except Exception:
        finish(traceback.format_exc())


if __name__ == '__main__':
    if '--verify' in sys.argv:
        print(json.dumps(verify(Path(sys.argv[sys.argv.index('--verify')+1]))))
    else:
        run()
