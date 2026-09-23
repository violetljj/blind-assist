"""UE source capture for the frozen CNH route comparison.

Run inside the editor with BA_CNH_CAPTURE_SPEC and BA_CNH_CAPTURE_OUTPUT.
Outside UE, ``finalize --output PATH`` converts native transport to the public
capture schema. No software renderer, collision proxy, or synthetic RGB fallback.
Requires PythonScriptPlugin, BlindAssistCapture and ProceduralMeshComponent.
"""
from __future__ import annotations

import argparse
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


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_npy(path, values, shape, dtype='<f4'):
    """Small stdlib transport writer; UE's embedded Python need not have numpy."""
    header = repr(dict(descr=dtype, fortran_order=False, shape=tuple(shape)))
    header += ' ' * ((64 - (10 + len(header) + 1) % 64) % 64) + '\n'
    code = {'<f4': 'f', '<u2': 'H', '|b1': 'B'}[dtype]
    data = array.array(code, values)
    if len(data) != math.prod(shape):
        raise ValueError('Transport shape mismatch')
    if sys.byteorder != 'little':
        data.byteswap()
    with Path(path).open('xb') as stream:
        stream.write(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header.encode())
        data.tofile(stream)


def basis(camera):
    p, y, r = (math.radians(camera.get(k, 0.)) for k in ('pitch', 'yaw', 'roll'))
    cp, sp, cy, sy, cr, sr = math.cos(p), math.sin(p), math.cos(y), math.sin(y), math.cos(r), math.sin(r)
    return ((cp*cy, cp*sy, sp),
            (sr*sp*cy-cr*sy, sr*sp*sy+cr*cy, -sr*cp),
            (-cr*sp*cy-sr*sy, -cr*sp*sy+sr*cy, cr*cp))


def camera_record(camera, rig):
    forward, right, up = basis(camera)
    origin = [camera[k] for k in ('x', 'y', 'z')]
    # Column vectors are optical right, down, forward in UE world X,Y,Z.
    matrix = [[right[k], -up[k], forward[k], origin[k]] for k in range(3)] + [[0, 0, 0, 1]]
    fx = rig['width']/(2*math.tan(math.radians(rig['hfov_deg']/2)))
    return dict(K=[[fx, 0, (rig['width']-1)/2], [0, fx, (rig['height']-1)/2], [0, 0, 1]],
                width=rig['width'], height=rig['height'], hfov_deg=rig['hfov_deg'],
                T_world_camera=matrix, baseline_m=rig['baseline_m'],
                camera_axes='X_RIGHT_Y_DOWN_Z_FORWARD', world_axes='UE_X_FORWARD_Y_RIGHT_Z_UP_METRES',
                T_camera_tof=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def id_color(identifier):
    if type(identifier) is not int or not 1 <= identifier <= 65534:
        raise ValueError('Instance ID must be an integer in 1..65534')
    lo, hi = identifier & 255, identifier >> 8
    # UE's deferred BaseColor GBuffer is quantized in sRGB. Choose its exact
    # codebook in linear space so high-byte IDs survive that quantization too.
    return [srgb_to_linear(v/255) for v in (lo, hi, lo ^ hi)]


def srgb_to_linear(value):
    return value/12.92 if value <= .04045 else ((value+.055)/1.055)**2.4


def linear_to_srgb(value):
    return 12.92*value if value <= .0031308 else 1.055*value**(1/2.4)-.055


def decode_id_color(rgb, known, tolerance=1.05/255):
    """Reject checksum failures and unknown/mixed pixels instead of inventing IDs."""
    if not all(math.isfinite(v) and -.001 <= v <= 1.001 for v in rgb):
        return 65535
    codes = [max(0, min(255, round(linear_to_srgb(v)*255))) for v in rgb]
    identifier = codes[0] + 256*codes[1]
    if identifier == 0 and max(abs(v) for v in rgb) <= tolerance:
        return 0
    if identifier not in known or codes[2] != (codes[0] ^ codes[1]):
        return 65535
    if max(abs(linear_to_srgb(a)-b/255) for a, b in zip(rgb, codes)) > tolerance:
        return 65535
    return identifier


def validate_spec(spec):
    if spec.get('scene_layer') != 'PRIMITIVE_UNIT_TEST' or spec.get('benchmark_eligible') is not False:
        raise ValueError('Primitive capture is SENSOR_UNIT_TEST_ONLY; legacy or benchmark specifications require a different admitted source implementation')
    rig = spec['rig']
    if (rig['width'], rig['height'], rig['hfov_deg'], rig['baseline_m']) != (640, 360, 100, .06):
        raise ValueError('This capture implements only the frozen 640x360/HFOV100/6cm rig')
    names = set()
    for frame in spec['frames']:
        name = frame['id']
        if name in names or name in ('.', '..') or not name or any(c in name for c in '/\\:'):
            raise ValueError('Frame IDs must be unique safe directory names')
        names.add(name)
        ids = [obj['id'] for obj in frame['objects']]
        if len(set(ids)) != len(ids):
            raise ValueError('Duplicate instance ID')
        for obj in frame['objects']:
            id_color(obj['id'])
            if obj['kind'] not in ('cube', 'cylinder'):
                raise ValueError('Capture currently supports explicitly declared primitive assets only')
            if len(obj['size_m']) != 3 or not all(math.isfinite(v) and v > 0 for v in obj['size_m']):
                raise ValueError('Invalid primitive scale')
            if not 0 < obj['nir_rho'] <= 1 or not all(0 <= c <= 1 for c in obj['albedo']):
                raise ValueError('Invalid separate visible/NIR material')
    if not names:
        raise ValueError('No frames')


def engine():
    import unreal as u
    from ue_capture_readiness import CaptureReadiness
    from ue_pair_export import PairExporter

    out = Path(os.environ['BA_CNH_CAPTURE_OUTPUT'])
    spec_path = Path(os.environ['BA_CNH_CAPTURE_SPEC'])
    spec = json.loads(spec_path.read_text(encoding='utf-8-sig'))
    validate_spec(spec)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'raw-manifest.json').exists() or (out/'engine-receipt.json').exists():
        raise FileExistsError('Fresh capture output required')
    rig = spec['rig']; width, height = rig['width'], rig['height']
    api = u.get_editor_subsystem(u.EditorActorSubsystem)
    world = u.EditorLoadingAndSavingUtils.new_blank_map(False)
    actors, scene, captures, rows, exported = [], [], [], [], {}
    started = time.monotonic()
    state = dict(index=0, stage='PREPARE', warm=0, finished=False, scene_signature=None, geometry=None)
    report = dict(status='RUNNING', source_sha256=sha(__file__), spec_sha256=sha(spec_path),
                  engine_version=u.SystemLibrary.get_engine_version(),
                  authority='UE_GPU_RENDERED_CONTROLLED_PROCEDURAL_LAB_DEVELOPMENT',
                  normal_encoding='RAW_SCS_NORMAL_WORLD_XYZ_REQUIRES_CANARY_VERIFICATION',
                  instance_encoding='LINEAR_BASE_COLOR_SRGB_CODEBOOK_TWO_BYTES_XOR_CHECKSUM_NOT_STENCIL',
                  sampling='POSED_STATIC_SCENE_CONTINUOUS_TRAJECTORY_METADATA_NOT_REALTIME')
    write_json(out/'progress.json', report)
    # GBuffer attribute outputs require deferred rendering; no AA for categorical pass.
    for command in ('r.ForwardShading 0', 'r.AntiAliasingMethod 0', 'r.MotionBlurQuality 0'):
        u.SystemLibrary.execute_console_command(world, command)
    if not hasattr(u, 'ProceduralMeshLibrary'):
        raise RuntimeError('Enable ProceduralMeshComponent for actual render-LOD mesh export')
    mesh_library = u.ProceduralMeshLibrary
    if not hasattr(mesh_library, 'get_section_from_static_mesh'):
        raise RuntimeError('Render LOD extraction API unavailable; no collision proxy fallback')
    readiness = CaptureReadiness(u, timeout=900)
    pairs = PairExporter(u, limit=4)

    # In-memory generated materials; no save_asset/save_level call touches shared Content.
    package = '/Game/CNHRouteTransient_' + str(os.getpid())
    material = u.AssetToolsHelpers.get_asset_tools().create_asset('CaptureMaterial', package, u.Material, u.MaterialFactoryNew())
    expression = u.MaterialEditingLibrary.create_material_expression(material, u.MaterialExpressionVectorParameter)
    expression.set_editor_property('parameter_name', 'CaptureColor')
    expression.set_editor_property('default_value', u.LinearColor(.5, .5, .5, 1))
    u.MaterialEditingLibrary.connect_material_property(expression, '', u.MaterialProperty.MP_BASE_COLOR)
    roughness = u.MaterialEditingLibrary.create_material_expression(material, u.MaterialExpressionConstant)
    roughness.r = .8
    u.MaterialEditingLibrary.connect_material_property(roughness, '', u.MaterialProperty.MP_ROUGHNESS)
    u.MaterialEditingLibrary.recompile_material(material)
    light = api.spawn_actor_from_class(u.DirectionalLight, u.Vector(0, 0, 500))
    light.set_actor_rotation(u.Rotator(pitch=-55, yaw=-25, roll=0), False)
    actors.append(light)
    fill = api.spawn_actor_from_class(u.PointLight, u.Vector(-200, 0, 270))
    fill.light_component.set_editor_property('attenuation_radius', 2000.)
    actors.append(fill)
    report['illumination_control'] = dict(authority='ASSUMED_UE_RENDERING_CONTROL_NOT_LUX_CALIBRATION',
        directional={'daylight': 3., 'indoor': .5, 'dusk': 1.},
        fixed_point_fill={'daylight': 1000., 'indoor': 5000., 'dusk': 500.},
        fixed_point_position_m=[-2., 0., 2.7], multiplier='spec.illumination.intensity')

    def make_capture(name, source, fmt):
        actor = api.spawn_actor_from_class(u.SceneCapture2D, u.Vector(0, 0, 0))
        actor.set_actor_label('CNH ' + name); actors.append(actor); captures.append(actor)
        c = actor.capture_component2d
        c.capture_every_frame = False; c.capture_on_movement = False
        c.always_persist_rendering_state = True; c.fov_angle = rig['hfov_deg']
        c.capture_source = source
        c.texture_target = u.RenderingLibrary.create_render_target2d(world, width, height, fmt)
        c.texture_target.target_gamma = 2.2 if name.startswith('rgb') else 1.
        pp = c.post_process_settings
        for key, value in [('override_auto_exposure_min_brightness', True), ('override_auto_exposure_max_brightness', True),
                           ('auto_exposure_min_brightness', 1.), ('auto_exposure_max_brightness', 1.),
                           ('override_motion_blur_amount', True), ('motion_blur_amount', 0.)]:
            setattr(pp, key, value)
        c.post_process_settings = pp
        return actor

    rgb_left = make_capture('rgb_left', u.SceneCaptureSource.SCS_FINAL_COLOR_LDR, u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
    rgb_right = make_capture('rgb_right', u.SceneCaptureSource.SCS_FINAL_COLOR_LDR, u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
    depth_left = make_capture('depth_left', u.SceneCaptureSource.SCS_SCENE_DEPTH, u.TextureRenderTargetFormat.RTF_RGBA32F)
    depth_right = make_capture('depth_right', u.SceneCaptureSource.SCS_SCENE_DEPTH, u.TextureRenderTargetFormat.RTF_RGBA32F)
    normal = make_capture('normal', u.SceneCaptureSource.SCS_NORMAL, u.TextureRenderTargetFormat.RTF_RGBA32F)
    albedo = make_capture('albedo', u.SceneCaptureSource.SCS_BASE_COLOR, u.TextureRenderTargetFormat.RTF_RGBA32F)

    def export_mesh(mesh):
        key = mesh.get_path_name()
        if key in exported:
            return exported[key]
        sections = []
        for section in range(mesh.get_num_sections(0)):
            vertices, triangles, normals, uv, tangents = mesh_library.get_section_from_static_mesh(mesh, 0, section)
            if not vertices or not triangles:
                raise RuntimeError('Empty render mesh section: ' + key)
            sections.append(dict(vertices_m=[[v.x/100, v.y/100, v.z/100] for v in vertices],
                                 triangles=list(triangles), normals=[[v.x, v.y, v.z] for v in normals],
                                 uv=[[v.x, v.y] for v in uv], section=section))
        relative = 'geometry/mesh-' + hashlib.sha256(key.encode()).hexdigest()[:16] + '.json'
        write_json(out/relative, dict(asset_path=key, lod_index=0, coordinates='UE_LOCAL_XYZ_METRES', sections=sections))
        exported[key] = dict(path=relative, sha256=sha(out/relative), asset_path=key, lod_index=0)
        return exported[key]

    def clear_scene():
        while scene:
            _, actor, _ = scene.pop()
            if not api.destroy_actor(actor):
                raise RuntimeError('Task actor cleanup failed')

    def prepare(frame):
        illumination = frame.get('illumination', dict(mode='daylight', intensity=1.))
        mode, gain = illumination['mode'], illumination['intensity']
        light.light_component.set_editor_property('intensity', report['illumination_control']['directional'][mode]*gain)
        fill.light_component.set_editor_property('intensity', report['illumination_control']['fixed_point_fill'][mode]*gain)
        signature = hashlib.sha256(json.dumps(frame['objects'], sort_keys=True).encode()).hexdigest()
        changed = signature != state['scene_signature']
        if changed:
            clear_scene()
        objects = [] if changed else state['geometry']
        for obj in frame['objects'] if changed else []:
            rot = obj.get('rotation', {})
            actor = api.spawn_actor_from_class(u.StaticMeshActor, u.Vector(*(100*x for x in obj['center_m'])),
                u.Rotator(pitch=rot.get('pitch', 0), yaw=rot.get('yaw', 0), roll=rot.get('roll', 0)))
            actor.set_actor_label('CNH ' + str(obj['id']) + ' ' + obj['name'])
            mesh = u.load_asset('/Engine/BasicShapes/' + {'cube': 'Cube', 'cylinder': 'Cylinder'}[obj['kind']])
            component = actor.static_mesh_component
            component.set_static_mesh(mesh); component.set_collision_profile_name('NoCollision')
            component.set_editor_property('forced_lod_model', 1)
            # These task actors use non-Nanite engine basic shapes. Do not alter assets.
            actor.set_actor_scale3d(u.Vector(*obj['size_m']))
            dynamic = component.create_dynamic_material_instance(0, material)
            dynamic.set_vector_parameter_value('CaptureColor', u.LinearColor(*obj['albedo'], 1))
            scene.append((obj, actor, dynamic))
            exported_mesh = export_mesh(mesh)
            transform = actor.get_actor_transform()
            center, extent = actor.get_actor_bounds(False)
            objects.append(dict(obj, mesh=exported_mesh, actual_translation_m=[actor.get_actor_location().x/100,
                actor.get_actor_location().y/100, actor.get_actor_location().z/100],
                actual_rotation_quaternion=list(transform.rotation.to_tuple()), actual_scale=list(actor.get_actor_scale3d().to_tuple()),
                actual_bounds_center_m=[center.x/100, center.y/100, center.z/100],
                actual_bounds_extent_m=[extent.x/100, extent.y/100, extent.z/100], render_lod_index=0))
        state.update(scene_signature=signature, geometry=objects)
        for obj, actor, dynamic in scene:
            dynamic.set_vector_parameter_value('CaptureColor', u.LinearColor(*obj['albedo'], 1))
        cam = frame['camera']; _, right, _ = basis(cam)
        for actor in captures:
            offset = rig['baseline_m'] if actor in (rgb_right, depth_right) else 0.
            actor.set_actor_location(u.Vector(*((cam[k]+offset*right[i])*100 for i, k in enumerate(('x', 'y', 'z')))), False, True)
            actor.set_actor_rotation(u.Rotator(pitch=cam['pitch'], yaw=cam['yaw'], roll=cam['roll']), False)
        folder = out/'frames'/frame['id']; folder.mkdir(parents=True, exist_ok=False)
        write_json(folder/'instances.json', dict(instances=objects, scene_static=True,
                   asset_scope='PROCEDURAL_ENGINE_PRIMITIVES_SHARED_CLASSES_DISTINCT_FULL_BUNDLES'))
        write_json(folder/'camera.json', camera_record(cam, rig))
        state['folder'] = folder
        readiness.begin(world, rgb_left.capture_component2d)

    def raw_rgb(actor, filename):
        actor.capture_component2d.capture_scene()
        values = u.RenderingLibrary.read_render_target_raw(world, actor.capture_component2d.texture_target, normalize=False)
        if len(values) != width*height:
            raise RuntimeError('Attribute readback pixel count mismatch')
        write_npy(filename, (v for pixel in values for v in (pixel.r, pixel.g, pixel.b)), (height, width, 3))

    def finish(error=None):
        if state['finished']:
            return
        state['finished'] = True
        try:
            report['pair_export'] = pairs.finish()
        except Exception:
            error = (error or '') + '\n' + traceback.format_exc()
        report.update(status='FAIL' if error else 'PASS_NATIVE_TRANSPORT_REQUIRES_FINALIZE_AND_CANARY',
                      frames=len(rows), expected_frames=len(spec['frames']), wall_s=time.monotonic()-started)
        if error:
            report['error'] = error
        write_json(out/'raw-manifest.json', dict(rig=rig, meshes=exported, frames=rows))
        write_json(out/'engine-receipt.json', report)
        cleanup_errors = []
        try:
            clear_scene()
        except Exception as exc:
            cleanup_errors.append(str(exc))
        for actor in captures:
            try:
                u.RenderingLibrary.release_render_target2d(actor.capture_component2d.texture_target)
            except Exception as exc:
                cleanup_errors.append(str(exc))
        for actor in reversed(actors):
            try:
                api.destroy_actor(actor)
            except Exception as exc:
                cleanup_errors.append(str(exc))
        write_json(out/'actor-release.json', dict(released=not cleanup_errors, errors=cleanup_errors))
        u.unregister_slate_post_tick_callback(handle)
        u.SystemLibrary.quit_editor()

    def tick(delta):
        try:
            if state['finished']:
                return
            frame = spec['frames'][state['index']]
            if state['stage'] == 'PREPARE':
                if not pairs.ready():
                    return
                prepare(frame); state.update(stage='READY', warm=0); return
            if state['stage'] == 'READY':
                if not readiness.poll():
                    return
                state['stage'] = 'SETTLE'
            if state['stage'] == 'SETTLE':
                rgb_left.capture_component2d.capture_scene(); rgb_right.capture_component2d.capture_scene()
                state['warm'] += 1
                if state['warm'] < 8:
                    return
                if pairs.profile()['pending'] > 2:
                    return
                folder = state['folder']
                for side, rgb, depth in [('left', rgb_left, depth_left), ('right', rgb_right, depth_right)]:
                    depth.capture_component2d.capture_scene()
                    pairs.export(world, rgb.capture_component2d.texture_target, depth.capture_component2d.texture_target,
                                 folder/(side+'.transport.png'), folder/('depth_'+side+'.transport.npy'), state['index']*2+(side=='right'))
                raw_rgb(normal, folder/'normal_left.transport.npy')
                raw_rgb(albedo, folder/'albedo_left.transport.npy')
                for obj, actor, dynamic in scene:
                    dynamic.set_vector_parameter_value('CaptureColor', u.LinearColor(*id_color(obj['id']), 1))
                state.update(stage='ID_SETTLE', warm=0); return
            if state['stage'] == 'ID_SETTLE':
                state['warm'] += 1
                if state['warm'] < 2:
                    return
                folder = state['folder']
                raw_rgb(albedo, folder/'instance_left.transport.npy')
                rows.append(dict((k, v) for k, v in frame.items() if k != 'objects'))
                rows[-1].update(folder=folder.relative_to(out).as_posix(), readiness=readiness.receipt(),
                                settling_ticks=8, id_settling_ticks=2, aa='NONE', motion_blur=False)
                state['index'] += 1; state['stage'] = 'PREPARE'
                write_json(out/'progress.json', dict(status='RUNNING', frames=len(rows), expected=len(spec['frames']), wall_s=time.monotonic()-started))
                if len(rows) == len(spec['frames']):
                    finish()
        except Exception:
            finish(traceback.format_exc())

    handle = u.register_slate_post_tick_callback(tick)


def finalize(out):
    """CPU serialization only. Preserve native evidence and reject missing passes."""
    import numpy as np
    import OpenEXR
    from PIL import Image

    out = Path(out)
    engine_receipt = json.loads((out/'engine-receipt.json').read_text())
    if engine_receipt['status'] != 'PASS_NATIVE_TRANSPORT_REQUIRES_FINALIZE_AND_CANARY':
        raise ValueError('Native capture did not finish')
    manifest = json.loads((out/'raw-manifest.json').read_text())
    h, w = manifest['rig']['height'], manifest['rig']['width']
    report = dict(status='PASS_FORMAT_ONLY_GEOMETRIC_CANARY_REQUIRED', backend='TASK_NOT_GPU_SUITABLE_SERIALIZATION', frames=[])
    for row in manifest['frames']:
        folder = out/row['folder']
        instances = json.loads((folder/'instances.json').read_text())['instances']
        known = {obj['id'] for obj in instances}
        depths = {}
        for side in ('left', 'right'):
            with Image.open(folder/(side+'.transport.png')) as image:
                if image.size != (w, h):
                    raise ValueError('RGB dimensions wrong')
                image.load()
                target = folder/(side+'.png')
                if target.exists():
                    raise FileExistsError(target)
                image.convert('RGB').save(target)
            depth = np.load(folder/('depth_'+side+'.transport.npy'), allow_pickle=False)
            if depth.dtype != np.dtype('<f4') or depth.shape != (h, w):
                raise ValueError('Native depth dtype/shape wrong')
            valid = np.isfinite(depth) & (depth > 0) & (depth < 100)
            depth = np.where(valid, depth, np.nan).astype(np.float32)
            depths[side] = depth
            target = folder/('depth_'+side+'.exr')
            if target.exists():
                raise FileExistsError(target)
            OpenEXR.File({'compression': OpenEXR.ZIP_COMPRESSION, 'type': OpenEXR.scanlineimage}, {'Z': depth}).write(str(target))
            np.save(folder/('depth_'+side+'_valid.npy'), valid, allow_pickle=False)
        normal = np.load(folder/'normal_left.transport.npy', allow_pickle=False)
        albedo = np.load(folder/'albedo_left.transport.npy', allow_pickle=False)
        encoded = np.load(folder/'instance_left.transport.npy', allow_pickle=False)
        if any(a.shape != (h, w, 3) or a.dtype != np.dtype('<f4') for a in (normal, albedo, encoded)):
            raise ValueError('Attribute dtype/shape mismatch')
        normal_length = np.linalg.norm(normal, axis=-1)
        # Preserve raw SCS_NORMAL values. Never silently guess/decode its convention.
        valid = np.isfinite(depths['left']) & np.isfinite(normal).all(axis=-1) & (np.abs(normal_length-1) < .04)
        if valid.sum() < .9*np.isfinite(depths['left']).sum():
            raise ValueError('SCS_NORMAL convention/coverage fails unit-vector check; inspect raw output')
        np.save(folder/'normal_left.npy', normal.astype(np.float16), allow_pickle=False)
        np.save(folder/'normal_left_valid.npy', valid, allow_pickle=False)
        np.save(folder/'albedo_left.npy', albedo.astype(np.float16), allow_pickle=False)
        finite = np.isfinite(encoded).all(-1)
        safe = np.where(np.isfinite(encoded), np.maximum(encoded, 0), 0)
        srgb = np.where(safe <= .0031308, safe*12.92, 1.055*np.power(safe, 1/2.4)-.055)
        codes = np.clip(np.rint(srgb*255), 0, 255).astype(np.uint16)
        identifiers = codes[..., 0] + 256*codes[..., 1]
        okay = finite & (codes[..., 2] == (codes[..., 0] ^ codes[..., 1])) & np.isin(identifiers, [0, *known])
        identifiers[~okay] = 65535
        identifiers[~np.isfinite(depths['left'])] = 0
        Image.fromarray(identifiers.astype(np.uint16)).save(folder/'instance_left.png')
        observed = set(np.unique(identifiers).tolist()) - {0, 65535}
        if not observed <= known:
            raise ValueError('Unknown mapped instance')
        hashes = {p.name: sha(p) for p in folder.iterdir() if p.is_file()}
        size = sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
        report['frames'].append(dict(id=row['id'], bytes=size, hashes=hashes,
            instance_unknown_fraction=float(np.mean(identifiers == 65535)), observed_ids=sorted(observed),
            normal_valid_fraction=float(valid.mean())))
    report.update(total_frame_bytes=sum(x['bytes'] for x in report['frames']), frame_count=len(report['frames']))
    write_json(out/'format-receipt.json', report)
    return report


if os.environ.get('BA_CNH_CAPTURE_SPEC') and os.environ.get('BA_CNH_CAPTURE_OUTPUT'):
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        engine()
    except Exception:
        write_json(Path(os.environ['BA_CNH_CAPTURE_OUTPUT'])/'startup-failure.json', dict(status='FAIL', error=traceback.format_exc()))
        import unreal
        unreal.SystemLibrary.quit_editor()
        raise
elif __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('finalize'); p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = finalize(args.output)
    print(json.dumps({k: v for k, v in result.items() if k != 'frames'}))
