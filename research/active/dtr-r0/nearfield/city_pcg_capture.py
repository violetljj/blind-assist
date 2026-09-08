"""Native City Sample slice capture. Never saves the source map or project."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import unreal as u

sys.path.insert(0, str(Path(__file__).parent))
from ue_pair_export import PairExporter
from ue_capture_readiness import CaptureReadiness

OUT = Path(os.environ['BA_CITY_OUT'])
SPEC = Path(os.environ['BA_CITY_SPEC'])
spec = json.loads(SPEC.read_text(encoding='utf-8-sig'))
api = u.get_editor_subsystem(u.EditorActorSubsystem)
levels = u.get_editor_subsystem(u.LevelEditorSubsystem)
editor = u.get_editor_subsystem(u.UnrealEditorSubsystem)
started = time.monotonic()
stage, index, warm = 0, 0, 0
after = 0.
finished = False
captures, objects, frames = [], [], []
pairs = PairExporter(u, 'native_probe')
map_file = Path(spec['map_file'])
report = dict(status='RUNNING', map_asset=spec['map_asset'],
              spec_sha256=hashlib.sha256(SPEC.read_bytes()).hexdigest(),
              map_sha256_before=hashlib.sha256(map_file.read_bytes()).hexdigest(),
              purpose='SAMPLE_PCG_INTEGRATION_NO_MODEL_SCORING')
readiness = CaptureReadiness(u, timeout=900)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def clear_objects():
    while objects:
        if not api.destroy_actor(objects.pop()):
            raise RuntimeError('Could not release controlled obstacle')


def component(source, fmt, width=640, height=360):
    actor = api.spawn_actor_from_class(u.SceneCapture2D, u.Vector(0, 0, 100000))
    captures.append(actor)
    c = actor.capture_component2d
    c.capture_every_frame = False
    c.capture_on_movement = False
    c.always_persist_rendering_state = True
    c.capture_source = source
    c.texture_target = u.RenderingLibrary.create_render_target2d(world, width, height, fmt)
    c.fov_angle = 100.
    if source == u.SceneCaptureSource.SCS_FINAL_COLOR_LDR:
        c.texture_target.target_gamma = 2.2
        c.set_editor_property('show_flag_settings', [
            u.EngineShowFlagsSetting(show_flag_name='TemporalAA', enabled=True),
            u.EngineShowFlagsSetting(show_flag_name='AntiAliasing', enabled=True)])
        volumes = [a for a in api.get_all_level_actors() if isinstance(a, u.PostProcessVolume)]
        if volumes:
            c.post_process_settings = max(volumes, key=lambda a: a.priority).settings
        settings = c.post_process_settings
        for name, value in [('motion_blur_amount', 0.), ('film_grain_intensity', 0.),
                            ('vignette_intensity', 0.), ('lens_flare_intensity', 0.),
                            ('bloom_intensity', 0.), ('lumen_scene_lighting_quality', 2.),
                            ('lumen_final_gather_quality', 2.)]:
            settings.set_editor_property('override_' + name, True)
            settings.set_editor_property(name, value)
        if 'exposure_ev100' in spec:
            for name in ('auto_exposure_min_brightness', 'auto_exposure_max_brightness'):
                settings.set_editor_property('override_' + name, True)
                settings.set_editor_property(name, float(spec['exposure_ev100']))
        c.post_process_settings = settings
        c.post_process_blend_weight = 1.
    return actor


def prepare(case):
    import math
    clear_objects()
    for obj in case.get('objects', []):
        explicit_asset = 'mesh_asset' in obj or 'primitive_asset' in obj
        if 'mesh_asset' in obj and 'primitive_asset' in obj:
            raise ValueError('Choose mesh_asset or primitive_asset, not both')
        if explicit_asset and 'size_m' in obj:
            raise ValueError('Explicit mesh assets use dimensionless scale, not size_m')
        asset_path = obj.get('mesh_asset', obj.get('primitive_asset', '/Engine/BasicShapes/Cube'))
        if not isinstance(asset_path, str) or not asset_path.startswith('/'):
            raise ValueError('Mesh asset must be an absolute Unreal asset path')
        if 'primitive_asset' in obj and not asset_path.startswith('/Engine/BasicShapes/'):
            raise ValueError('primitive_asset must name an Engine BasicShapes mesh')
        mesh = u.load_asset(asset_path)
        if not isinstance(mesh, u.StaticMesh):
            raise ValueError('Asset is missing or is not a StaticMesh: ' + asset_path)
        origin = obj['center_m']
        scale = obj.get('scale', 1.) if explicit_asset else obj['size_m']
        if isinstance(scale, (int, float)):
            scale = [scale] * 3
        rotation = obj.get('rotation_deg', {'pitch': 0., 'yaw': 0., 'roll': 0.})
        if set(rotation) != {'pitch', 'yaw', 'roll'}:
            raise ValueError('rotation_deg requires explicit pitch, yaw and roll')
        if len(origin) != 3 or len(scale) != 3 or not all(math.isfinite(float(v)) for v in [*origin, *scale, *rotation.values()]):
            raise ValueError('Object transform must contain finite three-dimensional values')
        if not all(float(v) > 0 for v in scale):
            raise ValueError('Object scale must be positive')
        actor = api.spawn_actor_from_class(u.StaticMeshActor, u.Vector(*(v * 100 for v in origin)))
        objects.append(actor)
        actor.set_actor_label('BA controlled ' + obj['name'])
        actor.static_mesh_component.set_static_mesh(mesh)
        actor.set_actor_scale3d(u.Vector(*scale))
        actor.set_actor_rotation(u.Rotator(**rotation), False)
        actor.static_mesh_component.set_collision_profile_name('BlockAll')
        placement = obj.get('placement', 'actor_origin')
        if placement != 'actor_origin':
            if placement not in ('bounds_front_center_floor', 'bounds_front_right_floor'):
                raise ValueError('Unknown placement: ' + placement)
            if not explicit_asset:
                raise ValueError('Bounds placement requires an explicit mesh')
            # Bounds align the prop only; depth remains the source of support labels.
            center, extent = actor.get_actor_bounds(False)
            anchor = [center.x-extent.x,
                      center.y if placement == 'bounds_front_center_floor' else center.y-extent.y,
                      center.z-extent.z]
            location = actor.get_actor_location()
            origin = [(v + target*100 - current)/100 for v,target,current in
                      zip([location.x,location.y,location.z],obj['center_m'],anchor)]
            actor.set_actor_location(u.Vector(*(v*100 for v in origin)), False, False)
        bounds = mesh.get_bounding_box()
        report.setdefault('controlled_objects', []).append(dict(case=case.get('name'), name=obj['name'],
            placement=placement, requested_anchor_m=obj['center_m'],
            mesh_asset=mesh.get_path_name(), actor_origin_m=list(origin), scale=list(scale), rotation_deg=rotation,
            mesh_local_bounds_cm=dict(min=[bounds.min.x,bounds.min.y,bounds.min.z], max=[bounds.max.x,bounds.max.y,bounds.max.z]),
            geometry_authority='Asset geometry; actor origin and dimensionless scale are not ground-truth object dimensions'
                if explicit_asset else 'Engine one-metre cube with requested size_m'))
    pose = case['camera']
    loc = u.Vector(*(pose[k] * 100 for k in ('x', 'y', 'z')))
    rot = u.Rotator(pitch=pose['pitch'], yaw=pose['yaw'], roll=pose.get('roll', 0.))
    for actor in captures:
        actor.set_actor_location(loc, False, False)
        actor.set_actor_rotation(rot, False)
    editor.set_level_viewport_camera_info(loc, rot)


def finish(error=None):
    global finished
    if finished:
        return
    finished = True
    report['status'] = 'FAIL' if error else 'PASS'
    if error:
        report['error'] = error
    try:
        report['pair_exports'] = pairs.finish()
        clear_objects()
        for actor in captures:
            u.RenderingLibrary.release_render_target2d(actor.capture_component2d.texture_target)
            if not api.destroy_actor(actor):
                raise RuntimeError('Could not release capture actor')
    except Exception:
        report.update(status='FAIL', cleanup_error=traceback.format_exc())
    report['map_sha256_after'] = hashlib.sha256(map_file.read_bytes()).hexdigest()
    report['source_unchanged'] = report['map_sha256_before'] == report['map_sha256_after'] == spec['map_sha256']
    if not report['source_unchanged']:
        report['status'] = 'FAIL'
    report['frame_count'] = len(frames)
    if spec.get('export_dependencies') and not error:
        try:
            from city_pcg_dependencies import export_dependencies
            roots = [spec['map_asset']] + [obj['mesh_asset'] for case in spec['cases'] for obj in case.get('objects',[]) if 'mesh_asset' in obj]
            dependencies = export_dependencies(OUT, roots)
            report['dependencies'] = dict(status=dependencies['status'], file_count=dependencies.get('file_count'), total_bytes=dependencies.get('total_bytes'))
            if dependencies['status'] != 'PASS':
                report['status'] = 'FAIL'
        except Exception:
            report.update(status='FAIL',dependency_error=traceback.format_exc())
    report['readiness'] = readiness.receipt()
    report['wall_elapsed_s'] = time.monotonic() - started
    write(OUT / 'evaluator/spec.json', spec)
    write(OUT / 'model/dataset.json', dict(schema='city-pcg-preview-v1', frames=frames,
          calibration=dict(width=640, height=360, horizontal_fov_degrees=100., depth_max_m=100.),
          authority='RGB_ONLY_NO_NATIVE_DEPTH_POSE_OR_OBJECTS'))
    write(OUT / 'receipt.json', report)
    u.unregister_slate_post_tick_callback(handle)
    u.SystemLibrary.quit_editor()


def tick(dt):
    global stage, index, warm, after, world, rgb, depth, beauty
    if finished or time.monotonic() < after:
        return
    if (OUT / 'stop.request').exists():
        finish('Task owner requested stop')
        return
    after = float('inf')
    try:
        if stage == 0:
            assert report['map_sha256_before'] == spec['map_sha256'], 'Map identity mismatch'
            assert levels.load_level(spec['map_asset']), 'Map load failed'
            world = editor.get_editor_world()
            if 'sun_source_angle_deg' in spec:
                for actor in api.get_all_level_actors():
                    if isinstance(actor, u.DirectionalLight):
                        actor.light_component.set_editor_property('light_source_angle', float(spec['sun_source_angle_deg']))
            report['loaded_world'] = world.get_path_name()
            pcg_rows = []
            for actor in api.get_all_level_actors():
                for c in actor.get_components_by_class(u.PCGComponent):
                    graph = c.get_graph()
                    pcg_rows.append(dict(actor=actor.get_actor_label(), graph=graph.get_path_name() if graph else None,
                                         seed=c.get_editor_property('seed'), generated=c.get_editor_property('generated')))
                    if actor.get_actor_label() in spec.get('generate_labels', []):
                        c.generate_local(True)
            report['pcg_components'] = pcg_rows
            mesh_rows = []
            for actor in api.get_all_level_actors():
                components = actor.get_components_by_class(u.StaticMeshComponent)
                if components:
                    center, extent = actor.get_actor_bounds(False)
                    mesh_rows.append(dict(actor=actor.get_actor_label(), components=len(components),
                        bounds_center_cm=[center.x,center.y,center.z], bounds_extent_cm=[extent.x,extent.y,extent.z],
                        instances=sum(c.get_instance_count() if isinstance(c,u.InstancedStaticMeshComponent) else 1 for c in components)))
            report['mesh_inventory'] = mesh_rows
            report['generation_requested'] = spec.get('generate_labels', [])
            report['map_load_s'] = time.monotonic() - started
            rgb = component(u.SceneCaptureSource.SCS_FINAL_COLOR_LDR, u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
            depth = component(u.SceneCaptureSource.SCS_SCENE_DEPTH, u.TextureRenderTargetFormat.RTF_RGBA32F)
            beauty = component(u.SceneCaptureSource.SCS_FINAL_COLOR_LDR, u.TextureRenderTargetFormat.RTF_RGBA8_SRGB, 1280, 720)
            stage = 1
        if stage == 2:
            if not readiness.poll():
                write(OUT / 'progress.json', dict(phase='READINESS', index=index, readiness=readiness.receipt()))
                after = time.monotonic() + .1
                return
            stage = 3
        if warm == 0:
            prepare(spec['cases'][index])
        rgb.capture_component2d.capture_scene()
        beauty.capture_component2d.capture_scene()
        warm += 1
        write(OUT / 'progress.json', dict(phase='SETTLING', index=index, ticks=warm, elapsed_s=time.monotonic()-started))
        if warm < spec.get('settling_ticks', 120):
            after = time.monotonic() + .05
            return
        if stage == 1:
            readiness.begin(world, rgb.capture_component2d)
            stage = 2
            after = 0.
            return
        depth.capture_component2d.capture_scene()
        pairs.export(world, rgb.capture_component2d.texture_target, depth.capture_component2d.texture_target,
                     OUT / f'model/sample/{index:04d}.png', OUT / f'evaluator/native/{index:04d}.npy', index)
        (OUT / 'appearance').mkdir(exist_ok=True)
        u.RenderingLibrary.export_render_target(world, beauty.capture_component2d.texture_target,
                                                str(OUT/'appearance'), f'{index:04d}.png')
        frames.append(dict(sample_index=index, rgb_path=f'sample/{index:04d}.png'))
        index += 1
        warm = 0
        stage = 1
        if index == len(spec['cases']):
            finish()
        after = 0.
    except Exception:
        finish(traceback.format_exc())


handle = u.register_slate_post_tick_callback(tick)
