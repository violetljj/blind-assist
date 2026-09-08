"""Native City Sample slice capture. Never saves the source map or project."""
import hashlib
import json
import math
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
settling_ticks = spec.get('settling_ticks', 120)
first_use_ticks = spec.get('first_use_settling_ticks', settling_ticks)
settling_interval = float(spec.get('settling_interval_s', .05))
appearance_enabled = spec.get('export_appearance', True)
if type(appearance_enabled) is not bool:
    raise ValueError('export_appearance must be boolean')
if type(settling_ticks) is not int or settling_ticks < 1:
    raise ValueError('settling_ticks must be a positive integer')
if type(first_use_ticks) is not int or first_use_ticks < settling_ticks:
    raise ValueError('first_use_settling_ticks must be an integer >= settling_ticks')
if not math.isfinite(settling_interval) or settling_interval < 0:
    raise ValueError('settling_interval_s must be finite and nonnegative')
api = u.get_editor_subsystem(u.EditorActorSubsystem)
levels = u.get_editor_subsystem(u.LevelEditorSubsystem)
editor = u.get_editor_subsystem(u.UnrealEditorSubsystem)
started = time.monotonic()
stage, index, warm = 0, 0, 0
warm_target = settling_ticks
progress_written = 0.
after = 0.
finished = False
captures, objects, frames = [], [], []
scene_lights = []
background_settings = None
background_settings_before = {}
isolated_actor = None
settled_mesh_assets = set()
pairs = PairExporter(u, spec.get('pair_export_mode', 'native_probe'))
map_file = Path(spec['map_file'])
report = dict(status='RUNNING', map_asset=spec['map_asset'],
              export_policy=dict(pair_mode=pairs.mode, appearance=appearance_enabled),
              spec_sha256=hashlib.sha256(SPEC.read_bytes()).hexdigest(),
              map_sha256_before=hashlib.sha256(map_file.read_bytes()).hexdigest(),
              purpose='SAMPLE_PCG_INTEGRATION_NO_MODEL_SCORING',
              settling_policy=dict(ticks=settling_ticks, first_use_ticks=first_use_ticks, interval_s=settling_interval,
                                   readiness_gate='UNCHANGED_NATIVE_ASSET_SHADER_STREAMING'))
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
    if spec.get('native_full_detail_only', False) and not spec.get('background_hlod_min_distance_m'):
        for hidden in api.get_all_level_actors():
            if isinstance(hidden, u.WorldPartitionHLOD):
                c.hide_actor_components(hidden, True)
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
    for light, original, kind in scene_lights:
        scale = float(case.get(kind+'_intensity_scale', 1.))
        if not math.isfinite(scale) or not 0 < scale <= 10:
            raise ValueError('Invalid scene light scale')
        light.set_editor_property('intensity', original*scale)
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
        if obj.get('material_asset'):
            material = u.load_asset(obj['material_asset'])
            if material is None:
                raise ValueError('Missing explicit material: ' + obj['material_asset'])
            for slot in range(max(1, actor.static_mesh_component.get_num_materials())):
                actor.static_mesh_component.set_material(slot, material)
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
        if (actor.capture_component2d.capture_source == u.SceneCaptureSource.SCS_FINAL_COLOR_LDR
                and ('exposure_ev100' in case or 'exposure_ev100' in spec)):
            settings = actor.capture_component2d.post_process_settings
            ev = float(case.get('exposure_ev100', spec.get('exposure_ev100', 13.2)))
            for key in ('auto_exposure_min_brightness', 'auto_exposure_max_brightness'):
                settings.set_editor_property('override_' + key, True)
                settings.set_editor_property(key, ev)
            actor.capture_component2d.post_process_settings = settings
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
            roots += [obj['material_asset'] for case in spec['cases'] for obj in case.get('objects',[]) if 'material_asset' in obj]
            roots += spec.get('dependency_roots', [])
            dependencies = export_dependencies(OUT, roots)
            report['dependencies'] = dict(status=dependencies['status'], file_count=dependencies.get('file_count'), total_bytes=dependencies.get('total_bytes'))
            if dependencies['status'] != 'PASS':
                report['status'] = 'FAIL'
        except Exception:
            report.update(status='FAIL',dependency_error=traceback.format_exc())
    report['readiness'] = readiness.receipt()
    if background_settings is not None:
        for key, value in background_settings_before.items():
            background_settings.set_editor_property(key, value)
    report['wall_elapsed_s'] = time.monotonic() - started
    write(OUT / 'evaluator/spec.json', spec)
    write(OUT / 'model/dataset.json', dict(schema='city-pcg-preview-v1', frames=frames,
          calibration=dict(width=640, height=360, horizontal_fov_degrees=100., depth_max_m=100.),
          authority='RGB_ONLY_NO_NATIVE_DEPTH_POSE_OR_OBJECTS'))
    write(OUT / 'receipt.json', report)
    if spec.get('native_targets'):
        write(OUT/'evaluator/target-raycheck.json',dict(schema='city-native-target-raycheck-v1',
            rows=report.get('native_target_checks',[])))
    u.unregister_slate_post_tick_callback(handle)
    u.SystemLibrary.quit_editor()


def tick(dt):
    global stage, index, warm, warm_target, progress_written, after, world, rgb, depth, beauty, background_settings, isolated_actor
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
            if spec.get('background_hlod_min_distance_m'):
                distance = float(spec['background_hlod_min_distance_m'])
                if not math.isfinite(distance) or distance < 50:
                    raise ValueError('Background HLOD distance must be finite and at least 50 metres')
                settings_class = u.load_class(None, '/Script/WorldPartitionEditor.WorldPartitionEditorSettings')
                background_settings = u.get_default_object(settings_class)
                settings = dict(bShowHLODsInEditor=True, bShowHLODsOverLoadedRegions=False,
                                HLODMinDrawDistance=distance*100, HLODMaxDrawDistance=0.)
                for key, value in settings.items():
                    background_settings_before[key] = background_settings.get_editor_property(key)
                    background_settings.set_editor_property(key, value)
                u.SystemLibrary.execute_console_command(world, 'wp.Editor.HLOD.AllowShowingHLODsInEditor 1')
                report['background_hlod'] = dict(min_distance_m=distance, settings=settings,
                    authority='DISTANT_APPEARANCE_ONLY_NOT_NEAR_TARGET_GEOMETRY')
            elif spec.get('native_full_detail_only', False):
                u.SystemLibrary.execute_console_command(world, 'wp.Editor.HLOD.AllowShowingHLODsInEditor 0')
            if 'world_partition_region_m' in spec:
                region = spec['world_partition_region_m']
                lo, hi = region['min'], region['max']
                if len(lo) != 3 or len(hi) != 3 or not all(
                        math.isfinite(float(a)) and math.isfinite(float(b)) and a < b
                        for a, b in zip(lo, hi)):
                    raise ValueError('World Partition region requires finite ordered bounds')
                box = u.Box(min=u.Vector(*(v*100 for v in lo)),
                            max=u.Vector(*(v*100 for v in hi)))
                descs = u.WorldPartitionBlueprintLibrary.get_intersecting_actor_descs(box)
                if not descs:
                    raise RuntimeError('No native actors intersect the requested region')
                write(OUT / 'progress.json', dict(phase='LOAD_NATIVE_REGION', actors=len(descs)))
                if spec.get('native_full_detail_only', False):
                    descs = [d for d in descs if 'HLOD' not in str(d.native_class)]
                    if not descs:
                        raise RuntimeError('Requested region contains no full-detail actors')
                u.WorldPartitionBlueprintLibrary.load_actors([d.guid for d in descs])
                if spec.get('native_full_detail_only', False) and not spec.get('background_hlod_min_distance_m'):
                    hidden = []
                    for actor in api.get_all_level_actors():
                        if 'HLOD' in actor.get_class().get_name():
                            actor.set_is_temporarily_hidden_in_editor(True)
                            actor.set_actor_hidden_in_game(True)
                            actor.set_actor_enable_collision(False)
                            hidden.append(actor.get_path_name())
                    report['hidden_native_hlod_actors'] = hidden
                report['native_region'] = dict(bounds_m=region, actor_descriptors=len(descs),
                    policy='LOAD_EXISTING_ACTORS_NO_MAP_SAVE_DEPENDENCIES_MAY_EXTEND_REGION')
            for actor in api.get_all_level_actors():
                for cls, kind in ((u.DirectionalLightComponent,'sun'),(u.SkyLightComponent,'skylight')):
                    for light in actor.get_components_by_class(cls):
                        scene_lights.append((light,float(light.get_editor_property('intensity')),kind))
            report['base_light_intensities']=[dict(kind=k,intensity=v) for _,v,k in scene_lights]
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
                    mesh_rows.append(dict(actor=actor.get_actor_label(), actor_class=actor.get_class().get_name(), components=len(components),
                        bounds_center_cm=[center.x,center.y,center.z], bounds_extent_cm=[extent.x,extent.y,extent.z],
                        instances=sum(c.get_instance_count() if isinstance(c,u.InstancedStaticMeshComponent) else 1 for c in components)))
            report['mesh_inventory'] = mesh_rows
            report['generation_requested'] = spec.get('generate_labels', [])
            report['map_load_s'] = time.monotonic() - started
            rgb = component(u.SceneCaptureSource.SCS_FINAL_COLOR_LDR, u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
            depth = component(u.SceneCaptureSource.SCS_SCENE_DEPTH, u.TextureRenderTargetFormat.RTF_RGBA32F)
            beauty = component(u.SceneCaptureSource.SCS_FINAL_COLOR_LDR, u.TextureRenderTargetFormat.RTF_RGBA8_SRGB, 1280, 720) if appearance_enabled else None
            stage = 1
        # Pump completed readbacks every tick and bound outstanding GPU exports.
        if not pairs.ready():
            after = 0.
            return
        if stage == 2:
            if not readiness.poll():
                write(OUT / 'progress.json', dict(phase='READINESS', index=index, readiness=readiness.receipt()))
                after = time.monotonic() + .1
                return
            stage = 3
            # Pre-readiness renders may contain incomplete resources. Give the
            # first view or a view that waited for assets a complete settled pass.
            ready_wait_s = readiness.receipt()['elapsed_s']
            case_assets = {obj.get('mesh_asset', obj.get('primitive_asset', '/Engine/BasicShapes/Cube'))
                           for obj in spec['cases'][index].get('objects', [])}
            new_assets = case_assets - settled_mesh_assets
            camera = spec['cases'][index]['camera']
            previous = spec['cases'][max(0,index-1)]['camera']
            view_jump = math.hypot(camera['x']-previous['x'], camera['y']-previous['y']) > 10.
            report.setdefault('view_readiness', []).append(dict(index=index, **readiness.receipt()))
            if index == 0 or ready_wait_s > .5 or new_assets or view_jump:
                warm = 0
                warm_target = first_use_ticks
                report.setdefault('post_ready_settling', []).append(dict(
                    index=index, ticks=warm_target, ready_wait_s=ready_wait_s, view_jump=view_jump,
                    first_use_assets=sorted(new_assets)))
            settled_mesh_assets.update(case_assets)
        if warm == 0 and stage == 1:
            prepare(spec['cases'][index])
        rgb.capture_component2d.capture_scene()
        if beauty is not None:
            beauty.capture_component2d.capture_scene()
        warm += 1
        if warm == 1 or time.monotonic()-progress_written >= 1.:
            write(OUT / 'progress.json', dict(phase='SETTLING', index=index, ticks=warm, elapsed_s=time.monotonic()-started))
            progress_written = time.monotonic()
        if warm < warm_target:
            after = time.monotonic() + settling_interval
            return
        if stage == 1:
            readiness.begin(world, rgb.capture_component2d)
            stage = 2
            after = 0.
            return
        depth.capture_component2d.capture_scene()
        if index == 0 and spec.get('export_hlod_membership'):
            from city_hlod_membership import export_membership
            export_membership(u, OUT/'evaluator/hlod-membership.json')
        if index == 0 and spec.get('source_floor_grid'):
            from city_source_floor import probe
            write(OUT/'evaluator/source-floor-grid.json', probe(u, world, spec['source_floor_grid']))
        if spec.get('export_native_inventory') and index in spec.get('inventory_indices', [0]):
            from city_native_inspect import inventory
            native_inventory = inventory(u, api, spec['cases'][index]['camera'], 35.)
            write(OUT/f'evaluator/native-inventory-{index:04d}.json', native_inventory)
            if index == 0:
                write(OUT/'evaluator/native-inventory.json', native_inventory)
        case = spec['cases'][index]
        if spec.get('export_controlled_targets'):
            measured = {}
            for kind, actor in (('rgb', rgb), ('depth', depth)):
                position = actor.get_actor_location()
                rotation = actor.get_actor_rotation()
                measured[kind] = dict(x=position.x/100, y=position.y/100, z=position.z/100,
                    yaw=rotation.yaw, pitch=rotation.pitch, roll=rotation.roll)
            report.setdefault('capture_poses', []).append(dict(sample_index=index,
                **measured, authority='ENGINE_ACTOR_TRANSFORMS_AT_PAIRED_STATIC_CAPTURE'))
        if case.get('probe_native_floor', False):
            pose = case['camera']
            hit = u.SystemLibrary.line_trace_single(world,
                u.Vector(pose['x']*100, pose['y']*100, pose['z']*100),
                u.Vector(pose['x']*100, pose['y']*100, (pose['z']-5)*100),
                u.TraceTypeQuery.TRACE_TYPE_QUERY1, True, [], u.DrawDebugTrace.NONE)
            point = hit.to_tuple()[5] if hit else None
            report.setdefault('native_floor_probes', []).append(dict(
                case=case.get('name'), hit=bool(hit),
                point_m=[point.x/100, point.y/100, point.z/100] if point else None,
                policy='AFTER_READINESS_DIAGNOSTIC_ONLY_NO_AUTOMATIC_HEIGHT_CHANGE'))
        pairs.export(world, rgb.capture_component2d.texture_target, depth.capture_component2d.texture_target,
                     OUT / f'model/sample/{index:04d}.png', OUT / f'evaluator/native/{index:04d}.npy', index)
        if 'route' in spec:
            report.setdefault('route_samples', []).append(dict(sample_index=index,
                camera=spec['cases'][index]['camera'],
                route_distance_m=spec['cases'][index]['route_distance_m'],
                capture_elapsed_s=time.monotonic()-started,
                timing='WALL_CLOCK_SETTLED_CAPTURE_NOT_SIMULATION_TIME'))
        if beauty is not None:
            (OUT / 'appearance').mkdir(exist_ok=True)
            u.RenderingLibrary.export_render_target(world, beauty.capture_component2d.texture_target,
                                                    str(OUT/'appearance'), f'{index:04d}.png')
        if spec.get('native_targets'):
            from city_native_targets import export
            if isolated_actor is None:
                isolated_actor=component(u.SceneCaptureSource.SCS_SCENE_DEPTH,u.TextureRenderTargetFormat.RTF_RGBA32F)
            target_spec = spec
            if spec.get('export_controlled_targets'):
                # Stable collection identities live in the spec; transient UE
                # components are resolved after this view's assembly is spawned.
                declared = {t['target_id']: t for t in spec['native_targets']}
                active = []
                for obj, actor in zip(case.get('objects', []), objects):
                    if not obj.get('target_part'):
                        continue
                    tid = obj['instance_id']
                    if tid not in declared:
                        raise ValueError('Undeclared controlled target: ' + tid)
                    mesh = actor.static_mesh_component
                    pos = actor.get_actor_location()
                    active.append(dict(declared[tid], component_path=mesh.get_path_name(),
                        mesh_asset=mesh.static_mesh.get_path_name(), instance_index=None,
                        position_m=[pos.x/100,pos.y/100,pos.z/100]))
                target_spec = dict(spec, native_targets=active)
                report.setdefault('controlled_target_bindings', []).append(dict(
                    sample_index=index, targets=active,
                    lifecycle='Stable configured assembly identities; actors recreated per settled view'))
            report.setdefault('native_target_checks',[]).extend(export(u,api,world,isolated_actor,target_spec,case,index,OUT))
        frames.append(dict(sample_index=index, rgb_path=f'sample/{index:04d}.png'))
        index += 1
        warm = 0
        warm_target = settling_ticks
        stage = 1
        if index == len(spec['cases']):
            finish()
        after = 0.
    except Exception:
        finish(traceback.format_exc())


handle = u.register_slate_post_tick_callback(tick)
