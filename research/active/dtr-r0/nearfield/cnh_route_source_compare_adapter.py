"""Map/pose adapter for a bounded two-layout source engineering comparison.

No rendering, admission, map saving or automatic candidate search occurs here.
The insertion collector owns capture/readiness/cleanup and calls these helpers.
Loading a region is evidence of a request, not proof of complete native geometry.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from cnh_route_capture import basis


SOURCES = {'/Game/BAResearchSlice/Street200V7', '/Game/Map/Small_City_LVL'}
SCOPE = 'TWO_LAYOUT_SOURCE_ENGINEERING_NOT_BENCHMARK'
DEVELOPMENT_SCOPE = 'STREET_DEVELOPMENT_PILOT_NOT_BENCHMARK'
ALLEY_SCOPE = 'ALLEY_DEVELOPMENT_PILOT_NOT_BENCHMARK'


def is_development(spec):
    return spec.get('scope') in (DEVELOPMENT_SCOPE, ALLEY_SCOPE)


def normalized_guid(value):
    if hasattr(value, 'get_editor_property'):
        return ''.join(f'{int(value.get_editor_property(k)) & 0xffffffff:08x}' for k in ('a','b','c','d'))
    return str(value).strip('{}').replace('-', '').lower()


def validated_pose(camera):
    keys = ('x', 'y', 'z', 'pitch', 'yaw', 'roll')
    if not all(k in camera for k in keys):
        raise ValueError('Explicit world metres and pitch/yaw/roll degrees required')
    result = {k: float(camera[k]) for k in keys}
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('Camera pose must be finite')
    return result


def validate_alley_manifest(spec):
    """Bind one authored map and its package evidence; no formal split admission."""
    manifest = spec.get('alley_manifest', {})
    asset = spec.get('map_asset', '')
    if (not asset.startswith('/Game/BAResearchAlley/') or '..' in asset
            or manifest.get('map_asset') != asset
            or not manifest.get('physical_site_id')
            or manifest.get('proposed_split') not in ('train', 'dev', 'test')
            or spec.get('data_role') != 'Development'
            or spec.get('native_material_policy') != 'ALLEY_FROZEN_STATIC_COMPILED'
            or spec.get('nominal_sample_interval_s') != .1
            or spec.get('render_recipe') != 'STATIC_SPATIAL_V1'
            or any(r['physical_site_id'] != manifest['physical_site_id'] for r in spec['layouts'])):
        raise ValueError('Alley requires one declared authored map/site, proposed split only, frozen static Development recipe')
    files = manifest.get('files', [])
    if not files or not any(Path(r['path']).resolve() == Path(spec['map_file']).resolve()
                            and r['sha256'] == spec['map_sha256'] for r in files):
        raise ValueError('Alley package manifest must include the source map')
    family = manifest.get('family_receipt', {})
    if not family.get('path') or not family.get('sha256'):
        raise ValueError('Frozen family receipt required; hash binding is not independent admission')
    for row in files + [family]:
        if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Alley frozen dependency or family receipt hash mismatch')


def validate_spec(spec):
    if spec.get('scope') not in (SCOPE, DEVELOPMENT_SCOPE, ALLEY_SCOPE) or spec.get('benchmark_eligible') is not False:
        raise ValueError('Explicit engineering-only comparison scope required')
    if spec.get('map_asset') not in SOURCES and spec.get('scope') != ALLEY_SCOPE:
        raise ValueError('Only the two declared source maps are supported')
    if spec.get('native_full_detail_only', True) is not True or spec.get('background_hlod_min_distance_m'):
        raise ValueError('This bounded adapter supports full-detail-only captures; distant HLOD requires the original collector')
    layouts = spec.get('layouts', [])
    valid_count = 1 <= len(layouts) <= 20 if is_development(spec) else len(layouts) == 2
    if not valid_count or len({r['layout_id'] for r in layouts}) != len(layouts):
        raise ValueError('Engineering requires two layouts; Development batches require 1..20 unique layouts')
    for row in layouts:
        validated_pose(row['camera'])
        candidates = row.get('candidates')
        if candidates is not None:
            if not isinstance(candidates, list) or not 1 <= len(candidates) <= 16:
                raise ValueError('Each layout requires 1..16 fixed candidates when supplied')
            for candidate in candidates:
                validated_pose(candidate['camera'])
        if not row.get('physical_site_id'):
            raise ValueError('Declared physical site identity required; independence remains unverified')
    if spec.get('scope') == ALLEY_SCOPE:
        validate_alley_manifest(spec)
    elif is_development(spec):
        if (spec['map_asset'] != '/Game/BAResearchSlice/Street200V7'
                or spec.get('data_role') != 'Development'
                or spec.get('native_material_policy') != 'STREET_TRANSIENT_ZERO_WPO_PDO'
                or spec.get('nominal_sample_interval_s') != .1
                or spec.get('render_recipe') not in (None,'STATIC_SPATIAL_V1')
                or any(r['physical_site_id'] != 'Street200V7-single-street-block' for r in layouts)
                or any(r.get('environment_category') not in ('sidewalk','intersection','plaza') for r in layouts)):
            raise ValueError('Development requires frozen static Street, declared categories, one shared street block and nominal 0.1s spacing')
    elif len({r['physical_site_id'] for r in layouts}) != 2:
        control = spec.get('city_derived_control', {})
        fresh_city1 = (spec.get('map_asset') == '/Game/Map/Small_City_LVL'
                       and spec.get('data_role') == 'Development'
                       and control.get('authority') == 'FRESH_CITY1_SAME_SITE_DEVELOPMENT'
                       and control.get('new_layouts') is True
                       and control.get('independent_site_count') == 1
                       and control.get('physical_site_id') == 'city-consumed-engineering-site-1'
                       and {r['physical_site_id'] for r in layouts} == {'city-consumed-engineering-site-1'}
                       and {r['layout_id'] for r in layouts} == {
                           'city1-fresh-development-00', 'city1-fresh-development-01'})
        if not fresh_city1:
            raise ValueError('Two camera views of one declared site are not two layouts')
    region = spec.get('world_partition_region_m')
    if spec['map_asset'] == '/Game/Map/Small_City_LVL' and region is None:
        raise ValueError('City requires an explicit bounded World Partition region')
    if region is not None:
        lo, hi = region['min'], region['max']
        if len(lo) != 3 or len(hi) != 3 or not all(
                math.isfinite(float(a)) and math.isfinite(float(b)) and a < b
                for a, b in zip(lo, hi)):
            raise ValueError('Finite ordered region bounds required')
        # Merely verifies requested region encloses each probe selection sphere.
        # It does not authenticate unloaded dependencies or a complete trajectory.
        poses = [row['camera'] for row in layouts]
        poses += [candidate['camera'] for row in layouts for candidate in row.get('candidates', [])]
        for camera in poses:
            pose = validated_pose(camera)
            if any(not lo[i] <= pose[k]-8. <= pose[k]+8. <= hi[i]
                   for i, k in enumerate(('x', 'y', 'z'))):
                raise ValueError('Requested region must contain each 8m probe sphere')
    path = Path(spec['map_file'])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != spec.get('map_sha256'):
        raise ValueError('Source map hash mismatch')
    return digest


def load_source(u, spec):
    """Load once in a task-owned editor; caller retains normal readiness gates."""
    digest = validate_spec(spec)
    levels = u.get_editor_subsystem(u.LevelEditorSubsystem)
    editor = u.get_editor_subsystem(u.UnrealEditorSubsystem)
    if not levels.load_level(spec['map_asset']):
        raise RuntimeError('Source map load failed')
    world = editor.get_editor_world()
    api = u.get_editor_subsystem(u.EditorActorSubsystem)
    u.SystemLibrary.execute_console_command(world, 'wp.Editor.HLOD.AllowShowingHLODsInEditor 0')
    receipt = dict(scope=spec['scope'], map_asset=spec['map_asset'], map_file=spec['map_file'],
                   map_sha256_before=digest, loaded_world=world.get_path_name(),
                   layouts=[r['layout_id'] for r in spec['layouts']],
                   native_coverage_complete=False, bounds_conservative=False,
                   deformation_bounded=False, asset_isolation='NOT_VERIFIED',
                   streaming_completeness='NOT_VERIFIED', admitted_layouts=0,
                   saved=False)
    if spec.get('scope') == ALLEY_SCOPE:
        receipt['alley_manifest'] = spec['alley_manifest']
        receipt['authored_nonpartitioned_map'] = 'FROZEN_PACKAGE_HASHES_VERIFIED'
    receipt['visibility_policy'] = 'TASK_OWNED_EDITOR_FULL_DETAIL_ONLY_NO_MAP_SAVE'
    region = spec.get('world_partition_region_m')
    if region is not None:
        box = u.Box(min=u.Vector(*(v*100 for v in region['min'])),
                    max=u.Vector(*(v*100 for v in region['max'])))
        all_descs = u.WorldPartitionBlueprintLibrary.get_intersecting_actor_descs(box)
        descs = [d for d in all_descs if 'HLOD' not in str(d.native_class)]
        if not descs:
            raise RuntimeError('Requested region contains no full-detail actor descriptors')
        u.WorldPartitionBlueprintLibrary.load_actors([d.guid for d in descs])
        receipt['native_region'] = dict(bounds_m=region,
            intersecting_descriptors=len(all_descs), requested_full_detail_actors=len(descs),
            requested_guids=[normalized_guid(d.guid) for d in descs],
            policy='LOAD_EXISTING_ACTORS_NO_MAP_SAVE_DEPENDENCIES_MAY_EXTEND_REGION')
    # Match the existing city collector: loaded HLOD actors otherwise remain
    # visible alongside native actors. These changes live only in this editor.
    hidden = []
    for actor in api.get_all_level_actors():
        if 'HLOD' in actor.get_class().get_name():
            actor.set_is_temporarily_hidden_in_editor(True)
            actor.set_actor_hidden_in_game(True)
            actor.set_actor_enable_collision(False)
            hidden.append(actor.get_path_name())
    receipt['hidden_native_hlod_actors'] = hidden
    receipt.update(freeze_native_ticks(u, api))
    receipt['visibility_release'] = 'CALLER_MUST_EXIT_TASK_OWNED_EDITOR_WITHOUT_SAVING'
    return world, receipt


def freeze_native_ticks(u, api):
    """Task-owned settled editor only; no PIE or simulated 10Hz dynamics claim."""
    actors_disabled, components_disabled, errors, actor_paths = 0, 0, [], []
    for actor in api.get_all_level_actors():
        try:
            actor.set_actor_tick_enabled(False)
            actors_disabled += 1
            actor_paths.append(actor.get_path_name())
        except Exception as exc:
            errors.append(dict(object=actor.get_path_name(), error=str(exc)))
        try:
            components = actor.get_components_by_class(u.ActorComponent)
        except Exception as exc:
            errors.append(dict(object=actor.get_path_name(), error=str(exc)))
            continue
        for component in components:
            try:
                component.set_component_tick_enabled(False)
                components_disabled += 1
            except Exception as exc:
                errors.append(dict(object=component.get_path_name(), error=str(exc)))
    return dict(motion_scope='TICK_DISABLED_SETTLED_EDITOR_NO_SIMULATION',
                native_tick_freeze=dict(actors_disabled=actors_disabled,
                    components_disabled=components_disabled, actor_paths=actor_paths, errors=errors),
                motion_scope_limitation='Spatial poses in one editor world; not dynamic or real-time 10Hz evidence')


def exclude_hlod_from_captures(u, captures):
    """Mirror city collector component() show-only/full-detail visibility policy."""
    api = u.get_editor_subsystem(u.EditorActorSubsystem)
    hidden = [a for a in api.get_all_level_actors() if isinstance(a, u.WorldPartitionHLOD)]
    for actor in captures.values():
        for hlod in hidden:
            actor.capture_component2d.hide_actor_components(hlod, True)
    return [a.get_path_name() for a in hidden]


def place_captures(u, captures, camera, baseline_m=.06):
    """captures: name->actor; names ending in '_right' use optical-right offset."""
    camera = validated_pose(camera)
    if not math.isfinite(baseline_m) or baseline_m <= 0:
        raise ValueError('Positive finite stereo baseline required')
    _, right, _ = basis(camera)
    result = {}
    for name, actor in captures.items():
        offset = baseline_m if name.endswith('_right') else 0.
        pos = [camera[k]+offset*right[i] for i, k in enumerate(('x', 'y', 'z'))]
        actor.set_actor_location(u.Vector(*(v*100 for v in pos)), False, False)
        actor.set_actor_rotation(u.Rotator(**{k: camera[k] for k in ('pitch', 'yaw', 'roll')}), False)
        result[name] = dict(position_m=pos, rotation_deg={k:camera[k] for k in ('pitch','yaw','roll')})
    u.get_editor_subsystem(u.UnrealEditorSubsystem).set_level_viewport_camera_info(
        u.Vector(*(camera[k]*100 for k in ('x', 'y', 'z'))),
        u.Rotator(**{k:camera[k] for k in ('pitch','yaw','roll')}))
    return result


def verify_source_unchanged(receipt):
    digest = hashlib.sha256(Path(receipt['map_file']).read_bytes()).hexdigest()
    return dict(map_sha256_after=digest,
                map_unchanged=digest == receipt['map_sha256_before'],
                authority='MAP_FILE_ONLY_NOT_DEPENDENCY_OR_ASSET_CLOSURE')
