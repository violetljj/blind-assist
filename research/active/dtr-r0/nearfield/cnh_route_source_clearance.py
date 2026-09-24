"""Loaded native-bound audit, not full-world completeness or collection admission.

Call before spawning inserted actors. Material capability is compiled-engine
evidence; missing support stays UNKNOWN. All loaded static components are audited,
including those outside the original probe sphere that WPO could move inward.
"""
from __future__ import annotations

import json
import math

from cnh_route_native_clearance import evaluate
from cnh_route_scene_probe import property_value, transformed_bounds, xyz
from cnh_route_source_compare_adapter import normalized_guid


def material_padding(report, primitive_extent=None):
    """Native helper schema; a configured material clamp alone is insufficient."""
    if report.get('data_status') != 'AVAILABLE':
        return None
    if report.get('uses_pdo') is not False or report.get('uses_displacement') is not False:
        return None
    if report.get('capability') == 'NO_COMPILED_MATERIAL_DEFORMATION' and report.get('uses_wpo') is False:
        return 0.
    if report.get('capability') != 'WPO_CLAMP_CONFIGURED' or report.get('uses_wpo') is not True:
        return None
    # This optional evidence must come from the compiled primitive aggregate API,
    # never from the maximum of individual material configuration values.
    if not primitive_extent or primitive_extent.get('status') != 'VERIFIED' or primitive_extent.get('all_component_materials') is not True:
        return None
    value = primitive_extent.get('max_wpo_extent_cm')
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return value/100 if math.isfinite(value) and value > 0 else None


def audit(u, api, layouts, probe_receipt, source_receipt):
    """Return per-layout manifest/result plus explicit unresolved authority.

    WorldPartition descriptor matching is necessary but cannot certify external
    dependencies, landscape, runtime-spawned geometry or full-world completeness.
    """
    from cnh_route_source_compare_adapter import is_development
    development=is_development(source_receipt)
    valid_count=1<=len(layouts)<=20 if development else len(layouts)==2
    if not valid_count or len({r['layout_id'] for r in layouts})!=len(layouts):
        raise ValueError('Invalid fixed layout batch')
    actors = list(api.get_all_level_actors())
    loaded_guids = {normalized_guid(property_value(a, 'actor_guid')) for a in actors
                    if property_value(a, 'actor_guid') is not None}
    requested = source_receipt.get('native_region', {}).get('requested_guids', [])
    missing_guids = [g for g in requested if normalized_guid(g) not in loaded_guids]
    row_index = {(r['component_path'], r.get('instance_index')):r
                 for r in probe_receipt.get('instances', [])}
    entities, unsupported, excluded, materials = [], [], [], {}
    capability = getattr(getattr(u, 'BlindAssistCaptureLibrary', None),
                         'get_material_geometry_capability', None)
    for actor in actors:
        for component in actor.get_components_by_class(u.PrimitiveComponent):
            path = component.get_path_name()
            # UE5.8 ZoneGraphRenderingComponent.cpp:140 and
            # ZoneGraphAnnotationComponent.cpp:20-40 gate these debug proxies.
            # Only these exact types, with both flags disabled in every capture.
            if (component.get_class().get_name() in (
                    'ZoneGraphRenderingComponent','ZoneGraphCrowdLaneAnnotations','ZoneGraphDisturbanceAnnotation')
                    and source_receipt.get('capture_debug_flags') == {'Navigation':False,'ZoneGraph':False}):
                excluded.append(dict(component=path,reason='DEBUG_VISUALIZATION_EXCLUDED_BY_CAPTURE_FLAGS'))
                continue
            if property_value(component, 'hidden_in_game') is True or property_value(component, 'visible') is False:
                excluded.append(dict(component=path, reason='HIDDEN_OR_INVISIBLE_IN_FIXED_SETTLED_CAPTURE'))
                continue
            # HLOD suppression is authenticated by the task-owned adapter receipt.
            if actor.get_path_name() in source_receipt.get('hidden_native_hlod_actors', []):
                excluded.append(dict(component=path, reason='ADAPTER_SUPPRESSED_HLOD'))
                continue
            is_static_mesh = isinstance(component, u.StaticMeshComponent) and component.static_mesh is not None
            is_dynamic_mesh = component.get_class().get_name() == 'DynamicMeshComponent'
            # SplineMeshComponent and other deforming subclasses are StaticMesh
            # instances too. Asset-box + rigid transform does not bound their
            # rendered vertices, even when every material reports no WPO.
            if is_static_mesh and component.get_class().get_name() not in (
                    'StaticMeshComponent', 'InstancedStaticMeshComponent', 'HierarchicalInstancedStaticMeshComponent'):
                unsupported.append(dict(component=path, reason='STATIC_MESH_SUBCLASS_DEFORMATION_NOT_AUDITED'))
                continue
            if not is_static_mesh and not is_dynamic_mesh:
                unsupported.append(dict(component=path, reason='NO_AUTHENTICATED_CONSERVATIVE_RENDER_BOUND'))
                continue
            padding, reasons, material_paths = [], [], []
            count = component.get_num_materials()
            if not count:
                reasons.append('MATERIAL_SLOTS_UNRESOLVED')
            for slot in range(count):
                material = component.get_material(slot)
                # UE 5.8 BaseDynamicMeshSceneProxy.cpp:120-123 uses this exact
                # engine fallback for null slots. Do not change the component.
                if material is None and is_dynamic_mesh:
                    fallback = getattr(getattr(u,'BlindAssistCaptureLibrary',None), 'get_default_surface_material', None)
                    material = fallback() if fallback else None
                key = material.get_path_name() if material is not None else path+':null:'+str(slot)
                material_paths.append(key)
                if key not in materials:
                    try:
                        materials[key] = json.loads(capability(material)) if capability and material else dict(status='UNKNOWN')
                    except Exception as exc:
                        materials[key] = dict(status='UNKNOWN', error=str(exc))
                value = material_padding(materials[key])
                if value is None:
                    reasons.append('MATERIAL_GEOMETRY_UNBOUNDED:'+key)
                else:
                    padding.append(value)
            # Static mobility excludes transform animation in the evaluated clips;
            # movable assets need an explicit swept motion bound, absent here.
            frozen_editor = (source_receipt.get('motion_scope') == 'TICK_DISABLED_SETTLED_EDITOR_NO_SIMULATION'
                and source_receipt.get('native_tick_freeze', {}).get('errors') == []
                and actor.get_path_name() in source_receipt.get('native_tick_freeze', {}).get('actor_paths', []))
            mobility_static = property_value(component, 'mobility') == u.ComponentMobility.STATIC or frozen_editor
            if not mobility_static:
                reasons.append('COMPONENT_MOTION_NOT_BOUNDED')
            instanced = isinstance(component, u.InstancedStaticMeshComponent)
            for index in range(component.get_instance_count() if instanced else 1):
                instance_index = index if instanced else None
                try:
                    transform = component.get_instance_transform(index, True) if instanced else component.get_world_transform()
                    # Recompute current engine mesh bounds; probe hashes/geometry
                    # are associated evidence, never a frozen transform substitute.
                    if is_static_mesh:
                        if xyz(transform.scale3d)==[0.,0.,0.]:
                            excluded.append(dict(component=path,instance_index=instance_index,
                                                 reason='ZERO_SCALE_NO_RENDER_SURFACE'))
                            continue
                        low, high = transformed_bounds(u, component.static_mesh, transform)
                    else:
                        scale = property_value(component, 'bounds_scale')
                        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 1.:
                            raise ValueError('Dynamic component bounds scale is missing or shrinks geometry')
                        origin, extent, _ = u.SystemLibrary.get_component_bounds(component)
                        low = [(getattr(origin,k)-getattr(extent,k))/100 for k in ('x','y','z')]
                        high = [(getattr(origin,k)+getattr(extent,k))/100 for k in ('x','y','z')]
                        if not all(math.isfinite(v) for v in low+high) or any(a>b for a,b in zip(low,high)):
                            raise ValueError('Invalid dynamic component bounds')
                    expansion = max(padding, default=0.)
                    row = row_index.get((path, instance_index))
                    entities.append(dict(id=path+':'+str(instance_index), kind='native',
                        bounds_min_m=[v-expansion for v in low], bounds_max_m=[v+expansion for v in high],
                        bounds_conservative=not reasons, deformation_bounded=not reasons,
                        material_paths=material_paths, world_padding_m=expansion,
                        probe_geometry=row.get('mesh') if row else None,
                        reasons=list(reasons), authority='CURRENT_ENGINE_BOUND_WITH_COMPILED_MATERIAL_GEOMETRY_AUDIT'))
                except Exception as exc:
                    unsupported.append(dict(component=path, instance_index=instance_index,
                                            reason='BOUND_EXTRACTION_FAILED', error=str(exc)))
    required = []
    if missing_guids:
        required.append('REQUESTED_WORLD_PARTITION_ACTORS_NOT_LOADED')
    authored_alley = (source_receipt.get('scope') == 'ALLEY_DEVELOPMENT_PILOT_NOT_BENCHMARK'
        and source_receipt.get('authored_nonpartitioned_map') == 'FROZEN_PACKAGE_HASHES_VERIFIED'
        and source_receipt.get('map_asset', '').startswith('/Game/BAResearchAlley/'))
    if not requested and source_receipt.get('map_asset') != '/Game/BAResearchSlice/Street200V7' and not authored_alley:
        required.append('NO_WORLD_PARTITION_DESCRIPTOR_AUDIT')
    if unsupported:
        required.append('UNSUPPORTED_PRIMITIVES_REQUIRE_CONSERVATIVE_BOUNDS_OR_AUTHENTICATED_EXCLUSION')
    if any(e['reasons'] for e in entities):
        required.append('STATIC_MATERIAL_OR_MOTION_BOUNDS_UNRESOLVED')
    candidates = []
    for layout in layouts:
        clips=layout.get('clips', [])
        if development:
            compact=[]
            for clip in clips:
                poses=clip['poses'];a,b=poses[0],poses[-1]
                linear=all(all(abs(p[k]-(a[k]+(b[k]-a[k])*i/(len(poses)-1)))<1e-8
                    for k in ('x','y','z','pitch','yaw','roll')) for i,p in enumerate(poses))
                compact.append(dict(clip,poses=[a,b],clearance_sweep='EXACT_LINEAR_PATH_ENDPOINT_ENVELOPE',
                    original_pose_count=len(poses)) if linear else clip)
            clips=compact
        manifest = dict(native_entities=entities, native_coverage_complete=not required,
                        coverage_scope='ENUMERATED_LOADED_WORLD_ONLY', clips=clips)
        result = evaluate(manifest)
        # AABB overlap is a prefilter rejection, not proven surface intrusion.
        candidates.append(dict(layout_id=layout['layout_id'], manifest=manifest, result=result,
            status=('FAIL_PREFILTER' if result['conflicts'] else
                    ('PASS_LOADED_WORLD_ONLY' if result['status']=='PASS' else 'UNKNOWN')),
            required_missing=required+result['reasons']))
    return dict(scope='LOADED_NATIVE_BOUND_AUDIT_ONLY', admitted_layouts=0,
        native_coverage_complete=False, loaded_world_coverage_complete=not required,
        scope_limitation='Does not certify full source, unloaded dependencies, future simulation or benchmark admission',
        requested_guids=requested, missing_requested_guids=missing_guids,
        requested_descriptors_loaded=bool(requested) and not missing_guids,
        unsupported_primitives=unsupported, excluded_primitives=excluded,
        material_capabilities=materials, candidates=candidates, required_missing=required)
