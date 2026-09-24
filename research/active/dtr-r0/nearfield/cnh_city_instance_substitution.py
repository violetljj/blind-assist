"""Unsaved per-instance City LOD0 substitution on two consumed layouts.

The first diagnostic keeps the prior vehicle zero-scale control and substitutes
only the implicated City0 curb. The full mode substitutes near vehicles too.
Neither mode establishes native visible instance IDs or benchmark admission.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from cnh_city_nearfield_derived import _materials, _state, camera_centres
from cnh_city_vehicle_mask import metadata, selected_indices, transform_state
from cnh_route_scene_probe import transformed_bounds


POLICY = 'CITY_NEAR_INSTANCE_DERIVED_LOD0_SUBSTITUTION'
CURB_MESH = ('/Game/Megascans/3D_Assets/Modular_Curb_5_M_00/'
             'Modular_Curb_5_M_LOD0_vcflbc0dw_YELLOW.Modular_Curb_5_M_LOD0_vcflbc0dw_YELLOW')
CURB_COMPONENT = ('/Game/Map/Small_City_LVL.Small_City_LVL:PersistentLevel.'
                  'TemplateActor_UAID_D45D6454B40F8AF200_1551189997.'
                  'StaticMesh__Game_Megascans_3D_Assets_Modular_Curb_5_M_00_'
                  'Modular_Curb_5_M_LOD0_vcflbc0dw_YELLOW_'
                  'Modular_Curb_5_M_LOD0_vcflbc0dw_YELLOW__11')
CURB_INDEX = 3
MODES = ('CURB_ONLY_ON_VEHICLE_MASK_CONTROL', 'NEAR_VEHICLES_AND_CURB')


def frozen_spec(capture, diagnostic, mode):
    """Pin both inputs and reject any changed curb-ray witness."""
    from cnh_city_nearfield_derived import frozen_spec as consumed_spec
    if mode not in MODES:
        raise ValueError('Unknown City substitution mode')
    diagnostic = Path(diagnostic)
    result = json.loads(diagnostic.read_text(encoding='utf-8-sig'))
    if result.get('status') != 'DIAGNOSTIC_ONLY' or result.get('benchmark_eligible') is not False:
        raise ValueError('Consumed City vehicle diagnostic required')
    witnesses = []
    for frame in result['frames']:
        if frame['layout_id'] != 'city-engineering-0' or frame['pose_index'] != 1:
            continue
        for ray in frame['worst_background_rays']:
            if (ray['x'], ray['y']) == (223, 277):
                attribution = next((row for row in frame['predicted_surface_attribution']
                                    if row['surface_index'] == ray['surface_index']), None)
                if (ray['surface_index'] != 15 or ray['mesh'] != CURB_MESH or attribution is None
                        or attribution['component'] != CURB_COMPONENT
                        or attribution['instance'] != CURB_INDEX
                        or attribution['mesh'] != CURB_MESH):
                    raise ValueError('City0 curb-ray attribution changed')
                witnesses.append(frame['clip_id'])
    if set(witnesses) != {'centre', 'boundary', 'outside', 'removed'} or len(witnesses) != 4:
        raise ValueError('Four repeated consumed City0 curb-ray witnesses required')
    spec = consumed_spec(capture)
    spec['native_geometry_policy'] = POLICY
    spec['city_derived_control'].update(substitution_mode=mode,
        consumed_diagnostic_sha256=hashlib.sha256(diagnostic.read_bytes()).hexdigest(),
        curb_attribution='EXPORTED_MESH_RAY_WINNER_NOT_NATIVE_VISIBLE_ID',
        curb_witness_clips=witnesses)
    return spec


def chosen_instances(path, mesh_path, bounds, centres, mode):
    """Freeze the curb identity; only vehicle assets use the 8 m rule."""
    if path == CURB_COMPONENT:
        if mesh_path != CURB_MESH or CURB_INDEX >= len(bounds):
            raise ValueError('Frozen curb component identity changed')
        if CURB_INDEX not in selected_indices(bounds, centres):
            raise ValueError('Frozen curb instance lies outside consumed 8 m union')
        return [CURB_INDEX]
    if mode == 'NEAR_VEHICLES_AND_CURB' and mesh_path.startswith('/Game/Vehicle/'):
        return selected_indices(bounds, centres)
    return []


class Session:
    def __init__(self, u, api, editor, receipt, vehicle_control=None):
        self.u, self.api, self.editor, self.receipt = u, api, editor, receipt
        self.vehicle_control = vehicle_control
        self.entries = []
        self.sources = {}
        self.clones = []
        self.native_reader = u.BlindAssistCaptureLibrary.get_ism_preservation_state

    def verify_sources(self):
        for mesh, before in self.sources.values():
            if _state(self.editor, mesh) != before:
                raise RuntimeError('Source mesh changed: ' + mesh.get_path_name())
        self.receipt['source_meshes_unchanged'] = True

    def restore(self):
        errors = []
        for component, instanced, originals, selected, before, overlays in reversed(self.entries):
            for actor in reversed(overlays):
                try:
                    if not self.api.destroy_actor(actor):
                        raise RuntimeError('Derived overlay actor release failed')
                except Exception as exc:
                    errors.append(str(exc))
            for index in selected:
                try:
                    if instanced:
                        if not component.update_instance_transform(index, originals[index], True, True, True):
                            raise RuntimeError('Original instance restoration failed')
                    else:
                        component.set_world_transform(originals[index], False, True)
                except Exception as exc:
                    errors.append(str(exc))
            try:
                current = ([component.get_instance_transform(i, True) for i in range(component.get_instance_count())]
                           if instanced else [component.get_world_transform()])
                if len(current) != len(originals):
                    raise RuntimeError('Original instance count changed')
                maxima = [0.] * 10
                for index, (actual, original) in enumerate(zip(current, originals)):
                    delta = [abs(a-b) for a, b in zip(transform_state(actual), transform_state(original))]
                    maxima = [max(a, b) for a, b in zip(maxima, delta)]
                    limits = ([.001]*3 + [1e-6]*7) if index in selected else [0.]*10
                    if any(d > limit for d, limit in zip(delta, limits)):
                        raise RuntimeError('Original transform restoration exceeds engine roundtrip precision')
                if metadata(component, instanced, self.native_reader) != before:
                    raise RuntimeError('Original instance metadata changed')
                self.receipt.setdefault('restoration_checks', []).append(dict(
                    component=component.get_path_name(), maximum_absolute_delta=maxima,
                    units='translation_cm,quaternion_xyzw,scale_xyz',
                    touched_translation_tolerance_cm=.001, touched_quaternion_scale_tolerance=1e-6,
                    untouched_tolerance=0.))
            except Exception as exc:
                errors.append(str(exc))
        try:
            self.verify_sources()
        except Exception as exc:
            errors.append(str(exc))
        if self.vehicle_control is not None:
            try:
                self.vehicle_control.restore()
            except Exception as exc:
                errors.append(str(exc))
        self.receipt['restoration'] = dict(restored=not errors, errors=errors,
            unsaved_assets_release='EXIT_TASK_OWNED_EDITOR_WITHOUT_SAVING')
        if errors:
            raise RuntimeError('City instance substitution restoration failed: ' + repr(errors))
        self.entries.clear()
        return self.receipt['restoration']


def apply(u, api, spec):
    if spec.get('map_asset') != '/Game/Map/Small_City_LVL' or spec.get('native_geometry_policy') != POLICY:
        raise ValueError('Explicit City instance substitution policy required')
    control = spec.get('city_derived_control', {})
    mode = control.get('substitution_mode')
    if (mode not in MODES or control.get('authority') != 'CONSUMED_TWO_LAYOUT_ENGINEERING_DIAGNOSTIC'
            or control.get('benchmark_eligible') is not False):
        raise ValueError('Only consumed City diagnostic substitution modes are supported')
    centres = camera_centres(spec)
    editor = u.get_editor_subsystem(u.StaticMeshEditorSubsystem)
    assets = u.AssetToolsHelpers.get_asset_tools()
    receipt = dict(schema='cnh_city_instance_substitution_v1', policy=POLICY, mode=mode,
        status='PREFLIGHT', benchmark_eligible=False, radius_m=8., camera_centres_m=centres,
        source_attribution='EXPORTED_MESH_RAY_WINNER_NOT_NATIVE_VISIBLE_ID',
        curb_witness=dict(pixel_xy=[223, 277], surface_index=15, instance_index=CURB_INDEX,
                          old_radial_error_mm=881.0783486771072, independent_rays=1,
                          repeated_clips=4),
        geometry_thresholds='UNCHANGED_V1', map_saved=False, source_assets_mutated=False,
        all_capture_modalities_share_overlays=True, components=[], meshes=[])
    planned = []
    curb_seen = False
    for actor in api.get_all_level_actors():
        if 'HLOD' in actor.get_class().get_name():
            if not (actor.is_hidden() and actor.is_temporarily_hidden_in_editor()):
                raise ValueError('HLOD must be excluded before City substitution')
        for component in actor.get_components_by_class(u.StaticMeshComponent):
            mesh = component.static_mesh
            if mesh is None:
                continue
            path = component.get_path_name()
            instanced = isinstance(component, u.InstancedStaticMeshComponent)
            transforms = ([component.get_instance_transform(i, True) for i in range(component.get_instance_count())]
                          if instanced else [component.get_world_transform()])
            if path == CURB_COMPONENT:
                curb_seen = True
                if not instanced:
                    raise ValueError('Frozen curb is no longer an instanced component')
            if path != CURB_COMPONENT and (mode != 'NEAR_VEHICLES_AND_CURB' or
                                            not mesh.get_path_name().startswith('/Game/Vehicle/')):
                continue
            indices = chosen_instances(path, mesh.get_path_name(),
                [transformed_bounds(u, mesh, t) for t in transforms], centres, mode)
            if indices:
                planned.append((component, mesh, instanced, transforms, indices,
                                metadata(component, instanced, u.BlindAssistCaptureLibrary.get_ism_preservation_state)))
    if not curb_seen or not any(p[0].get_path_name() == CURB_COMPONENT for p in planned):
        raise ValueError('Frozen City0 curb instance not present in loaded world')
    if mode == 'NEAR_VEHICLES_AND_CURB' and not any(p[1].get_path_name().startswith('/Game/Vehicle/') for p in planned):
        raise ValueError('No near vehicles found for full substitution')
    session = Session(u, api, editor, receipt)
    # Keep the consumed zero-scale vehicle run as the controlled curb-only baseline.
    if mode == 'CURB_ONLY_ON_VEHICLE_MASK_CONTROL':
        from cnh_city_vehicle_mask import apply as apply_vehicle_mask, POLICY as MASK_POLICY
        masked_spec = dict(spec, native_geometry_policy=MASK_POLICY)
        session.vehicle_control = apply_vehicle_mask(u, api, masked_spec)
        receipt['vehicle_control'] = session.vehicle_control.receipt
    package = '/Game/CNHCitySubstitution/D_' + uuid.uuid4().hex
    cache = {}
    try:
        for component, mesh, instanced, originals, selected, before in planned:
            key = mesh.get_path_name()
            if key not in cache:
                original_state = _state(editor, mesh)
                session.sources[key] = mesh, original_state
                clone = assets.duplicate_asset(mesh.get_name()+'_CNH_'+hashlib.sha256(key.encode()).hexdigest()[:12],
                                               package, mesh)
                if clone is None or clone == mesh:
                    raise RuntimeError('City mesh duplication failed: ' + key)
                session.clones.append(clone)
                settings = editor.get_nanite_settings(clone)
                settings.set_editor_property('enabled', False)
                editor.set_nanite_settings(clone, settings, True)
                if editor.get_nanite_settings(clone).get_editor_property('enabled'):
                    raise RuntimeError('Derived City mesh still enables Nanite')
                if _materials(clone) != _materials(mesh):
                    raise RuntimeError('Derived City material slots changed')
                session.verify_sources()
                cache[key] = clone
                receipt['meshes'].append(dict(source=key, derived=clone.get_path_name(),
                    source_state=original_state, derived_nanite_enabled=False, materials_same_objects=True))
            clone = cache[key]
            overlays = []
            session.entries.append((component, instanced, originals, selected, before, overlays))
            for index in selected:
                transform = originals[index]
                masked = u.Transform(location=transform.translation,
                    rotation=transform.rotation.rotator(), scale=u.Vector(0, 0, 0))
                if instanced:
                    if not component.update_instance_transform(index, masked, True, True, True):
                        raise RuntimeError('Original near instance masking failed')
                else:
                    component.set_world_transform(masked, False, True)
                overlay = api.spawn_actor_from_class(u.StaticMeshActor, transform.translation)
                if overlay is None:
                    raise RuntimeError('Derived City overlay actor spawn failed')
                overlays.append(overlay)
                overlay.set_actor_transform(transform, False, True)
                derived_component = overlay.static_mesh_component
                derived_component.set_static_mesh(clone)
                derived_component.set_editor_property('disallow_nanite', True,
                    notify_mode=u.PropertyAccessChangeNotifyMode.NEVER)
                derived_component.set_forced_lod_model(1)
                for slot in range(component.get_num_materials()):
                    derived_component.set_material(slot, component.get_material(slot))
                if (derived_component.static_mesh != clone or
                        derived_component.get_editor_property('forced_lod_model') != 1 or
                        not derived_component.get_editor_property('disallow_nanite')):
                    raise RuntimeError('Derived City component is not locked to non-Nanite LOD0')
                transform_delta = [abs(a-b) for a, b in zip(
                    transform_state(overlay.get_actor_transform()), transform_state(transform))]
                if any(d > limit for d, limit in zip(transform_delta, [.001]*3+[1e-6]*7)):
                    raise RuntimeError('Derived City overlay transform exceeds engine roundtrip precision')
                if any(derived_component.get_material(slot) != component.get_material(slot)
                       for slot in range(component.get_num_materials())):
                    raise RuntimeError('Derived City effective material differs')
                receipt['components'].append(dict(original_component=component.get_path_name(),
                    original_index=index, original_mesh=key, derived_actor=overlay.get_path_name(),
                    derived_mesh=clone.get_path_name(), original_transform=transform_state(transform),
                    original_index_retained=True, derived_transform_maximum_absolute_delta=transform_delta,
                    derived_forced_lod_model=1, derived_disallow_nanite=True))
            current = ([component.get_instance_transform(i, True) for i in range(component.get_instance_count())]
                       if instanced else [component.get_world_transform()])
            if len(current) != len(originals):
                raise RuntimeError('Original instance count changed')
            for index, (actual, original) in enumerate(zip(current, originals)):
                if index in selected:
                    if transform_state(actual)[7:] != [0., 0., 0.]:
                        raise RuntimeError('Original substituted instance still has render scale')
                elif transform_state(actual) != transform_state(original):
                    raise RuntimeError('Far original instance transform changed')
            if metadata(component, instanced, session.native_reader) != before:
                raise RuntimeError('Original custom data/material/seed/cull state changed')
        session.verify_sources()
        receipt.update(status='APPLIED_REQUIRES_RENDER_AND_GEOMETRY_DIAGNOSTIC',
            substituted_instances=len(receipt['components']),
            substituted_vehicle_instances=sum(x['original_mesh'].startswith('/Game/Vehicle/')
                                              for x in receipt['components']))
        return session
    except Exception:
        session.restore()
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = frozen_spec(args.capture, args.diagnostic, args.mode)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
