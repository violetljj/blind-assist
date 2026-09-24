"""Unsaved, material-preserving City mesh control for two consumed layouts.

Call apply before probes/readiness, keep the Session alive, and call restore in
the collector's finally path after its capture-integrity comparison. No save,
console-variable, material-graph or farfield component changes are performed.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import uuid
from cnh_route_derived_assets import _stable_state
from cnh_route_scene_probe import transformed_bounds, intersects_sphere

POLICY = 'CITY_NEARFIELD_DERIVED_LOD0_MATERIALS_UNCHANGED'


class NearFarPreflightError(ValueError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__('Near/far instance scope cannot be preserved without splitting')


def frozen_spec(capture):
    """Reuse recorded selected cameras/assets; never search new candidates."""
    capture = Path(capture)
    source = capture/'source/spec.json'
    selected = capture/'candidate-selection.json'
    spec = json.loads(source.read_text(encoding='utf-8-sig'))
    layouts = json.loads(selected.read_text())['selected_layouts']
    spec['layouts'] = layouts
    camera_centres(spec)
    for layout in layouts:
        layout.pop('candidates', None)
    spec['native_geometry_policy'] = POLICY
    spec['city_derived_control'] = dict(authority='CONSUMED_TWO_LAYOUT_ENGINEERING_DIAGNOSTIC',
        benchmark_eligible=False, new_layouts=False, radius_m=8.,
        source_spec_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        selected_layouts_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),
        geometry_thresholds='UNCHANGED_V1', required_global_render_toggles=False)
    return spec


def camera_centres(spec):
    layouts = spec['layouts']
    if len(layouts) != 2 or len({x['layout_id'] for x in layouts}) != 2:
        raise ValueError('Exactly two consumed City layouts required')
    centres = []
    for layout in layouts:
        poses = [layout['camera']]
        poses += [p for clip in layout['clips'] for p in clip['poses']]
        for pose in poses:
            point = tuple(float(pose[k]) for k in ('x', 'y', 'z'))
            if point not in centres:
                centres.append(point)
    return centres


def selection(bounds, centres):
    """All instances of a component must be near, or none; never alter far ones."""
    hits = [any(intersects_sphere(lo, hi, c, 8.) for c in centres) for lo, hi in bounds]
    if any(hits) and not all(hits):
        raise ValueError('Mixed near/far instanced component requires instance splitting')
    return bool(hits) and all(hits)


def _materials(mesh):
    return [s.get_editor_property('material_interface') for s in mesh.get_editor_property('static_materials')]


def _state(editor, mesh):
    return dict(nanite=_stable_state(editor.get_nanite_settings(mesh)),
                materials=[m.get_path_name() if m else None for m in _materials(mesh)])


def _component_state(component, instanced):
    transforms = ([component.get_instance_transform(i, True) for i in range(component.get_instance_count())]
                  if instanced else [component.get_world_transform()])
    return dict(transforms=[_stable_state(t) for t in transforms],
                overrides=_stable_state(component.get_editor_property('override_materials')),
                materials=[component.get_material(i) for i in range(component.get_num_materials())])


class Session:
    def __init__(self, editor, receipt):
        self.editor, self.receipt = editor, receipt
        self.sources, self.assignments, self.clones = {}, [], []

    def verify_sources(self):
        for source, before in self.sources.values():
            if _state(self.editor, source) != before:
                raise RuntimeError('Source mesh configuration changed: '+source.get_path_name())
        self.receipt['source_configuration_unchanged'] = True

    def restore(self):
        errors = []
        for component, source, lod, before, instanced in reversed(self.assignments):
            try:
                component.set_static_mesh(source)
                component.set_forced_lod_model(lod)
                if component.static_mesh != source or int(component.get_editor_property('forced_lod_model')) != lod:
                    raise RuntimeError('Mesh/LOD restoration differs')
                if _component_state(component, instanced) != before:
                    raise RuntimeError('Transform/material restoration differs')
            except Exception as exc:
                errors.append(str(exc))
        try:
            self.verify_sources()
        except Exception as exc:
            errors.append(str(exc))
        self.receipt['restoration'] = dict(restored=not errors, errors=errors,
            unsaved_clones_release='EXIT_TASK_OWNED_EDITOR_WITHOUT_SAVING')
        if errors:
            raise RuntimeError('City derived restoration failed: '+repr(errors))
        self.assignments.clear()
        return self.receipt['restoration']


def apply(u, api, spec):
    if spec.get('map_asset') != '/Game/Map/Small_City_LVL' or spec.get('native_geometry_policy') != POLICY:
        raise ValueError('Only explicitly selected City nearfield derived control is supported')
    control = spec.get('city_derived_control', {})
    if control.get('authority') != 'CONSUMED_TWO_LAYOUT_ENGINEERING_DIAGNOSTIC' or control.get('benchmark_eligible') is not False:
        raise ValueError('Consumed diagnostic receipt required; no benchmark admission')
    centres = camera_centres(spec)
    editor = u.get_editor_subsystem(u.StaticMeshEditorSubsystem)
    assets = u.AssetToolsHelpers.get_asset_tools()
    package = '/Game/CNHCityNearfield/D_'+uuid.uuid4().hex
    receipt = dict(schema='cnh_city_nearfield_derived_v1', policy=POLICY,
        radius_m=8., camera_centres_m=centres, saved=False, material_graph_edits=False,
        global_render_toggles=False, farfield_components_unchanged=True,
        physical_clearance_retains_translucent_geometry=True,
        depth_parity_material_filter='SEPARATE_FROM_PHYSICAL_CLEARANCE',
        all_capture_modalities_share_component_assignments=True, components=[], meshes=[])
    session = Session(editor, receipt)
    planned = []
    # Preflight every component before assigning any derived mesh.
    for actor in api.get_all_level_actors():
        for component in actor.get_components_by_class(u.StaticMeshComponent):
            mesh = component.static_mesh
            if mesh is None:
                continue
            instanced = isinstance(component, u.InstancedStaticMeshComponent)
            transforms = ([component.get_instance_transform(i, True) for i in range(component.get_instance_count())]
                          if instanced else [component.get_world_transform()])
            try:
                chosen = selection([transformed_bounds(u, mesh, t) for t in transforms], centres)
            except ValueError as exc:
                if 'Mixed near/far' in str(exc):
                    receipt.update(status='REJECTED_MIXED_NEAR_FAR_INSTANCES',
                        rejected_component=component.get_path_name(), mutations_started=False,
                        source_configuration_unchanged=True, planned_components=len(planned),
                        reason=str(exc), next_action='NO_CAPTURE_NO_SCOPE_EXPANSION')
                    raise NearFarPreflightError(receipt) from exc
                raise ValueError(component.get_path_name()+': '+str(exc)) from exc
            if chosen:
                planned.append((component, mesh, instanced))
    if not planned:
        raise ValueError('No nearfield static mesh components selected')
    cache = {}
    try:
        for component, source, instanced in planned:
            key = source.get_path_name()
            if key not in cache:
                before = _state(editor, source)
                session.sources[key] = source, before
                name = source.get_name()+'_CNH_'+hashlib.sha256(key.encode()).hexdigest()[:12]
                clone = assets.duplicate_asset(name, package, source)
                if clone is None or clone == source:
                    raise RuntimeError('Mesh duplication failed: '+key)
                session.clones.append(clone)
                settings = editor.get_nanite_settings(clone)
                settings.set_editor_property('enabled', False)
                editor.set_nanite_settings(clone, settings, True)
                if editor.get_nanite_settings(clone).get_editor_property('enabled'):
                    raise RuntimeError('Derived mesh still enables Nanite')
                if _materials(clone) != _materials(source):
                    raise RuntimeError('Mesh duplication changed material references')
                session.verify_sources()
                cache[key] = clone
                receipt['meshes'].append(dict(source=key, derived=clone.get_path_name(),
                    source_state=before, derived_nanite_enabled=False, materials_same_objects=True))
            clone = cache[key]
            before = _component_state(component, instanced)
            lod = int(component.get_editor_property('forced_lod_model'))
            session.assignments.append((component, source, lod, before, instanced))
            component.set_static_mesh(clone)
            component.set_forced_lod_model(1)
            if component.static_mesh != clone or int(component.get_editor_property('forced_lod_model')) != 1:
                raise RuntimeError('Derived mesh/LOD assignment failed')
            if _component_state(component, instanced) != before:
                raise RuntimeError('Assignment changed transforms or material overrides')
            receipt['components'].append(dict(component=component.get_path_name(), source_mesh=key,
                derived_mesh=clone.get_path_name(), instances=len(before['transforms']),
                original_forced_lod_model=lod, forced_lod_model=1, transforms_and_materials_unchanged=True))
        session.verify_sources()
        receipt['status'] = 'APPLIED_REQUIRES_READINESS_AND_GEOMETRY_DIAGNOSTIC'
        return session
    except Exception:
        session.restore()
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = frozen_spec(args.capture)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
