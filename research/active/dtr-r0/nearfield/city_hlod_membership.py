"""Read-only live-editor HLOD membership export; invoke from an owned UE session.

Call export_membership(unreal, output_path). No loading, spawning or source saving.
Unreflected/unknown fields remain explicit blockers, never an empty valid mapping.
"""
import json
import os
import hashlib
from pathlib import Path


def property_value(obj, name):
    try:
        return obj.get_editor_property(name)
    except Exception:
        return getattr(obj, name)


def guid_value(guid):
    # str(Guid) contains a transient address on this UE build.
    return ''.join(f'{int(property_value(guid, k)) & 0xffffffff:08X}' for k in ('a','b','c','d'))


def transform_value(transform):
    def vector(v):
        return [float(v.x), float(v.y), float(v.z)]
    rotation = transform.rotation
    return dict(translation_cm=vector(transform.translation), scale=vector(transform.scale3d),
                rotation_xyzw=[float(rotation.x),float(rotation.y),float(rotation.z),float(rotation.w)])


def guarded(row, name, operation):
    """Unknown stays null with an error, not false or an empty valid inventory."""
    try:
        row[name] = operation()
    except Exception as exc:
        row[name] = None
        row.setdefault('read_errors', {})[name] = str(exc)


def primitive_inventory(u, component):
    row = dict(component_path=component.get_path_name(),
               component_class=component.get_class().get_path_name())
    for name in ('visible', 'hidden_in_game', 'render_in_main_pass', 'render_in_depth_pass',
                 'owner_no_see', 'only_owner_see', 'cast_hidden_shadow',
                 'visible_in_reflection_captures', 'visible_in_ray_tracing', 'is_editor_only'):
        guarded(row, name, lambda name=name: bool(property_value(component, name)))
    guarded(row, 'is_visible', lambda: bool(component.is_visible()))
    def bounds():
        origin, extent, radius = u.SystemLibrary.get_component_bounds(component)
        return dict(min=[(getattr(origin,k)-getattr(extent,k))/100 for k in ('x','y','z')],
                    max=[(getattr(origin,k)+getattr(extent,k))/100 for k in ('x','y','z')],
                    sphere_radius_m=float(radius)/100)
    guarded(row, 'world_bounds_m', bounds)
    if isinstance(component, u.StaticMeshComponent):
        guarded(row, 'static_mesh_asset', lambda: (
            property_value(component, 'static_mesh').get_path_name()
            if property_value(component, 'static_mesh') else None))
    if isinstance(component, u.InstancedStaticMeshComponent):
        guarded(row, 'instance_count', lambda: int(component.get_instance_count()))
        row['instance_transforms_exported'] = False
    def materials():
        result = []
        for index in range(component.get_num_materials()):
            material = component.get_material(index)
            item = dict(slot=index, material_asset=material.get_path_name() if material else None)
            if material:
                # IsSky is a material property, not a mesh/actor label guess.
                guarded(item, 'base_material_asset', lambda: material.get_base_material().get_path_name())
                guarded(item, 'base_material_is_sky', lambda: bool(
                    property_value(material.get_base_material(), 'is_sky')))
            result.append(item)
        return result
    guarded(row, 'materials', materials)
    return row


def actor_inventory(u, actor):
    row = dict(actor_path=actor.get_path_name(), native_class=actor.get_class().get_path_name())
    guarded(row, 'actor_label', lambda: actor.get_actor_label())
    guarded(row, 'hidden_in_game', lambda: bool(property_value(actor, 'hidden')))
    guarded(row, 'temporarily_hidden_in_editor', lambda: bool(actor.is_temporarily_hidden_in_editor()))
    guarded(row, 'editor_only', lambda: bool(property_value(actor, 'is_editor_only_actor')))
    guarded(row, 'primitive_components', lambda: [primitive_inventory(u, component)
        for component in actor.get_components_by_class(u.PrimitiveComponent)])
    components = row['primitive_components']
    row['primitive_evidence'] = ('UNKNOWN' if components is None else
        'PRIMITIVE_COMPONENTS_PRESENT' if components else 'NO_PRIMITIVE_COMPONENTS_IN_LOADED_ACTOR')
    row['rendered_visibility'] = 'UNVERIFIED_FLAGS_ARE_NOT_RENDERER_VISIBILITY'
    return row


def export_membership(u, output_path):
    output_path = Path(output_path)
    # Captures execute a hashed snapshot, whose parents are not the checkout.
    # The launcher has already validated and bound this owned output directory.
    canonical = (Path(os.environ['BA_CITY_OUT']).resolve() if os.environ.get('BA_CITY_OUT')
                 else (Path(__file__).resolve().parents[4]/'artifacts.local').resolve())
    if output_path.exists() or not output_path.resolve().is_relative_to(canonical):
        raise ValueError('Fresh canonical membership output required')
    actors = u.get_editor_subsystem(u.EditorActorSubsystem).get_all_level_actors()
    native_export=None
    library=getattr(u,'BlindAssistCaptureLibrary',None)
    native_method=getattr(library,'export_world_hlod_source_membership',None) if library else None
    if native_method:
        native_path=output_path.with_name(output_path.stem+'-native.json')
        world=u.get_editor_subsystem(u.UnrealEditorSubsystem).get_editor_world()
        if native_method(world,str(native_path)):
            native_data=json.loads(native_path.read_bytes())
            native_export=dict(path=native_path.name,sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
                schema=native_data.get('schema'),status=native_data.get('status'),
                unresolved_hlod_count=native_data.get('unresolved_hlod_count'))
        else:native_export=dict(status='FAILED_TO_WRITE_NATIVE_MAPPING')
    rows, loaded = [], []
    for actor in actors:
        path = actor.get_path_name()
        loaded.append(actor_inventory(u, actor))
        if not isinstance(actor, u.WorldPartitionHLOD):
            continue
        row = dict(proxy_actor_path=path, status='UNVERIFIED', sources=[])
        try:
            source = property_value(actor, 'source_actors')
            row['source_mapping_class'] = source.get_class().get_path_name()
            mappings = property_value(source, 'actors')
            for mapping in mappings:
                item = {name: str(property_value(mapping, name)) for name in
                        ('package','path','container_package','world_package')}
                item['actor_instance_guid'] = guid_value(property_value(mapping,'actor_instance_guid'))
                item['container_transform'] = transform_value(property_value(mapping,'container_transform'))
                # Container ID serialization is version dependent. The instance GUID
                # plus owning package/transform are retained independently of repr.
                try:
                    item['container_id'] = guid_value(property_value(property_value(mapping,'container_id'),'id'))
                except Exception as exc:
                    item['container_id_error'] = str(exc)
                row['sources'].append(item)
            if row['sources'] and all(s['package'].startswith('/') and s['path'] and
                                     s['actor_instance_guid'] != '0'*32 for s in row['sources']):
                row['status'] = 'SOURCE_ACTOR_MAPPING_EXPORTED_NOT_COMPONENT_INSTANCE_ADMISSION'
            else:
                row['error'] = 'Empty or incomplete source mappings'
        except Exception as exc:
            row['error'] = str(exc)
        rows.append(row)
    result = dict(schema='city-live-hlod-membership-v2', status='UNVERIFIED',
                  scope='Currently loaded editor actors only; unloaded HLOD and nested source closure not established',
                  primitive_inventory_scope='Component types, flags, world bounds and material/mesh assets; no per-instance transforms, final renderer visibility or child/spawned actor closure',
                  loaded_actor_count=len(loaded), loaded_actor_paths=loaded, hlod=rows,
                  unresolved_hlod_count=sum(r['status']=='UNVERIFIED' for r in rows))
    result['native_export']=native_export
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
