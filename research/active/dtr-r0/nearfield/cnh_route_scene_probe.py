"""Read-only loaded-scene mesh capability probe, never benchmark admission.

LOD0 extraction does not establish Nanite, masked-material or WPO equivalence.
Only call on an already loaded map; this module neither loads nor changes assets.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import time


def xyz(value):
    return [float(getattr(value, key)) for key in ('x', 'y', 'z')]


def intersects_sphere(low, high, center, radius):
    if radius <= 0 or not all(math.isfinite(v) for v in [*low, *high, *center, radius]):
        raise ValueError('Finite bounds, centre and positive radius required')
    if any(a > b for a, b in zip(low, high)):
        raise ValueError('Reversed bounds')
    return sum(max(a-c, 0, c-b)**2 for a, b, c in zip(low, high, center)) <= radius**2


def transformed_bounds(u, mesh, transform):
    box = mesh.get_bounding_box()
    corners = [xyz(u.MathLibrary.transform_location(transform, u.Vector(*point)))
               for point in itertools.product(*[(getattr(box.min, k), getattr(box.max, k))
                                                for k in ('x', 'y', 'z')])]
    return ([min(p[i] for p in corners)/100 for i in range(3)],
            [max(p[i] for p in corners)/100 for i in range(3)])


def property_value(obj, name):
    try:
        return obj.get_editor_property(name)
    except Exception:
        return None


def material_record(material):
    if material is None:
        return dict(asset_path=None, blend_mode='UNKNOWN', masked='UNKNOWN',
                    world_position_offset='UNKNOWN', status='NOT_VERIFIED')
    # Instances may inherit/override properties. A successful property read alone
    # does not prove the compiled material has no deformation or opacity logic.
    blend = property_value(material, 'blend_mode')
    return dict(asset_path=material.get_path_name(), blend_mode=str(blend) if blend is not None else 'UNKNOWN',
                masked=('OBSERVED_MASKED' if blend is not None and 'MASKED' in str(blend).upper() else 'UNKNOWN'),
                world_position_offset='UNKNOWN', status='NOT_VERIFIED')


def write_fresh(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def probe(u, api, camera, radius_m, out):
    """Export loaded nearby static instances and LOD0; return a NOT_ADMITTED receipt.

    ``out`` must be a fresh task-owned directory. IDs identify exported entities,
    not pixels: native instance rasterisation is explicitly still unimplemented.
    All loaded non-static primitive types are reported conservatively, even when
    their intersection with the requested sphere cannot be established.
    """
    center = [float(camera[k]) for k in ('x', 'y', 'z')]
    intersects_sphere(center, center, center, radius_m)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out/'geometry').mkdir()
    started = time.monotonic()
    rows, errors, unsupported, meshes = [], [], [], {}
    library = getattr(u, 'ProceduralMeshLibrary', None)

    def export(mesh):
        key = mesh.get_path_name()
        if key in meshes:
            return meshes[key]
        item = dict(asset_path=key, lod_index=0, status='NOT_EXPORTED',
                    render_equivalence='NOT_VERIFIED')
        meshes[key] = item
        nanite = property_value(mesh, 'nanite_settings')
        enabled = property_value(nanite, 'enabled') if nanite is not None else None
        item['nanite'] = ('ENABLED' if enabled is True else 'DISABLED' if enabled is False else 'UNKNOWN')
        try:
            if library is None or not hasattr(library, 'get_section_from_static_mesh'):
                raise RuntimeError('Render LOD extraction API unavailable; no proxy fallback')
            sections = []
            for section in range(mesh.get_num_sections(0)):
                vertices, triangles, normals, uv, tangents = library.get_section_from_static_mesh(mesh, 0, section)
                if not vertices or not triangles or len(triangles) % 3:
                    raise ValueError('Empty or malformed render section')
                if any(i < 0 or i >= len(vertices) for i in triangles):
                    raise ValueError('Triangle index outside vertex buffer')
                sections.append(dict(section=section, vertices_m=[[v/100 for v in xyz(p)] for p in vertices],
                                     triangles=list(triangles), normals=[xyz(p) for p in normals],
                                     uv=[[float(p.x), float(p.y)] for p in uv]))
            if not sections:
                raise ValueError('No render sections')
            relative = 'geometry/' + hashlib.sha256(key.encode()).hexdigest() + '.json'
            write_fresh(out/relative, dict(asset_path=key, lod_index=0,
                        coordinates='UE_LOCAL_XYZ_METRES', sections=sections))
            item.update(status='EXPORTED_LOD0_NOT_RENDER_VERIFIED', path=relative,
                        sha256=hashlib.sha256((out/relative).read_bytes()).hexdigest(),
                        triangles=sum(len(s['triangles'])//3 for s in sections))
        except Exception as exc:
            item['error'] = str(exc)
        return item

    for actor in sorted(api.get_all_level_actors(), key=lambda a: a.get_path_name()):
        for component in sorted(actor.get_components_by_class(u.PrimitiveComponent), key=lambda c: c.get_path_name()):
            identity = dict(actor_path=actor.get_path_name(), component_path=component.get_path_name(),
                            component_class=component.get_class().get_name())
            if not isinstance(component, u.StaticMeshComponent):
                unsupported.append(dict(identity, status='UNSUPPORTED_PRIMITIVE', spatial_scope='NOT_ESTABLISHED'))
                continue
            try:
                mesh = component.static_mesh
                if mesh is None:
                    unsupported.append(dict(identity, status='STATIC_COMPONENT_WITHOUT_MESH'))
                    continue
                instanced = isinstance(component, u.InstancedStaticMeshComponent)
                count = component.get_instance_count() if instanced else 1
                for index in range(count):
                    transform = component.get_instance_transform(index, True) if instanced else component.get_world_transform()
                    low, high = transformed_bounds(u, mesh, transform)
                    if not intersects_sphere(low, high, center, radius_m):
                        continue
                    rotation = transform.rotation
                    rows.append(dict(identity, instance_index=index if instanced else None,
                        id=len(rows)+1, mesh=export(mesh), bounds_min_m=low, bounds_max_m=high,
                        actual_translation_m=[v/100 for v in xyz(transform.translation)],
                        actual_rotation_quaternion=[*xyz(rotation), float(rotation.w)],
                        actual_scale=xyz(transform.scale3d),
                        materials=[material_record(component.get_material(slot))
                                   for slot in range(component.get_num_materials())],
                        visibility='NOT_VERIFIED', material_surface_equivalence='NOT_VERIFIED'))
            except Exception as exc:
                errors.append(dict(identity, error=str(exc)))
    receipt = dict(schema='cnh-loaded-scene-probe-v1', status='NOT_ADMITTED',
        authority='LOADED_STATIC_MESH_LOD0_CAPABILITY_ONLY', camera_m=center, radius_m=radius_m,
        selection='TRANSFORMED_MESH_AABB_INTERSECTS_SPHERE', instances=rows, meshes=list(meshes.values()),
        unsupported_primitives=unsupported, errors=errors, instance_count=len(rows),
        uint16_capacity_ok=len(rows) <= 65534, instance_raster='NOT_IMPLEMENTED',
        full_scene_coverage='NOT_ESTABLISHED', streaming_completeness='NOT_ESTABLISHED',
        render_geometry_equivalence='NOT_VERIFIED', wall_s=time.monotonic()-started)
    write_fresh(out/'scene-probe.json', receipt)
    return receipt
