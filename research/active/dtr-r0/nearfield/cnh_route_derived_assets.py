"""Unsaved controlled-insertion mesh/material derivation inside Unreal Editor.

No save calls or source mutations. Derived identity retains the source asset family.
Material Attributes roots retain their complete input and override only WPO/PDO.
Call derive(unreal, source_object_path, '/Game/CNH_<task>/Derived').
"""
from __future__ import annotations
import hashlib
import re
import uuid


def _path(obj):
    return obj.get_path_name() if obj is not None else None


def _vec(value):
    return [float(value.x), float(value.y), float(value.z)]


def _stable_state(value):
    # PyWrapperStruct::Str includes the temporary wrapper's address before its
    # complete friendly property value. Addresses are not configuration changes.
    return re.sub(r"(<Struct '[^']+' )\([0-9a-fA-Fx]+\)", r'\1(WRAPPER)', str(value))


def derive(u, mesh_asset_path, package_root, *, material_only=False):
    def stage(name):
        message = 'CNH_DERIVE_STAGE '+name+' source='+mesh_asset_path
        u.log_warning(message)
        print(message, flush=True)

    if not re.fullmatch(r'/Game/CNH[A-Za-z0-9_/]*', package_root) or '..' in package_root:
        raise ValueError('Derived assets require a task-owned /Game/CNH... package root')
    source = u.load_asset(mesh_asset_path)
    if not material_only and not isinstance(source, u.StaticMesh):
        raise ValueError('Source is not a StaticMesh: '+mesh_asset_path)
    editor = u.get_editor_subsystem(u.StaticMeshEditorSubsystem)
    library = u.MaterialEditingLibrary
    assets = u.AssetToolsHelpers.get_asset_tools()
    run_root = package_root.rstrip('/')+'/D_'+uuid.uuid4().hex
    source_settings = None if material_only else _stable_state(editor.get_nanite_settings(source))
    source_enabled = False if material_only else bool(editor.get_nanite_settings(source).get_editor_property('enabled'))
    source_slots = [] if material_only else [_path(slot.get_editor_property('material_interface')) for slot in source.get_editor_property('static_materials')]
    records, cache, snapshots = [], {}, []
    # Do not enumerate every EMaterialProperty: the UE editing helper dereferences
    # a null input for unsupported enum members (e.g. diffuse/specular legacy).
    supported = ('EMISSIVE_COLOR OPACITY OPACITY_MASK BASE_COLOR METALLIC SPECULAR ROUGHNESS '
                 'ANISOTROPY NORMAL TANGENT WORLD_POSITION_OFFSET DISPLACEMENT SUBSURFACE_COLOR '
                 'CUSTOM_DATA_0 CUSTOM_DATA_1 AMBIENT_OCCLUSION REFRACTION MATERIAL_ATTRIBUTES '
                 'PIXEL_DEPTH_OFFSET SHADING_MODEL SURFACE_THICKNESS FRONT_MATERIAL').split()
    supported += ['CUSTOMIZED_U_VS_'+str(i) for i in range(8)]
    property_names = ['MP_'+name for name in supported if hasattr(u.MaterialProperty, 'MP_'+name)]
    if 'MP_WORLD_POSITION_OFFSET' not in property_names:
        raise RuntimeError('Material property graph inspection unavailable')

    def graph(material):
        result = {}
        for name in property_names:
            prop = getattr(u.MaterialProperty, name)
            node = library.get_material_property_input_node(material, prop)
            result[name] = (_path(node), library.get_material_property_input_node_output_name(material, prop))
        return result

    def instance_state(material):
        # Duplication retains inherited static switches and every override; these
        # exposed arrays are checked again after swapping only the parent pointer.
        names = ('scalar_parameter_values', 'vector_parameter_values', 'texture_parameter_values',
                 'font_parameter_values', 'base_property_overrides')
        return {name: _stable_state(material.get_editor_property(name)) for name in names}

    def duplicate(obj):
        stage('duplicate '+_path(obj))
        suffix = hashlib.sha256(_path(obj).encode()).hexdigest()[:12]
        clone = assets.duplicate_asset(obj.get_name()+'_CNH_'+suffix, run_root, obj)
        if clone is None or _path(clone) == _path(obj):
            raise RuntimeError('Asset duplication failed: '+_path(obj))
        return clone

    def material_clone(original):
        if original is None:
            raise ValueError('Missing source material slot')
        key = _path(original)
        if key in cache:
            if cache[key] is None:
                raise ValueError('Cyclic material parent chain')
            return cache[key]
        cache[key] = None
        if isinstance(original, u.MaterialInstanceConstant):
            parent = original.get_editor_property('parent')
            state = instance_state(original)
            snapshots.append(('instance', original, (_path(parent), state)))
            cloned_parent = material_clone(parent)
            clone = duplicate(original)
            copied = instance_state(clone)
            if copied != state:
                raise RuntimeError('Material duplication changed overrides')
            library.set_material_instance_parent(clone, cloned_parent)
            if instance_state(clone) != copied or _path(clone.get_editor_property('parent')) != _path(cloned_parent):
                raise RuntimeError('Material parent rewrite did not preserve overrides')
            library.update_material_instance(clone)
            records.append(dict(source=key, derived=_path(clone), kind='MaterialInstanceConstant',
                                source_parent=_path(parent), derived_parent=_path(cloned_parent), overrides_preserved=True))
        elif isinstance(original, u.Material):
            uses_attributes = bool(original.get_editor_property('use_material_attributes'))
            state = graph(original)
            snapshots.append(('root', original, state))
            clone = duplicate(original)
            before = graph(clone)
            stage('native_zero_deformation '+key)
            helper = getattr(u.BlindAssistCaptureLibrary, 'zero_derived_material_deformation', None)
            if helper is None:
                raise RuntimeError('Updated native ZeroDerivedMaterialDeformation helper is required')
            attr_prop = u.MaterialProperty.MP_MATERIAL_ATTRIBUTES
            retained_input = library.get_material_property_input_node(clone, attr_prop) if uses_attributes else None
            retained_output = library.get_material_property_input_node_output_name(clone, attr_prop) if uses_attributes else None
            if not helper(clone):
                raise RuntimeError('Native deformation wrapper rejected derived material')
            stage('recompile_derived_material '+key)
            library.recompile_material(clone)
            after = graph(clone)
            changed = {'MP_WORLD_POSITION_OFFSET', 'MP_PIXEL_DEPTH_OFFSET'}
            attribute_receipt = None
            if uses_attributes:
                changed.add('MP_MATERIAL_ATTRIBUTES')
                wrapper = library.get_material_property_input_node(clone, attr_prop)
                if not isinstance(wrapper, u.MaterialExpressionSetMaterialAttributes):
                    raise RuntimeError('Native attributes wrapper missing after recompile')
                connections = list(library.get_inputs_for_material_expression(clone, wrapper))
                if len(connections) != 3 or connections[0] != retained_input:
                    raise RuntimeError('Native wrapper did not preserve original full attributes')
                vector, scalar = connections[1:]
                if not isinstance(vector, u.MaterialExpressionConstant3Vector) or not isinstance(scalar, u.MaterialExpressionConstant):
                    raise RuntimeError('Native override nodes are not constants')
                color = vector.get_editor_property('constant')
                if any(float(getattr(color, n)) != 0. for n in ('r','g','b')) or float(scalar.get_editor_property('r')) != 0.:
                    raise RuntimeError('Native override nodes are nonzero')
                attribute_receipt = dict(retained_input=_path(retained_input), retained_output=retained_output,
                                         wrapper=_path(wrapper), native_preserves_original_output_index=True,
                                         override_input_names=['WorldPositionOffset','PixelDepthOffset'])
            zero = library.get_material_property_input_node(clone, u.MaterialProperty.MP_WORLD_POSITION_OFFSET)
            if not isinstance(zero, u.MaterialExpressionConstant3Vector):
                raise RuntimeError('Derived direct WPO is not a zero-vector node')
            color = zero.get_editor_property('constant')
            if any(float(getattr(color, n)) != 0. for n in ('r','g','b')):
                raise RuntimeError('Derived direct WPO is nonzero')
            for name in before:
                if name not in changed and before[name] != after[name]:
                    raise RuntimeError('Non-deformation material input changed: '+name)
            records.append(dict(source=key, derived=_path(clone), kind='Material',
                                wpo='EXPLICIT_ZERO_VECTOR', pixel_depth_offset='EXPLICIT_ZERO_SCALAR',
                                material_attributes=uses_attributes, attributes=attribute_receipt,
                                native_deformation_helper_verified=True,
                                non_deformation_inputs_unchanged=True))

        else:
            raise ValueError('Unsupported material interface: '+key)
        cache[key] = clone
        return clone

    if material_only:
        material = material_clone(source)
        for kind, original, state in snapshots:
            current = graph(original) if kind == 'root' else (_path(original.get_editor_property('parent')), instance_state(original))
            if current != state:
                raise RuntimeError('Source material changed: '+_path(original))
        return material, dict(schema='cnh_derived_material_v1', source_material=_path(source),
            derived_material=_path(material), saved=False, source_configuration_unchanged=True,
            materials=records, mesh_and_nanite_unchanged=True)

    mesh = duplicate(source)
    for index, slot in enumerate(source.get_editor_property('static_materials')):
        mesh.set_material(index, material_clone(slot.get_editor_property('material_interface')))
    settings = editor.get_nanite_settings(mesh)
    settings.set_editor_property('enabled', False)
    stage('disable_derived_nanite')
    editor.set_nanite_settings(mesh, settings, True)
    if editor.get_nanite_settings(mesh).get_editor_property('enabled'):
        raise RuntimeError('Derived Nanite is still enabled')
    if _stable_state(editor.get_nanite_settings(source)) != source_settings:
        raise RuntimeError('Source Nanite configuration changed')
    if [_path(slot.get_editor_property('material_interface')) for slot in source.get_editor_property('static_materials')] != source_slots:
        raise RuntimeError('Source mesh material slots changed')
    for kind, original, state in snapshots:
        current = graph(original) if kind == 'root' else (_path(original.get_editor_property('parent')), instance_state(original))
        if current != state:
            raise RuntimeError('Source material graph/overrides changed: '+_path(original))
    stage('source_unchanged_verified')
    bounds = mesh.get_bounds()
    receipt = dict(schema='cnh_derived_asset_v1', source_mesh=_path(source), derived_mesh=_path(mesh),
                   package_root=run_root, saved=False, asset_family_identity=_path(source), new_asset_family=False,
                   source_nanite_enabled=source_enabled, derived_nanite_enabled=False,
                   source_configuration_unchanged=True, source_unchanged=True, materials=records,
                   local_bounds_cm=dict(origin=_vec(bounds.origin), extent=_vec(bounds.box_extent)),
                   configuration=dict(wpo='ZERO_ON_CLONED_ROOTS', pixel_depth_offset='ZERO_ON_CLONED_ROOTS', nanite=False,
                                      component_lod='CALLER_MUST_FORCE_EXPORTED_LOD', textures_and_appearance='DUPLICATED_GRAPH_RETAINED'))
    return mesh, receipt
