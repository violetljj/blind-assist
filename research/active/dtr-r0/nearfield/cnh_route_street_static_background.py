"""Transient Street material intervention; never save source assets or the map.

Only compiled WPO/PDO materials are cloned. Native mesh, Nanite, mask and
appearance inputs remain unchanged. Unknown/null materials remain unresolved.
"""
from __future__ import annotations
import json
import os
from cnh_route_derived_assets import derive


def apply(u, api):
    cache, slots, unresolved = {}, [], []
    helper = u.BlindAssistCaptureLibrary.get_material_geometry_capability
    for actor in api.get_all_level_actors():
        for component in actor.get_components_by_class(u.PrimitiveComponent):
            if component.get_editor_property('hidden_in_game') or not component.get_editor_property('visible'):
                continue
            for index in range(component.get_num_materials()):
                original = component.get_material(index)
                if original is None:
                    unresolved.append(dict(component=component.get_path_name(), slot=index, reason='NULL_MATERIAL'))
                    continue
                key = original.get_path_name()
                capability = json.loads(helper(original))
                if capability.get('data_status') != 'AVAILABLE':
                    unresolved.append(dict(component=component.get_path_name(), slot=index, reason='CAPABILITY_UNKNOWN'))
                    continue
                if not (capability.get('uses_wpo') or capability.get('uses_pdo')):
                    continue
                if key not in cache:
                    material, receipt = derive(u, key, '/Game/CNHStreetStatic_'+str(os.getpid()), material_only=True)
                    after = json.loads(helper(material))
                    cache[key] = (material, receipt, capability, after)
                material, receipt, before, after = cache[key]
                component.set_material(index, material)
                if component.get_material(index) != material:
                    raise RuntimeError('Transient native material assignment failed')
                slots.append(dict(component=component.get_path_name(), slot=index, source=key,
                    derived=material.get_path_name()))
    return dict(schema='cnh_street_static_material_intervention_v1', saved=False,
        policy='STREET_TRANSIENT_ZERO_WPO_PDO', mesh_and_nanite_unchanged=True,
        all_capture_modalities_share_component_assignments=True, slots=slots,
        materials=[dict(derivation=r, before=b, after=a) for _,r,b,a in cache.values()],
        unresolved=unresolved, release='EXIT_TASK_OWNED_EDITOR_WITHOUT_SAVING')


def verify(u, receipt):
    """Called only after the normal asynchronous capture-readiness gate."""
    for row in receipt['materials']:
        material = u.load_asset(row['derivation']['derived_material'])
        row['after'] = json.loads(u.BlindAssistCaptureLibrary.get_material_geometry_capability(material))
        after = row['after']
        if after.get('data_status') != 'AVAILABLE' or after.get('capability') != 'NO_COMPILED_MATERIAL_DEFORMATION':
            raise RuntimeError('Derived native material not deformation-free: '+json.dumps(after))
    receipt['compiled_verification'] = 'PASS'
