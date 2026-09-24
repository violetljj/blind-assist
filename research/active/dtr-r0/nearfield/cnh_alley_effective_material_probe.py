"""UE read-only inspection of authored alley source mesh LOD0 material use.

No maps are loaded and no frames are rendered. The saved-map actor and source
identity links are checked separately by cnh_alley_asset_isolation_audit.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import traceback


def object_path(obj):
    return obj.get_path_name() if obj is not None else None


def inspect(unreal, spec):
    editor=unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    library=unreal.MaterialEditingLibrary
    if editor is None or not hasattr(library,'get_material_used_textures'):
        raise RuntimeError('Native LOD/material inspection API unavailable')
    result=[]
    for item in spec['assets']:
        mesh=unreal.load_asset(item['asset_path'])
        if not isinstance(mesh,unreal.StaticMesh):
            raise ValueError('StaticMesh unavailable: '+item['asset_path'])
        slots=mesh.get_editor_property('static_materials')
        sections=[]
        for section in range(mesh.get_num_sections(0)):
            slot=int(editor.get_lod_material_slot(mesh,0,section))
            if not 0<=slot<len(slots):
                raise ValueError('LOD0 section slot outside mesh materials: '+item['asset_path'])
            sections.append(dict(section=section,material_slot=slot))
        if not sections:
            raise ValueError('LOD0 has no sections: '+item['asset_path'])
        used_slots=sorted({row['material_slot'] for row in sections})
        materials=[]
        for slot in used_slots:
            material=slots[slot].get_editor_property('material_interface')
            if material is None:
                raise ValueError('Null LOD0 material: '+item['asset_path'])
            chain=[]
            seen=set()
            ancestor=material
            while ancestor is not None:
                path=object_path(ancestor)
                if path in seen:raise ValueError('Material parent cycle: '+path)
                seen.add(path)
                if isinstance(ancestor,unreal.MaterialInstanceConstant):
                    overrides=[]
                    for override in ancestor.get_editor_property('texture_parameter_values'):
                        info=override.get_editor_property('parameter_info')
                        overrides.append(dict(name=str(info.get_editor_property('name')),
                            association=str(info.get_editor_property('association')),
                            index=int(info.get_editor_property('index')),
                            texture=object_path(override.get_editor_property('parameter_value'))))
                    chain.append(dict(path=path,kind='MaterialInstanceConstant',
                        parent=object_path(ancestor.get_editor_property('parent')),
                        explicit_texture_overrides=sorted(overrides,key=lambda x:(x['name'],x['association'],x['index']))))
                    ancestor=ancestor.get_editor_property('parent')
                elif isinstance(ancestor,unreal.Material):
                    chain.append(dict(path=path,kind='Material',parent=None))
                    ancestor=None
                else:
                    raise ValueError('Unresolved material parent type: '+path)
            if not chain or chain[-1]['kind']!='Material':
                raise ValueError('Material parent chain has no root: '+object_path(material))
            parameters=[]
            if isinstance(material,unreal.MaterialInstanceConstant):
                for name in library.get_texture_parameter_names(material):
                    texture=library.get_material_instance_texture_parameter_value(material,name)
                    parameters.append(dict(name=str(name),texture=object_path(texture)))
            switches=[]
            if isinstance(material,unreal.MaterialInstanceConstant):
                for name in library.get_static_switch_parameter_names(material):
                    switches.append(dict(name=str(name),
                        value=bool(library.get_material_instance_static_switch_parameter_value(material,name))))
            textures=sorted({object_path(texture) for texture in library.get_material_used_textures(material)
                             if texture is not None})
            materials.append(dict(slot=slot,interface=object_path(material),parent_chain=chain,
                effective_texture_parameters=sorted(parameters,key=lambda x:x['name']),
                effective_static_switches=sorted(switches,key=lambda x:x['name']),
                editor_used_textures=textures))
        result.append(dict(split=item['split'],role=item['role'],key=item['key'],
            mesh=object_path(mesh),lod_index=0,sections=sections,
            source_nanite_enabled=bool(editor.get_nanite_settings(mesh).get_editor_property('enabled')),
            materials=materials))
    return dict(schema='cnh-alley-native-effective-material-v1',
        status='NATIVE_SOURCE_LOD0_MATERIAL_INSPECTION_COMPLETE',
        scope='READ_ONLY_SOURCE_MESHES_NO_MAP_LOAD_NO_TEST_CAPTURE',
        assets=result)


def main():
    import unreal
    out=Path(os.environ['BA_CNH_EFFECTIVE_OUTPUT'])
    try:
        spec=json.loads(Path(os.environ['BA_CNH_EFFECTIVE_SPEC']).read_text(encoding='utf-8'))
        report=inspect(unreal,spec)
    except Exception:
        report=dict(schema='cnh-alley-native-effective-material-v1',status='FAIL',
                    error=traceback.format_exc())
    out.mkdir(parents=True,exist_ok=True)
    (out/'native-materials.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    unreal.SystemLibrary.quit_editor()


if __name__=='__main__':main()
