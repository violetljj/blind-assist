"""One-session, unsaved material-only MFPD switch probe for six R3a inserts.

No map load, mesh edit, asset save, render, or frame capture.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cnh_route_derived_assets import derive


SHARED = {
    '/Game/Material/TextureCellBombing/Voronoi_Perturbed_2k_RGBA8_nonVT.Voronoi_Perturbed_2k_RGBA8_nonVT',
    '/Game/Textures/SurfaceFeature/T_RooftopPuddle.T_RooftopPuddle',
    '/Game/Textures/SurfaceFeature/T_tiled_shoeprints_random.T_tiled_shoeprints_random',
}


def path(obj):
    return obj.get_path_name() if obj is not None else None


def used(u, material):
    return sorted({path(t) for t in u.MaterialEditingLibrary.get_material_used_textures(material) if t})


def probe(u, spec):
    lib = u.MaterialEditingLibrary
    rows = []
    for item in spec['assets']:
        if item['role'] != 'insert':
            continue
        source_mesh = u.load_asset(item['asset_path'])
        slots = source_mesh.get_editor_property('static_materials')
        if source_mesh.get_num_sections(0) != 1:
            raise RuntimeError('Expected one LOD0 section: ' + item['asset_path'])
        slot = int(u.get_editor_subsystem(u.StaticMeshEditorSubsystem).get_lod_material_slot(source_mesh, 0, 0))
        if len(slots) != 1 or slot != 0:
            raise RuntimeError('Expected one source material slot: ' + item['asset_path'])
        source = slots[slot].get_editor_property('material_interface')
        if not isinstance(source, u.MaterialInstanceConstant):
            raise RuntimeError('Expected material instance: ' + item['asset_path'])
        switch = 'Enable MFPD'
        source_switch_before = bool(lib.get_material_instance_static_switch_parameter_value(source, switch))
        if not source_switch_before:
            raise RuntimeError('Source switch not enabled: ' + path(source))
        original = used(u, source)
        params = {str(name): path(lib.get_material_instance_texture_parameter_value(source, name))
                  for name in lib.get_texture_parameter_names(source)}
        keys = ('Albedo', 'Normal', 'Roughness')
        specific = {name: params.get(name) for name in keys}
        if not all(specific.values()) or not set(specific.values()).issubset(original):
            raise RuntimeError('Source-specific appearance textures not all used: ' + path(source))
        derived, receipt = derive(u, path(source), '/Game/CNHMFPDProbe', material_only=True)
        before = used(u, derived)
        derived_switch_before = bool(lib.get_material_instance_static_switch_parameter_value(derived, switch))
        if not derived_switch_before or not SHARED.issubset(before):
            raise RuntimeError('Derived pre-switch material does not retain source MFPD dependency: ' + path(derived))
        lib.set_material_instance_static_switch_parameter_value(derived, switch, False)
        lib.update_material_instance(derived)
        derived_switch_after = bool(lib.get_material_instance_static_switch_parameter_value(derived, switch))
        after = used(u, derived)
        source_switch_after = bool(lib.get_material_instance_static_switch_parameter_value(source, switch))
        source_used_after = used(u, source)
        rows.append(dict(split=item['split'], key=item['key'], source_mesh=path(source_mesh),
                         lod_index=0, material_slot=slot, source_material=path(source),
                         derived_material=path(derived), source_used_textures=original,
                         derived_before_used_textures=before, derived_after_used_textures=after,
                         removed_used_textures=sorted(set(before)-set(after)),
                         added_used_textures=sorted(set(after)-set(before)),
                         shared_three_remaining=sorted(SHARED.intersection(after)),
                         source_specific_appearance=specific,
                         source_specific_retained=all(v in after for v in specific.values()),
                         source_switch_before=source_switch_before, source_switch_after=source_switch_after,
                         derived_switch_before=derived_switch_before, derived_switch_after=derived_switch_after,
                         source_used_textures_unchanged=original == source_used_after,
                         derivation=receipt))
    if len(rows) != 6 or {r['split'] for r in rows} != {'train', 'dev', 'test'}:
        raise RuntimeError('Expected two inserts per split')
    split_used = {split: set().union(*(set(r['derived_after_used_textures']) for r in rows if r['split']==split))
                  for split in ('train', 'dev', 'test')}
    intersections = {a+'__'+b: sorted(split_used[a] & split_used[b])
                     for a,b in (('train','dev'),('train','test'),('dev','test'))}
    complete = (all(not r['shared_three_remaining'] and r['source_specific_retained']
                    and not r['derived_switch_after'] and r['source_switch_after']
                    and r['source_used_textures_unchanged'] for r in rows)
                and all(not x for x in intersections.values()))
    return dict(schema='cnh-alley-mfpd-unsaved-material-probe-v1',
                status='PASS_ZERO_CROSS_SPLIT_EDITOR_USED_TEXTURES' if complete else 'INCOMPLETE_ISOLATION',
                scope='UNSAVED_DERIVED_MATERIAL_ONLY_NO_MAP_LOAD_NO_RENDER_NO_TEST_CAPTURE',
                shared_three=sorted(SHARED), assets=rows,
                cross_split_editor_used_texture_intersections=intersections)


def main():
    import unreal
    out = Path(os.environ['BA_CNH_MFPD_OUTPUT'])
    try:
        spec = json.loads(Path(os.environ['BA_CNH_MFPD_SPEC']).read_text(encoding='utf-8'))
        result = probe(unreal, spec)
    except Exception:
        result = dict(schema='cnh-alley-mfpd-unsaved-material-probe-v1', status='FAIL', error=traceback.format_exc())
    out.mkdir(parents=True, exist_ok=True)
    (out/'mfpd-materials.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    unreal.SystemLibrary.quit_editor()


if __name__ == '__main__':
    main()
