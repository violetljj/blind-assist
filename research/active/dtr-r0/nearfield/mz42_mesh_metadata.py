"""Read four existing UE mesh assets without editing or saving project content."""
import json
import os
from pathlib import Path
import traceback

import unreal as u


ASSETS = {
    'pipe': '/Game/Prop/Kit_ventilation_RR/Mesh/SM_roof_Ac_pipe_long_N1',
    'ladder': '/Game/Prop/CONSTRUCTION/Kit_Ladder/Mesh/SM_Ladder_A01_N1',
    'pouch': '/Game/Prop/FWY/Kit_Trash/Mesh/SM_FWYTrash_Pouch_A01_N1',
    'birch': '/Game/Prop/Kit_Tree_Birch/Mesh/SM_Tree_Birch_a',
}


def main():
    output = Path(os.environ['BA_MZ42_METADATA'])
    if output.exists():
        raise FileExistsError(output)
    result = dict(status='STARTED', assets={}, scope='Read-only actual Unreal asset load and bounds/material inspection')
    try:
        for family, path in ASSETS.items():
            mesh = u.load_asset(path)
            if not isinstance(mesh, u.StaticMesh):
                raise ValueError('Missing StaticMesh: ' + path)
            bounds = mesh.get_bounding_box()
            row = dict(asset=mesh.get_path_name(), bounds_min_cm=[bounds.min.x, bounds.min.y, bounds.min.z],
                       bounds_max_cm=[bounds.max.x, bounds.max.y, bounds.max.z], materials=[])
            for slot in mesh.static_materials:
                material = slot.material_interface
                info = dict(asset=material.get_path_name() if material else None)
                try:
                    base = material.get_base_material()
                    info.update(base=base.get_path_name(), blend_mode=str(base.get_editor_property('blend_mode')),
                                two_sided=bool(base.get_editor_property('two_sided')))
                except Exception as error:
                    info.update(properties_status='UNKNOWN', error=str(error))
                row['materials'].append(info)
            result['assets'][family] = row
        result['status'] = 'PASS'
    except Exception:
        result.update(status='FAIL', error=traceback.format_exc())
        raise
    finally:
        output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        u.SystemLibrary.quit_editor()


main()
