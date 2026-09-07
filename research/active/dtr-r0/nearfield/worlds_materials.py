"""Create only G14-owned world-aligned PBR materials; never edit existing assets."""
import hashlib
import json
from pathlib import Path

BASE = '/Game/G14RealismV1'


def ensure_materials(u):
    data = Path(u.Paths.project_dir()).parent / 'g14-realism-assets'
    receipt = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    if receipt['status'] != 'PASS' or len(receipt['files']) != 14:
        raise RuntimeError('G14 surface downloads are incomplete')
    ml = u.MaterialEditingLibrary
    at = u.AssetToolsHelpers.get_asset_tools()
    # Verify even when imported assets exist: the receipt remains auditable.
    for entry in receipt['files']:
        payload = (data / entry['path']).read_bytes()
        if len(payload) != entry['bytes'] or hashlib.sha256(payload).hexdigest() != entry['sha256']:
            raise RuntimeError('G14 surface payload mismatch: ' + entry['path'])
    for name in sorted({e['asset'] for e in receipt['files']}):
        material_path = BASE + '/Materials/' + name
        if u.EditorAssetLibrary.does_asset_exist(material_path):
            continue
        mat = at.create_asset(name, BASE + '/Materials', u.Material, u.MaterialFactoryNew())
        mat.set_editor_property('used_with_nanite', True)
        foliage = name == 'shrub_01'
        mat.set_editor_property('tangent_space_normal', foliage)
        if foliage:
            mat.set_editor_property('blend_mode', u.BlendMode.BLEND_MASKED)
            mat.set_editor_property('two_sided', True)
            mat.set_editor_property('shading_model', u.MaterialShadingModel.MSM_TWO_SIDED_FOLIAGE)
        for entry in (e for e in receipt['files'] if e['asset'] == name and e['role'] != 'mesh'):
            path = data / entry['path']
            role = entry['role']
            texture_path = BASE + '/Textures/' + path.stem
            if u.EditorAssetLibrary.does_asset_exist(texture_path):
                tex = u.load_asset(texture_path)
            else:
                task = u.AssetImportTask()
                task.filename = str(path)
                task.destination_path = BASE + '/Textures'
                task.automated = True
                task.save = False
                task.replace_existing = False
                at.import_asset_tasks([task])
                imported = task.get_objects()
                if not imported:
                    raise RuntimeError('G14 texture import failed: ' + str(path))
                tex = imported[0]
                tex.set_editor_property('srgb', role == 'color')
                if role == 'normal':
                    tex.set_editor_property('compression_settings', u.TextureCompressionSettings.TC_NORMALMAP)
                    tex.set_editor_property('flip_green_channel', 'nor_gl' in path.name)
                u.EditorAssetLibrary.save_loaded_asset(tex)
            obj = ml.create_material_expression(mat, u.MaterialExpressionTextureSample if foliage else u.MaterialExpressionTextureObject)
            obj.texture = tex
            obj.sampler_type = (u.MaterialSamplerType.SAMPLERTYPE_NORMAL if role == 'normal' else
                                u.MaterialSamplerType.SAMPLERTYPE_COLOR if role == 'color' else
                                u.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
            if foliage:
                fn, output = obj, 'RGB' if role in ('color', 'normal') else 'R'
            else:
                size = ml.create_material_expression(mat, u.MaterialExpressionConstant3Vector)
                size.constant = u.LinearColor(entry['tile_cm'], entry['tile_cm'], entry['tile_cm'], 1)
                fn = ml.create_material_expression(mat, u.MaterialExpressionMaterialFunctionCall)
                fn.set_material_function(u.load_asset('/Engine/Functions/Engine_MaterialFunctions01/Texturing/' +
                                                     ('WorldAlignedNormal' if role == 'normal' else 'WorldAlignedTexture')))
                ml.connect_material_expressions(obj, '', fn, 'TextureObject')
                ml.connect_material_expressions(size, '', fn, 'TextureSize')
                output = 'XYZ Texture'
            prop = {'color': u.MaterialProperty.MP_BASE_COLOR, 'normal': u.MaterialProperty.MP_NORMAL,
                    'rough': u.MaterialProperty.MP_ROUGHNESS, 'alpha': u.MaterialProperty.MP_OPACITY_MASK}[role]
            ml.connect_material_property(fn, output, prop)
            if foliage and role == 'color':
                scatter = ml.create_material_expression(mat, u.MaterialExpressionMultiply)
                amount = ml.create_material_expression(mat, u.MaterialExpressionConstant)
                amount.set_editor_property('r', .2)
                ml.connect_material_expressions(fn, output, scatter, 'A')
                ml.connect_material_expressions(amount, '', scatter, 'B')
                ml.connect_material_property(scatter, '', u.MaterialProperty.MP_SUBSURFACE_COLOR)
        ml.recompile_material(mat)
        u.EditorAssetLibrary.save_loaded_asset(mat)
    mesh_path = BASE + '/Meshes/shrub_01_2k'
    if not u.EditorAssetLibrary.does_asset_exist(mesh_path):
        entry = next(e for e in receipt['files'] if e['role'] == 'mesh')
        options = u.FbxImportUI()
        options.import_mesh = True
        options.import_materials = False
        options.import_textures = False
        options.import_as_skeletal = False
        options.mesh_type_to_import = u.FBXImportType.FBXIT_STATIC_MESH
        options.automated_import_should_detect_type = False
        options.static_mesh_import_data.combine_meshes = True
        task = u.AssetImportTask()
        task.filename = str(data / entry['path'])
        task.destination_path = BASE + '/Meshes'
        task.automated = True
        task.save = False
        task.replace_existing = False
        task.options = options
        at.import_asset_tasks([task])
        objects = task.get_objects()
        if not objects:
            raise RuntimeError('G14 shrub import failed')
        mesh = objects[0]
        for index in range(len(mesh.static_materials)):
            mesh.set_material(index, u.load_asset(BASE + '/Materials/shrub_01'))
        u.EditorAssetLibrary.save_loaded_asset(mesh)
    return {'namespace': BASE, 'manifest_sha256': hashlib.sha256((data / 'manifest.json').read_bytes()).hexdigest(),
            'displacement': False, 'pavement_tile_cm': 180, 'leafy_ground_tile_cm': 200}
