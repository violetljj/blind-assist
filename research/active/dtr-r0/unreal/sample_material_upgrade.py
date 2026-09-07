"""Verified Poly Haven assets and correctly typed PBR maps for the sample only."""
import hashlib
import json
from pathlib import Path
import unreal as u

BASE='/Game/SampleMaterialsV2'
DATA=Path(u.Paths.project_dir()).parent/'sample-materials-v2'
ml=u.MaterialEditingLibrary
at=u.AssetToolsHelpers.get_asset_tools()


def imported(path,folder,options=None):
    destination=BASE+'/'+folder+'/'+path.stem
    if u.EditorAssetLibrary.does_asset_exist(destination):return u.load_asset(destination)
    task=u.AssetImportTask();task.filename=str(path);task.destination_path=BASE+'/'+folder
    task.automated=True;task.save=True;task.replace_existing=False
    if options:task.options=options
    at.import_asset_tasks([task])
    objects=task.get_objects()
    if not objects:raise RuntimeError('Import produced no asset: '+str(path))
    return objects[0]


def texture(path,role):
    tex=imported(path,'Textures')
    tex.set_editor_property('srgb',role=='color')
    if role=='normal':
        tex.set_editor_property('compression_settings',u.TextureCompressionSettings.TC_NORMALMAP)
        tex.set_editor_property('flip_green_channel','nor_gl' in path.name)
    u.EditorAssetLibrary.save_loaded_asset(tex)
    return tex


def material(name,maps,tile_cm=None,foliage=False,uv_channel=0):
    if u.EditorAssetLibrary.does_asset_exist(BASE+'/Materials/'+name):
        mat=u.load_asset(BASE+'/Materials/'+name)
        ml.delete_all_material_expressions(mat)
    else:
        mat=at.create_asset(name,BASE+'/Materials',u.Material,u.MaterialFactoryNew())
    mat.set_editor_property('used_with_nanite',True)
    properties={'color':u.MaterialProperty.MP_BASE_COLOR,'normal':u.MaterialProperty.MP_NORMAL,
                'rough':u.MaterialProperty.MP_ROUGHNESS,'metal':u.MaterialProperty.MP_METALLIC,
                'ao':u.MaterialProperty.MP_AMBIENT_OCCLUSION,'alpha':u.MaterialProperty.MP_OPACITY_MASK}
    for role,path in maps.items():
        tex=texture(path,role)
        if tile_cm:
            obj=ml.create_material_expression(mat,u.MaterialExpressionTextureObject)
            obj.texture=tex
            obj.sampler_type=(u.MaterialSamplerType.SAMPLERTYPE_NORMAL if role=='normal' else
                u.MaterialSamplerType.SAMPLERTYPE_COLOR if role=='color' else u.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
            size=ml.create_material_expression(mat,u.MaterialExpressionConstant3Vector)
            size.constant=u.LinearColor(tile_cm,tile_cm,tile_cm,1)
            fn=ml.create_material_expression(mat,u.MaterialExpressionMaterialFunctionCall)
            fn.set_material_function(u.load_asset('/Engine/Functions/Engine_MaterialFunctions01/Texturing/'+
                ('WorldAlignedNormal' if role=='normal' else 'WorldAlignedTexture')))
            ml.connect_material_expressions(obj,'',fn,'TextureObject')
            ml.connect_material_expressions(size,'',fn,'TextureSize')
            node,output=fn,'XYZ Texture'
        else:
            node=ml.create_material_expression(mat,u.MaterialExpressionTextureSample)
            node.texture=tex
            uv=ml.create_material_expression(mat,u.MaterialExpressionTextureCoordinate)
            uv.set_editor_property('coordinate_index',uv_channel)
            ml.connect_material_expressions(uv,'',node,'UVs')
            node.sampler_type=(u.MaterialSamplerType.SAMPLERTYPE_NORMAL if role=='normal' else
                u.MaterialSamplerType.SAMPLERTYPE_COLOR if role=='color' else u.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
            output='RGB' if role in ('normal','color') else 'R'
        ml.connect_material_property(node,output,properties[role])
        if foliage and role=='color':
            scatter=ml.create_material_expression(mat,u.MaterialExpressionMultiply)
            amount=ml.create_material_expression(mat,u.MaterialExpressionConstant)
            amount.set_editor_property('r',.2)
            ml.connect_material_expressions(node,output,scatter,'A')
            ml.connect_material_expressions(amount,'',scatter,'B')
            ml.connect_material_property(scatter,'',u.MaterialProperty.MP_SUBSURFACE_COLOR)
    if tile_cm:mat.set_editor_property('tangent_space_normal',False)
    if foliage:
        specular=ml.create_material_expression(mat,u.MaterialExpressionConstant)
        specular.set_editor_property('r',.1)
        ml.connect_material_property(specular,'',u.MaterialProperty.MP_SPECULAR)
        mat.set_editor_property('blend_mode',u.BlendMode.BLEND_MASKED)
        mat.set_editor_property('two_sided',True)
        mat.set_editor_property('shading_model',u.MaterialShadingModel.MSM_TWO_SIDED_FOLIAGE)
    ml.recompile_material(mat)
    u.EditorAssetLibrary.save_loaded_asset(mat)
    return mat


def mesh(path):
    options=u.FbxImportUI();options.import_mesh=True;options.import_materials=False
    options.import_textures=False;options.import_as_skeletal=False
    options.mesh_type_to_import=u.FBXImportType.FBXIT_STATIC_MESH
    options.automated_import_should_detect_type=False
    options.static_mesh_import_data.combine_meshes=True
    return imported(path,'Meshes',options)


def apply(api,model,box):
    receipt=json.loads((DATA/'manifest.json').read_text())
    if receipt['status']!='PASS':raise RuntimeError('Asset download not complete')
    # The provider provenance receipt is copied into the build output separately.
    surfaces={}
    for asset,tile in [('concrete_pavement',180),('concrete_wall_007',216)]:
        folder=DATA/asset
        maps={r:next(folder.glob('*'+suffix+'*')) for r,suffix in
              [('color','_diff_'),('normal','_nor_dx_'),('rough','_rough_'),('ao','_ao_')]}
        surfaces[asset]=material(asset,maps,tile)
    clean=DATA/'modular_street_seating/modular_street_seating_assembled_4k.fbx'
    furniture=mesh(clean)
    maps_folder=DATA/'modular_street_seating/textures'
    seating={}
    for part in ('timber','supports','armrests','connectors'):
        maps={r:next(maps_folder.glob('*_'+part+'_'+suffix+'*')) for r,suffix in
              [('color','diff_'),('normal','nor_gl_'),('rough','rough_')]}
        metal=list(maps_folder.glob('*_'+part+'_metal_*'))
        if metal:maps['metal']=metal[0]
        seating[part]=material('seating_'+part,maps)
    slots=[]
    for i,slot in enumerate(furniture.static_materials):
        label=str(slot.material_slot_name).lower();slots.append(label)
        part=next((p for p in seating if p in label),None)
        if part is None:raise RuntimeError('Unmapped seating material slot: '+label)
        furniture.set_material(i,seating[part])
    u.EditorAssetLibrary.save_loaded_asset(furniture)
    tree=mesh(DATA/'tree_small_02/tree_small_02_2k.fbx')
    folder=DATA/'tree_small_02/textures'
    tree_mats={}
    for part,prefix in [('bark','tree_small_02'),('branch','tree_small_02_branch'),('leaves','tree_small_02_leaves')]:
        maps={r:next(folder.glob(prefix+'_'+suffix+'*')) for r,suffix in
              [('color','diff_'),('normal','nor_gl_'),('rough','rough_')]}
        if part=='leaves':maps['alpha']=next(folder.glob(prefix+'_alpha_*'))
        tree_mats[part]=material('tree_'+part,maps,foliage=part=='leaves',uv_channel=1 if part=='branch' else 0)
    tree_slots=[]
    for i,slot in enumerate(tree.static_materials):
        label=str(slot.material_slot_name).lower();tree_slots.append(label)
        part='leaves' if 'lea' in label else 'branch' if 'branch' in label else 'bark'
        tree.set_material(i,tree_mats[part])
    settings=tree.get_editor_property('nanite_settings');settings.enabled=True
    tree.set_editor_property('nanite_settings',settings)
    u.EditorAssetLibrary.save_loaded_asset(tree)
    for a in list(api.get_all_level_actors()):
        label=a.get_actor_label()
        if isinstance(a,u.PostProcessVolume):
            pp=a.settings
            for key in ('auto_exposure_min_brightness','auto_exposure_max_brightness'):
                pp.set_editor_property('override_'+key,True)
                pp.set_editor_property(key,10.5)
            a.settings=pp
        if label.startswith(('Sample limestone paving','Sample timber bench','Sample oak',
                             'Sample scanned concrete pavement','Sample modular seating','Sample scanned tree')):
            api.destroy_actor(a)
        elif label.startswith(('Sample tree island rim','Sample curb coping')):
            a.static_mesh_component.set_material(0,surfaces['concrete_wall_007'])
        elif label.startswith('Sample furniture paving'):
            a.static_mesh_component.set_material(0,surfaces['concrete_pavement'])
    paving=box('scanned concrete pavement',(3400,0,10),(2400,480,4),'Pavers')
    paving.static_mesh_component.set_material(0,surfaces['concrete_pavement'])
    for side in (-1,1):
        model(furniture.get_path_name(),'modular seating',(3370,side*450,26),height=85,yaw=0 if side<0 else 180)
        for x in (2900,4100):
            model(tree.get_path_name(),'scanned tree',(x,side*425,29),height=510,yaw=x%270)
    return {'assets_manifest_sha256':hashlib.sha256((DATA/'manifest.json').read_bytes()).hexdigest(),
            'material_builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'clean_seating_sha256':hashlib.sha256(clean.read_bytes()).hexdigest(),
            'seating_slots':slots,'tree_slots':tree_slots,'paving_tile_cm':180,'curb_tile_cm':216,
            'normal_convention':'DX surfaces; GL model normal green channel inverted',
            'tree_uv_channels':{'branch':1,'leaves':0,'bark':0},
            'tree_material_nanite_usage':{key:bool(mat.get_editor_property('used_with_nanite')) for key,mat in tree_mats.items()},
            'map_namespace':BASE,'material_displacement':False,
            'claim':'Surface normal detail affects shading, not evaluator geometry'}
