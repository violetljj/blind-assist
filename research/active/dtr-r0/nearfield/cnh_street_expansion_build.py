"""Build one new Street-derived concept-A preview map, never modify Street200.

In task-owned UE Editor: set BA_CNH_EXPANSION_SPEC and BA_CNH_EXPANSION_OUTPUT,
then exec this file. Requires existing BlindAssistCapture native capability API.
No external assets, original material changes, original map saves or captures.
The controller owns editor launch/exit. This script does not schedule work.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import math
import os
from pathlib import Path


def validate(spec):
    if spec.get('schema')!='cnh-street-expansion-design-v1' or spec.get('automatic_admission') is not False:
        raise ValueError('Authoring-only expansion contract required')
    if spec.get('map_asset') not in ('/Game/BAResearchExpansion/BrickServiceA_V1','/Game/BAResearchExpansion/BrickServiceA_V2','/Game/BAResearchExpansion/BrickServiceA_V3'):
        raise ValueError('Only bounded first concept-A map is authorized')
    if spec.get('asset_root')!=spec['map_asset']+'_Assets':
        raise ValueError('Dedicated new asset namespace required')
    if (spec.get('alley_clear_width_m'),spec.get('wall_modules_per_side'),spec.get('wall_storeys'))!=(3.,4,2):
        raise ValueError('Frozen first-block geometry differs')
    if (spec.get('plaza_width_m'),spec.get('plaza_length_m'),spec.get('entry_length_m'))!=(12.,12.,6.):
        raise ValueError('Frozen first-block footprint differs')


def rotated_bounds(low,high,yaw):
    angle=math.radians(yaw);c,s=math.cos(angle),math.sin(angle)
    points=[(c*x-s*y,s*x+c*y,z) for x,y,z in itertools.product(*zip(low,high))]
    return ([min(p[k] for p in points) for k in range(3)],
            [max(p[k] for p in points) for k in range(3)])


def design(spec,wall_bounds,paving_bounds):
    validate(spec)
    low,high=wall_bounds
    width=high[1]-low[1];height=high[2]-low[2]
    v2=spec['map_asset'].endswith(('_V2','_V3'))
    if not ((.9<width<8. and 2.<height<6.) if v2 else (5.<width<6. and 3.<height<4.)):
        raise ValueError('Actual native wall bounds differ from reviewed asset')
    p_low,p_high=paving_bounds
    if any(abs((p_high[i]-p_low[i])-1.)>1e-4 for i in (0,1)):
        raise ValueError('Reviewed one-metre paving asset required')
    modules=math.ceil(22./width) if v2 else 4
    length=width*modules;half=1.5;rows=[]
    for side,yaw in [('north',-90),('south',90)]:
        rlow,rhigh=rotated_bounds(low,high,yaw)
        for module in range(modules):
            for storey in range(2):
                target_y=half if side=='north' else -half-(rhigh[1]-rlow[1])
                minimum=[module*width,target_y,storey*height]
                rows.append(dict(name=f'brick_{side}_{module}_{storey}',asset='wall',yaw=yaw,
                    location_m=[minimum[i]-rlow[i] for i in range(3)],scale=[1,1,1]))
    if v2:
        # Native unscaled opaque modules enclose the plaza, leaving its alley
        # entrance open. End modules can extend outside the paved rectangle.
        for side,yaw in [('north',-90),('south',90),('back',180)]:
            rlow,rhigh=rotated_bounds(low,high,yaw)
            for index in range(math.ceil(12./width)):
                if side=='back':minimum=[length+12.,-6.+index*width,0.]
                else:minimum=[length+index*width,6. if side=='north' else -6.-(rhigh[1]-rlow[1]),0.]
                rows.append(dict(name=f'plaza_boundary_{side}_{index}',asset='wall',yaw=yaw,
                    location_m=[minimum[i]-rlow[i] for i in range(3)],scale=[1,1,1]))
        rows.append(dict(name='surrounding_ground',asset='paving',yaw=0.,
            location_m=[-1000.-p_low[0]*2000.,-1000.-p_low[1]*2000.,-.03-p_high[2]],scale=[2000.,2000.,1.]))
    # Tile on three non-overlapping rectangular regions. Last alley row may
    # overhang the plaza start by <1m; avoid overlap by shortening only its X scale.
    regions=[('entry',-6.,0.,-6.,6.),('alley',0.,length,-half,half),('plaza',length,length+12.,-6.,6.)]
    for region,x0,x1,y0,y1 in regions:
        for ix in range(math.ceil(x1-x0)):
            for iy in range(int(y1-y0)):
                sx=min(1.,x1-x0-ix)
                rows.append(dict(name=f'paving_{region}_{ix}_{iy}',asset='paving',yaw=0.,
                    location_m=[x0+ix-p_low[0]*sx,y0+iy-p_low[1],-p_high[2]],scale=[sx,1,1]))
    return dict(actors=rows,dimensions_authority='AUTHORED_CONFIGURATION_AND_NATIVE_MESH_BOUNDS_NOT_IMAGE_MEASUREMENT',
        alley_clear_width_m=3.,alley_length_m=length,plaza_size_m=[12.,12.],
        geometry_acceptance='NOT_RUN_RENDER_EXPORT_AND_WALKTHROUGH_REQUIRED',
        views=[dict(name='entry_alley',position_m=[-4.,0.,1.7],yaw=0.),
               dict(name='alley_plaza',position_m=[length-4.,0.,1.7],yaw=0.),
               dict(name='plaza_back',position_m=[length+10.,4.,2.],yaw=-155.)])


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(u,spec,out):
    validate(spec);out=Path(out)
    if out.exists():
        raise FileExistsError('Do not overwrite expansion evidence: '+str(out))
    project=Path(u.Paths.project_dir())
    source_map=project/'Content'/Path(spec['preserved_source_map'].removeprefix('/Game/')+'.umap')
    source_hash=_sha(source_map)
    if u.EditorAssetLibrary.does_asset_exist(spec['map_asset']) or u.EditorAssetLibrary.does_directory_exist(spec['asset_root']):
        raise FileExistsError('Expansion destination already exists; inspect instead of overwrite')
    meshes={};source_files={};bounds={};brick_override=None
    if spec['map_asset'].endswith('_V3'):
        material_path='/Game/Building/CH/I/Material/MI_Bldg_BrickOffset_Brown'
        brick_override=u.load_asset(material_path)
        if brick_override is None:raise ValueError('Existing brown brick material unavailable')
        material_file=project/'Content'/Path(material_path.removeprefix('/Game/')+'.uasset')
        source_files[str(material_file)]=_sha(material_file)
        capability=json.loads(u.BlindAssistCaptureLibrary.get_material_geometry_capability(brick_override))
        if capability.get('capability')!='NO_COMPILED_MATERIAL_DEFORMATION' or capability.get('blend_mode')!=0:
            raise ValueError('Brown brick override must be compiled static opaque: '+str(capability))
    for key in ('wall','paving'):
        path=spec[key+'_asset'];mesh=u.load_asset(path)
        if not isinstance(mesh,u.StaticMesh):raise ValueError('Actual reviewed mesh unavailable: '+path)
        source_file=project/'Content'/Path(path.removeprefix('/Game/')+'.uasset')
        source_files[str(source_file)]=_sha(source_file)
        for slot in mesh.get_editor_property('static_materials'):
            material=slot.get_editor_property('material_interface')
            capability=json.loads(u.BlindAssistCaptureLibrary.get_material_geometry_capability(material))
            if capability.get('capability')!='NO_COMPILED_MATERIAL_DEFORMATION':
                raise ValueError('Preserved source material not verified static: '+str(capability))
            if key=='wall' and spec['map_asset'].endswith(('_V2','_V3')) and capability.get('blend_mode')!=0:
                raise ValueError('V2 brick wall requires actually opaque material: '+str(capability))
        box=mesh.get_bounding_box()
        bounds[key]=([float(getattr(box.min,k))/100 for k in ('x','y','z')],
                     [float(getattr(box.max,k))/100 for k in ('x','y','z')])
        meshes[key]=mesh
    plan=design(spec,bounds['wall'],bounds['paving'])
    out.mkdir(parents=True)
    receipt=dict(status='BUILDING',spec=spec,source_map_sha256=source_hash,
        source_mesh_sha256=source_files,plan=plan,source_materials='ORIGINAL_REFERENCES_UNMODIFIED',
        new_site='NEW_AUTHORED_BLOCK_NOT_INDEPENDENT_ASSET_FAMILY_OR_TEST_SET',actors=[])
    def record():
        (out/'build-receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    record()
    try:
        levels=u.get_editor_subsystem(u.LevelEditorSubsystem)
        editor=u.get_editor_subsystem(u.UnrealEditorSubsystem)
        api=u.get_editor_subsystem(u.EditorActorSubsystem)
        tools=u.AssetToolsHelpers.get_asset_tools();mesh_editor=u.get_editor_subsystem(u.StaticMeshEditorSubsystem)
        # Never load the original map. Create an independent empty level.
        if not levels.new_level(spec['map_asset']):raise RuntimeError('New level creation failed')
        world=editor.get_editor_world()
        if world.get_path_name().split('.')[0]!=spec['map_asset']:raise RuntimeError('Active world differs from dedicated destination')
        derived={}
        neutral_ground=None
        if brick_override is not None:
            neutral_ground=tools.create_asset('M_NeutralGround',spec['asset_root'],u.Material,u.MaterialFactoryNew())
            if neutral_ground is None:raise RuntimeError('New neutral ground material creation failed')
            ml=u.MaterialEditingLibrary
            color=ml.create_material_expression(neutral_ground,u.MaterialExpressionConstant3Vector)
            color.set_editor_property('constant',u.LinearColor(.16,.17,.18,1.))
            rough=ml.create_material_expression(neutral_ground,u.MaterialExpressionConstant)
            rough.set_editor_property('r',.95)
            ml.connect_material_property(color,'',u.MaterialProperty.MP_BASE_COLOR)
            ml.connect_material_property(rough,'',u.MaterialProperty.MP_ROUGHNESS)
            ml.recompile_material(neutral_ground)
            if not u.EditorAssetLibrary.save_loaded_asset(neutral_ground,only_if_is_dirty=False):
                raise RuntimeError('New neutral material save failed')
            receipt['component_material_overrides']=dict(wall=brick_override.get_path_name(),
                surrounding_ground=neutral_ground.get_path_name(),ground_textures='NONE_CONSTANT_BASE_COLOR_AND_ROUGHNESS',
                source_material_assets_edited=False)
        for key,source in meshes.items():
            clone=tools.duplicate_asset('SM_'+key,spec['asset_root'],source)
            if clone is None:raise RuntimeError('New mesh duplication failed')
            settings=mesh_editor.get_nanite_settings(clone);settings.set_editor_property('enabled',False)
            mesh_editor.set_nanite_settings(clone,settings,True)
            if mesh_editor.get_nanite_settings(clone).get_editor_property('enabled'):raise RuntimeError('Derived Nanite remained enabled')
            source_materials=[s.get_editor_property('material_interface') for s in source.get_editor_property('static_materials')]
            if [s.get_editor_property('material_interface') for s in clone.get_editor_property('static_materials')]!=source_materials:
                raise RuntimeError('Material references changed')
            if not u.EditorAssetLibrary.save_loaded_asset(clone,only_if_is_dirty=False):raise RuntimeError('New clone save failed')
            derived[key]=clone
        for row in plan['actors']:
            actor=api.spawn_actor_from_class(u.StaticMeshActor,u.Vector(*(v*100 for v in row['location_m'])),u.Rotator(yaw=row['yaw']))
            actor.set_actor_label('CNH_A_'+row['name']);actor.set_folder_path('CNH_Expansion_A')
            component=actor.static_mesh_component;component.set_static_mesh(derived[row['asset']])
            if brick_override is not None and row['asset']=='wall':
                for index in range(component.get_num_materials()):component.set_material(index,brick_override)
            if neutral_ground is not None and row['name']=='surrounding_ground':
                for index in range(component.get_num_materials()):component.set_material(index,neutral_ground)
            component.set_forced_lod_model(1);component.set_collision_profile_name('BlockAll')
            actor.set_actor_scale3d(u.Vector(*row['scale']))
            actor.tags=[u.Name('CNH_AUTHORED_SITE_A')]
            receipt['actors'].append(dict(name=row['name'],path=actor.get_path_name()))
        sun=api.spawn_actor_from_class(u.DirectionalLight,u.Vector(0,0,1000),u.Rotator(pitch=-35,yaw=-25))
        sun.light_component.set_editor_property('intensity',5.)
        sun.light_component.set_editor_property('atmosphere_sun_light',True)
        api.spawn_actor_from_class(u.SkyAtmosphere,u.Vector(0,0,0))
        sky=api.spawn_actor_from_class(u.SkyLight,u.Vector(0,0,500))
        sky.light_component.set_editor_property('real_time_capture',True)
        if not u.EditorLoadingAndSavingUtils.save_map(world,spec['map_asset']):raise RuntimeError('New map save failed')
        receipt['status']='SAVED_AUTHORING_PREVIEW_REQUIRES_VISUAL_AND_GEOMETRY_VALIDATION'
    except Exception:
        import traceback
        receipt.update(status='BUILD_FAILED_RETAIN_PARTIAL_NEW_ASSETS_NO_AUTO_DELETE',error=traceback.format_exc())
        raise
    finally:
        receipt['source_map_unchanged']=_sha(source_map)==source_hash
        receipt['source_mesh_files_unchanged']=all(_sha(Path(p))==digest for p,digest in source_files.items())
        record()
        if not receipt['source_map_unchanged'] or not receipt['source_mesh_files_unchanged']:
            raise RuntimeError('Original source file integrity changed')
    return receipt


if __name__=='__main__':
    import unreal
    import sys
    sys.path.insert(0,str(Path(__file__).parent))
    from cnh_street_expansion_preview import run
    run(unreal)
