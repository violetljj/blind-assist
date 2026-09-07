"""Native audit and bounded visual finishing of the independent sample map."""
import unreal as u
import hashlib
from pathlib import Path


def audit(api):
    result={'actors':[],'materials':{}}
    for a in api.get_all_level_actors():
        row={'label':a.get_actor_label(),'class':a.get_class().get_name(),
             'location':list(a.get_actor_location().to_tuple())}
        meshes=a.get_components_by_class(u.StaticMeshComponent)
        if meshes:
            row['components']=[]
            for c in meshes:
                if not c.static_mesh:continue
                mats=[]
                for i in range(c.get_num_materials()):
                    m=c.get_material(i)
                    if not m:continue
                    path=m.get_path_name();mats.append(path)
                    if path not in result['materials'] and isinstance(m,u.MaterialInstanceConstant):
                        result['materials'][path]={key:str(m.get_editor_property(key)) for key in
                            ('parent','scalar_parameter_values','vector_parameter_values','texture_parameter_values')}
                row['components'].append({'mesh':c.static_mesh.get_path_name(),'materials':mats})
        if isinstance(a,u.SkyLight):row['intensity']=a.light_component.intensity
        if isinstance(a,u.DirectionalLight):row['intensity']=a.light_component.intensity
        if meshes or isinstance(a,(u.SkyLight,u.DirectionalLight)):result['actors'].append(row)
    return result


def apply(api,model,box):
    report={'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'window_parameters':{},'building_bounds':[]}
    actors=list(api.get_all_level_actors())
    for a in actors:
        if isinstance(a,u.PostProcessVolume):
            pp=a.settings
            for name,value in {'dynamic_global_illumination_method':u.DynamicGlobalIlluminationMethod.LUMEN,
                               'reflection_method':u.ReflectionMethod.LUMEN,
                               'lumen_scene_lighting_quality':4.,'lumen_final_gather_quality':4.,
                               'lumen_reflection_quality':4.}.items():
                pp.set_editor_property('override_'+name,True)
                pp.set_editor_property(name,value)
            a.settings=pp
        if a.get_actor_label().startswith('Sample Finish west gallery'):api.destroy_actor(a)
    actors=[a for a in api.get_all_level_actors()]
    templates={a.get_actor_label():a for a in actors}
    report['window_overrides']=[]
    for a in actors:
        if isinstance(a,u.SkyLight):
            a.light_component.set_editor_property('intensity',140.)
            a.light_component.recapture_sky()
        if isinstance(a,u.DirectionalLight):a.light_component.set_editor_property('intensity',4500.)
        if a.get_actor_label().startswith('V4 CitySample'):
            o,e=a.get_actor_bounds(False)
            report['building_bounds'].append({'label':a.get_actor_label(),'center':list(o.to_tuple()),'extent':list(e.to_tuple())})
            for c in a.get_components_by_class(u.StaticMeshComponent):
                for i in range(c.get_num_materials()):
                    m=c.get_material(i)
                    if not isinstance(m,u.MaterialInstanceConstant):continue
                    p=m.get_editor_property('parent')
                    if not p or 'glass' not in p.get_name().lower():continue
                    params=report['window_parameters'].get(m.get_path_name(),{})
                    if not params:
                        for name in u.MaterialEditingLibrary.get_scalar_parameter_names(m):
                            params[str(name)]=u.MaterialEditingLibrary.get_material_instance_scalar_parameter_value(m,name)
                    report['window_parameters'][m.get_path_name()]=params
                    if a.get_actor_label().endswith('ground floor'):
                        key=hashlib.sha256(m.get_path_name().encode()).hexdigest()[:12]
                        path='/Game/SampleFinish/Windows/W_'+key
                        variant=u.load_asset(path) if u.EditorAssetLibrary.does_asset_exist(path) else u.AssetToolsHelpers.get_asset_tools().create_asset('W_'+key,'/Game/SampleFinish/Windows',u.MaterialInstanceConstant,u.MaterialInstanceConstantFactoryNew())
                        u.MaterialEditingLibrary.set_material_instance_parent(variant,m)
                        for name,value in {'Exposure':.25,'LightsOff':1.,'AmountOff':.9}.items():
                            if name in params:u.MaterialEditingLibrary.set_material_instance_scalar_parameter_value(variant,name,value)
                        u.EditorAssetLibrary.save_loaded_asset(variant)
                        c.set_material(i,variant)
                        report['window_overrides'].append(path)
    # Close the unfinished western horizon with a native building assembly,
    # including its separately packaged ground floor. Entire footprint is outside
    # the 22..46 m measurement segment.
    source=templates['V4 CitySample North modern gallery']
    floor_source=templates['V4 CitySample North modern gallery ground floor']
    a=api.spawn_actor_from_class(source.get_class(),u.Vector(),u.Rotator(yaw=90))
    a.set_actor_label('Sample Finish west gallery')
    o,e=a.get_actor_bounds(False)
    loc=u.Vector(-2200-o.x-e.x,-o.y,42)
    a.set_actor_location(loc,False,False)
    floor=api.spawn_actor_from_class(floor_source.get_class(),loc,u.Rotator(yaw=90))
    floor.set_actor_label('Sample Finish west gallery ground floor')
    report['west_gallery_front_x_cm']=-2200
    # Replace the remaining conspicuous red kit benches in the visible backdrop.
    replaced=[]
    for a in actors:
        if a.get_actor_label().startswith('Scanned timber bench'):
            o,e=a.get_actor_bounds(False)
            model('/Game/SampleMaterialsV2/Meshes/modular_street_seating_assembled_4k',
                  'Finish backdrop bench',(o.x,o.y,max(26,o.z-e.z)),height=85,yaw=a.get_actor_rotation().yaw)
            replaced.append(a.get_actor_label());api.destroy_actor(a)
    report['replaced_backdrop_benches']=replaced
    # Packed building construction can restore its template material overrides
    # on reload. Preserve the near facade's native mesh/instance transforms as
    # regular level actors so the sample-specific window treatment persists.
    facade=templates.get('V4 CitySample South heritage corner ground floor')
    report['unpacked_storefront_meshes']=0
    if facade:
        for c in facade.get_components_by_class(u.StaticMeshComponent):
            if not c.static_mesh:continue
            transforms=([c.get_instance_transform(i,world_space=True) for i in range(c.get_instance_count())]
                        if isinstance(c,u.InstancedStaticMeshComponent) else [c.get_world_transform()])
            for transform in transforms:
                part=api.spawn_actor_from_class(u.StaticMeshActor,transform.translation)
                part.set_actor_label('Sample Finish storefront '+str(report['unpacked_storefront_meshes']))
                part.set_actor_transform(transform,False,False)
                target=part.static_mesh_component
                target.set_static_mesh(c.static_mesh)
                target.set_collision_profile_name(c.get_collision_profile_name())
                for slot in range(c.get_num_materials()):target.set_material(slot,c.get_material(slot))
                report['unpacked_storefront_meshes']+=1
        api.destroy_actor(facade)
    # Near-field panes use a coated, opaque glazing approximation: reflected
    # surroundings replace the source kit's repeated bedroom impostors.
    path='/Game/SampleFinish/DayGlazing'
    glazing=u.load_asset(path) if u.EditorAssetLibrary.does_asset_exist(path) else u.AssetToolsHelpers.get_asset_tools().create_asset('DayGlazing','/Game/SampleFinish',u.Material,u.MaterialFactoryNew())
    ml=u.MaterialEditingLibrary
    ml.delete_all_material_expressions(glazing)
    color=ml.create_material_expression(glazing,u.MaterialExpressionConstant3Vector)
    color.set_editor_property('constant',u.LinearColor(.014,.022,.026,1))
    ml.connect_material_property(color,'',u.MaterialProperty.MP_BASE_COLOR)
    for prop,value in ((u.MaterialProperty.MP_ROUGHNESS,.18),(u.MaterialProperty.MP_SPECULAR,.7)):
        node=ml.create_material_expression(glazing,u.MaterialExpressionConstant)
        node.set_editor_property('r',value);ml.connect_material_property(node,'',prop)
    glazing.set_editor_property('used_with_nanite',True)
    ml.recompile_material(glazing);u.EditorAssetLibrary.save_loaded_asset(glazing)
    report['near_glazing_slots']=0
    for a in api.get_all_level_actors():
        if not a.get_actor_label().startswith('Sample Finish storefront'):continue
        c=a.static_mesh_component
        for slot in range(c.get_num_materials()):
            mat=c.get_material(slot)
            if not mat:continue
            parent=mat.get_editor_property('parent') if isinstance(mat,u.MaterialInstanceConstant) else None
            if '/Game/SampleFinish/' in mat.get_path_name() or parent and 'glass' in parent.get_name().lower():
                c.set_material(slot,glazing);report['near_glazing_slots']+=1
    report['sky_intensity']=140.
    report['sun_lux']=4500.
    report['capture_lumen_explicit_override']=True
    return report
