"""Build new research maps from official PCG graphs and explicit layout rules."""
import json
import os
from pathlib import Path
import time
import traceback
import unreal as u

OUT = Path(os.environ['BA_CITY_OUT'])
MAP = os.environ['BA_CITY_MAP']
assert MAP.startswith('/Game/BAResearchSlice/')
MAP_FILE = (Path(u.Paths.project_dir()) / ('Content/' + MAP[len('/Game/'):] + '.umap')).resolve()
assert not MAP_FILE.exists(), 'Refuse to overwrite a map'
api = u.get_editor_subsystem(u.EditorActorSubsystem)
levels = u.get_editor_subsystem(u.LevelEditorSubsystem)
editor = u.get_editor_subsystem(u.UnrealEditorSubsystem)
started = time.monotonic()
stage, ticks, done = 0, 0, False
components = []
graphs = ['/CitySamplePCG/Examples/Plaza/Plaza_Square_Tree_lines_and_benches_on_each_side']
layout = json.loads(Path(os.environ['BA_CITY_LAYOUT']).read_text(encoding='utf-8-sig')) if os.environ.get('BA_CITY_LAYOUT') else None
graph_rows = layout['graphs'] if layout else [dict(asset=p) for p in graphs]
graphs = [row['asset'] for row in graph_rows]
overrides = []


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def finish(error=None):
    global done
    done = True
    report = dict(status='FAIL' if error else 'PASS', error=error, map_file=str(MAP_FILE),
                  map_asset=MAP, graphs=graphs, seed=42, elapsed_s=time.monotonic()-started,
                  scope='Official PCG plaza rule reuse; engineering scene only')
    report['layout'] = layout
    report['graph_overrides'] = overrides
    if not error:
        meshes=[]
        actor_bounds=[]
        for actor in api.get_all_level_actors():
            cs=actor.get_components_by_class(u.StaticMeshComponent)
            if cs:
                center,extent=actor.get_actor_bounds(False)
                actor_bounds.append(dict(actor=actor.get_actor_label(),center_m=[v/100 for v in (center.x,center.y,center.z)],extent_m=[v/100 for v in (extent.x,extent.y,extent.z)]))
            for c in cs:
                if c.static_mesh:
                    row=dict(actor=actor.get_actor_label(), mesh=c.static_mesh.get_path_name(),
                             instances=c.get_instance_count() if isinstance(c,u.InstancedStaticMeshComponent) else 1)
                    if isinstance(c,u.InstancedStaticMeshComponent):
                        row['instance_transforms']=[str(c.get_instance_transform(i,world_space=True)) for i in range(min(2,c.get_instance_count()))]
                    row['mesh_bounds']=str(c.static_mesh.get_bounding_box())
                    meshes.append(row)
        report['meshes']=meshes
        report['actor_bounds']=actor_bounds
        report['instance_count']=sum(row['instances'] for row in meshes)
        if not report['instance_count']:
            report.update(status='FAIL',error='No generated mesh instances')
        else:
            assert levels.save_current_level(), 'Save failed'
            world=editor.get_editor_world()
            floors=[]
            for x in (-20,-10,0,10,20):
                for y in (-20,-10,0,10,20):
                    hit=u.SystemLibrary.line_trace_single(world,u.Vector(x*100,y*100,150),u.Vector(x*100,y*100,-200),u.TraceTypeQuery.TRACE_TYPE_QUERY1,True,[],u.DrawDebugTrace.NONE)
                    if hit:
                        point=hit.to_tuple()[5]
                        floors.append(dict(x=point.x/100,y=point.y/100,z=point.z/100))
            report['floor_probes']=floors
    write('build-receipt.json',report)
    u.unregister_slate_post_tick_callback(handle)
    u.SystemLibrary.quit_editor()


def tick(dt):
    global stage, ticks
    if done: return
    if (OUT/'stop.request').exists():
        finish('Owner requested stop');return
    try:
        if stage == 0:
            stage = -1  # Level loading can reenter callbacks.
            base=(layout or {}).get('base_map')
            assert levels.new_level_from_template(MAP,base) if base else levels.new_level(MAP), 'Could not create isolated map'
            for recipe in (layout or {}).get('materials',[]):
                dest=MAP+'_Materials/'+recipe['name']
                assert not u.EditorAssetLibrary.does_asset_exist(dest), dest
                mat=u.AssetToolsHelpers.get_asset_tools().create_asset(recipe['name'],MAP+'_Materials',u.Material,u.MaterialFactoryNew())
                mat.set_editor_property('used_with_nanite',True)
                ml=u.MaterialEditingLibrary
                tex=u.load_asset(recipe['color_texture'])
                assert isinstance(tex,u.Texture2D), recipe['color_texture']
                sample=ml.create_material_expression(mat,u.MaterialExpressionTextureSample)
                sample.texture=tex
                sample.sampler_type=u.MaterialSamplerType.SAMPLERTYPE_VIRTUAL_COLOR if tex.virtual_texture_streaming else u.MaterialSamplerType.SAMPLERTYPE_COLOR
                ml.connect_material_property(sample,'RGB',u.MaterialProperty.MP_BASE_COLOR)
                rough=ml.create_material_expression(mat,u.MaterialExpressionConstant)
                rough.set_editor_property('r',.95)
                ml.connect_material_property(rough,'',u.MaterialProperty.MP_ROUGHNESS)
                ml.recompile_material(mat)
                assert u.EditorAssetLibrary.save_loaded_asset(mat), 'Material save failed'
            for change in (layout or {}).get('material_overrides',[]):
                material=u.load_asset(change['material'].replace('{MAP}',MAP))
                assert isinstance(material,u.MaterialInterface), change['material']
                matched=0
                for actor in api.get_all_level_actors():
                    if actor.get_actor_label()!=change['actor_label']: continue
                    for c in actor.get_components_by_class(u.StaticMeshComponent):
                        for slot in range(c.get_num_materials()):c.set_material(slot,material)
                        matched+=1
                assert matched>0, 'No actors matched material override'
                overrides.append(dict(material_override=change,matched_components=matched))
            sun=api.spawn_actor_from_class(u.DirectionalLight,u.Vector(0,0,1000)) if not base else next(a for a in api.get_all_level_actors() if isinstance(a,u.DirectionalLight))
            sun.set_actor_rotation(u.Rotator(pitch=-40,yaw=-30,roll=0),False)
            sun.light_component.set_editor_property('intensity',50000.)
            sky=api.spawn_actor_from_class(u.SkyLight,u.Vector(0,0,500)) if not base else next(a for a in api.get_all_level_actors() if isinstance(a,u.SkyLight))
            sky.light_component.set_editor_property('real_time_capture',True)
            if not base:
                api.spawn_actor_from_class(u.SkyAtmosphere,u.Vector(0,0,0))
                api.spawn_actor_from_class(u.ExponentialHeightFog,u.Vector(0,0,0))
            pp=api.spawn_actor_from_class(u.PostProcessVolume,u.Vector(0,0,0)) if not base else next(a for a in api.get_all_level_actors() if isinstance(a,u.PostProcessVolume))
            pp.set_editor_property('unbound',True)
            for group in (layout or {}).get('mesh_groups',[]):
                mesh=u.load_asset(group['asset'])
                assert isinstance(mesh,u.StaticMesh), group['asset']
                for pos in group['positions_m']:
                    loc=[v*100 for v in pos]
                    scale=group.get('scale',[1,1,1])
                    if group.get('center_xy_top_z'):
                        assert not group.get('yaw',0), 'Bounds alignment requires zero yaw'
                        bounds=mesh.get_bounding_box()
                        loc[0]-=(bounds.min.x+bounds.max.x)*.5*scale[0]
                        loc[1]-=(bounds.min.y+bounds.max.y)*.5*scale[1]
                        loc[2]-=bounds.max.z*scale[2]
                    actor=api.spawn_actor_from_class(u.StaticMeshActor,u.Vector(*loc))
                    actor.set_actor_label(group['label'])
                    actor.static_mesh_component.set_static_mesh(mesh)
                    actor.set_actor_scale3d(u.Vector(*group.get('scale',[1,1,1])))
                    actor.set_actor_rotation(u.Rotator(pitch=0,yaw=group.get('yaw',0),roll=0),False)
                    actor.static_mesh_component.set_collision_profile_name('BlockAll')
            for n,row in enumerate(graph_rows):
                path=row['asset']
                graph=u.load_asset(path)
                assert isinstance(graph,u.PCGGraph), path
                if row.get('parameters'):
                    dest=MAP+'_Rules/Graph_'+str(n)
                    assert not u.EditorAssetLibrary.does_asset_exist(dest), dest
                    graph=u.EditorAssetLibrary.duplicate_asset(path,dest)
                    assert graph, 'Graph duplicate failed'
                    for change in row['parameters']:
                        node=next(v for v in graph.nodes if v.get_name()==change['node'])
                        instance=node.get_settings().subgraph_instance
                        helper=u.PCGGraphParametersHelpers
                        before=helper.get_double_parameter(instance,change['name'])
                        helper.set_double_parameter(instance,change['name'],float(change['value']))
                        after=helper.get_double_parameter(instance,change['name'])
                        assert abs(after-float(change['value']))<.001, 'Parameter override did not apply'
                        overrides.append(dict(graph=dest,node=change['node'],name=change['name'],before=before,after=after))
                    assert u.EditorAssetLibrary.save_loaded_asset(graph), 'Graph save failed'
                actor=api.spawn_actor_from_class(u.PCGVolume,u.Vector(*(v*100 for v in row.get('location_m',[0,0,0]))))
                actor.set_actor_rotation(u.Rotator(pitch=0,yaw=row.get('yaw',0),roll=0),False)
                actor.set_actor_label(row.get('label','BA official plaza rule '+str(n)))
                comp=actor.get_component_by_class(u.PCGComponent)
                comp.set_editor_property('is_component_partitioned',False)
                comp.set_editor_property('seed',row.get('seed',42))
                comp.set_graph(graph)
                components.append(comp)
                comp.generate_local(True)
            editor.set_level_viewport_camera_info(u.Vector(-4000,-4000,3000),u.Rotator(pitch=-30,yaw=45,roll=0))
            stage=1
        if stage != 1: return
        ticks+=1
        generated=[bool(c.generated) for c in components]
        write('progress.json',dict(phase='PCG_GENERATION',generated=generated,ticks=ticks,elapsed_s=time.monotonic()-started))
        if ticks>30 and all(generated):
            finish()
        elif time.monotonic()-started>600:
            finish('PCG generation did not complete within 600 seconds')
    except Exception:
        finish(traceback.format_exc())


handle=u.register_slate_post_tick_callback(tick)
