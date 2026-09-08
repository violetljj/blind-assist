"""Build a new small map from unmodified City Sample plaza PCG graphs."""
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


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def finish(error=None):
    global done
    done = True
    report = dict(status='FAIL' if error else 'PASS', error=error, map_file=str(MAP_FILE),
                  map_asset=MAP, graphs=graphs, seed=42, elapsed_s=time.monotonic()-started,
                  scope='Official PCG plaza rule reuse; engineering scene only')
    if not error:
        meshes=[]
        for actor in api.get_all_level_actors():
            cs=actor.get_components_by_class(u.StaticMeshComponent)
            for c in cs:
                if c.static_mesh:
                    row=dict(actor=actor.get_actor_label(), mesh=c.static_mesh.get_path_name(),
                             instances=c.get_instance_count() if isinstance(c,u.InstancedStaticMeshComponent) else 1)
                    if isinstance(c,u.InstancedStaticMeshComponent):
                        row['instance_transforms']=[str(c.get_instance_transform(i,world_space=True)) for i in range(min(2,c.get_instance_count()))]
                    row['mesh_bounds']=str(c.static_mesh.get_bounding_box())
                    meshes.append(row)
        report['meshes']=meshes
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
            assert levels.new_level(MAP), 'Could not create isolated map'
            sun=api.spawn_actor_from_class(u.DirectionalLight,u.Vector(0,0,1000))
            sun.set_actor_rotation(u.Rotator(pitch=-40,yaw=-30,roll=0),False)
            sun.light_component.set_editor_property('intensity',50000.)
            sky=api.spawn_actor_from_class(u.SkyLight,u.Vector(0,0,500))
            sky.light_component.set_editor_property('real_time_capture',True)
            api.spawn_actor_from_class(u.SkyAtmosphere,u.Vector(0,0,0))
            api.spawn_actor_from_class(u.ExponentialHeightFog,u.Vector(0,0,0))
            pp=api.spawn_actor_from_class(u.PostProcessVolume,u.Vector(0,0,0))
            pp.set_editor_property('unbound',True)
            for n,path in enumerate(graphs):
                graph=u.load_asset(path)
                assert isinstance(graph,u.PCGGraph), path
                actor=api.spawn_actor_from_class(u.PCGVolume,u.Vector(0,0,0))
                actor.set_actor_label('BA official plaza rule '+str(n))
                comp=actor.get_component_by_class(u.PCGComponent)
                comp.set_editor_property('is_component_partitioned',False)
                comp.set_editor_property('seed',42)
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
