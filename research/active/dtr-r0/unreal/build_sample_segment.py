"""A separately saved, furnished 24 m sample with native geometry and RGB-D exports."""
import array
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback
import unreal as u

sys.path.insert(0,str(Path(__file__).parent))
from sample_segment_contract import convert_components,evaluate_route
ROOT=Path(u.Paths.project_dir())
OUT=Path(os.environ['BA_SAMPLE_OUTPUT'])
MAP='/Game/StreetLab/WillowSampleV1'
SOURCE=ROOT/'Content/StreetLab/StreetLabV4.umap'
EYE_HEIGHT_M=1.70
FLOOR_M=.12
CAMERA_Z_M=FLOOR_M+EYE_HEIGHT_M
api=u.get_editor_subsystem(u.EditorActorSubsystem)
levels=u.get_editor_subsystem(u.LevelEditorSubsystem)
editor=u.get_editor_subsystem(u.UnrealEditorSubsystem)
assets=u.AssetToolsHelpers.get_asset_tools()
report={'status':'RUNNING','source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'map':MAP,'region_m':[22,46,-6,6],'floor_m':.12,'views':[],
        'builder_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'view_resolution':[3840,2160]}
report.update(capture_mode='sensors_only' if os.environ['BA_SAMPLE_ACTION']=='sensors' else 'showcase_and_sensors',
              sensor_eye_height_m=EYE_HEIGHT_M,sensor_world_z_m=CAMERA_Z_M)
captures=[]
world=None


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def box(name,xyz,size,material):
    a=api.spawn_actor_from_class(u.StaticMeshActor,u.Vector(*xyz))
    a.set_actor_label('Sample '+name);a.set_folder_path('Sample/Street')
    a.static_mesh_component.set_static_mesh(u.load_asset('/Engine/BasicShapes/Cube'))
    a.static_mesh_component.set_material(0,u.load_asset('/Game/StreetLab/Materials/'+material))
    a.static_mesh_component.set_collision_profile_name('BlockAll')
    a.set_actor_scale3d(u.Vector(*(v/100 for v in size)))
    return a


def model(path,name,xyz,height=None,length=None,yaw=0):
    mesh=u.load_asset(path);assert mesh,path
    b=mesh.get_bounds()
    scale=height/(b.box_extent.z*2) if height else length/(b.box_extent.x*2) if length else 1.
    angle=math.radians(yaw)
    x=xyz[0]-scale*(b.origin.x*math.cos(angle)-b.origin.y*math.sin(angle))
    y=xyz[1]-scale*(b.origin.x*math.sin(angle)+b.origin.y*math.cos(angle))
    z=xyz[2]-scale*(b.origin.z-b.box_extent.z)
    a=api.spawn_actor_from_class(u.StaticMeshActor,u.Vector(x,y,z),u.Rotator(yaw=yaw))
    a.set_actor_label('Sample '+name);a.set_folder_path('Sample/Furniture')
    a.static_mesh_component.set_static_mesh(mesh)
    a.static_mesh_component.set_collision_profile_name('BlockAll')
    a.set_actor_scale3d(u.Vector(scale,scale,scale))
    return a


def build():
    global world
    if u.EditorAssetLibrary.does_asset_exist(MAP):
        raise RuntimeError('Sample map exists; inspect it or use a new version')
    assert levels.load_level('/Game/StreetLab/StreetLabV4')
    assert u.EditorLoadingAndSavingUtils.save_map(editor.get_editor_world(),MAP)
    assert levels.load_level(MAP)
    world=editor.get_editor_world()
    remove=('Stone curb','Curb joints','Tactile','Tree planter','Tree soil','Epic oak tree',
            'Scanned planter','Shrub in planter','Bollard','Epic street light','Scanned timber bench',
            'A-board','Board wood','Cafe board','Crosswalk paver','V3 tactile','V2 tactile')
    removed=[]
    for a in list(api.get_all_level_actors()):
        label=a.get_actor_label();o,e=a.get_actor_bounds(False)
        if isinstance(a,(u.SkeletalMeshActor,u.LevelSequenceActor)) or (
            label.startswith(remove) and o.x+e.x>=2100 and o.x-e.x<=4700):
            removed.append(label);api.destroy_actor(a)
    report['removed']=removed
    # A broad flush central route. Fine physical joints read at walking height;
    # no floor feature exceeds the declared 25 mm relief tolerance.
    box('paving joint bed',(3400,0,6.8),(2400,480,9),'Charcoal')
    for ix in range(20):
        for iy in range(8):
            box('limestone paving %02d %02d'%(ix,iy),
                (2260+ix*120,-210+iy*60,10),(119.5,59.5,4),'Pavers')
    for side in (-1,1):
        box('flush drainage edge',(3400,side*244,10),(2400,8,4),'Charcoal')
        box('furniture paving',(3400,side*445,19),(2400,394,14),'Pavers')
        box('curb coping',(3400,side*253,18),(2400,10,16),'Limestone')
        for x in range(2225,4600,50):
            box('drain grate',(x,side*244,12),(36,5,.3),'Bronze')
        for x in (2900,4100):
            # Low, furnished tree islands sit beyond the walking envelope.
            box('tree island soil',(x,side*425,28),(160,150,4),'Soil')
            for dx in (-87,87):box('tree island rim',(x+dx,side*425,33),(14,178,14),'Limestone')
            for dy in (-82,82):box('tree island rim',(x,side*425+dy,33),(160,14,14),'Limestone')
            model('/Game/ArchVis/SampleScene/Tree/HillTree_02','oak',(x,side*425,29),height=730,yaw=x%270)
        model('/Game/StreetLab/Props/painted_wooden_bench_2k','timber bench',
              (3470,side*430,26),height=90,yaw=0 if side<0 else 180)
        model('/Game/StreetLab/Props/planter_box_02_2k','herb planter',
              (3720,side*495,26),length=130,yaw=90)
        model('/Game/ArchVis/SampleScene/Tree/HillTree_02','planter foliage',
              (3720,side*495,55),height=115,yaw=side*39)
        model('/Game/Building/Geometry/SM_StreetLight','street lantern',
              (3800,side*535,26),height=430)
    # Tactile direction remains an actual low-relief material feature.
    box('tactile inset',(3400,-165,12.15),(2360,28,.3),'Tactile')
    for y in (-173,-165,-157):box('tactile guide rib',(3400,y,12.4),(2360,1.8,.4),'Tactile')
    for a in api.get_all_level_actors():
        if a.get_actor_label()=='Pedestrian paving':
            loc=a.get_actor_location();loc.z=-4.7;a.set_actor_location(loc,False,False)
        if isinstance(a,u.DirectionalLight):
            a.set_actor_rotation(u.Rotator(pitch=-42,yaw=-32),False)
            a.light_component.set_editor_property('intensity',7500.)
            a.light_component.set_editor_property('light_color',u.Color(r=255,g=243,b=224,a=255))
            a.light_component.set_editor_property('light_source_angle',1.2)
        if isinstance(a,u.ExponentialHeightFog):
            a.component.set_editor_property('fog_density',.004)
    editor.set_level_viewport_camera_info(u.Vector(2520,-80,172),u.Rotator(pitch=-2,yaw=3))
    assert levels.save_current_level()


def polish():
    """Correct observed first-build scale/coplanarity in this task-owned map."""
    assert levels.load_level(MAP)
    for a in list(api.get_all_level_actors()):
        if a.get_actor_label().startswith('Sample timber bench'):
            api.destroy_actor(a)
        elif a.get_actor_label()=='Pedestrian paving':
            loc=a.get_actor_location();loc.z=-4.7;a.set_actor_location(loc,False,False)
        elif isinstance(a,u.DirectionalLight):
            a.light_component.set_editor_property('intensity',7500.)
    for side in (-1,1):
        model('/Game/StreetLab/Props/painted_wooden_bench_2k','timber bench',
              (3470,side*430,26),height=90,yaw=0 if side<0 else 180)
    assert levels.save_current_level()


def inventory():
    rows=[];components=instances=0
    for actor in api.get_all_level_actors():
        for c in actor.get_components_by_class(u.StaticMeshComponent):
            mesh=c.static_mesh
            if mesh is None:continue
            components+=1
            common={'asset':mesh.get_path_name(),'collision_enabled':str(c.get_collision_enabled())}
            if isinstance(c,u.InstancedStaticMeshComponent):
                bounds=mesh.get_bounds()
                for index in range(c.get_instance_count()):
                    t=c.get_instance_transform(index,world_space=True)
                    corners=[u.MathLibrary.transform_location(t,u.Vector(
                        bounds.origin.x+sx*bounds.box_extent.x,bounds.origin.y+sy*bounds.box_extent.y,
                        bounds.origin.z+sz*bounds.box_extent.z)) for sx,sy,sz in itertools.product((-1,1),repeat=3)]
                    low=[min(p.to_tuple()[i] for p in corners)/100 for i in range(3)]
                    high=[max(p.to_tuple()[i] for p in corners)/100 for i in range(3)]
                    rows.append(dict(common,id=c.get_path_name()+'#'+str(index),
                        center_m=[(a+b)/2 for a,b in zip(low,high)],extent_m=[(b-a)/2 for a,b in zip(low,high)]))
                    instances+=1
            else:
                o,e,radius=u.SystemLibrary.get_component_bounds(c)
                rows.append(dict(common,id=c.get_path_name(),center_m=[v/100 for v in o.to_tuple()],extent_m=[v/100 for v in e.to_tuple()]))
    contract=convert_components(rows)
    contract['export_counts']={'static_mesh_components':components,'instances':instances,'rows':len(rows)}
    contract['coverage']['export_completeness']='ENUMERATED_ALL_LOADED_STATIC_MESH_COMPONENTS_AND_INSTANCES'
    write(OUT/'evaluator/static_geometry.json',contract)
    witnesses={'center':[(26,0),(34,0)],'left_bypass':[(26,0),(27,-1.2),(34,-1.2)],
               'right_bypass':[(26,0),(27,1.2),(34,1.2)],'contact_control':[(27,4.25),(30,4.25)]}
    checks={}
    for name,path in witnesses.items():
        proxy=evaluate_route(path,contract)
        hits=[]
        for start,end in zip(path,path[1:]):
            z=118 if name=='contact_control' else 104
            hit=u.SystemLibrary.capsule_trace_single(world,u.Vector(start[0]*100,start[1]*100,z),
                u.Vector(end[0]*100,end[1]*100,z),28,90,u.TraceTypeQuery.TRACE_TYPE_QUERY1,
                False,captures,u.DrawDebugTrace.NONE)
            fields=hit.to_tuple() if hit is not None else None
            owner=fields[9] if fields else None
            hits.append(owner.get_actor_label() if owner else None)
        checks[name]={'path_world_xy_m':path,'proxy':proxy,'native_capsule_hit_actors':hits}
    report['witnesses']=checks
    report['geometry_counts']=contract['export_counts']
    write(OUT/'evaluator/witnesses.json',checks)


def capture_component(width,height,source,fmt):
    a=api.spawn_actor_from_class(u.SceneCapture2D,u.Vector(0,0,100000))
    c=a.capture_component2d
    c.capture_every_frame=False;c.capture_on_movement=False;c.always_persist_rendering_state=True
    c.capture_source=source;c.texture_target=u.RenderingLibrary.create_render_target2d(world,width,height,fmt)
    if source==u.SceneCaptureSource.SCS_FINAL_COLOR_LDR:
        c.texture_target.target_gamma=2.2
        pp=max((a for a in api.get_all_level_actors() if isinstance(a,u.PostProcessVolume)),key=lambda a:a.priority)
        c.post_process_settings=pp.settings;c.post_process_blend_weight=1.
        report['color_capture_settings']={
            'gi_override':bool(c.post_process_settings.override_dynamic_global_illumination_method),
            'gi_method':str(c.post_process_settings.dynamic_global_illumination_method),
            'reflection_override':bool(c.post_process_settings.override_reflection_method),
            'reflection_method':str(c.post_process_settings.reflection_method)}
    captures.append(a)
    return a


views=[('Arrival',(2500,-80,172),(-2,3,0),68),
       ('Furniture',(3050,-30,145),(-4,-50,0),64),
       ('Reverse',(4350,80,172),(-2,183,0),68),
       ('Overview',(2450,-1450,1850),(-43,52,0),62)]
stage=0;index=0;warm=0;after=time.monotonic()+8


def finish(error=None):
    report['status']='FAIL' if error else 'PASS'
    if error:report['error']=error
    report['source_unchanged']=hashlib.sha256(SOURCE.read_bytes()).hexdigest()==report['source_sha256']
    if not report['source_unchanged']:report['status']='FAIL'
    write(OUT/'receipt.json',report)
    for a in captures:
        u.RenderingLibrary.release_render_target2d(a.capture_component2d.texture_target)
        api.destroy_actor(a)
    u.unregister_slate_post_tick_callback(handle)
    u.SystemLibrary.quit_editor()


def tick(delta):
    global stage,index,warm,after,world,hero,rgb,depth
    try:
        if time.monotonic()<after:return
        # Level loading/saving pumps Slate callbacks; prevent reentrant stages.
        after=float('inf')
        if stage==0:
            if os.environ['BA_SAMPLE_ACTION']=='audit':
                assert levels.load_level(MAP)
                from sample_scene_finish import audit
                write(OUT/'scene_audit.json',audit(api))
                finish();return
            if os.environ['BA_SAMPLE_ACTION']=='build':build()
            elif os.environ['BA_SAMPLE_ACTION']=='polish':polish()
            elif os.environ['BA_SAMPLE_ACTION']=='materials':
                assert levels.load_level(MAP)
                from sample_material_upgrade import apply
                report['material_upgrade']=apply(api,model,box)
                assert levels.save_current_level()
            elif os.environ['BA_SAMPLE_ACTION']=='finish':
                assert levels.load_level(MAP)
                from sample_scene_finish import apply
                report['scene_finish']=apply(api,model,box)
                assert levels.save_current_level()
            else:assert levels.load_level(MAP)
            world=editor.get_editor_world();stage=1;after=time.monotonic()+30;return
        if stage==1:
            assert levels.load_level(MAP);world=editor.get_editor_world()
            report['map_sha256']=hashlib.sha256((ROOT/'Content/StreetLab/WillowSampleV1.umap').read_bytes()).hexdigest()
            inventory()
            if os.environ['BA_SAMPLE_ACTION']=='sensors':
                stage=3;after=time.monotonic()+1;return
            hero=capture_component(3840,2160,u.SceneCaptureSource.SCS_FINAL_COLOR_LDR,u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
            stage=2;after=time.monotonic()+5;return
        if stage==2:
            name,xyz,rot,fov=views[index]
            hero.set_actor_location(u.Vector(*xyz),False,False)
            hero.set_actor_rotation(u.Rotator(pitch=rot[0],yaw=rot[1],roll=rot[2]),False)
            hero.capture_component2d.fov_angle=fov
            if warm<48:
                hero.capture_component2d.capture_scene();warm+=1;after=time.monotonic()+.08;return
            u.RenderingLibrary.export_render_target(world,hero.capture_component2d.texture_target,str(OUT),name+'.png')
            if name=='Arrival' and os.environ['BA_SAMPLE_ACTION']=='finish':
                debug=capture_component(1920,1080,u.SceneCaptureSource.SCS_BASE_COLOR,u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
                debug.set_actor_location(u.Vector(*xyz),False,False)
                debug.set_actor_rotation(u.Rotator(pitch=rot[0],yaw=rot[1],roll=rot[2]),False)
                debug.capture_component2d.fov_angle=fov
                debug.capture_component2d.capture_scene()
                u.RenderingLibrary.export_render_target(world,debug.capture_component2d.texture_target,str(OUT),'BaseColor.png')
            report['views'].append({'name':name,'location_cm':xyz,'rotation':rot,'fov':fov,'path':name+'.png'})
            index+=1;warm=0
            if index==len(views):stage=3;index=0
            after=time.monotonic()+1;return
        if stage==3:
            rgb=capture_component(640,360,u.SceneCaptureSource.SCS_FINAL_COLOR_LDR,u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
            depth=capture_component(640,360,u.SceneCaptureSource.SCS_SCENE_DEPTH,u.TextureRenderTargetFormat.RTF_RGBA32F)
            report['sensor_frames']=[];stage=4;warm=0
        if stage==4:
            x=26+index*.1
            for a in (rgb,depth):
                a.set_actor_location(u.Vector(x*100,0,CAMERA_Z_M*100),False,False)
                a.set_actor_rotation(u.Rotator(pitch=-10),False)
                a.capture_component2d.fov_angle=100
            if warm<(32 if index==0 else 2):
                rgb.capture_component2d.capture_scene();warm+=1;after=time.monotonic()+.1;return
            rgb.capture_component2d.capture_scene();depth.capture_component2d.capture_scene()
            folder=OUT/'model/sample';folder.mkdir(parents=True,exist_ok=True)
            u.RenderingLibrary.export_render_target(world,rgb.capture_component2d.texture_target,str(folder),f'{index:04d}.png')
            values=u.RenderingLibrary.read_render_target_raw(world,depth.capture_component2d.texture_target,normalize=False)
            header=str({'descr':'<f4','fortran_order':False,'shape':(360,640)})
            header+=' '*((64-(10+len(header)+1)%64)%64)+'\n'
            with (folder/f'{index:04d}.npy').open('wb') as f:
                f.write(b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header.encode())
                array.array('f',(v.r/100 if math.isfinite(v.r) and 0<v.r<10000 else 0. for v in values)).tofile(f)
            pose={'x':x,'y':0.,'z':.12,'pitch':0.,'yaw':0.,'roll':0.}
            report['sensor_frames'].append({'sample_index':index,'time_s':round(index*.1,5),
                'rgb_path':f'sample/{index:04d}.png','depth_path':f'sample/{index:04d}.npy',
                'camera_transform':dict(pose,z=CAMERA_Z_M,pitch=-10.),'wearer_transform':pose,
                'command_velocity':{'x':1.,'y':0.,'z':0.}})
            index+=1;warm=0;after=time.monotonic()+.1
            if index==11:
                write(OUT/'model/sensor_manifest.json',{'calibration':{'width':640,'height':360,'horizontal_fov_degrees':100.,'depth_max_m':100.},
                    'frames':report['sensor_frames'],'route_origin_world_m':[26,0,.12],
                    'authority':'SENSOR_ONLY_RGB_FORWARD_DEPTH_AND_EGO_POSES'})
                finish()
    except Exception:finish(traceback.format_exc())


handle=u.register_slate_post_tick_callback(tick)
