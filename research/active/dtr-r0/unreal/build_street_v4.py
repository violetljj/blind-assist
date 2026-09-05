"""Build a City Sample pedestrian quarter in a separate V4 map.

Uses native, complete Epic hero buildings at their original centimeter scale.
Preserves V3 map bytes; saves a map-specific activity sequence. Reloads, verifies
PIE walking, then captures three real UE views before writing the final receipt.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import shutil
import unreal as u

ROOT=Path(u.Paths.project_dir())
OUT=ROOT/'Saved/street-v4'
OUT.mkdir(parents=True,exist_ok=True)
MAP='/Game/StreetLab/StreetLabV4'
BASE='/Game/StreetLabV4'
SEQ=BASE+'/StreetActivity'
api=u.get_editor_subsystem(u.EditorActorSubsystem)
editor=u.get_editor_subsystem(u.UnrealEditorSubsystem)
levels=u.get_editor_subsystem(u.LevelEditorSubsystem)
assets=u.AssetToolsHelpers.get_asset_tools()
ml=u.MaterialEditingLibrary
report={'status':'RUNNING','map':MAP,'source_map':'/Game/StreetLab/StreetLabV3','changes':[]}
routes={
 'Alley crossing pedestrian':(0,[(0,(2150,-940,27)),(150,(2150,-940,27)),(530,(2150,580,27)),(899,(2150,580,27))]),
 'Approaching pedestrian':(90,[(0,(4000,-350,27)),(899,(400,-350,27))]),
 'Parallel pedestrian':(-90,[(0,(-500,380,27)),(899,(3100,380,27))]),
 'Plaza pedestrian':(90,[(0,(5800,420,27)),(899,(3500,420,27))])}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(obj):assert u.EditorAssetLibrary.save_loaded_asset(obj,only_if_is_dirty=False)
def bind_people():
    if u.EditorAssetLibrary.does_asset_exist(SEQ):seq=u.load_asset(SEQ)
    else:seq=assets.create_asset('StreetActivity',BASE,u.LevelSequence,u.LevelSequenceFactoryNew())
    for b in seq.get_bindings():b.remove()
    seq.set_display_rate(u.FrameRate(30,1));seq.set_playback_start(0);seq.set_playback_end(900)
    people={a.get_actor_label():a for a in api.get_all_level_actors() if isinstance(a,u.SkeletalMeshActor)}
    for name,(yaw,points) in routes.items():
        a=people[name];c=a.skeletal_mesh_component
        anim=u.load_asset(c.skeletal_mesh_asset.get_path_name().split('.')[0].rsplit('/',1)[0]+'/NeutralWalk')
        c.set_position(.15,False);c.override_animation_data(anim,True,False,.15,0.)
        c.set_position(.15,False);c.override_animation_data(anim,True,False,.15,0.);c.set_update_animation_in_editor(True)
        a.set_actor_location(u.Vector(*points[0][1]),False,False);a.set_actor_rotation(u.Rotator(yaw=yaw),False)
        b=seq.add_possessable(a);section=b.add_track(u.MovieScene3DTransformTrack).add_section();section.set_range(0,900)
        channels=section.get_all_channels()
        for c,v in zip(channels,[*points[0][1],0,0,yaw,1,1,1]):c.set_default(v)
        for frame,xyz in points:
            for c,v in zip(channels[:3],xyz):c.add_key(u.FrameNumber(frame),v,interpolation=u.MovieSceneKeyInterpolation.LINEAR)
        section=b.add_track(u.MovieSceneSkeletalAnimationTrack).add_section();section.set_range(0,900)
        params=section.get_editor_property('params');params.animation=anim;section.set_editor_property('params',params)
    for a in api.get_all_level_actors():
        if isinstance(a,u.LevelSequenceActor):
            a.set_sequence(seq);s=a.get_editor_property('playback_settings');s.auto_play=True
            s.loop_count=u.MovieSceneSequenceLoopCount(-1);a.set_editor_property('playback_settings',s)
    save(seq)

def state(a):
    c=a.skeletal_mesh_component
    return {'position':list(a.get_actor_location().to_tuple()),'left_foot':list(c.get_socket_transform('Bip01-L-Foot',u.RelativeTransformSpace.RTS_COMPONENT).translation.to_tuple()),'materials':[c.get_material(i).get_path_name() for i in range(c.get_num_materials())]}
# Complete native hero buildings. Their assembled meshes, trim, doors and roofs
# are kept intact; no non-uniform or miniature scaling is used.
LIB='/Game/CitySampleBuildings/Building/Library/Kit_Hero_Bldg/LevelInstance/'
BUILDINGS=[
 ('North heritage corner','BPP_Bldg_Hero_CHA_A01_N1',0,'north_end',1900),
 ('North modern gallery','BPP_Bldg_Hero_Low_SFD_Long_N1',0,'north_start',2450),
 ('South heritage corner','BPP_Bldg_Hero_CHA_A01_N1',180,'south_start',2450),
 ('Plaza terminus','BPP_Bldg_Hero_CHA_A01_N1',-90,'east',7800),
]
spawned=[]
groundfloors={}
source=ROOT/'Content/StreetLab/StreetLabV3.umap'

def build():
    report['source_map_sha256']=sha(source)
    backup=OUT/('StreetLabV3-source-'+report['source_map_sha256']+'.umap')
    if not backup.exists():shutil.copy2(source,backup)
    report['source_backup']=str(backup)
    (ROOT/'Saved/lab-visual-v4.json').write_text(json.dumps(report,indent=2))
    assert levels.load_level('/Game/StreetLab/StreetLabV3')
    assert u.EditorLoadingAndSavingUtils.save_map(editor.get_editor_world(),MAP)
    # SaveMap writes a copy; it does NOT switch the active editor world.
    assert levels.load_level(MAP)
    assert editor.get_editor_world().get_path_name().split('.')[0]==MAP
    prefixes=('Moulded cornice','Facade pilaster','Shop window','Window warm',
      'Window vertical','Shop fascia','Store sign','Fabric canopy','Canopy valance',
      'Upper window','Upper dark','Window sill','Window mullion','Window transom',
      'Balcony','Downpipe','Roof coping','Distant city block','V2 masonry reveal',
      'V2 parapet','V2 raised dormer','V2 dormer','V2 timber fascia',
      'V2 terminating civic','V2 civic','Street identity','District identity')
    removed=[]
    for a in list(api.get_all_level_actors()):
        label=a.get_actor_label()
        if label.endswith(' building') or label.startswith(prefixes):
            removed.append(label);api.destroy_actor(a)
    report['removed_procedural_actor_count']=len(removed)
    report['buildings']=[]
    for label,name,yaw,edge,coordinate in BUILDINGS:
        cls=u.EditorAssetLibrary.load_blueprint_class(LIB+name)
        assert cls,name
        a=api.spawn_actor_from_class(cls,u.Vector(),u.Rotator(yaw=yaw))
        a.set_actor_label('V4 CitySample '+label);a.set_folder_path('Street/V4/CitySample')
        spawned.append((a,LIB+name,edge,coordinate))
        # Epic packages the first storey separately from the upper assembly.
        floor_path=LIB+name.removesuffix('_N1')+'_Level01_N1'
        floor_cls=u.EditorAssetLibrary.load_blueprint_class(floor_path)
        assert floor_cls,floor_path
        floor=api.spawn_actor_from_class(floor_cls,u.Vector(),u.Rotator(yaw=yaw))
        floor.set_actor_label(a.get_actor_label()+' ground floor')
        floor.set_folder_path('Street/V4/CitySample')
        groundfloors[a.get_actor_label()]=(floor,floor_path)
    ground=api.spawn_actor_from_class(u.StaticMeshActor,u.Vector(5000,0,-35))
    ground.set_actor_label('V4 district ground');ground.set_folder_path('Street/V4')
    ground.static_mesh_component.set_static_mesh(u.load_asset('/Engine/BasicShapes/Cube'))
    ground.set_actor_scale3d(u.Vector(400,400,.5))
    ground.static_mesh_component.set_material(0,u.load_asset('/Game/StreetLab/Materials/Pavers'))
    report['asset_source']='https://www.fab.com/listings/008fe959-5511-428e-93bd-f99b1179f6d5'
    report['asset_publisher']='Epic Games; City Sample Buildings'
    report['virtual_textures']=True
    report['changes']=['complete native City Sample heritage and modern buildings replace procedural facades',
      'original building scale and native material assignments',
      'preserved 5.5m alley and existing walking/sensor corridor',
      'separate V4 map and activity bindings; actual reload, PIE and three render views']

def place_buildings():
    for a,path,edge,coordinate in spawned:
        o,e=a.get_actor_bounds(False)
        assert min(e.x,e.y,e.z)>100,(path,str(e))
        # Bounds measured at world origin AFTER mesh compilation and component
        # registration. PackedLevelActor's origin would distort translated bounds.
        x=coordinate-o.x+(e.x if edge.endswith('start') or edge=='east' else -e.x)
        y=(-o.y if edge=='east' else (650-o.y+e.y) if edge.startswith('north') else (-650-o.y-e.y))
        z=26-o.z+e.z
        a.set_actor_location(u.Vector(x,y,z),False,False)
        floor,floor_path=groundfloors[a.get_actor_label()]
        floor.set_actor_location(u.Vector(x,y,z),False,False)
        report['buildings'].append({'label':a.get_actor_label(),'asset':path,
          'ground_floor_asset':floor_path,'ground_floor_instances':sum(c.get_instance_count() for c in floor.get_components_by_class(u.InstancedStaticMeshComponent)),
          'native_origin_cm':list(o.to_tuple()),'native_extent_cm':list(e.to_tuple()),
          'location_cm':[x,y,z],'scale':[1,1,1],
          'rotation_degrees':list(a.get_actor_rotation().to_tuple()),
          'static_mesh_components':len(a.get_components_by_class(u.StaticMeshComponent)),
          'mesh_instances':sum(c.get_instance_count() for c in a.get_components_by_class(u.InstancedStaticMeshComponent))})
    bind_people()
    editor.set_level_viewport_camera_info(u.Vector(-300,-70,172),u.Rotator(pitch=-2,yaw=4))
    assert levels.save_current_level()
    assert sha(source)==report['source_map_sha256'],'V3 map changed'
    report['map_sha256']=sha(ROOT/'Content/StreetLab/StreetLabV4.umap')
    (OUT/'build.json').write_text(json.dumps(report,indent=2))

stage=0
if os.environ.get('BA_UE_V4_VALIDATE_ONLY')=='1':
    report=json.loads((OUT/'build.json').read_text())
    assert sha(ROOT/'Content/StreetLab/StreetLabV4.umap')==report['map_sha256']
    assert levels.load_level(MAP)
    stage=2
if os.environ.get('BA_UE_V4_RENDER_ONLY')=='1':
    report=json.loads((ROOT/'Saved/lab-visual-v4.json').read_text())
    assert min(report['movement_cm'].values())>10
    assert sha(ROOT/'Content/StreetLab/StreetLabV4.umap')==report['map_sha256']
    report.pop('error',None)
    stage=5
next_time=time.monotonic()+8
render_out=ROOT.parent/'street-v4-visual/release'

def finish(error=None):
    u.unregister_slate_post_tick_callback(handle)
    levels.editor_request_end_play()
    report['status']='FAIL' if error else 'PASS'
    if error:report['error']=error
    report['source_map_unchanged']=sha(source)==report.get('source_map_sha256')
    if not report['source_map_unchanged']:
        report['status']='FAIL';report['source_integrity_error']='V3 map changed'
    (ROOT/'Saved/lab-visual-v4.json').write_text(json.dumps(report,indent=2))
    if os.environ.get('BA_UE_V4_QUIT','1')=='1':u.SystemLibrary.quit_editor()

def tick(delta):
    global stage,next_time
    try:
        if time.monotonic()<next_time:return
        next_time=float('inf')
        if stage==0:
            build();stage=1;next_time=time.monotonic()+45
        elif stage==1:
            place_buildings()
            assert levels.load_level('/Game/StreetLab/StreetLabV3');assert levels.load_level(MAP)
            report['reloaded_people']={a.get_actor_label():state(a) for a in api.get_all_level_actors() if a.get_actor_label() in routes}
            stage=2;next_time=time.monotonic()+4
        elif stage==2:
            stage=3;next_time=time.monotonic()+1;levels.editor_request_begin_play()
        elif stage==3:
            game=editor.get_game_world()
            if not game or u.GameplayStatics.get_time_seconds(game)<7:
                next_time=time.monotonic()+1;return
            report['pie_first_simulation_seconds']=u.GameplayStatics.get_time_seconds(game)
            report['pie_first']={a.get_actor_label():state(a) for a in u.GameplayStatics.get_all_actors_of_class(game,u.SkeletalMeshActor) if a.get_actor_label() in routes}
            stage=4;next_time=time.monotonic()+2.3
        elif stage==4:
            game=editor.get_game_world()
            if u.GameplayStatics.get_time_seconds(game)<report['pie_first_simulation_seconds']+2.3:
                next_time=time.monotonic()+.5;return
            report['pie_second_simulation_seconds']=u.GameplayStatics.get_time_seconds(game)
            report['pie_second']={a.get_actor_label():state(a) for a in u.GameplayStatics.get_all_actors_of_class(game,u.SkeletalMeshActor) if a.get_actor_label() in routes}
            report['movement_cm']={}
            for name in routes:
                a,b=report['pie_first'][name],report['pie_second'][name]
                d=sum((x-y)**2 for x,y in zip(a['position'],b['position']))**.5
                report['movement_cm'][name]=d
                assert d>10,(name,d)
                assert a['left_foot']!=b['left_foot'],name
            levels.editor_request_end_play();stage=5;next_time=time.monotonic()+4
        elif stage==5:
            import runpy
            os.environ['BA_UE_V3_RENDER_MAP']=MAP
            os.environ['BA_UE_V3_RENDER_OUTPUT']=str(render_out)
            os.environ['BA_UE_V3_RENDER_QUIT']='0'
            if (render_out/'receipt.json').exists():
                (render_out/'receipt.json').rename(render_out/('receipt-previous-'+str(time.time_ns())+'.json'))
            runpy.run_path(str(Path(tick.__code__.co_filename).with_name('render_street_v3.py')))
            stage=6;next_time=time.monotonic()+5
        else:
            receipt=render_out/'receipt.json'
            if not receipt.exists():next_time=time.monotonic()+5;return
            report['render']=json.loads(receipt.read_text())
            assert report['render']['status']=='PASS',report['render']
            assert report['render']['map_sha256']==report['map_sha256']
            assert len(report['render']['views'])==3
            report['render_directory']=str(render_out)
            finish()
    except Exception:finish(traceback.format_exc())
handle=u.register_slate_post_tick_callback(tick)
