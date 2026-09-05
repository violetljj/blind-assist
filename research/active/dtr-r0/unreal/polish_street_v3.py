"""Build a separate V3 visual demo; preserve V2 and all experiment evidence.

Run with UnrealEditor -ExecCmds="py <this file>" and an RHI. The default
finishes with reloaded, measured PIE verification and quits its owned editor.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import unreal as u

ROOT=Path(u.Paths.project_dir())
OUT=ROOT/'Saved/street-v3'
OUT.mkdir(parents=True,exist_ok=True)
MAP='/Game/StreetLab/StreetLabV3'
BASE='/Game/StreetLabV3'
SEQ=BASE+'/StreetActivity'
api=u.get_editor_subsystem(u.EditorActorSubsystem)
editor=u.get_editor_subsystem(u.UnrealEditorSubsystem)
levels=u.get_editor_subsystem(u.LevelEditorSubsystem)
assets=u.AssetToolsHelpers.get_asset_tools()
ml=u.MaterialEditingLibrary
report={'status':'RUNNING','map':MAP,'source_map':'/Game/StreetLab/StreetLabV2','changes':[]}
routes={
 'Alley crossing pedestrian':(0,[(0,(2150,-940,27)),(150,(2150,-940,27)),(530,(2150,580,27)),(899,(2150,580,27))]),
 'Approaching pedestrian':(90,[(0,(4000,-350,27)),(899,(400,-350,27))]),
 'Parallel pedestrian':(-90,[(0,(-500,380,27)),(899,(3100,380,27))]),
 'Plaza pedestrian':(90,[(0,(5800,420,27)),(899,(3500,420,27))])}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(obj):assert u.EditorAssetLibrary.save_loaded_asset(obj,only_if_is_dirty=False)
def spawn(mesh,name,xyz,yaw=0):
    a=api.spawn_actor_from_class(u.StaticMeshActor,u.Vector(*xyz),u.Rotator(yaw=yaw))
    a.set_actor_label(name);a.set_folder_path('Street/V3')
    a.static_mesh_component.set_static_mesh(mesh)
    return a
def material(name,color,rough=.7):
    path=BASE+'/'+name
    m=u.load_asset(path) if u.EditorAssetLibrary.does_asset_exist(path) else assets.create_asset(name,BASE,u.Material,u.MaterialFactoryNew())
    ml.delete_all_material_expressions(m)
    n=ml.create_material_expression(m,u.MaterialExpressionConstant3Vector);n.constant=u.LinearColor(*color,1)
    ml.connect_material_property(n,'',u.MaterialProperty.MP_BASE_COLOR)
    n=ml.create_material_expression(m,u.MaterialExpressionConstant);n.r=rough
    ml.connect_material_property(n,'',u.MaterialProperty.MP_ROUGHNESS)
    ml.recompile_material(m);save(m);return m
def imported(path,name):
    dest=BASE+'/'+name
    if u.EditorAssetLibrary.does_asset_exist(dest):return u.load_asset(dest)
    task=u.AssetImportTask();task.filename=str(path);task.destination_path=BASE
    task.destination_name=name;task.automated=True;task.save=True
    assets.import_asset_tasks([task])
    return u.load_asset(dest)
def cafe_furniture():
    directory=ROOT.parent/'asset-downloads/outdoor_table_chair_set_01'
    path=BASE+'/ScannedCafeSet'
    if not u.EditorAssetLibrary.does_asset_exist(path):
        task=u.AssetImportTask();task.filename=str(directory/'outdoor_table_chair_set_01_2k.fbx')
        task.destination_path=BASE;task.destination_name='ScannedCafeSet';task.automated=True;task.save=True
        options=u.FbxImportUI();options.automated_import_should_detect_type=False
        options.import_mesh=True;options.import_as_skeletal=False;options.import_materials=False;options.import_textures=False
        options.mesh_type_to_import=u.FBXImportType.FBXIT_STATIC_MESH
        options.static_mesh_import_data.combine_meshes=True
        task.options=options;task.factory=u.FbxFactory();assets.import_asset_tasks([task])
    mesh=u.load_asset(path);assert mesh
    for i,slot in enumerate(mesh.static_materials):
        role='chair' if 'chair' in str(slot.material_slot_name).lower() else 'table'
        mat=material('ScannedCafe_'+role,(.3,.25,.18));ml.delete_all_material_expressions(mat)
        for suffix in ('diff','nor_gl','arm'):
            tex=imported(directory/'textures'/f'outdoor_table_chair_set_01_{role}_{suffix}_2k.jpg',f'Cafe_{role}_{suffix}')
            if suffix!='diff':tex.set_editor_property('srgb',False)
            if suffix=='nor_gl':
                tex.set_editor_property('compression_settings',u.TextureCompressionSettings.TC_NORMALMAP)
                tex.set_editor_property('flip_green_channel',True)
            save(tex)
            node=ml.create_material_expression(mat,u.MaterialExpressionTextureSample);node.texture=tex
            node.set_editor_property('sampler_type',u.MaterialSamplerType.SAMPLERTYPE_COLOR if suffix=='diff' else u.MaterialSamplerType.SAMPLERTYPE_NORMAL if suffix=='nor_gl' else u.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
            if suffix=='diff':ml.connect_material_property(node,'RGB',u.MaterialProperty.MP_BASE_COLOR)
            elif suffix=='nor_gl':ml.connect_material_property(node,'RGB',u.MaterialProperty.MP_NORMAL)
            else:
                ml.connect_material_property(node,'G',u.MaterialProperty.MP_ROUGHNESS)
                ml.connect_material_property(node,'B',u.MaterialProperty.MP_METALLIC)
                ml.connect_material_property(node,'R',u.MaterialProperty.MP_AMBIENT_OCCLUSION)
        ml.recompile_material(mat);save(mat);mesh.set_material(i,mat)
    save(mesh)
    for a in list(api.get_all_level_actors()):
        if a.get_actor_label().startswith(('Cafe table','Cafe chair','Chair leg','Table cup')):api.destroy_actor(a)
    b=mesh.get_bounds();scale=90/(b.box_extent.z*2)
    report['cafe_furniture']={'mesh':mesh.get_path_name(),'source':'https://polyhaven.com/a/outdoor_table_chair_set_01','license':'CC0','source_sha256':sha(directory/'outdoor_table_chair_set_01_2k.fbx'),'native_extent_cm':list(b.box_extent.to_tuple()),'scale':scale,'slots':[str(s.material_slot_name) for s in mesh.static_materials]}
    for x in (-650,200,1050):
        a=spawn(mesh,'V3 scanned cafe furniture',(x-scale*b.origin.x,-720-scale*b.origin.y,26-scale*(b.origin.z-b.box_extent.z)))
        a.set_actor_scale3d(u.Vector(scale,scale,scale))
def tactile():
    # Actual centimeter geometry, unit actor scale. Rounded 3mm relief avoids
    # the 7100cm stretched BasicShapes cube's thin shadow/normal discontinuity.
    verts=[];faces=[]
    def quad(points):
        start=len(verts);verts.extend(points);faces.append(tuple(start+i+1 for i in range(len(points))))
    quad([(-50,-15,0),(50,-15,0),(50,15,0),(-50,15,0)])
    for y in (-9,0,9):
        profile=[(y+1.25*math.cos(k*math.pi/8),.1+.3*math.sin(k*math.pi/8)) for k in range(9)]
        for (y1,z1),(y2,z2) in zip(profile,profile[1:]):quad([(-49,y1,z1),(49,y1,z1),(49,y2,z2),(-49,y2,z2)])
    path=OUT/'TactileTile.obj'
    path.write_text('\n'.join(['o TactileTile',*[f'v {x} {y} {z}' for x,y,z in verts],*['f '+' '.join(map(str,f)) for f in faces]]))
    mesh=imported(path,'TactileTile');assert mesh
    b=mesh.get_bounds().box_extent
    assert abs(b.x-50)<.01 and abs(b.y-15)<.01 and abs(b.z-.2)<.01,b
    mat=material('MatteOchre',(.32,.23,.075),.95)
    for a in list(api.get_all_level_actors()):
        if a.get_actor_label().startswith(('Tactile guidance','Tactile rib')):api.destroy_actor(a)
    for x in range(-1050,6001,100):
        a=spawn(mesh,'V3 tactile tile',(x,-390,26.02));c=a.static_mesh_component
        c.set_material(0,mat);c.set_collision_profile_name('NoCollision');c.set_editor_property('cast_shadow',False)
    report['tactile']={'mesh':mesh.get_path_name(),'native_bounds':list(mesh.get_bounds().box_extent.to_tuple()),'relief_cm':.4,'collision':'NoCollision','casts_shadow':False}

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

def build():
    source=ROOT/'Content/StreetLab/StreetLabV2.umap'
    report['source_map_sha256']=sha(source)
    if not u.EditorAssetLibrary.does_asset_exist(MAP):
        assert levels.load_level('/Game/StreetLab/StreetLabV2')
        assert u.EditorLoadingAndSavingUtils.save_map(editor.get_editor_world(),MAP)
    assert levels.load_level(MAP)
    for a in list(api.get_all_level_actors()):
        if a.get_actor_label().startswith('V3 '):api.destroy_actor(a)
    u.AssetRegistryHelpers.get_asset_registry().scan_paths_synchronous(['/Game/ConceptCar'])
    hdr=imported(ROOT.parent/'asset-downloads/environment/overcast_industrial_courtyard_2k.hdr','OvercastCourtyard')
    assert isinstance(hdr,u.TextureCube),str(hdr)
    report['lighting']={'environment':'https://polyhaven.com/a/overcast_industrial_courtyard','license':'CC0','hdr_sha256':sha(ROOT.parent/'asset-downloads/environment/overcast_industrial_courtyard_2k.hdr'),'skylight_intensity':350.,'sun_lux':2200.,'fixed_ev100':9.5}
    for a in list(api.get_all_level_actors()):
        label=a.get_actor_label();p=a.get_actor_location()
        if isinstance(a,u.DirectionalLight):
            if 'fill' in label:api.destroy_actor(a);continue
            c=a.light_component;c.set_editor_property('intensity',2200.);c.set_editor_property('light_source_angle',1.0)
            c.set_editor_property('light_color',u.Color(r=255,g=249,b=240,a=255))
        if isinstance(a,u.SkyLight):
            c=a.light_component;c.set_editor_property('source_type',u.SkyLightSourceType.SLS_SPECIFIED_CUBEMAP)
            c.set_editor_property('real_time_capture',False);c.set_editor_property('cubemap',hdr)
            c.set_editor_property('intensity',350.);c.set_editor_property('lower_hemisphere_is_black',False);c.recapture_sky()
        # Replace the first three repetitive south blocks with one official
        # freestanding cafe pavilion; all changes stay behind the sidewalk.
        if -1250<p.x<1950 and -1650<p.y<-530 and isinstance(a,(u.StaticMeshActor,u.TextRenderActor)):api.destroy_actor(a)
        elif -1250<p.x<1950 and -1650<p.y<-450 and p.z>170 and isinstance(a,(u.StaticMeshActor,u.TextRenderActor)):api.destroy_actor(a)
        if label.startswith(('Epic parked vehicle','Vehicle glass','Vehicle wheel','Vehicle tire')):api.destroy_actor(a)
    pavilion=spawn(u.load_asset('/Game/Building/Geometry/SM_Building'),'V3 Epic cafe pavilion',(400,-2080,-65),90)
    for i,name in enumerate(['MI_Columns','MI_Roof','MI_Slab','MI_Structure']):pavilion.static_mesh_component.set_material(i,u.load_asset('/Game/Building/Materials/'+name))
    terrace=spawn(u.load_asset('/Engine/BasicShapes/Cube'),'V3 cafe terrace',(400,-4500,10))
    terrace.set_actor_scale3d(u.Vector(100,80,.32));terrace.static_mesh_component.set_material(0,u.load_asset('/Game/StreetLab/Materials/Pavers'))
    # Mature tree crowns close the side-garden horizon revealed by removing
    # the old solid shop blocks. Native bounds keep every trunk on the terrace.
    tree=u.load_asset('/Game/ArchVis/SampleScene/Tree/HillTree_02');b=tree.get_bounds()
    scale=1500/(2*b.box_extent.z)
    for i,x in enumerate((-1400,-650,100,850,1600,2350)):
        a=spawn(tree,'V3 cafe garden oak',(x,-3650,26-scale*(b.origin.z-b.box_extent.z)),i*61)
        a.set_actor_scale3d(u.Vector(scale,scale,scale))
    for y in (-1500,-2450):
        a=spawn(tree,'V3 cafe garden oak',(-1450,y,26-scale*(b.origin.z-b.box_extent.z)),137)
        a.set_actor_scale3d(u.Vector(scale,scale,scale))
    # Continuous low crowns and the extended ground close the exposed western
    # and southern map edge at walking eye height, rather than isolated trees.
    hedge_scale=900/(2*b.box_extent.z)
    border=[(-3100,y) for y in range(-1000,-5700,-150)]+[(x,-5700) for x in range(-3100,2900,150)]
    for i,(x,y) in enumerate(border):
        a=spawn(tree,'V3 cafe garden hedge',(x,y,-170-hedge_scale*(b.origin.z-b.box_extent.z)),i*43)
        a.set_actor_scale3d(u.Vector(hedge_scale,hedge_scale,hedge_scale));a.static_mesh_component.set_collision_profile_name('NoCollision')
    for x,y in ((2320,-1250),(2200,-2100)):
        scale=1250/(2*b.box_extent.z)
        a=spawn(tree,'V3 endwall garden oak',(x,y,26-scale*(b.origin.z-b.box_extent.z)),73)
        a.set_actor_scale3d(u.Vector(scale,scale,scale))
    roof=material('PavilionStone',(.36,.365,.34),.8)
    pavilion.static_mesh_component.set_material(1,roof)
    carmesh=u.load_asset('/Game/ConceptCar/Car/SM_AutomotiveTP_Car');assert carmesh
    b=carmesh.get_bounds();scale=450/(b.box_extent.x*2)
    car=spawn(carmesh,'V3 Epic concept car',(2040-scale*b.origin.x,155-scale*b.origin.y,12-scale*(b.origin.z-b.box_extent.z)))
    car.set_actor_scale3d(u.Vector(scale,scale,scale))
    report['official_assets']=[pavilion.static_mesh_component.static_mesh.get_path_name(),carmesh.get_path_name()]
    tactile()
    cafe_furniture()
    # Reduce physically implausible white albedo, without editing shared V2 materials.
    stone=material('WarmLimestone',(.28,.265,.23),.83)
    for a in api.get_all_level_actors():
        if isinstance(a,u.StaticMeshActor):
            c=a.static_mesh_component
            for i in range(c.get_num_materials()):
                m=c.get_material(i)
                if m and m.get_name() in ('Cream','SoftStone'):c.set_material(i,stone)
        if isinstance(a,u.PostProcessVolume) and a.priority>=20:
            s=a.settings
            for name,value in [('auto_exposure_min_brightness',9.5),('auto_exposure_max_brightness',9.5),('auto_exposure_bias',0.),('lumen_final_gather_quality',4.),('lumen_scene_lighting_quality',4.),('bloom_intensity',.04)]:
                s.set_editor_property('override_'+name,True);s.set_editor_property(name,value)
            a.settings=s
        if isinstance(a,u.CameraActor):a.camera_component.post_process_blend_weight=0.
    bind_people();assert levels.save_current_level()
    assert sha(source)==report['source_map_sha256'],'V2 must remain byte-identical'
    report['map_sha256']=sha(ROOT/'Content/StreetLab/StreetLabV3.umap')
    report['changes']=['native-scale rounded tactile tiles','single HDR sky with one weak directional light','official curved cafe pavilion replaces three repeated south blocks','complete official concept car replaces malformed sports car','lower albedo limestone','scanned cafe table and chair sets with normal and packed ARM maps','mature garden trees close exposed side boundary','fresh map-specific human sequence bindings']
    (OUT/'build.json').write_text(json.dumps(report,indent=2))

def state(a):
    c=a.skeletal_mesh_component
    return {'position':list(a.get_actor_location().to_tuple()),'left_foot':list(c.get_socket_transform('Bip01-L-Foot',u.RelativeTransformSpace.RTS_COMPONENT).translation.to_tuple()),'materials':[c.get_material(i).get_path_name() for i in range(c.get_num_materials())]}
def finish(error=None):
    u.unregister_slate_post_tick_callback(handle)
    levels.editor_request_end_play()
    report['status']='FAIL' if error else 'PASS'
    if error:report['error']=error
    (ROOT/'Saved/lab-visual-v3.json').write_text(json.dumps(report,indent=2))
    if os.environ.get('BA_UE_V3_QUIT','1')=='1':u.SystemLibrary.quit_editor()
stage=0;next_time=time.monotonic()+10
def tick(delta):
    global stage,next_time
    try:
        if time.monotonic()<next_time:return
        next_time=float('inf')
        if stage==0:
            build()
            assert levels.load_level('/Game/StreetLab/StreetLabV2');assert levels.load_level(MAP)
            report['reloaded_people']={a.get_actor_label():state(a) for a in api.get_all_level_actors() if a.get_actor_label() in routes}
            stage=1;next_time=time.monotonic()+4
        elif stage==1:
            stage=2;next_time=time.monotonic()+7;levels.editor_request_begin_play()
        elif stage==2:
            game=editor.get_game_world()
            report['pie_first']={a.get_actor_label():state(a) for a in u.GameplayStatics.get_all_actors_of_class(game,u.SkeletalMeshActor) if a.get_actor_label() in routes}
            stage=3;next_time=time.monotonic()+2.3
        else:
            game=editor.get_game_world()
            report['pie_second']={a.get_actor_label():state(a) for a in u.GameplayStatics.get_all_actors_of_class(game,u.SkeletalMeshActor) if a.get_actor_label() in routes}
            report['movement_cm']={}
            for name in routes:
                a,b=report['pie_first'][name],report['pie_second'][name]
                d=sum((x-y)**2 for x,y in zip(a['position'],b['position']))**.5
                report['movement_cm'][name]=d
                assert d>10,(name,d)
                assert a['left_foot']!=b['left_foot'],name
            finish()
    except Exception:finish(traceback.format_exc())
handle=u.register_slate_post_tick_callback(tick)



