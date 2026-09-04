"""Build Willow Walk, a furnished pedestrian street using Epic assets and CC0 scans."""
import json
import math
from pathlib import Path
import unreal as u

ROOT = Path(u.Paths.project_dir())
DOWNLOADS = ROOT.parent / 'asset-downloads'
BASE = '/Game/StreetLab'
MAP = BASE + '/StreetLab'
levels = u.get_editor_subsystem(u.LevelEditorSubsystem)
editor = u.get_editor_subsystem(u.UnrealEditorSubsystem)
actors = u.get_editor_subsystem(u.EditorActorSubsystem)
assets = u.AssetToolsHelpers.get_asset_tools()
matlib = u.MaterialEditingLibrary
if u.EditorAssetLibrary.does_asset_exist(MAP):
    raise RuntimeError('StreetLab already exists. Preserve edits; build into a new project to regenerate.')
levels.new_level(MAP)
world = editor.get_editor_world()
materials = {}


def import_file(path, dest, options=None):
    task = u.AssetImportTask()
    task.filename = str(path)
    task.destination_path = dest
    task.automated = True
    task.save = True
    task.replace_existing = False
    if options:
        task.options = options
    assets.import_asset_tasks([task])
    objects = task.get_objects()
    if not objects:
        raise RuntimeError(f'Import failed: {path}')
    return objects[0]


def node(mat, cls, **props):
    obj = matlib.create_material_expression(mat, cls)
    for key, value in props.items():
        obj.set_editor_property(key, value)
    return obj


def solid(name, color, roughness=.65, metallic=0, glow=0):
    mat = assets.create_asset(name, BASE+'/Materials', u.Material, u.MaterialFactoryNew())
    col = node(mat,u.MaterialExpressionConstant3Vector,constant=u.LinearColor(*color,1))
    matlib.connect_material_property(col,'',u.MaterialProperty.MP_BASE_COLOR)
    r = node(mat,u.MaterialExpressionConstant,r=roughness)
    m = node(mat,u.MaterialExpressionConstant,r=metallic)
    matlib.connect_material_property(r,'',u.MaterialProperty.MP_ROUGHNESS)
    matlib.connect_material_property(m,'',u.MaterialProperty.MP_METALLIC)
    if glow:
        emission = node(mat,u.MaterialExpressionConstant3Vector,
                        constant=u.LinearColor(*(c*glow for c in color),1))
        matlib.connect_material_property(emission,'',u.MaterialProperty.MP_EMISSIVE_COLOR)
    matlib.recompile_material(mat)
    materials[name]=mat
    return mat


for name, rgb, rough, metal in [
    ('Limestone',(.42,.37,.28),.82,0), ('Cream',(.63,.59,.47),.8,0),
    ('Charcoal',(.025,.032,.03),.45,.5), ('Bronze',(.25,.14,.065),.32,.8),
    ('Sage',(.08,.17,.12),.8,0), ('Terracotta',(.35,.12,.065),.83,0),
    ('Window',(.025,.045,.05),.16,0), ('WarmInterior',(.19,.13,.065),.5,0),
    ('Soil',(.035,.021,.01),1,0), ('Tactile',(.46,.31,.075),.82,0),
    ('Paint',(.72,.7,.59),.9,0), ('Asphalt',(.055,.065,.065),.93,0),
    ('Leaf',(.075,.14,.027),.85,0), ('Rubber',(.009,.012,.011),.95,0)]:
    solid(name,rgb,rough,metal)
solid('WarmLight',(1,.63,.27),.5,0,.8)


def texture(path, role, flip=False):
    tex=import_file(path,BASE+'/Textures')
    if role!='color':
        tex.set_editor_property('srgb',False)
    if role=='normal':
        tex.set_editor_property('compression_settings',u.TextureCompressionSettings.TC_NORMALMAP)
        tex.set_editor_property('flip_green_channel',flip)
    u.EditorAssetLibrary.save_loaded_asset(tex)
    return tex


def pbr(name, maps, tile_cm=200, world_aligned=True):
    mat=assets.create_asset(name,BASE+'/Materials',u.Material,u.MaterialFactoryNew())
    for role, path in maps.items():
        tex=texture(path,role,'nor_gl' in str(path))
        if world_aligned:
            obj=node(mat,u.MaterialExpressionTextureObject,texture=tex)
            size=node(mat,u.MaterialExpressionConstant3Vector,constant=u.LinearColor(tile_cm,tile_cm,tile_cm,1))
            func=node(mat,u.MaterialExpressionMaterialFunctionCall)
            func.set_material_function(u.load_asset('/Engine/Functions/Engine_MaterialFunctions01/Texturing/'+
                                                   ('WorldAlignedNormal' if role=='normal' else 'WorldAlignedTexture')))
            matlib.connect_material_expressions(obj,'',func,'TextureObject')
            matlib.connect_material_expressions(size,'',func,'TextureSize')
            source, output=func,'XYZ Texture'
        else:
            sample=node(mat,u.MaterialExpressionTextureSample,texture=tex)
            sample.set_editor_property('sampler_type',u.MaterialSamplerType.SAMPLERTYPE_NORMAL if role=='normal'
                                       else u.MaterialSamplerType.SAMPLERTYPE_COLOR if role=='color'
                                       else u.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
            source,output=sample,'RGB'
        prop={'color':u.MaterialProperty.MP_BASE_COLOR,'normal':u.MaterialProperty.MP_NORMAL,
              'roughness':u.MaterialProperty.MP_ROUGHNESS,'arm':u.MaterialProperty.MP_ROUGHNESS}[role]
        if role=='arm':
            output='G'
        matlib.connect_material_property(source,output,prop)
    if world_aligned:
        mat.set_editor_property('tangent_space_normal',False)
    matlib.recompile_material(mat)
    materials[name]=mat
    return mat


sources=json.loads((DOWNLOADS/'materials/sources.json').read_text())
for item, name, tile in zip(sources,['Pavers','Brick','Plaster','Wood'],[240,240,300,150]):
    pbr(name,{r:v['path'] for r,v in item['maps'].items()},tile)


def prop_mesh(asset):
    folder=DOWNLOADS/asset
    options=u.FbxImportUI()
    options.import_mesh=True
    options.import_materials=False
    options.import_textures=False
    options.import_as_skeletal=False
    options.mesh_type_to_import=u.FBXImportType.FBXIT_STATIC_MESH
    options.automated_import_should_detect_type=False
    options.static_mesh_import_data.set_editor_property('combine_meshes',True)
    mesh=import_file(folder/f'{asset}_2k.fbx',BASE+'/Props',options)
    mat=pbr(asset,{r:folder/'textures'/f'{asset}_{suffix}_2k.jpg'
                   for r,suffix in [('color','diff'),('normal','nor_gl'),('arm','arm')]},world_aligned=False)
    for i in range(len(mesh.static_materials)):
        mesh.set_material(i,mat)
    u.EditorAssetLibrary.save_loaded_asset(mesh)
    return mesh


bench_mesh=prop_mesh('painted_wooden_bench')
planter_mesh=prop_mesh('planter_box_02')
tree_mesh=u.load_asset('/Game/ArchVis/SampleScene/Tree/HillTree_02')
lamp_mesh=u.load_asset('/Game/Building/Geometry/SM_StreetLight')


def spawn(cls,name,xyz,rotation=(0,0,0),folder='Street/Architecture'):
    a=actors.spawn_actor_from_class(cls,u.Vector(*xyz),
        u.Rotator(pitch=rotation[0],yaw=rotation[1],roll=rotation[2]))
    a.set_actor_label(name)
    a.set_folder_path(folder)
    return a


def box(name,xyz,size,mat='Cream',rotation=(0,0,0),mesh='Cube',collision=True):
    a=spawn(u.StaticMeshActor,name,xyz,rotation)
    c=a.static_mesh_component
    c.set_static_mesh(u.load_asset('/Engine/BasicShapes/'+mesh))
    c.set_material(0,materials[mat])
    c.set_collision_profile_name('BlockAll' if collision else 'NoCollision')
    a.set_actor_scale3d(u.Vector(*(s/100 for s in size)))
    return a


def model(mesh,name,xyz,height=None,yaw=0,length=None):
    b=mesh.get_bounds()
    scale=(height/(b.box_extent.z*2)) if height else (length/(b.box_extent.x*2)) if length else 1
    x,y,z=xyz
    angle=math.radians(yaw)
    x-=scale*(b.origin.x*math.cos(angle)-b.origin.y*math.sin(angle))
    y-=scale*(b.origin.x*math.sin(angle)+b.origin.y*math.cos(angle))
    z-=scale*(b.origin.z-b.box_extent.z)
    a=spawn(u.StaticMeshActor,name,(x,y,z),(0,yaw,0),'Street/Furniture and vegetation')
    a.static_mesh_component.set_static_mesh(mesh)
    a.static_mesh_component.set_collision_profile_name('BlockAll')
    a.set_actor_scale3d(u.Vector(scale,scale,scale))
    return a


def text(name,word,xyz,size=26,yaw=-90,color=(235,219,178)):
    a=spawn(u.TextRenderActor,name,xyz,(0,yaw,0),'Street/Signage')
    a.text_render.set_text(word)
    a.text_render.set_world_size(size)
    a.text_render.set_horizontal_alignment(u.HorizTextAligment.EHTA_CENTER)
    a.text_render.set_text_render_color(u.Color(r=color[0],g=color[1],b=color[2],a=255))
    return a


# A continuous urban block; experiments live in normal street geometry.
box('District ground',(2400,0,-35),(16000,10000,60),'Asphalt')
box('Pedestrian paving',(2500,0,-4),(8000,1260,32),'Pavers')
box('South sidewalk',(2500,-455,10),(8000,320,32),'Pavers')
box('North sidewalk',(2500,455,10),(8000,320,32),'Pavers')
for side in [-1,1]:
    box('Stone curb',(2500,side*282,18),(8000,18,32),'Limestone')
    for x in range(-1100,6500,160):
        box('Curb joints',(x,side*282,34),(1.2,19,1),'Charcoal',collision=False)
# A restrained physical tactile strip, not a debug trajectory overlay.
box('Tactile guidance strip',(2450,-390,28),(7100,29,3),'Tactile')
for y in [-399,-390,-381]:
    box('Tactile rib',(2450,y,30),(7100,2.5,2),'Tactile')
for x in [1800,3350]:
    for dx in [-45,-15,15,45]:
        for dy in [-25,0,25]:
            box('Tactile warning stud',(x+dx,-390+dy,30),(7,7,3),'Tactile',mesh='Cylinder')

shops=['WILLOW COFFEE','FIELD NOTES','ATELIER 08','BOTANICA','PANTRY & CO.','NORTH GALLERY']
blocks=[(-650,950,1150),(380,980,1370),(1400,900,1080),(2950,1050,1450),(4100,1060,1170),(5250,1050,1330)]
for side in [-1,1]:
    face=side*640
    for i,(x,width,height) in enumerate(blocks):
        name=shops[(i+(1 if side==1 else 0))%len(shops)]
        wall='Brick' if (i+(side==1))%3==0 else 'Plaster'
        box(name+' building',(x,side*1110,height/2),(width,940,height),wall)
        for z,depth,h in [(48,40,70),(365,45,22),(height-30,65,45),(height,80,25)]:
            box('Moulded cornice',(x,face-side*depth/2,z),(width+25,depth,h),'Cream')
        for edge in [-1,1]:
            box('Facade pilaster',(x+edge*(width/2-24),face-side*14,height/2),(32,28,height),'Limestone')
        # Shopfront recesses, mullions, warm inner panels and projecting awnings.
        for wx in [-width*.29,0,width*.29]:
            box('Shop window frame',(x+wx,face-side*23,185),(width*.25,40,284),'Charcoal')
            box('Shop window glass',(x+wx,face-side*46,183),(width*.25-14,5,260),'Window')
            for wz in [70,260]:
                box('Window warm reflection',(x+wx,face-side*50,wz),(width*.21,1,4),'WarmLight',collision=False)
            for vx in [-width*.07,width*.07]:
                box('Window vertical mullion',(x+wx+vx,face-side*52,184),(5,6,268),'Bronze')
        box('Shop fascia',(x,face-side*36,333),(width-55,46,62),'Sage' if i%2==0 else 'Charcoal')
        text('Store sign',name,(x,face-side*63,320),30,90 if side==-1 else -90)
        box('Fabric canopy',(x,face-side*107,289),(width-85,160,12),'Sage' if i%2==0 else 'Terracotta',rotation=(0,0,side*7))
        box('Canopy valance',(x,face-side*186,280),(width-80,9,28),'Sage' if i%2==0 else 'Terracotta')
        for wx in [-width*.28,0,width*.28]:
            for z in range(525,int(height)-140,290):
                box('Upper window stone surround',(x+wx,face-side*16,z),(170,35,215),'Cream')
                box('Upper dark glazing',(x+wx,face-side*38,z),(143,8,187),'Window')
                box('Window sill',(x+wx,face-side*54,z-109),(198,60,17),'Limestone')
                box('Window mullion',(x+wx,face-side*48,z),(6,8,188),'Cream')
                box('Window transom',(x+wx,face-side*48,z+26),(146,8,7),'Cream')
                if i%2==0 and z==525:
                    box('Balcony platform',(x+wx,face-side*78,z-110),(206,128,19),'Limestone')
                    box('Balcony handrail',(x+wx,face-side*137,z-18),(201,5,5),'Charcoal')
                    for rail in range(-90,91,30):
                        box('Balcony rail',(x+wx+rail,face-side*137,z-65),(3,3,93),'Charcoal')
        for pipe_x in [-width/2+70,width/2-70]:
            box('Downpipe',(x+pipe_x,face-side*36,height/2),(9,9,height-30),'Charcoal',mesh='Cylinder')
        box('Roof coping',(x,side*1110,height+8),(width+50,990,18),'Charcoal')

# Alley crossing, a small plaza, and distant building silhouettes close the view.
for side in [-1,1]:
    box('Alley paving',(2170,side*1400,2),(500,2000,25),'Pavers')
    box('Distant city block',(6900,side*1100,700),(1100,1400,1400),'Plaster')
    for x in [-1900,6800]:
        model(tree_mesh,'Background oak',(x,side*1450,0),height=1250,yaw=x%360)
box('End plaza',(6450,0,3),(850,2000,25),'Pavers')
text('Street identity','WILLOW WALK',(6350,-20,335),66,180)
text('District identity','PEDESTRIAN QUARTER',(6348,-20,285),22,180)

# Official Epic oak trees and lights, plus CC0 photo-scanned bench/planter props.
for side in [-1,1]:
    for i,x in enumerate([-700,900,3200,4900,6100]):
        y=side*210
        box('Tree planter stone',(x,y,28),(185,185,60),'Limestone')
        box('Tree soil',(x,y,61),(158,158,8),'Soil')
        model(tree_mesh,'Epic oak tree',(x,y,57),height=820+(i%3)*70,yaw=i*83+side*20)
        lamp=model(lamp_mesh,'Epic street light',(x+280,side*238,34),height=440,yaw=0)
        lamp.static_mesh_component.set_material(0,materials['Charcoal'])
        lamp.static_mesh_component.set_material(1,materials['WarmLight'])
        model(bench_mesh,'Scanned timber bench',(x+80,side*460,26),height=90,yaw=0 if side==-1 else 180)
        model(planter_mesh,'Scanned planter',(x-220,side*490,26),length=145,yaw=90)
        # Small real leaf clusters use the official oak crown rather than colored spheres.
        shrub=model(tree_mesh,'Shrub in planter',(x-220,side*490,40),height=140,yaw=i*43)
        shrub.static_mesh_component.set_collision_profile_name('NoCollision')
        for dx in [-50,50]:
            box('Bollard',(x+dx,side*313,67),(14,14,82),'Charcoal',mesh='Cylinder')
            box('Bollard reflector',(x+dx,side*313,99),(15,15,5),'Bronze',mesh='Cylinder')

# Cafe furniture, real dimensions, and obstacles that belong to the environment.
for x in [-360,-40,260]:
    box('Cafe table top',(x,-410,105),(75,75,7),'Wood',mesh='Cylinder')
    box('Cafe table pedestal',(x,-410,67),(8,8,72),'Charcoal',mesh='Cylinder')
    box('Cafe table foot',(x,-410,31),(43,43,5),'Charcoal',mesh='Cylinder')
    for dy in [-85,85]:
        box('Cafe chair seat',(x,-410+dy,72),(42,42,6),'Wood')
        box('Cafe chair back',(x,-410+dy+(20 if dy>0 else -20),95),(42,5,45),'Wood')
        for dx in [-16,16]:
            for oy in [-16,16]:
                box('Chair leg',(x+dx,-410+dy+oy,48),(3,3,42),'Charcoal')
    box('Table cup',(x+12,-405,114),(7,7,11),'Cream',mesh='Cylinder')
for x in [600,3500]:
    box('A-board',(x,-390,93),(65,7,112),'Charcoal',rotation=(8,0,0))
    text('Cafe board','COFFEE\n&\nPASTRIES',(x,-398,115),13,-90)
    for dx in [-36,36]:
        box('Board wood frame',(x+dx,-390,90),(6,10,130),'Wood')
# Delivery loading at the alley: a detailed official sports car creates occlusion.
car_mesh=u.load_asset('/Game/Vehicles/SportsCar/SM_SportsCar')
car=model(car_mesh,'Epic parked vehicle',(2040,180,26),length=450,yaw=0)
glass=u.load_asset('/Game/Vehicles/SportsCar/SM_SportsCar_Glass')
glass_actor=spawn(u.StaticMeshActor,'Vehicle glass',tuple(car.get_actor_location().to_tuple()))
glass_actor.static_mesh_component.set_static_mesh(glass)
glass_actor.set_actor_scale3d(car.get_actor_scale3d())
for dx in [-145,143]:
    for dy in [-87,87]:
        box('Vehicle tire',(2040+dx,180+dy,58),(66,66,23),'Rubber',rotation=(90,0,0),mesh='Cylinder')
        box('Wheel hub',(2040+dx,180+dy+(13 if dy>0 else -13),58),(40,40,3),'Bronze',rotation=(90,0,0),mesh='Cylinder')
for dx in [-110,-55,0,55,110]:
    box('Crosswalk paver',(2230+dx,0,15),(30,540,2),'Paint',collision=False)

# Warm late-afternoon lighting and atmospheric depth, rendered with Lumen.
sun=spawn(u.DirectionalLight,'Late afternoon sun',(1000,-1000,2000),(-52,-20,0),'Street/Lighting')
sun.light_component.set_editor_property('intensity',50000.0)
sun.light_component.set_editor_property('light_color',u.Color(r=255,g=235,b=207,a=255))
sun.light_component.set_editor_property('light_source_angle',2.0)
sun.light_component.set_editor_property('atmosphere_sun_light',True)
sky=spawn(u.SkyLight,'Sky illumination',(2000,0,1000),folder='Street/Lighting')
sky.light_component.set_editor_property('real_time_capture',True)
sky.light_component.set_editor_property('intensity',1.0)
spawn(u.SkyAtmosphere,'Atmosphere',(0,0,0),folder='Street/Lighting')
fog=spawn(u.ExponentialHeightFog,'Distance haze',(0,0,-250),folder='Street/Lighting')
fog.component.set_editor_property('fog_density',.012)
fog.component.set_editor_property('fog_height_falloff',.3)
fog.component.set_editor_property('start_distance',1500)
for side in [-1,1]:
    for x in [-650,1400,4100]:
        light=spawn(u.PointLight,'Shop warm light',(x,side*510,260),folder='Street/Lighting')
        light.light_component.set_editor_property('intensity',50)
        light.light_component.set_editor_property('light_color',u.Color(r=255,g=193,b=113,a=255))
        light.light_component.set_editor_property('attenuation_radius',430)

# Metric walking controls and scripted dynamic scenarios retain the experiment seam.
spawn(u.PlayerStart,'Pedestrian start',(-1100,-370,125),folder='Street/Experiment')
character=u.get_default_object(u.load_class(None,'/Game/FirstPerson/Blueprints/BP_FirstPersonCharacter.BP_FirstPersonCharacter_C'))
character.character_movement.set_editor_property('max_walk_speed',120.0)
u.EditorAssetLibrary.save_asset('/Game/FirstPerson/Blueprints/BP_FirstPersonCharacter')
sequence=assets.create_asset('StreetActivity',BASE,u.LevelSequence,u.LevelSequenceFactoryNew())
sequence.set_display_rate(u.FrameRate(30,1))
sequence.set_playback_start(0)
sequence.set_playback_end(900)


def person(name,points,yaw):
    a=spawn(u.SkeletalMeshActor,name,points[0][1],(0,yaw,0),'Street/Experiment')
    a.skeletal_mesh_component.set_skeletal_mesh_asset(u.load_asset('/Game/Characters/Mannequins/Meshes/SKM_Quinn_Simple'))
    a.skeletal_mesh_component.set_collision_profile_name('BlockAll')
    binding=sequence.add_possessable(a)
    section=binding.add_track(u.MovieScene3DTransformTrack).add_section()
    section.set_range(0,900)
    channels=section.get_all_channels()
    for c,v in zip(channels,[*points[0][1],0,0,yaw,1,1,1]): c.set_default(v)
    for frame,xyz in points:
        for c,v in zip(channels[:3],xyz):
            c.add_key(u.FrameNumber(frame),v,interpolation=u.MovieSceneKeyInterpolation.LINEAR)
    animation=binding.add_track(u.MovieSceneSkeletalAnimationTrack).add_section()
    animation.set_range(0,900)
    params=animation.get_editor_property('params')
    params.animation=u.load_asset('/Game/Characters/Mannequins/Anims/Unarmed/Walk/MF_Unarmed_Walk_Fwd')
    animation.set_editor_property('params',params)
    a.tags=[u.Name('scripted_dynamic_obstacle'),u.Name('evaluator_truth_only')]


person('Alley crossing pedestrian',[(0,(2150,-940,27)),(150,(2150,-940,27)),(530,(2150,580,27)),(899,(2150,580,27))],0)
person('Approaching pedestrian',[(0,(4000,-350,27)),(899,(400,-350,27))],90)
person('Parallel pedestrian',[(0,(-500,380,27)),(899,(3100,380,27))],-90)
person('Plaza pedestrian',[(0,(5800,420,27)),(899,(3500,420,27))],90)
seq_actor=spawn(u.LevelSequenceActor,'Street activity - 30 second loop',(0,0,0),folder='Street/Experiment')
seq_actor.set_sequence(sequence)
settings=seq_actor.get_editor_property('playback_settings')
settings.auto_play=True
settings.loop_count=u.MovieSceneSequenceLoopCount(-1)
seq_actor.set_editor_property('playback_settings',settings)

views=[('Hero',(-300,-70,185),(-2,4,0),67),
       ('Pedestrian',(-850,-370,187),(0,0,0),80),
       ('Cafe',(540,130,175),(-1,-132,0),68),
       ('Overview',(1800,0,3100),(-68,0,0),75),
       ('Crossing',(1430,-360,187),(0,9,0),75)]
for name,xyz,rot,fov in views:
    cam=spawn(u.CameraActor,name,xyz,rot,'Street/Cameras')
    cam.camera_component.set_field_of_view(fov)
editor.set_level_viewport_camera_info(u.Vector(*views[0][1]),u.Rotator(pitch=-2,yaw=4,roll=0))
u.EditorAssetLibrary.save_directory(BASE,only_if_is_dirty=False,recursive=True)
levels.save_current_level()
receipt={'status':'MAP_BUILT','engine':u.SystemLibrary.get_engine_version(),'map':MAP,
         'actor_count':len(actors.get_all_level_actors()),'title':'Willow Walk pedestrian quarter',
         'official_assets':['Epic ArchVis HillTree_02','Epic Building street lights',
                            'Epic Vehicles sports car','Epic First Person and Quinn walk animation'],
         'cc0_assets':['Poly Haven scanned pavement, brick, plaster, wood, bench and planter'],
         'rendering':'Lumen GI and reflections; virtual shadow maps; dynamic atmosphere',
         'scenario_intents':['static street furniture','narrow cafe sidewalk','occluded alley crossing',
                             'parallel near miss','approaching pedestrian'],
         'units':'centimeters','camera_height_m':1.6,'sidewalk_elevation_m':.27,
         'evidence_role':'synthetic Development scene, no DTR evaluation',
         'sensor_status':'camera poses only; RGB-D exporter and DTR bridge not connected'}
(ROOT/'Saved/lab-build.json').write_text(json.dumps(receipt,indent=2))
u.log('WILLOW_WALK_BUILD_OK')
