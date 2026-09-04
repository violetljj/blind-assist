"""Run inside Unreal Editor. Generated assets stay in the local UE project."""
import json
from pathlib import Path
import unreal as u

ASSET = '/Game/ObstacleLab'
MAP = ASSET + '/ObstacleLab'
editor = u.get_editor_subsystem(u.UnrealEditorSubsystem)
levels = u.get_editor_subsystem(u.LevelEditorSubsystem)
actors = u.get_editor_subsystem(u.EditorActorSubsystem)
assets = u.AssetToolsHelpers.get_asset_tools()
root = Path(u.Paths.project_dir())
if u.EditorAssetLibrary.does_asset_exist(MAP):
    raise RuntimeError('Lab map already exists; preserve manual edits. Use a new output project for rebuilding.')
levels.new_level(MAP)
world = editor.get_editor_world()
materials = {}
records = []


def material(name, rgb):
    mat = assets.create_asset(name, ASSET + '/Materials', u.Material, u.MaterialFactoryNew())
    color = u.MaterialEditingLibrary.create_material_expression(mat, u.MaterialExpressionConstant3Vector)
    color.set_editor_property('constant', u.LinearColor(*rgb, 1))
    u.MaterialEditingLibrary.connect_material_property(color, '', u.MaterialProperty.MP_BASE_COLOR)
    u.MaterialEditingLibrary.recompile_material(mat)
    materials[name] = mat


for name, color in {'Floor':(.13,.18,.22), 'Wall':(.55,.61,.65),
                    'Teal':(.02,.65,.58), 'Amber':(1,.43,.04),
                    'Red':(.8,.07,.06), 'White':(.82,.89,.92),
                    'Blue':(.07,.3,.75)}.items():
    material(name, color)


def spawn(cls, name, xyz, rotation=(0, 0, 0)):
    actor = actors.spawn_actor_from_class(cls, u.Vector(*xyz),
        u.Rotator(pitch=rotation[0], yaw=rotation[1], roll=rotation[2]))
    actor.set_actor_label(name)
    actor.set_folder_path('ObstacleLab')
    return actor


def shape(name, xyz, size, color='Wall', mesh='Cube', collision=True):
    actor = spawn(u.StaticMeshActor, name, xyz)
    comp = actor.static_mesh_component
    comp.set_static_mesh(u.load_asset('/Engine/BasicShapes/' + mesh))
    comp.set_material(0, materials[color])
    comp.set_collision_profile_name('BlockAll' if collision else 'NoCollision')
    actor.set_actor_scale3d(u.Vector(*(v / 100 for v in size)))
    actor.tags = [u.Name('lab_geometry')]
    return actor


def label(name, text, xyz, size=34):
    actor = spawn(u.TextRenderActor, name, xyz, (0, 180, 0))
    comp = actor.text_render
    comp.set_text(text)
    comp.set_world_size(size)
    comp.set_text_render_color(u.Color(230, 245, 255, 255))
    return actor


shape('Campus foundation', (1300, 0, -25), (4200, 6000, 50), 'Floor')
shape('Back wall', (3250, 0, 140), (40, 6000, 280))
shape('West boundary', (1200, -2970, 100), (4100, 35, 200))
shape('East boundary', (1200, 2970, 100), (4100, 35, 200))
label('Welcome', 'BLINDASSIST  /  OBSTACLE LAB', (3165, -2250, 360), 125)
label('Subtitle', '6 SCENARIOS   |   FIRST-PERSON WALK   |   SYNTHETIC DEVELOPMENT',
      (3160, -2230, 300), 38)

# Each lane has a 22 m route, visible distance ticks and independent start camera.
lanes = [(-2400, '01  STATIC', 'Bollards / low barrier / head clearance'),
         (-1450, '02  NARROW', '1.2 m passage / offset obstacles'),
         (-500, '03  OCCLUDED CROSSING', 'Moving mannequin behind an opaque screen'),
         (450, '04  NEAR MISS', 'Parallel motion / no route crossing'),
         (1400, '05  HEAD ON', 'Approaching mannequin / early warning'),
         (2350, '06  CROSS TRAFFIC', 'Crossing box cart / intermittent visibility')]
for index, (y, title, subtitle) in enumerate(lanes):
    shape(f'Lane {index+1}', (1400,y,1), (2800,760,2), 'Wall')
    shape(f'Route {index+1}', (1300,y,3), (2400,12,2), 'Teal', collision=False)
    for x in range(200, 2601, 200):
        shape(f'Distance {index+1} {x//100}m', (x,y,4), (5,80,2), 'White', collision=False)
    label(f'Title {index+1}', title, (2850,y-340,205), 43)
    label(f'Detail {index+1}', subtitle, (2848,y-340,153), 18)
    camera = spawn(u.CameraActor, f'Sensor_{index+1}_RGB_160cm', (100,y,160))
    camera.camera_component.set_field_of_view(90)
    camera.tags = [u.Name('sensor_pose'), u.Name('not_connected_to_dtr')]
    records.append({'id':index+1, 'name':title, 'start_cm':[100,y,160],
                    'route_end_cm':[2500,y,160], 'camera_fov_deg':90,
                    'camera_height_m':1.6, 'nominal_wearer_speed_mps':1.2})

# Static corridor with three obstacle scales.
for x, dy in [(750,-100),(1050,100),(1350,-80)]:
    shape(f'Bollard {x}', (x,-2400+dy,50), (35,35,100), 'Amber', 'Cylinder')
shape('Low barrier', (1750,-2400,20), (45,220,40), 'Red')
shape('Overhead beam', (2200,-2400,170), (60,290,30), 'Amber')
for y in [-2560,-2240]:
    shape('Beam support', (2200,y,100), (25,25,200), 'Blue')
# Two slabs leave exactly 120 cm clear width.
for y in [-1670,-1230]:
    shape('Narrow passage wall', (1350,y,105), (1000,320,210), 'Blue')
shape('Offset block', (2200,-1560,55), (95,95,110), 'Amber')
shape('Occluding screen', (1000,-710,115), (70,320,230), 'Blue')
shape('Cart occluder', (950,2120,110), (65,310,220), 'Blue')

sun = spawn(u.DirectionalLight, 'Sun', (0,0,1500), (-48,-35,0))
sun.light_component.set_editor_property('intensity', 3.0)
sky = spawn(u.SkyLight, 'Ambient', (1000,0,1200))
sky.light_component.set_editor_property('intensity', 1.2)
spawn(u.SkyAtmosphere, 'Sky', (0,0,0))
for y in [-1900,0,1900]:
    light = spawn(u.PointLight, 'Fill', (1250,y,700))
    light.light_component.set_editor_property('intensity', 100)
    light.light_component.set_editor_property('attenuation_radius', 2500)

spawn(u.PlayerStart, 'Pedestrian start', (100,-2400,100))
# Use the official template movement, scaled to ordinary walking speed.
character_class = u.load_class(None, '/Game/FirstPerson/Blueprints/BP_FirstPersonCharacter.BP_FirstPersonCharacter_C')
character = u.get_default_object(character_class)
character.character_movement.set_editor_property('max_walk_speed', 120.0)
u.EditorAssetLibrary.save_asset('/Game/FirstPerson/Blueprints/BP_FirstPersonCharacter')

sequence = assets.create_asset('DynamicObstacles', ASSET, u.LevelSequence, u.LevelSequenceFactoryNew())
sequence.set_display_rate(u.FrameRate(30, 1))
sequence.set_playback_start(0)
sequence.set_playback_end(600)


def animate(actor, positions):
    binding = sequence.add_possessable(actor)
    track = binding.add_track(u.MovieScene3DTransformTrack)
    section = track.add_section()
    section.set_range(0, 600)
    channels = section.get_all_channels()
    rot = actor.get_actor_rotation()
    scale = actor.get_actor_scale3d()
    defaults = [*positions[0][1], rot.roll, rot.pitch, rot.yaw, scale.x,scale.y,scale.z]
    for channel, value in zip(channels, defaults):
        channel.set_default(value)
    for frame, xyz in positions:
        for channel, value in zip(channels[:3], xyz):
            channel.add_key(u.FrameNumber(frame), value,
                            interpolation=u.MovieSceneKeyInterpolation.LINEAR)
    return binding


def mannequin(name, positions, yaw):
    actor = spawn(u.SkeletalMeshActor, name, positions[0][1], (0,yaw,0))
    mesh = actor.skeletal_mesh_component
    mesh.set_skeletal_mesh_asset(u.load_asset('/Game/Characters/Mannequins/Meshes/SKM_Quinn_Simple'))
    mesh.set_collision_profile_name('BlockAll')
    binding = animate(actor, positions)
    track = binding.add_track(u.MovieSceneSkeletalAnimationTrack)
    section = track.add_section()
    section.set_range(0,600)
    params = section.get_editor_property('params')
    params.animation = u.load_asset('/Game/Characters/Mannequins/Anims/Unarmed/Walk/MF_Unarmed_Walk_Fwd')
    section.set_editor_property('params', params)
    actor.tags = [u.Name('scripted_dynamic_obstacle'), u.Name('evaluator_truth_only')]
    return actor


mannequin('Occluded crossing pedestrian', [(0,(1250,-850,5)),(150,(1250,-850,5)),
           (330,(1250,-150,5)),(599,(1250,-150,5))], 0)
mannequin('Parallel near miss pedestrian', [(0,(550,700,5)),(450,(2350,700,5)),
           (599,(2350,700,5))], -90)
mannequin('Head on pedestrian', [(0,(2400,1400,5)),(500,(400,1400,5)),
           (599,(400,1400,5))], 90)
cart = shape('Crossing cart', (1350,2010,60), (120,85,120), 'Amber')
cart.static_mesh_component.set_mobility(u.ComponentMobility.MOVABLE)
animate(cart, [(0,(1350,2010,60)),(180,(1350,2010,60)),
               (420,(1350,2690,60)),(599,(1350,2690,60))])
sequence_actor = spawn(u.LevelSequenceActor, 'Play dynamic scenarios (20s loop)', (0,0,0))
sequence_actor.set_sequence(sequence)
settings = sequence_actor.get_editor_property('playback_settings')
settings.auto_play = True
settings.loop_count = u.MovieSceneSequenceLoopCount(-1)
sequence_actor.set_editor_property('playback_settings', settings)

overview = spawn(u.CameraActor, 'Overview', (-2000,-4000,3500), (-33,48,0))
overview.camera_component.set_field_of_view(70)
editor.set_level_viewport_camera_info(u.Vector(-2000,-4000,3500), u.Rotator(pitch=-33,yaw=48,roll=0))
u.EditorAssetLibrary.save_directory(ASSET, only_if_is_dirty=False, recursive=True)
levels.save_current_level()
all_actors = actors.get_all_level_actors()
receipt = {'status':'MAP_BUILT', 'engine':u.SystemLibrary.get_engine_version(),
           'map':MAP, 'actor_count':len(all_actors), 'lanes':records,
           'sequence':sequence.get_path_name(), 'duration_seconds':20,
           'official_assets':['Epic First Person BP template', 'Epic Quinn mannequin and walk animation',
                              'Epic Engine BasicShapes'],
           'units':'centimeters', 'coordinates':'UE left-handed: +X forward, +Y right, +Z up',
           'evidence_role':'synthetic Development playground; no DTR evaluation performed',
           'sensor_status':'named RGB camera poses only; depth/export/online DTR bridge not connected'}
(root/'Saved/lab-build.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
u.log('BLINDASSIST_LAB_BUILD_OK ' + str(len(all_actors)))
