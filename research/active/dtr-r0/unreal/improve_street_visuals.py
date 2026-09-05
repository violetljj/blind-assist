"""Create StreetLabV2: clothed people, distinct facades and balanced daylight.

Run download_street_humans.py first, then execute this inside the UE editor.
All edits target the copied map and new assets; StreetLab is retained.
"""
import json
import math
from pathlib import Path
import unreal as u

ROOT = Path(u.Paths.project_dir())
DATA = ROOT.parent / 'asset-downloads/rocketbox'
BASE = '/Game/StreetLab'
MAP = BASE + '/StreetLabV2'
MAT = BASE + '/MaterialsV2'
HUM = BASE + '/Humans'
levels = u.get_editor_subsystem(u.LevelEditorSubsystem)
editor = u.get_editor_subsystem(u.UnrealEditorSubsystem)
actors = u.get_editor_subsystem(u.EditorActorSubsystem)
assets = u.AssetToolsHelpers.get_asset_tools()
ml = u.MaterialEditingLibrary
receipt = {'map': MAP, 'source_map': BASE + '/StreetLab', 'humans': [],
           'evidence_role': 'synthetic Development visual scene',
           'source': 'https://github.com/microsoft/Microsoft-Rocketbox', 'license': 'MIT',
           'native_human_forward_axis': '+Y',
           'velocity_heading_to_actor_yaw_offset_degrees': -90}


def expression(mat, cls, **props):
    node = ml.create_material_expression(mat, cls)
    for key, value in props.items():
        node.set_editor_property(key, value)
    return node


def solid(name, rgb, rough=.8, metallic=0):
    path = MAT + '/' + name
    if u.EditorAssetLibrary.does_asset_exist(path):
        return u.load_asset(path)
    mat = assets.create_asset(name, MAT, u.Material, u.MaterialFactoryNew())
    col = expression(mat, u.MaterialExpressionConstant3Vector, constant=u.LinearColor(*rgb, 1))
    ml.connect_material_property(col, '', u.MaterialProperty.MP_BASE_COLOR)
    for value, prop in [(rough, u.MaterialProperty.MP_ROUGHNESS), (metallic, u.MaterialProperty.MP_METALLIC)]:
        node = expression(mat, u.MaterialExpressionConstant, r=value)
        ml.connect_material_property(node, '', prop)
    ml.recompile_material(mat)
    return mat


def import_file(path, dest, options=None, name=None, replace=False):
    expected = dest + '/' + (name or Path(path).stem)
    if not replace and u.EditorAssetLibrary.does_asset_exist(expected):
        return [u.load_asset(expected)]
    task = u.AssetImportTask()
    task.filename = str(path)
    task.destination_path = dest
    task.automated = True
    task.save = True
    task.replace_existing = replace
    if name:
        task.destination_name = name
    if options:
        task.options = options
        task.factory = u.FbxFactory()
    assets.import_asset_tasks([task])
    u.EditorAssetLibrary.save_directory(dest, only_if_is_dirty=True, recursive=True)
    return list(task.get_objects())


def plaster(name, tint):
    path = MAT + '/' + name
    if u.EditorAssetLibrary.does_asset_exist(path):
        return u.load_asset(path)
    mat = assets.create_asset(name, MAT, u.Material, u.MaterialFactoryNew())
    mat.set_editor_property('tangent_space_normal', False)
    for role, suffix in [('color', 'diff'), ('normal', 'nor_dx'), ('roughness', 'rough')]:
        tex = u.load_asset(BASE + '/Textures/plastered_wall_02_' + suffix + '_2k')
        obj = expression(mat, u.MaterialExpressionTextureObject, texture=tex)
        size = expression(mat, u.MaterialExpressionConstant3Vector, constant=u.LinearColor(300, 300, 300, 1))
        func = expression(mat, u.MaterialExpressionMaterialFunctionCall)
        func.set_material_function(u.load_asset('/Engine/Functions/Engine_MaterialFunctions01/Texturing/' +
                                                ('WorldAlignedNormal' if role == 'normal' else 'WorldAlignedTexture')))
        ml.connect_material_expressions(obj, '', func, 'TextureObject')
        ml.connect_material_expressions(size, '', func, 'TextureSize')
        if role == 'color':
            multiply = expression(mat, u.MaterialExpressionMultiply)
            col = expression(mat, u.MaterialExpressionConstant3Vector, constant=u.LinearColor(*tint, 1))
            ml.connect_material_expressions(func, 'XYZ Texture', multiply, 'A')
            ml.connect_material_expressions(col, '', multiply, 'B')
            ml.connect_material_property(multiply, '', u.MaterialProperty.MP_BASE_COLOR)
        else:
            ml.connect_material_property(func, 'XYZ Texture', u.MaterialProperty.MP_NORMAL if role == 'normal'
                                         else u.MaterialProperty.MP_ROUGHNESS)
    ml.recompile_material(mat)
    return mat


def human_material(name, folder, color_file, normal_file=None, masked=False):
    path = folder + '/' + name
    if u.EditorAssetLibrary.does_asset_exist(path):
        return u.load_asset(path)
    mat = assets.create_asset(name, folder, u.Material, u.MaterialFactoryNew())
    color = import_file(color_file, folder + '/Textures')[0]
    sample = expression(mat, u.MaterialExpressionTextureSample, texture=color)
    ml.connect_material_property(sample, 'RGB', u.MaterialProperty.MP_BASE_COLOR)
    rough = expression(mat, u.MaterialExpressionConstant, r=.72)
    ml.connect_material_property(rough, '', u.MaterialProperty.MP_ROUGHNESS)
    if normal_file and normal_file.exists():
        normal = import_file(normal_file, folder + '/Textures')[0]
        normal.set_editor_property('srgb', False)
        normal.set_editor_property('compression_settings', u.TextureCompressionSettings.TC_NORMALMAP)
        n = expression(mat, u.MaterialExpressionTextureSample, texture=normal,
                       sampler_type=u.MaterialSamplerType.SAMPLERTYPE_NORMAL)
        ml.connect_material_property(n, 'RGB', u.MaterialProperty.MP_NORMAL)
    if masked:
        mat.set_editor_property('blend_mode', u.BlendMode.BLEND_MASKED)
        mat.set_editor_property('two_sided', True)
        ml.connect_material_property(sample, 'A', u.MaterialProperty.MP_OPACITY_MASK)
    ml.recompile_material(mat)
    return mat


def import_human(name):
    directory = HUM + '/' + name
    local = DATA / 'Assets/Avatars/Adults' / name
    known = [u.load_asset(p) for p in u.EditorAssetLibrary.list_assets(directory, recursive=False)]
    meshes = [o for o in known if isinstance(o, u.SkeletalMesh)]
    if not meshes or not meshes[0].skeleton:
        options = u.FbxImportUI()
        options.automated_import_should_detect_type = False
        options.import_mesh = True
        options.import_as_skeletal = True
        options.mesh_type_to_import = u.FBXImportType.FBXIT_SKELETAL_MESH
        options.import_materials = False
        options.import_textures = False
        options.import_animations = False
        options.create_physics_asset = True
        objects = import_file(local / 'Export' / (name + '.fbx'), directory, options, replace=bool(meshes))
        meshes = [o for o in objects if isinstance(o, u.SkeletalMesh)]
    assert meshes, 'No skeletal meshes imported for ' + name
    mesh = next((m for m in meshes if 'hipoly' in m.get_name().lower()), meshes[0])
    # Match the source FBX material slot names, preserving skin/clothing/hair UVs.
    materials = {}
    for color in sorted((local / 'Textures').glob('*_color.tga')):
        stem = color.stem[:-6]
        materials[stem.lower()] = human_material(stem, directory, color,
                                                 color.with_name(stem + '_normal.tga'), 'opacity' in stem)
    for index, slot in enumerate(mesh.materials):
        slot_name = str(slot.material_slot_name).lower()
        match = next((m for key, m in materials.items() if key in slot_name or slot_name in key), None)
        if match is None:
            match = next((m for key, m in materials.items() if ('head' in key) == ('head' in slot_name)
                          and ('opacity' in key) == ('opacity' in slot_name)), None)
        assert match, f'Material slot unresolved: {name} {slot_name}'
        slot.material_interface = match
        slots = list(mesh.materials)
        slots[index] = slot
        mesh.materials = slots
    anim_name = 'NeutralWalk'
    anim = u.load_asset(directory + '/' + anim_name)
    if not anim:
        options = u.FbxImportUI()
        options.automated_import_should_detect_type = False
        options.import_mesh = False
        options.import_animations = True
        options.mesh_type_to_import = u.FBXImportType.FBXIT_ANIMATION
        options.skeleton = mesh.skeleton
        options.import_materials = False
        options.import_textures = False
        options.override_animation_name = anim_name
        sex = 'f' if name.startswith('Female') else 'm'
        results = import_file(DATA / f'Assets/Animations/all_animations_max_motextr_xy/{sex}_walk_neutral.max.fbx',
                              directory, options, anim_name)
        anim = next((o for o in results if isinstance(o, u.AnimSequence)), None)
    assert anim, 'Compatible walk animation missing: ' + name
    anim.set_editor_property('enable_root_motion', False)
    anim.set_editor_property('force_root_lock', True)
    u.EditorAssetLibrary.save_loaded_asset(mesh)
    u.EditorAssetLibrary.save_loaded_asset(anim)
    bounds = mesh.get_bounds()
    info = {'name': name, 'mesh': mesh.get_path_name(), 'animation': anim.get_path_name(),
            'native_height_cm': bounds.box_extent.z * 2,
            'mesh_material_slots': [str(s.material_slot_name) for s in mesh.materials]}
    receipt['humans'].append(info)
    return mesh, anim


def spawn(cls, name, xyz, yaw=0):
    a = actors.spawn_actor_from_class(cls, u.Vector(*xyz), u.Rotator(yaw=yaw))
    a.set_actor_label(name)
    a.set_folder_path('Street/V2 details')
    return a


def box(name, xyz, size, material, yaw=0, shape='Cube'):
    a = spawn(u.StaticMeshActor, name, xyz, yaw)
    a.static_mesh_component.set_static_mesh(u.load_asset('/Engine/BasicShapes/' + shape))
    a.static_mesh_component.set_material(0, material)
    a.static_mesh_component.set_collision_profile_name('BlockAll')
    a.set_actor_scale3d(u.Vector(*(v / 100 for v in size)))
    return a


def balanced_post_process():
    """Same fixed exposure for the editor viewport and SceneCapture2D sensors."""
    settings = u.PostProcessSettings()
    values = {'auto_exposure_method': u.AutoExposureMethod.AEM_HISTOGRAM,
              'auto_exposure_min_brightness': 10.0, 'auto_exposure_max_brightness': 10.0,
              'auto_exposure_bias': 0.0, 'bloom_intensity': .12, 'vignette_intensity': .12,
              'motion_blur_amount': 0.0, 'color_saturation': u.Vector4(1.02, 1.02, 1.02, 1.0)}
    for name, value in values.items():
        settings.set_editor_property('override_' + name, True)
        settings.set_editor_property(name, value)
    return settings


def main():
    if not (DATA / 'sources.json').exists():
        raise RuntimeError('Run download_street_humans.py first')
    if u.EditorAssetLibrary.does_asset_exist(MAP) and (ROOT / 'Saved/lab-visual-upgrade.json').exists():
        raise RuntimeError('StreetLabV2 already exists; preserve it and choose a new destination for rebuild')
    levels.load_level(BASE + '/StreetLab')
    world = editor.get_editor_world()
    assert u.EditorLoadingAndSavingUtils.save_map(world, MAP)
    levels.load_level(MAP)
    world = editor.get_editor_world()
    people = [import_human(name) for name in json.loads((DATA / 'sources.json').read_text())['humans']]
    all_actors = actors.get_all_level_actors()
    old_people = sorted([a for a in all_actors if isinstance(a, u.SkeletalMeshActor)],
                        key=lambda a: a.get_actor_label())
    seq = u.load_asset(BASE + '/StreetActivityV2')
    if seq:
        for binding in seq.get_bindings():
            binding.remove()
    else:
        seq = assets.create_asset('StreetActivityV2', BASE, u.LevelSequence, u.LevelSequenceFactoryNew())
    seq.set_display_rate(u.FrameRate(30, 1))
    seq.set_playback_start(0)
    seq.set_playback_end(900)
    routes = {
        'Alley crossing pedestrian': [(0, (2150, -940, 27)), (150, (2150, -940, 27)), (530, (2150, 580, 27)), (899, (2150, 580, 27))],
        'Approaching pedestrian': [(0, (4000, -350, 27)), (899, (400, -350, 27))],
        'Parallel pedestrian': [(0, (-500, 380, 27)), (899, (3100, 380, 27))],
        'Plaza pedestrian': [(0, (5800, 420, 27)), (899, (3500, 420, 27))],
    }
    for index, actor in enumerate(old_people):
        mesh, anim = people[index % len(people)]
        component = actor.skeletal_mesh_component
        component.set_skeletal_mesh_asset(mesh)
        component.set_animation_mode(u.AnimationMode.ANIMATION_SINGLE_NODE)
        component.set_animation(anim)
        component.set_update_animation_in_editor(True)
        component.set_position(index * .21, False)
        component.set_play_rate(1.0)
        component.play_animation(anim, True)
        binding = seq.add_possessable(actor)
        transform = binding.add_track(u.MovieScene3DTransformTrack).add_section()
        transform.set_range(0, 900)
        points = routes[actor.get_actor_label()]
        yaw = actor.get_actor_rotation().yaw
        channels = transform.get_all_channels()
        for c, value in zip(channels, [*points[0][1], 0, 0, yaw, 1, 1, 1]):
            c.set_default(value)
        for frame, xyz in points:
            for c, value in zip(channels[:3], xyz):
                c.add_key(u.FrameNumber(frame), value, interpolation=u.MovieSceneKeyInterpolation.LINEAR)
        section = binding.add_track(u.MovieSceneSkeletalAnimationTrack).add_section()
        section.set_range(0, 900)
        params = section.get_editor_property('params')
        params.animation = anim
        section.set_editor_property('params', params)
    for a in all_actors:
        if isinstance(a, u.StaticMeshActor) and a.get_actor_label().startswith('Tactile'):
            label = a.get_actor_label()
            relief = .1 if 'guidance strip' in label else .4
            loc, scale = a.get_actor_location(), a.get_actor_scale3d()
            a.set_actor_location(u.Vector(loc.x, loc.y, 26 + relief / 2), False, False)
            a.set_actor_scale3d(u.Vector(scale.x, scale.y, relief / 100))
            a.static_mesh_component.set_collision_profile_name('NoCollision')
        if isinstance(a, u.LevelSequenceActor):
            a.set_sequence(seq)
        if isinstance(a, u.DirectionalLight):
            a.set_actor_rotation(u.Rotator(pitch=-58, yaw=-35), False)
            a.light_component.set_editor_property('intensity', 7000.0)
            a.light_component.set_editor_property('light_color', u.Color(r=255, g=248, b=235, a=255))
            a.light_component.set_editor_property('light_source_angle', 8.0)
        if isinstance(a, u.SkyLight):
            a.light_component.set_editor_property('real_time_capture', False)
            a.light_component.set_editor_property('source_type', u.SkyLightSourceType.SLS_SPECIFIED_CUBEMAP)
            a.light_component.set_editor_property('cubemap', u.load_asset('/Engine/MapTemplates/Sky/DaylightAmbientCubemap'))
            a.light_component.set_editor_property('intensity', 2.5)
            a.light_component.set_editor_property('lower_hemisphere_is_black', False)
            a.light_component.recapture_sky()
        if isinstance(a, u.CameraActor):
            a.camera_component.set_editor_property('post_process_settings', balanced_post_process())
            a.camera_component.set_editor_property('post_process_blend_weight', 1.0)
    pp = spawn(u.PostProcessVolume, 'V2 balanced daylight exposure', (2000, 0, 500))
    pp.set_editor_property('unbound', True)
    pp.set_editor_property('priority', 20.0)
    pp.set_editor_property('settings', balanced_post_process())
    fill = spawn(u.DirectionalLight, 'V2 broad sky fill', (0, 0, 2500), yaw=145)
    fill.set_actor_rotation(u.Rotator(pitch=-70, yaw=145), False)
    fill.light_component.set_editor_property('intensity', 4000.0)
    fill.light_component.set_editor_property('cast_shadows', False)
    fill.light_component.set_editor_property('light_color', u.Color(r=217, g=230, b=255, a=255))
    palette = [plaster('Chalk', (1.0, .94, .80)), plaster('RoseRender', (.85, .55, .42)),
               plaster('WarmGrey', (.68, .74, .71)), plaster('PaleSage', (.67, .83, .61)),
               plaster('OchreRender', (.94, .72, .40)), plaster('BlueSlate', (.39, .58, .65))]
    trim = solid('SoftStone', (.48, .44, .36))
    glass = solid('BlueGreyGlass', (.095, .14, .17), .24, .25)
    metal = solid('PaintedMetal', (.08, .10, .09), .5, .4)
    paving = u.load_asset(BASE + '/Materials/Pavers')
    wood = u.load_asset(BASE + '/Materials/Wood')
    # Distinct frontage colours, window proportions and rhythms replace exact repeated templates.
    buildings = sorted([a for a in all_actors if a.get_actor_label().endswith(' building')],
                       key=lambda a: (a.get_actor_location().y, a.get_actor_location().x))
    for i, building in enumerate(buildings):
        loc = building.get_actor_location()
        width = building.get_actor_scale3d().x * 100
        height = building.get_actor_scale3d().z * 100
        face = -640 if loc.y < 0 else 640
        side = -1 if loc.y < 0 else 1
        if i % 3:
            building.static_mesh_component.set_material(0, palette[i % len(palette)])
        # Finely spaced ground-floor masonry and rooftop parapets catch soft light.
        for z in (92, 146, 200, 254):
            box('V2 masonry reveal', (loc.x, face - side * 2, z), (width - 45, 2, 1.6), trim)
        box('V2 parapet', (loc.x, face + side * 18, height + 38), (width + 35, 36, 66), palette[i % 6])
        for offset in (-.42, .42):
            box('V2 parapet cap', (loc.x + width * offset, face + side * 18, height + 75), (55, 60, 17), trim)
        if i % 3 == 1:
            for xx in (-width * .30, 0, width * .30):
                box('V2 raised dormer', (loc.x + xx, face + side * 100, height + 105), (195, 165, 180), palette[i % 6])
                box('V2 dormer glass', (loc.x + xx, face + side * 14, height + 108), (112, 7, 120), glass)
        if i % 3 == 2:
            # Vertical timber slats create a recognisable contemporary shop frontage.
            for xx in range(-int(width / 2) + 55, int(width / 2) - 30, 44):
                box('V2 timber fascia slat', (loc.x + xx, face - side * 67, 333), (17, 13, 54), wood)
        for a in all_actors:
            p = a.get_actor_location()
            if abs(p.x - loc.x) > width / 2 or abs(p.y - face) > 220 or not isinstance(a, u.StaticMeshActor):
                continue
            label = a.get_actor_label()
            if 'glazing' in label or 'Shop window glass' in label:
                a.static_mesh_component.set_material(0, glass)
            if label.startswith(('Fabric canopy', 'Canopy valance', 'Shop fascia')):
                a.static_mesh_component.set_material(0, palette[(i + 3) % 6])
            if i % 3 == 1 and label.startswith(('Upper window', 'Upper dark', 'Window sill', 'Window mullion', 'Window transom')):
                scale = a.get_actor_scale3d()
                a.set_actor_scale3d(u.Vector(scale.x * .84, scale.y, scale.z * 1.1))
    # Realistic ground services break broad uninterrupted pavement without obstructing experiments.
    for x in (-1000, 700, 2600, 4500, 5900):
        for side in (-1, 1):
            box('V2 drainage frame', (x, side * 262, 13), (84, 26, 3), metal)
            for dx in range(-35, 36, 10):
                box('V2 drain grate', (x + dx, side * 262, 15), (3, 24, 1), trim)
        box('V2 utility cover', (x + 190, 150, 13), (65, 65, 2), metal, shape='Cylinder')
    # Close the distant view with an articulated civic facade, not a horizon of blank cubes.
    back = solid('CivicSandstone', (.43, .40, .31))
    box('V2 terminating civic building', (7600, 0, 850), (700, 4300, 1700), back)
    for y in range(-1950, 2000, 260):
        for z in (370, 750, 1130, 1470):
            box('V2 civic window surround', (7235, y, z), (35, 170, 230), trim)
            box('V2 civic glazing', (7212, y, z), (7, 138, 197), glass)
        box('V2 civic pilaster', (7240, y + 115, 820), (45, 28, 1600), trim)
    box('V2 civic cornice', (7220, 0, 1700), (95, 4350, 70), trim)
    # Additional tree crowns close lateral alley views while preserving the central street.
    tree = u.load_asset('/Game/ArchVis/SampleScene/Tree/HillTree_02')
    for x, y in ((2170, -2400), (2170, 2400), (6830, -950), (6830, 950)):
        a = spawn(u.StaticMeshActor, 'V2 background oak', (x, y, 0), yaw=x % 360)
        a.static_mesh_component.set_static_mesh(tree)
        b = tree.get_bounds()
        scale = 1200 / (2 * b.box_extent.z)
        a.set_actor_scale3d(u.Vector(scale, scale, scale))
    editor.set_level_viewport_camera_info(u.Vector(-300, -70, 172), u.Rotator(pitch=-2, yaw=4))
    receipt.update(status='PASS', actor_count=len(actors.get_all_level_actors()),
                   sequence=seq.get_path_name(), fixed_exposure_ev100=10.0, sun_lux=7000,
                   skylight_intensity=2.5, broad_sky_fill_lux=4000,
                   source_commit=json.loads((DATA / 'sources.json').read_text())['commit'])
    u.EditorAssetLibrary.save_directory(BASE, only_if_is_dirty=True, recursive=True)
    levels.save_current_level()
    (ROOT / 'Saved/lab-visual-upgrade.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    u.log('STREET_VISUAL_UPGRADE_OK')


if __name__ == '__main__':
    main()
