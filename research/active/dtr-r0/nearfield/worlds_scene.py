"""Transient G14 research worlds; metres in specs, centimetres at the UE boundary.

New surface/shrub assets use only /Game/G14RealismV1; no levels are saved.
Importable outside UE for geometry checks.
The caller owns the returned actor list and must destroy it after acquisition.
These deliberately structured primitive scenes are synthetic Development worlds,
not natural streets, photorealism claims, or copies of the saved Willow layout.
"""
import math
import random


MATERIAL_ROOT = '/Game/StreetLab/Materials/'
WORLD_IDS = ('sidewalk', 'corridor')


def world_spec(world_id, layout_seed):
    """Return reproducible full placements, including native-depth-visible scenery."""
    if world_id not in WORLD_IDS:
        raise ValueError('Unknown research world: ' + str(world_id))
    rng = random.Random(int(layout_seed))
    origin = (1000., 0. if world_id == 'sidewalk' else 100., 0.)
    objects, lights, models, buildings = [], [], [], []

    def box(name, center, size, material, kind='cube'):
        objects.append(dict(name=name, kind=kind,
                            center_m=[round(origin[i] + center[i], 5) for i in range(3)],
                            size_m=list(size), material=MATERIAL_ROOT + material))

    def point(name, center, intensity, radius, color):
        lights.append(dict(name=name, kind='point',
                           center_m=[origin[i] + center[i] for i in range(3)],
                           intensity=intensity, radius_m=radius, color=list(color)))

    def model(name, mesh, base, height, yaw=0., max_width=None):
        models.append(dict(name=name, mesh=mesh,
                           base_m=[origin[i] + base[i] for i in range(3)],
                           height_m=height, yaw_deg=yaw, max_width_m=max_width,
                           anchor_xy='mesh_pivot' if ('tree' in name or 'plant' in name and 'planter' not in name) else 'bounds_center'))

    def bench(x, y):
        model('scanned timber bench', '/Game/SampleMaterialsV2/Meshes/modular_street_seating_assembled_4k',
              (x, y, .12), .85, 0 if y < 0 else 180, .8)

    wallmat = rng.choice(['Cream', 'Plaster', 'Limestone'])
    if world_id == 'sidewalk':
        # Continuous support extends under every native building and past the horizon.
        box('district foundation', (100, 0, -.22), (600, 600, .4), 'Limestone')
        box('frontage forecourt', (30, 6.5, .04), (100, 8.2, .16), 'Pavers')
        box('opposite pavement', (30, -10.5, .04), (100, 4, .16), 'Pavers')
        box('opposite curb', (30, -8.52, .05), (100, .16, .22), 'Limestone')
        box('far plaza', (45, 0, .04), (46, 25, .16), 'Pavers')
        box('continuous sidewalk slab', (8, 0, -.04), (28, 4.8, .32), 'Pavers')
        box('road', (30, -5.5, -.04), (100, 6, .16), 'Asphalt')
        box('curb', (8, -2.48, .05), (28, .16, .22), 'Limestone')
        box('garden soil bed', (8, 3.8, .18), (28, 2.5, .12), 'Soil')
        # Exposed soil avoids the flat green rectangles of the initial shell.
        for x in range(-4, 24, 2):
            model('scanned understory plant', '/Game/G14RealismV1/Meshes/shrub_01_2k',
                  (x + rng.uniform(-.25, .25), 3.8 + (.15 if x % 4 else -.15), .24),
                  rng.uniform(.9, 1.4), rng.uniform(0, 360), 2.2)
        box('garden edging', (8, 2.55, .17), (28, .09, .1), 'Limestone')
        # Thin contrasting joints are real geometry, flush with the .12 m floor.
        for x in range(-5, 23):
            box('paving cross joint', (x, 0, .119), (.018, 4.75, .002), 'Limestone')
        for y in (-1.6, -.8, .8, 1.6):
            box('paving longitudinal joint', (8, y, .119), (28, .015, .002), 'Limestone')
        for x in (-3, 4, 11, 18):
            x += rng.uniform(-.3, .3)
            for dx in (-.83, .83):
                box('tree island side rim', (x + dx, 3.8, .22), (.12, 1.72, .2), 'Limestone')
            for dy in (-.8, .8):
                box('tree island end rim', (x, 3.8 + dy, .22), (1.6, .12, .2), 'Limestone')
            box('planter soil', (x, 3.8, .20), (1.55, 1.5, .16), 'Soil')
            model('scanned leafy tree', '/Game/SampleMaterialsV2/Meshes/tree_small_02_2k',
                  (x, 3.8, .28), 6.1 + rng.uniform(-.8, 1.2), rng.uniform(0, 360), 5.5)
        for x in (2., 12.):
            bench(x + rng.uniform(-.4, .4), 1.98)
        # A planted courtyard terminates the sightline instead of exposing the
        # foundation plane at the horizon. It lies beyond the acquisition path.
        box('courtyard enclosing garden wall', (30, 0, 1.25), (.35, 200, 2.5), 'Brick')
        box('courtyard wall stone coping', (30, 0, 2.53), (.48, 200, .12), 'Limestone')
        for y in (-16., -10., -4., 2., 8., 14.):
            box('courtyard masonry pier', (29.8, y, 1.42), (.65, .65, 2.84), 'Limestone')
            model('scanned courtyard tree', '/Game/SampleMaterialsV2/Meshes/tree_small_02_2k',
                  (31.5 + rng.uniform(0, 1), y, .12), rng.uniform(7.5, 10), rng.uniform(0, 360), 7.)
        for x in (1., 9., 17.):
            model('scanned opposite tree', '/Game/SampleMaterialsV2/Meshes/tree_small_02_2k',
                  (x, -10.1, .12), rng.uniform(6., 8.), rng.uniform(0, 360), 5.)
        for x in (6., 16.):
            box('campus litter bin', (x, 2.02, .55), (.42, .42, .86), 'Charcoal')
            box('litter bin rim', (x, 2.02, .97), (.46, .46, .06), 'Bronze')
        for x in range(-3, 25, 6):
            box('road broken center marking', (x, -5.6, .043), (2.5, .10, .004), 'Paint')
        for y in (-2.25, 2.25):
            box('sidewalk stone border', (8, y, .121), (28, .22, .006), 'Limestone')
        for x in (-1., 8., 17.):
            model('native street lantern', '/Game/Building/Geometry/SM_StreetLight',
                  (x, -2.13, .12), 4.2, max_width=.55)
            point('sidewalk lamp', (x, -1.55, 3.88), 45., 5., (255, 224, 186))
        # Full native City Sample assemblies retain their trim, doors and materials.
        lib = '/Game/CitySampleBuildings/Building/Library/Kit_Hero_Bldg/LevelInstance/'
        for name, asset, yaw, edge, coordinate in (
                ('side heritage frontage', 'BPP_Bldg_Hero_CHA_A01_N1', 0., 'north', 5.3),
                ('opposite gallery', 'BPP_Bldg_Hero_Low_SFD_Long_N1', 180., 'south', -12.5),
                ('vista heritage', 'BPP_Bldg_Hero_CHA_A01_N1', -90., 'east', 26.)):
            buildings.append(dict(name=name, asset=lib + asset,
                                  ground_asset=lib + asset.removesuffix('_N1') + '_Level01_N1',
                                  yaw_deg=yaw, edge=edge, coordinate_m=coordinate))
        point('outdoor diffuse fill', (4, 0, 7), 350., 20., (225, 239, 255))
    else:
        box('continuous corridor floor', (8, 0, -.04), (28, 4.8, .32), 'Limestone')
        box('ceiling', (8, 0, 3.52), (28, 5, .22), 'Paint')
        for side in (-1, 1):
            box('corridor wall', (8, side * 2.5, 1.82), (28, .2, 3.4), wallmat)
            box('wall skirting', (8, side * 2.385, .25), (28, .05, .26), 'Charcoal')
            box('wall dado rail', (8, side * 2.37, 1.12), (28, .07, .08), 'Wood')
            box('painted lower wall panel', (8, side * 2.386, .69), (28, .028, .78), 'Sage')
            box('ceiling cornice', (8, side * 2.35, 3.35), (28, .13, .10), 'Paint')
        for x in range(-5, 23):
            box('tile cross grout', (x, 0, .119), (.018, 4.8, .002), 'Cream')
        for y in (-1.6, -.8, .8, 1.6):
            box('tile longitudinal grout', (8, y, .119), (28, .018, .002), 'Cream')
        for side in (-1, 1):
            for x in (-2., 4., 10., 16.):
                x += rng.uniform(-.3, .3)
                box('closed door leaf', (x, side * 2.365, 1.22), (1.02, .06, 2.2), 'Wood')
                for dx in (-.57, .57):
                    box('door jamb', (x + dx, side * 2.32, 1.26), (.09, .14, 2.28), 'Paint')
                box('door lintel', (x, side * 2.32, 2.42), (1.23, .14, .1), 'Paint')
                box('door handle', (x + .35, side * 2.25, 1.12), (.15, .08, .035), 'Bronze')
                box('room sign', (x + .92, side * 2.37, 1.62), (.27, .035, .16), 'Charcoal')
        for x, side in ((1., 1), (8., -1), (15., 1)):
            bench(x, side * 1.98)
            px = x + 1.5
            model('scanned planter', '/Game/StreetLab/Props/planter_box_02_2k',
                  (px, side * 2., .12), .5, 90., .65)
            model('native leafy indoor plant', '/Game/SampleMaterialsV2/Meshes/tree_small_02_2k',
                  (px, side * 2., .52), 1.45, rng.uniform(0, 360), .8)
        for x, side in ((1., -1), (7., 1), (13., -1)):
            box('noticeboard timber frame', (x, side * 2.33, 1.90), (1.55, .10, .92), 'Wood')
            box('noticeboard backing', (x, side * 2.265, 1.90), (1.41, .025, .78), 'Sage')
            for dx, dz, material in ((-.46, .10, 'Cream'), (0., -.08, 'Paint'), (.46, .10, 'Cream')):
                box('pinned campus notice', (x + dx, side * 2.242, 1.90 + dz), (.33, .009, .44), material)
        for x, side in ((5.8, -1), (11.8, 1)):
            # Use the assembled seating mesh already verified in native capture;
            # ScannedCafe_chair is a material asset, not a StaticMesh.
            bench(x, side * 1.98)
            box('waiting side table top', (x + .85, side * 2., .65), (.48, .48, .06), 'Wood')
            box('waiting side table pedestal', (x + .85, side * 2., .36), (.12, .12, .52), 'Charcoal')
        for x in (-3., 2., 7., 12., 17., 21.):
            box('ceiling luminaire', (x, 0, 3.38), (1.2, .46, .05), 'WarmLight')
            point('corridor ceiling light', (x, 0, 3.15), 120., 5., (255, 242, 221))
        box('end wall', (22, 0, 1.82), (.22, 5, 3.4), wallmat)
        box('end double door', (21.86, 0, 1.32), (.06, 1.8, 2.4), 'Sage')
        box('end door division', (21.81, 0, 1.32), (.05, .04, 2.4), 'Bronze')
        box('end transom', (21.85, 0, 2.78), (.06, 1.8, .35), 'WarmLight')
    for obj in objects:
        if obj['name'] in ('continuous sidewalk slab', 'frontage forecourt', 'opposite pavement', 'far plaza'):
            obj['material'] = '/Game/G14RealismV1/Materials/concrete_pavement_02'
        elif obj['name'] == 'garden soil bed':
            obj['material'] = '/Game/G14RealismV1/Materials/leafy_grass'
        elif obj['name'] == 'continuous corridor floor':
            obj['material'] = '/Game/G14RealismV1/Materials/terrazzo_tiles'
        elif obj['name'] in ('corridor wall', 'end wall') and int(layout_seed) % 2 == 0:
            obj['material'] = '/Game/SampleMaterialsV2/Materials/concrete_wall_007'
    return dict(schema='nearfield-research-world-v3', world_id=world_id,
                layout_seed=int(layout_seed), origin_m=list(origin),
                floor_z_m=.12, camera_nominal_m=[origin[0], origin[1], 1.82],
                camera_yaw_deg=0., reserved_path_half_width_m=.9,
                objects=objects, lights=lights, models=models, buildings=buildings,
                provenance={'construction': 'deterministic architectural shell with existing scanned/native furnished meshes',
                            'geometry_assets': sorted({'/Engine/BasicShapes/Cube', '/Engine/BasicShapes/Cylinder'} | {m['mesh'] for m in models} | {b[k] for b in buildings for k in ('asset', 'ground_asset')}),
                            'material_source': MATERIAL_ROOT,
                            'furnishing_materials': 'original mesh material slots; no flat-color overrides',
                            'truth': 'full native scene depth; placements are provenance only',
                            'saved_willow_modified': False,
                            'lighting': 'owned local point lights; capture controls and records environment overrides'})


def destroy_world(api, owned_actors):
    """Destroy explicit task-owned handles; retain any failing handles for retry."""
    failed = []
    for actor in reversed(owned_actors[:]):
        try:
            if api.destroy_actor(actor):
                owned_actors.remove(actor)
            else:
                failed.append(actor)
        except Exception:
            failed.append(actor)
    if failed:
        raise RuntimeError('Could not destroy %d owned world actors' % len(failed))


def build_world(u, api, world_id, layout_seed):
    """Spawn once per layout. Return (owned actor list, serializable scene spec)."""
    spec = world_spec(world_id, layout_seed)
    owned = []
    try:
        from worlds_materials import ensure_materials
        spec['surface_assets'] = ensure_materials(u)
        meshes = {kind: u.load_asset('/Engine/BasicShapes/' + name)
                  for kind, name in (('cube', 'Cube'), ('cylinder', 'Cylinder'))}
        materials = {path: u.load_asset(path) for path in sorted({o['material'] for o in spec['objects']})}
        if not all(meshes.values()) or not all(materials.values()):
            raise RuntimeError('Missing required local research-world mesh/material')
        for obj in spec['objects']:
            actor = api.spawn_actor_from_class(u.StaticMeshActor, u.Vector(*(v * 100 for v in obj['center_m'])))
            if not actor:
                raise RuntimeError('World actor spawn failed: ' + obj['name'])
            owned.append(actor)
            actor.set_actor_label('G14 %s %s' % (world_id, obj['name']))
            actor.static_mesh_component.set_static_mesh(meshes[obj['kind']])
            actor.set_actor_scale3d(u.Vector(*obj['size_m']))
            actor.static_mesh_component.set_material(0, materials[obj['material']])
            actor.static_mesh_component.set_collision_profile_name('BlockAll')
        for obj in spec['models']:
            mesh = u.load_asset(obj['mesh'])
            if not mesh:
                raise RuntimeError('Missing local native furnishing: ' + obj['mesh'])
            if not isinstance(mesh, u.StaticMesh):
                raise RuntimeError('Native furnishing asset is not a StaticMesh: ' + obj['mesh'])
            bounds = mesh.get_bounds()
            scale = obj['height_m'] * 100 / (bounds.box_extent.z * 2)
            angle = math.radians(obj['yaw_deg'])
            # Cap transverse width uniformly, preserving the scanned proportions.
            native_y = 2 * (abs(math.sin(angle)) * bounds.box_extent.x + abs(math.cos(angle)) * bounds.box_extent.y)
            if obj['anchor_xy'] == 'mesh_pivot':
                native_y += 2 * abs(math.sin(angle) * bounds.origin.x + math.cos(angle) * bounds.origin.y)
            if obj['max_width_m']:
                scale = min(scale, obj['max_width_m'] * 100 / native_y)
            bx, by, bz = (v * 100 for v in obj['base_m'])
            anchor_x = bounds.origin.x if obj['anchor_xy'] == 'bounds_center' else 0.
            anchor_y = bounds.origin.y if obj['anchor_xy'] == 'bounds_center' else 0.
            location = u.Vector(bx - scale * (anchor_x * math.cos(angle) - anchor_y * math.sin(angle)),
                                by - scale * (anchor_x * math.sin(angle) + anchor_y * math.cos(angle)),
                                bz - scale * (bounds.origin.z - bounds.box_extent.z))
            actor = api.spawn_actor_from_class(u.StaticMeshActor, location, u.Rotator(yaw=obj['yaw_deg']))
            if not actor:
                raise RuntimeError('Native furnishing spawn failed: ' + obj['name'])
            owned.append(actor)
            actor.set_actor_label('G14 %s %s' % (world_id, obj['name']))
            actor.static_mesh_component.set_static_mesh(mesh)
            actor.set_actor_scale3d(u.Vector(scale, scale, scale))
            actor.static_mesh_component.set_collision_profile_name('BlockAll')
            obj['resolved_uniform_scale'] = scale
            obj['native_bounds_cm'] = dict(origin=list(bounds.origin.to_tuple()), extent=list(bounds.box_extent.to_tuple()))
        for building in spec['buildings']:
            pair = []
            for path in (building['asset'], building['ground_asset']):
                cls = u.EditorAssetLibrary.load_blueprint_class(path)
                if not cls:
                    raise RuntimeError('Missing native building: ' + path)
                actor = api.spawn_actor_from_class(cls, u.Vector(), u.Rotator(yaw=building['yaw_deg']))
                if not actor:
                    raise RuntimeError('Native building spawn failed: ' + path)
                owned.append(actor)
                pair.append(actor)
                actor.set_actor_label('G14 %s %s' % (world_id, building['name']))
            center, extent = pair[0].get_actor_bounds(False)
            if min(extent.x, extent.y, extent.z) <= 100:
                raise RuntimeError('Native packed building bounds are not ready')
            ox, oy, _ = (v * 100 for v in spec['origin_m'])
            if building['edge'] == 'north':
                position = u.Vector(ox - 1600 - center.x + extent.x,
                                    oy + building['coordinate_m'] * 100 - center.y + extent.y,
                                    12.)
            elif building['edge'] == 'south':
                position = u.Vector(ox - 1800 - center.x + extent.x,
                                    oy + building['coordinate_m'] * 100 - center.y - extent.y, 12.)
            else:
                position = u.Vector(ox + building['coordinate_m'] * 100 - center.x + extent.x,
                                    oy - center.y, 12.)
            for actor in pair:
                actor.set_actor_location(position, False, False)
            # These CitySample assemblies use their native z=0 street datum;
            # -128cm bounds include below-grade geometry and must not lift facades.
            building['ground_alignment'] = 'native blueprint street datum z=0 aligned to floor z=.12m'
            building['resolved_location_cm'] = list(position.to_tuple())
            building['native_bounds_cm'] = dict(origin=list(center.to_tuple()), extent=list(extent.to_tuple()))
        for light in spec['lights']:
            actor = api.spawn_actor_from_class(u.PointLight, u.Vector(*(v * 100 for v in light['center_m'])))
            if not actor:
                raise RuntimeError('World light spawn failed: ' + light['name'])
            owned.append(actor)
            actor.set_actor_label('G14 %s %s' % (world_id, light['name']))
            actor.light_component.set_editor_property('intensity', light['intensity'])
            actor.light_component.set_editor_property('attenuation_radius', light['radius_m'] * 100)
            r, g, b = light['color']
            actor.light_component.set_editor_property('light_color', u.Color(r=r, g=g, b=b, a=255))
        spec['spawned_actor_count'] = len(owned)
        return owned, spec
    except BaseException:
        destroy_world(api, owned)
        raise
