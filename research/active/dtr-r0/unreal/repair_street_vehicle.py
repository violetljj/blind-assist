"""Repair only StreetLabV2's modular Epic vehicle, preserving source assets.

Exec in the UE editor with StreetLabV2 loaded. Captures unchanged/body-only/
complete views, records actual imported rig wheel coordinates and mesh settings,
then saves the V2 map. Original StreetLab and /Game/Vehicles assets are untouched.
"""
import json
import math
import time
import traceback
from pathlib import Path
import unreal as u

ROOT = Path(u.Paths.project_dir())
OUT = ROOT.parent / 'vehicle-repair-20260905'
OUT.mkdir(exist_ok=True)
MAP = '/Game/StreetLab/StreetLabV2'
DEST = '/Game/StreetLab/VehicleV2'
actors = u.get_editor_subsystem(u.EditorActorSubsystem)
meshes = u.get_editor_subsystem(u.StaticMeshEditorSubsystem)
levels = u.get_editor_subsystem(u.LevelEditorSubsystem)
world = u.get_editor_subsystem(u.UnrealEditorSubsystem).get_editor_world()
assert world.get_path_name().split('.')[0] == MAP, 'Repair requires StreetLabV2'
car = next(a for a in actors.get_all_level_actors() if a.get_actor_label() == 'Epic parked vehicle')
component = car.static_mesh_component
original = component.static_mesh
report = {'status': 'RUNNING', 'map': MAP, 'original_mesh': original.get_path_name()}


def mesh_settings(mesh):
    settings = meshes.get_nanite_settings(mesh)
    return {'nanite_enabled': settings.enabled,
            'fallback_percent_triangles': settings.fallback_percent_triangles,
            'fallback_relative_error': settings.fallback_relative_error,
            'lods': [{'vertices': meshes.get_number_verts(mesh, i),
                      'reduction': str(meshes.get_lod_reduction_settings(mesh, i))}
                     for i in range(meshes.get_lod_count(mesh))],
            'bounds': {'origin': list(mesh.get_bounds().origin.to_tuple()),
                       'extent': list(mesh.get_bounds().box_extent.to_tuple())}}


def duplicate_full_mesh(source, name):
    destination = DEST + '/' + name
    mesh = u.load_asset(destination) if u.EditorAssetLibrary.does_asset_exist(destination) else u.EditorAssetLibrary.duplicate_asset(source.get_path_name().split('.')[0], destination)
    settings = meshes.get_nanite_settings(mesh)
    settings.enabled = False
    settings.fallback_percent_triangles = 1.0
    settings.fallback_relative_error = 0.0
    meshes.set_nanite_settings(mesh, settings, True)
    u.EditorAssetLibrary.save_loaded_asset(mesh)
    return mesh


report['before'] = mesh_settings(original)
report['actor_scale'] = list(car.get_actor_scale3d().to_tuple())
report['actor_location'] = list(car.get_actor_location().to_tuple())
(OUT/'inspection.json').write_text(json.dumps(report,indent=2))
camera = actors.spawn_actor_from_class(u.SceneCapture2D, u.Vector(3420,-115,155), u.Rotator(pitch=-6,yaw=165))
c = camera.capture_component2d
c.capture_every_frame = True
c.capture_on_movement = False
c.always_persist_rendering_state = True
c.fov_angle = 45
c.capture_source = u.SceneCaptureSource.SCS_FINAL_COLOR_LDR
c.texture_target = u.RenderingLibrary.create_render_target2d(world, 1100, 700, u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
c.texture_target.target_gamma = 2.2
pp = max((a for a in actors.get_all_level_actors() if isinstance(a,u.PostProcessVolume)), key=lambda a:a.priority)
c.post_process_settings = pp.settings
c.post_process_blend_weight = 1.0
stage = 0
busy = False
deadline = time.monotonic() + 10


def add_wheels():
    wheel = duplicate_full_mesh(u.load_asset('/Game/Vehicles/SportsCar/SM_SportsCar_Wheel'), 'SM_SportsCar_Wheel_Full')
    rig = actors.spawn_actor_from_class(u.SkeletalMeshActor, u.Vector(0,0,0), u.Rotator())
    rig.skeletal_mesh_component.set_skeletal_mesh_asset(u.load_asset('/Game/Vehicles/SportsCar/SKM_SportsCar'))
    try:
        locations = {name:list(rig.skeletal_mesh_component.get_socket_transform('Phys_Wheel_'+name,u.RelativeTransformSpace.RTS_COMPONENT).translation.to_tuple())
                     for name in ['FL','FR','BL','BR']}
    finally:
        actors.destroy_actor(rig)
    report['wheel_rig_local_cm'] = locations
    report['wheel_mesh'] = mesh_settings(wheel)
    # Remove only the old approximation and previous repair actors in V2.
    for actor in actors.get_all_level_actors():
        if actor.get_actor_label().startswith(('Vehicle tire', 'Wheel hub', 'Epic official wheel ')):
            actors.destroy_actor(actor)
    scale = car.get_actor_scale3d()
    yaw = car.get_actor_rotation().yaw
    angle = math.radians(yaw)
    origin = car.get_actor_location()
    extent = wheel.get_bounds().box_extent
    axle_rotation = 90 if extent.x < extent.y else 0
    report['wheel_actors'] = []
    for name, local in locations.items():
        x,y,z = local[0]*scale.x, local[1]*scale.y, local[2]*scale.z
        location = u.Vector(origin.x+x*math.cos(angle)-y*math.sin(angle), origin.y+x*math.sin(angle)+y*math.cos(angle), origin.z+z)
        rotation = u.Rotator(yaw=yaw+axle_rotation+(180 if local[1]<0 else 0))
        actor = actors.spawn_actor_from_class(u.StaticMeshActor,location,rotation)
        actor.set_actor_label('Epic official wheel '+name)
        actor.set_folder_path('Street/VehicleV2')
        actor.static_mesh_component.set_static_mesh(wheel)
        actor.static_mesh_component.set_forced_lod_model(1)
        actor.static_mesh_component.set_collision_profile_name('BlockAll')
        actor.set_actor_scale3d(scale)
        report['wheel_actors'].append({'name':name,'location':list(location.to_tuple()),'rotation':list(rotation.to_tuple())})


def finish(error=None):
    report['status'] = 'FAIL' if error else 'COMPLETE_REQUIRES_VISUAL_INSPECTION'
    if error: report['error'] = error
    (OUT/'receipt.json').write_text(json.dumps(report,indent=2))
    u.unregister_slate_post_tick_callback(handle)
    u.RenderingLibrary.release_render_target2d(c.texture_target)
    actors.destroy_actor(camera)
    if not error:
        levels.save_current_level()
        (ROOT/'Saved/lab-vehicle-repair.json').write_text(json.dumps(report,indent=2))
    u.SystemLibrary.quit_editor()


def tick(delta):
    global stage,deadline,busy
    if busy:
        return
    busy = True
    try:
        if time.monotonic()<deadline:
            c.capture_scene()
            return
        u.RenderingLibrary.export_render_target(world,c.texture_target,str(OUT),['before.png','body-only.png','after.png','side.png'][stage])
        if stage==0:
            copied = duplicate_full_mesh(original,'SM_SportsCar_Full')
            component.set_static_mesh(copied)
            component.set_forced_lod_model(1)
            report['after_body'] = mesh_settings(copied)
        elif stage==1:
            add_wheels()
        elif stage==2:
            camera.set_actor_location(u.Vector(2090,-580,130),False,False)
            camera.set_actor_rotation(u.Rotator(pitch=-4,yaw=85),False)
            c.fov_angle=55
        else:
            finish()
            return
        stage+=1
        deadline=time.monotonic()+8
    except Exception:
        finish(traceback.format_exc())
    finally:
        busy = False


handle = u.register_slate_post_tick_callback(tick)
