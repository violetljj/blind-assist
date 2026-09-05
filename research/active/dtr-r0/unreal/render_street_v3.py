"""Render repeatable Hero/Cafe/Walking views through actual UE SceneCapture2D.

Defaults to reloading V3 and quitting the task-owned editor. Development may
set BA_UE_V3_RENDER_MAP, BA_UE_V3_RENDER_OUTPUT and BA_UE_V3_RENDER_QUIT=0.
No map or asset is saved.
"""
import json
import hashlib
import os
from pathlib import Path
import time
import traceback
import unreal as u

OUT=Path(os.environ.get('BA_UE_V3_RENDER_OUTPUT',str(Path(u.Paths.project_dir())/'Saved/street-v3-views')))
OUT.mkdir(parents=True,exist_ok=True)
editor=u.get_editor_subsystem(u.UnrealEditorSubsystem)
api=u.get_editor_subsystem(u.EditorActorSubsystem)
assert u.get_editor_subsystem(u.LevelEditorSubsystem).load_level(os.environ.get('BA_UE_V3_RENDER_MAP','/Game/StreetLab/StreetLabV3'))
world=editor.get_editor_world()
views=[('Hero',(-300,-70,172),(-2,4,0),67),
       ('Cafe',(540,130,175),(-1,-132,0),68),
       ('Walking',(2800,-355,186),(-4,0,0),70)]
actor=api.spawn_actor_from_class(u.SceneCapture2D,u.Vector(*views[0][1]))
c=actor.capture_component2d
c.capture_every_frame=False;c.capture_on_movement=False;c.always_persist_rendering_state=True
c.capture_source=u.SceneCaptureSource.SCS_FINAL_COLOR_LDR
c.texture_target=u.RenderingLibrary.create_render_target2d(world,1920,1080,u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
c.texture_target.target_gamma=2.2
pp=max((a for a in api.get_all_level_actors() if isinstance(a,u.PostProcessVolume)),key=lambda a:a.priority)
settings=pp.settings
settings.override_dynamic_global_illumination_method=True
settings.dynamic_global_illumination_method=u.DynamicGlobalIlluminationMethod.LUMEN
settings.override_reflection_method=True;settings.reflection_method=u.ReflectionMethod.LUMEN
c.post_process_settings=settings;c.post_process_blend_weight=1.
index=0;frames=0;next_time=time.monotonic()+8
report={'status':'RUNNING','map':world.get_path_name(),'views':[],'pipeline':'1920x1080 FinalColorLDR sRGB gamma2.2; component Lumen GI/reflections; 48 accumulated frames'}
map_file=Path(u.Paths.project_dir())/'Content'/(world.get_path_name().split('.')[0].removeprefix('/Game/')+'.umap')
report['map_sha256']=hashlib.sha256(map_file.read_bytes()).hexdigest()


def finish(error=None):
    report['status']='FAIL' if error else 'PASS'
    if error:report['error']=error
    (OUT/'receipt.json').write_text(json.dumps(report,indent=2))
    u.unregister_slate_post_tick_callback(handle)
    u.RenderingLibrary.release_render_target2d(c.texture_target)
    api.destroy_actor(actor)
    if os.environ.get('BA_UE_V3_RENDER_QUIT','1')=='1':u.SystemLibrary.quit_editor()


def tick(delta):
    global index,frames,next_time
    try:
        if time.monotonic()<next_time:return
        name,xyz,rot,fov=views[index]
        if frames==0:
            actor.set_actor_location(u.Vector(*xyz),False,False)
            actor.set_actor_rotation(u.Rotator(pitch=rot[0],yaw=rot[1],roll=rot[2]),False)
            c.fov_angle=fov
        if frames<48:
            c.capture_scene();frames+=1;next_time=time.monotonic()+.08
            return
        u.RenderingLibrary.export_render_target(world,c.texture_target,str(OUT),name+'.png')
        report['views'].append({'name':name,'location_cm':xyz,'rotation_degrees':rot,'fov':fov,'png':name+'.png','actual_capture_location_cm':list(actor.get_actor_location().to_tuple())})
        index+=1;frames=0;next_time=time.monotonic()+1
        if index==len(views):finish()
    except Exception:finish(traceback.format_exc())


handle=u.register_slate_post_tick_callback(tick)
