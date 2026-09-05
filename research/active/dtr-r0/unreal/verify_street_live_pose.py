"""Verify live human animation with actual UE renders and deterministic bones.

Execute in a task-owned UE editor on /Game/StreetLab/StreetLabV2. Output is
BA_UE_POSE_QA_OUTPUT or project Saved/live-pose-qa. This test samples the live
capture implementation at 0.15, 0.45, then 0.15 seconds, holds each pose for four
wall-clock seconds, and exports three actual SceneCapture2D PNGs. Component-space
bones must change between sampled times and remain identical during holds and
repeat sampling. Inspect the PNGs to verify framing, clothing, pose and feet.

The test modifies only the transient level, never saves, and exits its editor.
"""
import ast, json, math, os, time, traceback
from pathlib import Path
from types import SimpleNamespace
import unreal as u
OUT=Path(os.environ.get('BA_UE_POSE_QA_OUTPUT',str(Path(u.Paths.project_dir())/'Saved/live-pose-qa')))
OUT.mkdir(parents=True,exist_ok=True)
source=Path(__file__).with_name('capture_street_closed_loop.py')
api=u.get_editor_subsystem(u.EditorActorSubsystem)
world=u.get_editor_subsystem(u.UnrealEditorSubsystem).get_editor_world()
human_assets=json.loads((Path(u.Paths.project_dir())/'Saved/lab-visual-upgrade.json').read_text())['humans']
spawned=[]
ANCHOR_X,ANCHOR_Y=26.,0.
floor=.12
last_yaw={}
spec={}
scenarios=SimpleNamespace(actors_at=lambda spec,t:[{'id':'qa','kind':'pedestrian','base_m':0,'x_m':4+t,'y_m':0}])
module=ast.parse(source.read_text())
functions=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ('spawn_actor','move_visuals')]
exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),globals())
for a in api.get_all_level_actors():
    if isinstance(a,(u.SkeletalMeshActor,u.LevelSequenceActor)):api.destroy_actor(a)
p=spawn_actor({'id':'qa','kind':'pedestrian'})
visuals={'qa':p}
comp=p.skeletal_mesh_component
names=[str(comp.get_bone_name(i)) for i in range(comp.get_num_bones())]
selected=[n for n in names if any(k in n.lower() for k in ['foot','toe','hand','calf'])]
camera=api.spawn_actor_from_class(u.SceneCapture2D,u.Vector(3420,-115,155),u.Rotator(pitch=-6,yaw=165))
c=camera.capture_component2d
c.capture_every_frame=True
c.capture_on_movement=False
c.always_persist_rendering_state=True
c.fov_angle=45
c.capture_source=u.SceneCaptureSource.SCS_FINAL_COLOR_LDR
c.texture_target=u.RenderingLibrary.create_render_target2d(world,800,900,u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
c.texture_target.target_gamma=2.2
pp=max((a for a in api.get_all_level_actors() if isinstance(a,u.PostProcessVolume)),key=lambda a:a.priority)
s=pp.settings
s.override_dynamic_global_illumination_method=True;s.dynamic_global_illumination_method=u.DynamicGlobalIlluminationMethod.LUMEN
s.override_reflection_method=True;s.reflection_method=u.ReflectionMethod.LUMEN
c.post_process_settings=s;c.post_process_blend_weight=1.
poses=[.15,.45,.15]
index=0
stage='pose'
next_time=time.monotonic()+8
report={'status':'RUNNING','bones':names,'frames':[],'method':'actual live spawn_actor/move_visuals from source AST'}
def snapshot():
    return {'animation_position':comp.get_position(),'playing':comp.is_playing(), 'play_rate':comp.get_play_rate(),
        'bones':{n:list(comp.get_socket_transform(n,u.RelativeTransformSpace.RTS_COMPONENT).translation.to_tuple()) for n in selected},
        'location':list(p.get_actor_location().to_tuple()),'yaw':p.get_actor_rotation().yaw}
def finish(error=None):
    report['status']='FAIL' if error else 'PASS'
    if error:report['error']=error
    (OUT/'receipt.json').write_text(json.dumps(report,indent=2))
    u.unregister_slate_post_tick_callback(handle)
    u.RenderingLibrary.release_render_target2d(c.texture_target)
    u.SystemLibrary.quit_editor()
def tick(delta):
    global index,stage,next_time
    try:
        if time.monotonic()<next_time:
            c.capture_scene();return
        if stage=='pose':
            move_visuals(poses[index])
            report['frames'].append({'time':poses[index],'before':snapshot()})
            stage='capture';next_time=time.monotonic()+4
        else:
            report['frames'][-1]['after']=snapshot()
            u.RenderingLibrary.export_render_target(world,c.texture_target,str(OUT),f'pose-{index}.png')
            assert report['frames'][-1]['before']==report['frames'][-1]['after'],'Wall clock pose drift'
            index+=1
            if index==len(poses):
                assert report['frames'][0]['before']==report['frames'][2]['before'],'Resampling same time changed pose'
                assert report['frames'][0]['before']['bones']!=report['frames'][1]['before']['bones'],'No pose change'; assert abs(report['frames'][0]['before']['animation_position']-.15)<.001,'Animation seek failed'
                finish();return
            stage='pose';next_time=time.monotonic()+.1
    except Exception:finish(traceback.format_exc())
handle=u.register_slate_post_tick_callback(tick)
