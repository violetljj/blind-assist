"""Render actual street views and check its first-person/dynamic runtime."""
import json
from pathlib import Path
import time
import traceback
import unreal as u

ROOT=Path(u.Paths.project_dir())/'Saved'
levels=u.get_editor_subsystem(u.LevelEditorSubsystem)
editor=u.get_editor_subsystem(u.UnrealEditorSubsystem)
actor_api=u.get_editor_subsystem(u.EditorActorSubsystem)
report={'status':'RUNNING','scene':'Willow Walk'}
names=['Hero','Cafe','Crossing','Overview']
stage=0
started=time.monotonic()
next_action=started+20
task=None
pending=None
shot_time=0
u.log('STREET_VISUAL_SMOKE_STARTED')


def finish():
    u.unregister_slate_post_tick_callback(handle)
    levels.editor_request_end_play()
    (ROOT/'lab-smoke.json').write_text(json.dumps(report,indent=2))
    u.SystemLibrary.quit_editor()


def tick(delta):
    global stage,next_action,task,pending,shot_time
    try:
        now=time.monotonic()
        if now-started>420: raise RuntimeError('Street render smoke exceeded 420 seconds')
        if now<next_action: return
        if stage<len(names):
            if pending and (not pending.exists() or pending.stat().st_mtime<shot_time): return
            cameras={a.get_actor_label():a for a in actor_api.get_all_level_actors() if isinstance(a,u.CameraActor)}
            name=names[stage]
            pending=ROOT/(name.lower()+'.png')
            shot_time=time.time()
            u.SystemLibrary.execute_console_command(editor.get_editor_world(),'r.HighResScreenshotDelay 64')
            task=u.AutomationLibrary.take_high_res_screenshot(1920,1080,str(pending),camera=cameras[name],delay=4.0)
            u.log('STREET_CAPTURE_REQUEST '+name)
            stage+=1
            next_action=now+4
        elif stage==len(names):
            if not pending.exists() or pending.stat().st_mtime<shot_time: return
            levels.editor_request_begin_play()
            stage+=1
            next_action=now+7
        elif stage==len(names)+1:
            game=editor.get_game_world()
            pawn=u.GameplayStatics.get_player_pawn(game,0)
            assert pawn,'First person pawn missing'
            report['pawn_class']=pawn.get_class().get_name()
            report['walk_speed_cm_s']=pawn.character_movement.max_walk_speed
            assert abs(report['walk_speed_cm_s']-120)<.1
            a=next(a for a in u.GameplayStatics.get_all_actors_of_class(game,u.SkeletalMeshActor)
                   if a.get_actor_label()=='Approaching pedestrian')
            report['dynamic_x_first_cm']=a.get_actor_location().x
            stage+=1
            next_action=now+2
        elif stage==len(names)+2:
            game=editor.get_game_world()
            a=next(a for a in u.GameplayStatics.get_all_actors_of_class(game,u.SkeletalMeshActor)
                   if a.get_actor_label()=='Approaching pedestrian')
            report['dynamic_x_second_cm']=a.get_actor_location().x
            assert abs(report['dynamic_x_second_cm']-report['dynamic_x_first_cm'])>10
            assert all((ROOT/(n.lower()+'.png')).is_file() for n in names)
            report['status']='PASS'
            finish()
    except Exception:
        report['status']='FAIL'
        report['error']=traceback.format_exc()
        u.log_error(report['error'])
        finish()


handle=u.register_slate_post_tick_callback(tick)
