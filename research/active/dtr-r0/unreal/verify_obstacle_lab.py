"""Editor-only visual/PIE smoke check, invoked through -ExecCmds=py <script>."""
import json
from pathlib import Path
import time
import traceback
import unreal as u

root = Path(u.Paths.project_dir()) / 'Saved'
root.mkdir(exist_ok=True)
levels = u.get_editor_subsystem(u.LevelEditorSubsystem)
editor = u.get_editor_subsystem(u.UnrealEditorSubsystem)
actor_api = u.get_editor_subsystem(u.EditorActorSubsystem)
report = {'status':'RUNNING'}
u.log('BLINDASSIST_VISUAL_SMOKE_STARTED')
stage = 0
started = time.monotonic()
next_action = started + 15
task = None


def finish():
    u.unregister_slate_post_tick_callback(handle)
    levels.editor_request_end_play()
    (root / 'lab-smoke.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    u.SystemLibrary.quit_editor()


def tick(delta):
    global stage, next_action, task
    try:
        now = time.monotonic()
        if now - started > 240:
            raise RuntimeError('Visual smoke check exceeded 240 seconds')
        if now < next_action:
            return
        if stage == 0:
            cams = {a.get_actor_label():a for a in actor_api.get_all_level_actors()
                    if isinstance(a,u.CameraActor)}
            report['camera_count'] = len(cams)
            assert len(cams) == 7
            task = u.AutomationLibrary.take_high_res_screenshot(1600,1000,
                str(root/'overview.png'), camera=cams['Overview'])
            stage = 1
            next_action = now+5
        elif stage == 1 and task.is_task_done():
            cams = {a.get_actor_label():a for a in actor_api.get_all_level_actors()
                    if isinstance(a,u.CameraActor)}
            task = u.AutomationLibrary.take_high_res_screenshot(1280,720,
                str(root/'pedestrian.png'), camera=cams['Sensor_1_RGB_160cm'])
            stage = 2
            next_action = now+5
        elif stage == 2 and task.is_task_done():
            levels.editor_request_begin_play()
            stage = 3
            next_action = now+7
        elif stage == 3:
            game = editor.get_game_world()
            assert game, 'PIE world missing'
            pawn = u.GameplayStatics.get_player_pawn(game,0)
            assert pawn, 'First person pawn missing'
            report['pawn_class'] = pawn.get_class().get_name()
            report['walk_speed_cm_s'] = pawn.character_movement.max_walk_speed
            assert abs(report['walk_speed_cm_s']-120) < .1
            report['pawn_z_cm'] = pawn.get_actor_location().z
            dynamic = [a for a in u.GameplayStatics.get_all_actors_of_class(game,u.SkeletalMeshActor)
                       if a.get_actor_label() == 'Head on pedestrian']
            assert len(dynamic)==1
            report['dynamic_x_first_cm'] = dynamic[0].get_actor_location().x
            stage = 4
            next_action = now+2
        elif stage == 4:
            game = editor.get_game_world()
            actor = next(a for a in u.GameplayStatics.get_all_actors_of_class(game,u.SkeletalMeshActor)
                         if a.get_actor_label() == 'Head on pedestrian')
            report['dynamic_x_second_cm'] = actor.get_actor_location().x
            assert abs(report['dynamic_x_second_cm']-report['dynamic_x_first_cm']) > 10
            assert (root/'overview.png').is_file() and (root/'pedestrian.png').is_file()
            report['status'] = 'PASS'
            finish()
    except Exception:
        report['status'] = 'FAIL'
        report['error'] = traceback.format_exc()
        u.log_error(report['error'])
        finish()


handle = u.register_slate_post_tick_callback(tick)
