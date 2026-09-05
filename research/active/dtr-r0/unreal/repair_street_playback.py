"""Repair the V2 demo's saved human materials, poses and Sequencer bindings.

Runs in a task-owned editor, preserves previous asset bytes in the QA directory,
saves only the demo successor, reloads it, captures a clothed pedestrian and
verifies automatic PIE translation and skeletal motion. Experiment data is untouched.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import unreal as u

ROOT=Path(u.Paths.project_dir())
OUT=Path(os.environ.get('BA_UE_PLAYBACK_QA_OUTPUT',str(ROOT/'Saved/playback-repair')))
OUT.mkdir(parents=True,exist_ok=True)
MAP='/Game/StreetLab/StreetLabV2'
SEQ='/Game/StreetLab/StreetActivityV3'
levels=u.get_editor_subsystem(u.LevelEditorSubsystem)
editor=u.get_editor_subsystem(u.UnrealEditorSubsystem)
api=u.get_editor_subsystem(u.EditorActorSubsystem)
assets=u.AssetToolsHelpers.get_asset_tools()
report={'status':'RUNNING','map':MAP,'sequence':SEQ,'before':{},'after':{},'evidence_role':'demo successor only'}
routes={
    'Alley crossing pedestrian':(0,[(0,(2150,-940,27)),(150,(2150,-940,27)),(530,(2150,580,27)),(899,(2150,580,27))]),
    'Approaching pedestrian':(90,[(0,(4000,-350,27)),(899,(400,-350,27))]),
    'Parallel pedestrian':(-90,[(0,(-500,380,27)),(899,(3100,380,27))]),
    'Plaza pedestrian':(90,[(0,(5800,420,27)),(899,(3500,420,27))]),
}


def disk_path(asset):
    return ROOT/'Content'/(asset.removeprefix('/Game/')+('.umap' if asset==MAP else '.uasset'))


def preserve(asset):
    path=disk_path(asset)
    if path.exists():
        report['before'][asset]=hashlib.sha256(path.read_bytes()).hexdigest()
        backup=OUT/'before'/path.relative_to(ROOT/'Content')
        backup.parent.mkdir(parents=True,exist_ok=True)
        if not backup.exists():shutil.copy2(path,backup)


def force_save(asset):
    assert u.EditorAssetLibrary.save_loaded_asset(asset,only_if_is_dirty=False)


def actor_state(a):
    comp=a.skeletal_mesh_component
    return {'location':list(a.get_actor_location().to_tuple()),
            'mesh':comp.skeletal_mesh_asset.get_path_name(),
            'materials':[comp.get_material(i).get_path_name() if comp.get_material(i) else None
                         for i in range(comp.get_num_materials())],
            'left_foot':list(comp.get_socket_transform('Bip01-L-Foot',u.RelativeTransformSpace.RTS_COMPONENT).translation.to_tuple()),
            'right_foot':list(comp.get_socket_transform('Bip01-R-Foot',u.RelativeTransformSpace.RTS_COMPONENT).translation.to_tuple())}


def repair():
    for name in ['lab-smoke.json','hero.png','cafe.png','crossing.png','overview.png']:
        source=ROOT/'Saved'/name
        backup=OUT/'before'/name
        if source.exists() and not backup.exists():
            backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,backup)
    assert levels.load_level(MAP)
    world=editor.get_editor_world()
    preserve(MAP)
    humans={a.get_actor_label():a for a in api.get_all_level_actors() if isinstance(a,u.SkeletalMeshActor)}
    report['actors_before']={name:actor_state(humans[name]) for name in routes}
    old=u.load_asset('/Game/StreetLab/StreetActivityV2')
    report['old_binding_resolution']={str(b.get_display_name()):[o.get_path_name() for o in
        u.MovieSceneSequenceExtensions.locate_bound_objects(old,b,world)] for b in old.get_bindings()}
    if u.EditorAssetLibrary.does_asset_exist(SEQ):
        raise RuntimeError('StreetActivityV3 already exists; preserve the completed repair')
    seq=assets.create_asset('StreetActivityV3','/Game/StreetLab',u.LevelSequence,u.LevelSequenceFactoryNew())
    seq.set_display_rate(u.FrameRate(30,1))
    seq.set_playback_start(0);seq.set_playback_end(900)
    for name,(yaw,points) in routes.items():
        actor=humans[name]
        comp=actor.skeletal_mesh_component
        mesh=comp.skeletal_mesh_asset
        package=mesh.get_path_name().split('.')[0]
        directory=package.rsplit('/',1)[0]
        preserve(package)
        slots=list(mesh.materials)
        comp.set_editor_property('override_materials',[])
        for i,slot in enumerate(slots):
            matpath=directory+'/'+str(slot.material_slot_name)
            material=u.load_asset(matpath)
            assert material,matpath
            preserve(matpath)
            u.MaterialEditingLibrary.set_material_usage(material,u.MaterialUsage.MATUSAGE_SKELETAL_MESH)
            u.MaterialEditingLibrary.recompile_material(material)
            force_save(material)
            slot.material_interface=material
            comp.set_material(i,material)
        mesh.materials=slots
        force_save(mesh)
        anim=u.load_asset(directory+'/NeutralWalk')
        assert anim
        # AnimationData is serialized; SetAnimation/PlayAnimation alone are transient.
        comp.set_position(.15,False)
        comp.override_animation_data(anim,True,False,.15,0.)
        comp.set_position(.15,False)
        comp.override_animation_data(anim,True,False,.15,0.)
        comp.set_update_animation_in_editor(True)
        actor.set_actor_location(u.Vector(*points[0][1]),False,False)
        actor.set_actor_rotation(u.Rotator(yaw=yaw),False)
        binding=seq.add_possessable(actor)
        section=binding.add_track(u.MovieScene3DTransformTrack).add_section()
        section.set_range(0,900)
        channels=section.get_all_channels()
        for channel,value in zip(channels,[*points[0][1],0,0,yaw,1,1,1]):channel.set_default(value)
        for frame,xyz in points:
            for channel,value in zip(channels[:3],xyz):
                channel.add_key(u.FrameNumber(frame),value,interpolation=u.MovieSceneKeyInterpolation.LINEAR)
        animation=binding.add_track(u.MovieSceneSkeletalAnimationTrack).add_section()
        animation.set_range(0,900)
        params=animation.get_editor_property('params');params.animation=anim
        animation.set_editor_property('params',params)
    for a in api.get_all_level_actors():
        if isinstance(a,u.LevelSequenceActor):
            a.set_sequence(seq)
            settings=a.get_editor_property('playback_settings')
            settings.auto_play=True
            settings.loop_count=u.MovieSceneSequenceLoopCount(-1)
            a.set_editor_property('playback_settings',settings)
    report['new_binding_resolution']={str(b.get_display_name()):[o.get_path_name() for o in
        u.MovieSceneSequenceExtensions.locate_bound_objects(seq,b,world)] for b in seq.get_bindings()}
    assert all(len(v)==1 for v in report['new_binding_resolution'].values())
    force_save(seq)
    assert levels.save_current_level()
    for asset in [*report['before'],SEQ]:
        report['after'][asset]=hashlib.sha256(disk_path(asset).read_bytes()).hexdigest()
    assert levels.load_level('/Game/StreetLab/StreetLab')
    assert levels.load_level(MAP)
    current={a.get_actor_label():a for a in api.get_all_level_actors() if isinstance(a,u.SkeletalMeshActor)}
    report['actors_reloaded']={name:actor_state(current[name]) for name in routes}
    assert all(all('/Humans/' in m for m in state['materials']) for state in report['actors_reloaded'].values())
    (OUT/'repair-state.json').write_text(json.dumps(report,indent=2))


def finish(error=None):
    report['status']='FAIL' if error else 'PASS'
    if error:report['error']=error
    (OUT/'receipt.json').write_text(json.dumps(report,indent=2))
    (ROOT/'Saved/lab-playback-repair.json').write_text(json.dumps(report,indent=2))
    u.unregister_slate_post_tick_callback(handle)
    levels.editor_request_end_play()
    u.SystemLibrary.quit_editor()


stage=0
next_time=time.monotonic()+12
shot=None


def tick(delta):
    global stage,next_time,shot
    try:
        if time.monotonic()<next_time:return
        if stage==0:
            # Loading/saving can pump Slate; do not re-enter this mutation stage.
            next_time=float('inf')
            repair()
            camera=api.spawn_actor_from_class(u.CameraActor,u.Vector(3550,-430,165),u.Rotator(pitch=-4,yaw=10))
            camera.camera_component.set_field_of_view(48)
            pp=max((a for a in api.get_all_level_actors() if isinstance(a,u.PostProcessVolume)),key=lambda a:a.priority)
            camera.camera_component.post_process_settings=pp.settings
            camera.camera_component.post_process_blend_weight=1.
            shot=u.AutomationLibrary.take_high_res_screenshot(1280,720,str(OUT/'clothed-reloaded.png'),camera=camera,delay=4.)
            stage=1;next_time=time.monotonic()+9
        elif stage==1:
            assert (OUT/'clothed-reloaded.png').exists(),'Reloaded human screenshot missing'
            stage=2;next_time=time.monotonic()+7
            levels.editor_request_begin_play()
        elif stage==2:
            game=editor.get_game_world()
            pawn=u.GameplayStatics.get_player_pawn(game,0)
            report['walk_speed_cm_s']=pawn.character_movement.max_walk_speed
            report['pie_first']={a.get_actor_label():actor_state(a) for a in
                u.GameplayStatics.get_all_actors_of_class(game,u.SkeletalMeshActor) if a.get_actor_label() in routes}
            stage=3;next_time=time.monotonic()+2.3
        else:
            game=editor.get_game_world()
            report['pie_second']={a.get_actor_label():actor_state(a) for a in
                u.GameplayStatics.get_all_actors_of_class(game,u.SkeletalMeshActor) if a.get_actor_label() in routes}
            for name in routes:
                first,second=report['pie_first'][name],report['pie_second'][name]
                distance=sum((a-b)**2 for a,b in zip(first['location'],second['location']))**.5
                assert distance>10,(name,distance)
                assert first['left_foot']!=second['left_foot'],(name,'skeletal pose did not animate')
            assert abs(report['walk_speed_cm_s']-120)<.1
            finish()
    except Exception:finish(traceback.format_exc())


handle=u.register_slate_post_tick_callback(tick)
