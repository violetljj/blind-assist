"""One editor, sequential authoring maps, three native views each, clean exit."""
import json
import os
from pathlib import Path
import traceback
from ue_capture_readiness import CaptureReadiness


def preview_sequence(u,plans,build_map,out):
    """build_map(plan) returns a receipt with three explicit camera views."""
    out=Path(out);api=u.get_editor_subsystem(u.EditorActorSubsystem)
    handle=[None];camera=[None];finished=[False];busy=[False]
    state=dict(map_index=0,view_index=0,warm=0,stage='BUILD')
    report=dict(status='RUNNING',scope='DEVELOPMENT_AUTHORING_NOT_DATA_COLLECTION',
        benchmark_eligible=False,formal_split_assignment='NONE_ENGINEERING_DEMOS_ONLY',expected_maps=len(plans),maps=[])
    ready=CaptureReadiness(u,timeout=300)

    def release_camera():
        if camera[0] is not None:
            u.RenderingLibrary.release_render_target2d(camera[0].capture_component2d.texture_target)
            if not api.destroy_actor(camera[0]):raise RuntimeError('Preview camera release failed')
            camera[0]=None

    def finish(error=None):
        if finished[0]:return
        finished[0]=True
        try:
            profile=u.BlindAssistCaptureLibrary.drain_rgb_writes()
            if profile.failed or profile.pending:raise RuntimeError('Preview PNG writes incomplete')
            release_camera()
        except Exception:error=(error or '')+traceback.format_exc()
        report.update(status='FAIL' if error else 'PASS_MULTI_MAP_AUTHORING_PREVIEW_ONLY',error=error)
        (out/'preview-receipt.json').write_text(json.dumps(report,indent=2)+'\n')
        if handle[0] is not None:u.unregister_slate_post_tick_callback(handle[0])
        u.SystemLibrary.quit_editor()

    def tick(delta):
        if busy[0] or finished[0]:return
        busy[0]=True
        try:
            if state['stage']=='BUILD':
                receipt=build_map(plans[state['map_index']])
                if len(receipt['views'])!=3:raise ValueError('Exactly overview plus two eye-height views required')
                report['maps'].append(dict(build=receipt,views=[]))
                camera[0]=api.spawn_actor_from_class(u.SceneCapture2D,u.Vector(0,0,170))
                c=camera[0].capture_component2d;c.capture_every_frame=False;c.capture_on_movement=False
                c.always_persist_rendering_state=True;c.fov_angle=70.
                world=u.get_editor_subsystem(u.UnrealEditorSubsystem).get_editor_world()
                c.capture_source=u.SceneCaptureSource.SCS_FINAL_COLOR_LDR
                # R1 actual entry views washed lit brick, sky and pale equipment
                # toward white. Keep scene materials/lights intact, reduce the
                # preview exposure by 1.5 stops and remove additive bloom glare.
                pp=c.post_process_settings
                preview_postprocess=dict(auto_exposure_bias=-1.5,bloom_intensity=0.,
                    lens_flare_intensity=0.,motion_blur_amount=0.)
                for name,value in preview_postprocess.items():
                    pp.set_editor_property('override_'+name,True)
                    pp.set_editor_property(name,value)
                c.post_process_settings=pp;c.post_process_blend_weight=1.
                report['maps'][-1]['preview_postprocess']=dict(preview_postprocess,
                    authority='R1_VISUAL_WASHOUT_CORRECTION_NOT_MATERIAL_RECOLORING')
                c.texture_target=u.RenderingLibrary.create_render_target2d(world,1280,720,u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
                c.texture_target.target_gamma=2.2
                state['stage']='PREPARE';return
            c=camera[0].capture_component2d
            world=u.get_editor_subsystem(u.UnrealEditorSubsystem).get_editor_world()
            current=report['maps'][-1];view=current['build']['views'][state['view_index']]
            if state['stage']=='PREPARE':
                camera[0].set_actor_location(u.Vector(*(v*100 for v in view['position_m'])),False,True)
                camera[0].set_actor_rotation(u.Rotator(pitch=view.get('pitch',0),yaw=view.get('yaw',0)),False)
                ready.begin(world,c);state.update(stage='WAIT',warm=0);return
            if not ready.poll():return
            c.capture_scene();state['warm']+=1
            if state['warm']<48:return
            path=out/(f'map-{state["map_index"]:02d}-'+view['name']+'.png')
            if not u.BlindAssistCaptureLibrary.export_rgb_png(world,c.texture_target,str(path)):
                raise RuntimeError('Native PNG export rejected')
            current['views'].append(dict(view,path=str(path),authority='ACTUAL_UE_RENDER_NOT_CONCEPT_IMAGE'))
            state['view_index']+=1
            if state['view_index']<3:state['stage']='PREPARE';return
            profile=u.BlindAssistCaptureLibrary.drain_rgb_writes()
            if profile.failed or profile.pending:raise RuntimeError('Map preview writes incomplete')
            release_camera();state.update(map_index=state['map_index']+1,view_index=0,stage='BUILD')
            if state['map_index']==len(plans):finish()
        except Exception:finish(traceback.format_exc())
        finally:busy[0]=False
    handle[0]=u.register_slate_post_tick_callback(tick)


def run(u):
    from cnh_street_alley_build import prepare
    out=Path(os.environ['BA_CNH_ALLEY_OUTPUT'])
    try:
        spec=json.loads(Path(os.environ['BA_CNH_ALLEY_SPEC']).read_text())
        plans,builder=prepare(u,spec,out)
        preview_sequence(u,plans,builder,out)
    except Exception:
        (out/'preview-receipt.json').write_text(json.dumps(dict(status='FAIL',error=traceback.format_exc()),indent=2))
        u.SystemLibrary.quit_editor()
