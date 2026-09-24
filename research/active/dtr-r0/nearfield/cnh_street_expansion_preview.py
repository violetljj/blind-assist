"""Bounded native UE screenshots after expansion authoring; no dataset capture."""
import json
import os
from pathlib import Path
import traceback
import math
from cnh_street_expansion_build import build
from ue_capture_readiness import CaptureReadiness


def run(u):
    out=Path(os.environ['BA_CNH_EXPANSION_OUTPUT'])
    actor=None;handle=[None];finished=[False]
    api=u.get_editor_subsystem(u.EditorActorSubsystem)
    report=dict(status='RUNNING',benchmark_eligible=False,views=[])
    def finish(error=None):
        if finished[0]:return
        finished[0]=True
        try:
            profile=u.BlindAssistCaptureLibrary.drain_rgb_writes()
            if profile.failed or profile.pending:raise RuntimeError('Native PNG write incomplete')
            if actor is not None:
                u.RenderingLibrary.release_render_target2d(actor.capture_component2d.texture_target)
                if not api.destroy_actor(actor):raise RuntimeError('Preview camera not released')
        except Exception:
            error=(error or '')+traceback.format_exc()
        report.update(status='FAIL' if error else 'PASS_AUTHORING_PREVIEW_ONLY',error=error)
        (out/'preview-receipt.json').write_text(json.dumps(report,indent=2)+'\n')
        if handle[0] is not None:u.unregister_slate_post_tick_callback(handle[0])
        u.SystemLibrary.quit_editor()
    try:
        spec=json.loads(Path(os.environ['BA_CNH_EXPANSION_SPEC']).read_text())
        receipt=build(u,spec,out/'build')
        world=u.get_editor_subsystem(u.UnrealEditorSubsystem).get_editor_world()
        actor=api.spawn_actor_from_class(u.SceneCapture2D,u.Vector(0,0,170))
        c=actor.capture_component2d;c.capture_every_frame=False;c.capture_on_movement=False
        c.always_persist_rendering_state=True;c.fov_angle=70.
        c.capture_source=u.SceneCaptureSource.SCS_FINAL_COLOR_LDR
        c.texture_target=u.RenderingLibrary.create_render_target2d(world,1280,720,u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
        c.texture_target.target_gamma=2.2
        length=receipt['plan']['alley_length_m']
        # At yaw=90 the long world-X axis spans the wide image dimension.
        # Include entry, plaza, rear wall depth and a generous image border.
        extent_x=length+24.
        height=max(extent_x/(2*math.tan(math.radians(35))),20./(2*math.tan(math.radians(35))*720/1280))+10.
        views=[receipt['plan']['views'][0],dict(name='top_overview',position_m=[(length+6.)/2,0,height],yaw=90,pitch=-90)]
        readiness=CaptureReadiness(u,timeout=300)
        state=dict(index=0,warm=0,prepared=False)
        def tick(delta):
            try:
                view=views[state['index']]
                if not state['prepared']:
                    actor.set_actor_location(u.Vector(*(v*100 for v in view['position_m'])),False,True)
                    actor.set_actor_rotation(u.Rotator(pitch=view.get('pitch',0),yaw=view['yaw']),False)
                    readiness.begin(world,c);state['prepared']=True
                if not readiness.poll():return
                c.capture_scene();state['warm']+=1
                if state['warm']<48:return
                path=out/(view['name']+'.png')
                if not u.BlindAssistCaptureLibrary.export_rgb_png(world,c.texture_target,str(path)):
                    raise RuntimeError('Native preview export failed')
                report['views'].append(dict(view,path=str(path),authority='ACTUAL_UE_RENDER_NOT_GENERATED_CONCEPT'))
                state.update(index=state['index']+1,warm=0,prepared=False)
                if state['index']==len(views):finish()
            except Exception:finish(traceback.format_exc())
        handle[0]=u.register_slate_post_tick_callback(tick)
    except Exception:
        finish(traceback.format_exc())
