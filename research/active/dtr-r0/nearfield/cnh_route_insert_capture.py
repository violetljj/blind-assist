"""Actual-asset seven-pass/occlusion canary in a task-owned blank UE world.

This is an engineering fixture, never a realistic benchmark layout. Real source
assets retain their family identities; control floor/occluder are explicit cubes.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))


def engine():
    import unreal as u
    from cnh_route_capture import camera_record, write_json, write_npy
    from cnh_route_derived_assets import derive
    from ue_capture_readiness import CaptureReadiness
    from ue_pair_export import PairExporter

    out = Path(os.environ['BA_CNH_INSERT_OUTPUT'])
    write_json(out/'api-capabilities.json', dict(
        material_properties=[n for n in dir(u.MaterialProperty) if n.startswith('MP_')],
        expression_input=hasattr(u,'ExpressionInput'),
        set_material_attributes=hasattr(u,'MaterialExpressionSetMaterialAttributes')))
    spec = json.loads(Path(os.environ['BA_CNH_INSERT_SPEC']).read_text(encoding='utf-8-sig'))
    if spec.get('scene_layer') != 'INSERTED_ASSET_ENGINEERING_CANARY' or spec.get('benchmark_eligible') is not False:
        raise ValueError('Explicit engineering-only source required')
    if len(spec['assets']) != 2 or {x['id'] for x in spec['assets']} != {1,254}:
        raise ValueError('Two real assets with IDs 1 and 254 required')
    api = u.get_editor_subsystem(u.EditorActorSubsystem)
    world = u.EditorLoadingAndSavingUtils.new_blank_map(False)
    rig = dict(width=640, height=360, hfov_deg=100., baseline_m=.06)
    camera = dict(x=0., y=0., z=1.6, pitch=0., yaw=0., roll=0.)
    actors, captures, frames, derived = [], [], [], []
    state = dict(index=0, stage='PREPARE', warm=0, finished=False)
    handle = [None]
    started = time.monotonic()
    pairs = PairExporter(u, 'native_async', limit=4)
    readiness = CaptureReadiness(u, timeout=300)
    report = dict(status='RUNNING', scope='REAL_ASSET_INSERTION_UNIT_CANARY_NOT_BENCHMARK',
                  instance_method='ISOLATED_NATIVE_DEPTH_AGREEMENT_NOT_STENCIL',
                  source_unchanged=False, expected_frames=6,
                  engine_version=u.SystemLibrary.get_engine_version())

    def finish(error=None):
        if state['finished']:
            return
        state['finished'] = True
        try:
            report['pair_export'] = pairs.finish()
        except Exception:
            error = (error or '') + traceback.format_exc()
        cleanup = []
        for actor in captures:
            try:
                u.RenderingLibrary.release_render_target2d(actor.capture_component2d.texture_target)
            except Exception as exc:
                cleanup.append(str(exc))
        for actor in reversed(actors):
            try:
                if not api.destroy_actor(actor):
                    cleanup.append('Actor destroy failed')
            except Exception as exc:
                cleanup.append(str(exc))
        release = dict(released=not cleanup, errors=cleanup)
        write_json(out/'actor-release.json', release)
        report.update(status='FAIL' if error or cleanup else 'PASS_NATIVE_TRANSPORT',
                      actor_release=release, frames=len(frames), wall_s=time.monotonic()-started,
                      source_unchanged=len(derived)==2 and all(r['source_configuration_unchanged'] for _,r in derived))
        if error:
            report['error'] = error
        write_json(out/'raw-manifest.json', dict(rig=rig, frames=frames, derived_assets=[r for _,r in derived]))
        write_json(out/'engine-receipt.json', report)
        if handle[0] is not None:
            u.unregister_slate_post_tick_callback(handle[0])
        u.SystemLibrary.quit_editor()

    def actor_mesh(mesh, position, scale):
        actor = api.spawn_actor_from_class(u.StaticMeshActor, u.Vector(*[v*100 for v in position]))
        actors.append(actor)
        actor.static_mesh_component.set_static_mesh(mesh)
        actor.static_mesh_component.set_collision_profile_name('NoCollision')
        actor.static_mesh_component.set_editor_property('forced_lod_model', 1)
        actor.set_actor_scale3d(u.Vector(*scale))
        return actor

    def capture(name, source, fmt):
        offset = .06 if name.endswith('right') else 0.
        actor = api.spawn_actor_from_class(u.SceneCapture2D, u.Vector(0., offset*100,160.))
        actors.append(actor); captures.append(actor)
        c = actor.capture_component2d
        c.capture_every_frame=False; c.capture_on_movement=False; c.always_persist_rendering_state=True
        c.fov_angle=100.; c.capture_source=source
        c.texture_target=u.RenderingLibrary.create_render_target2d(world,640,360,fmt)
        c.texture_target.target_gamma=2.2 if name.startswith('rgb') else 1.
        pp=c.post_process_settings
        for key,value in [('override_auto_exposure_min_brightness',True),('override_auto_exposure_max_brightness',True),
            ('auto_exposure_min_brightness',1.),('auto_exposure_max_brightness',1.),
            ('override_motion_blur_amount',True),('motion_blur_amount',0.),
            ('override_bloom_intensity',True),('bloom_intensity',0.)]:
            setattr(pp,key,value)
        c.post_process_settings=pp
        return actor

    try:
        for cmd in ('r.AntiAliasingMethod 0','r.MotionBlurQuality 0'):
            u.SystemLibrary.execute_console_command(world,cmd)
        light=api.spawn_actor_from_class(u.DirectionalLight,u.Vector(0,0,500)); actors.append(light)
        light.set_actor_rotation(u.Rotator(pitch=-40,yaw=-25,roll=0),False)
        light.light_component.set_editor_property('intensity',3.)
        fill=api.spawn_actor_from_class(u.PointLight,u.Vector(0,-100,220)); actors.append(fill)
        fill.light_component.set_editor_property('intensity',5.)
        fill.light_component.set_editor_property('attenuation_radius',2000.)
        for asset in spec['assets']:
            mesh, receipt=derive(u,asset['mesh_asset'],'/Game/CNHInsertion_'+str(os.getpid()))
            derived.append((mesh,receipt))
            write_json(out/f'derived-{asset["id"]}.json',receipt)
        cube=u.load_asset('/Engine/BasicShapes/Cube')
        actor_mesh(cube,(0,0,-.1),(20,20,.2))
        occluder=actor_mesh(cube,(1.25,0,1.5),(.1,2.,3.5))
        target=actor_mesh(derived[0][0],(2.5,0,1.2),(1,1,1))
        rgb_left=capture('rgb_left',u.SceneCaptureSource.SCS_FINAL_COLOR_LDR,u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
        rgb_right=capture('rgb_right',u.SceneCaptureSource.SCS_FINAL_COLOR_LDR,u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
        depth_left=capture('depth_left',u.SceneCaptureSource.SCS_SCENE_DEPTH,u.TextureRenderTargetFormat.RTF_RGBA32F)
        depth_right=capture('depth_right',u.SceneCaptureSource.SCS_SCENE_DEPTH,u.TextureRenderTargetFormat.RTF_RGBA32F)
        normal=capture('normal_left',u.SceneCaptureSource.SCS_NORMAL,u.TextureRenderTargetFormat.RTF_RGBA32F)
        albedo=capture('albedo_left',u.SceneCaptureSource.SCS_BASE_COLOR,u.TextureRenderTargetFormat.RTF_RGBA32F)
        isolated=capture('isolated_left',u.SceneCaptureSource.SCS_SCENE_DEPTH,u.TextureRenderTargetFormat.RTF_RGBA32F)
        isolated.capture_component2d.primitive_render_mode=u.SceneCapturePrimitiveRenderMode.PRM_USE_SHOW_ONLY_LIST
        isolated.capture_component2d.show_only_component(target.static_mesh_component)
        cases=[(asset_index,kind) for asset_index in range(2) for kind in ('visible','occluded','partial')]

        def prepare():
            asset_index,kind=cases[state['index']]
            mesh,receipt=derived[asset_index]
            component=target.static_mesh_component
            component.set_static_mesh(mesh)
            # Preserve shape: uniform scaling and centre from source bounds.
            box=mesh.get_bounding_box()
            height=(box.max.z-box.min.z)/100
            if not math.isfinite(height) or height<=0:
                raise ValueError('Invalid real-asset height')
            gain=1.8/height
            target.set_actor_scale3d(u.Vector(gain,gain,gain))
            yaw=90. if box.max.x-box.min.x > box.max.y-box.min.y else 0.
            target.set_actor_rotation(u.Rotator(pitch=0,yaw=yaw,roll=0),False)
            target.set_actor_location(u.Vector(0,0,0),False,True)
            centre=u.Vector((box.min.x+box.max.x)/2,(box.min.y+box.max.y)/2,(box.min.z+box.max.z)/2)
            offset=u.MathLibrary.transform_location(target.get_actor_transform(),centre)
            target.set_actor_location(u.Vector(250-offset.x,-offset.y,120-offset.z),False,True)
            occluder.set_actor_hidden_in_game(kind=='visible')
            occluder.set_actor_scale3d(u.Vector(.1,1. if kind=='partial' else 2.,3.5))
            occluder.set_actor_location(u.Vector(125,50 if kind=='partial' else 0,150),False,True)
            name=f'asset-{spec["assets"][asset_index]["id"]}-{kind}'
            folder=out/'frames'/name;folder.mkdir(parents=True,exist_ok=False)
            state['folder']=folder
            write_json(folder/'camera.json',camera_record(camera,rig))
            transform=target.get_actor_transform()
            sections=[]
            for section in range(mesh.get_num_sections(0)):
                vertices,triangles,normals,uv,tangents=u.ProceduralMeshLibrary.get_section_from_static_mesh(mesh,0,section)
                if not vertices or not triangles:
                    raise ValueError('Empty derived render section')
                sections.append(dict(vertices_m=[[v.x/100,v.y/100,v.z/100] for v in vertices],triangles=list(triangles)))
            write_json(folder/'target-geometry.json',dict(sections=sections,mesh=mesh.get_path_name(),
                actual_translation_m=[target.get_actor_location().x/100,target.get_actor_location().y/100,target.get_actor_location().z/100],
                actual_rotation_quaternion=[float(getattr(transform.rotation,k)) for k in ('x','y','z','w')],
                actual_scale=[float(getattr(target.get_actor_scale3d(),k)) for k in ('x','y','z')],
                source_asset=receipt['source_mesh'],derived_nanite_enabled=False,material_wpo='EXPLICIT_ZERO',
                inserted_id=spec['assets'][asset_index]['id'],nir_rho=.5,nir_authority='ASSUMED_CANARY_ONLY'))
            readiness.begin(world,rgb_left.capture_component2d)

        def raw(actor,path):
            actor.capture_component2d.capture_scene()
            values=u.RenderingLibrary.read_render_target_raw(world,actor.capture_component2d.texture_target,normalize=False)
            if len(values)!=640*360:
                raise ValueError('Native attribute dimensions')
            write_npy(path,(v for p in values for v in (p.r,p.g,p.b)),(360,640,3))

        def tick(delta):
            try:
                if state['finished']:
                    return
                if not pairs.ready():
                    return
                if state['stage']=='PREPARE':
                    prepare();state.update(stage='READY',warm=0);return
                if state['stage']=='READY':
                    if not readiness.poll():
                        return
                    state['stage']='SETTLE'
                rgb_left.capture_component2d.capture_scene();rgb_right.capture_component2d.capture_scene()
                state['warm']+=1
                if state['warm']<32:
                    return
                folder=state['folder']
                for side,rgb,depth in (('left',rgb_left,depth_left),('right',rgb_right,depth_right)):
                    depth.capture_component2d.capture_scene()
                    pairs.export(world,rgb.capture_component2d.texture_target,depth.capture_component2d.texture_target,
                                 folder/(side+'.png'),folder/('depth_'+side+'.transport.npy'),state['index']*2+int(side=='right'))
                raw(normal,folder/'normal_left.transport.npy');raw(albedo,folder/'albedo_left.transport.npy')
                isolated.capture_component2d.capture_scene()
                if not u.BlindAssistCaptureLibrary.export_depth_npy(world,isolated.capture_component2d.texture_target,str(folder/'isolated_depth.transport.npy')):
                    raise RuntimeError('Isolated native depth export failed')
                asset_index,kind=cases[state['index']]
                frames.append(dict(folder=folder.relative_to(out).as_posix(),id=folder.name,
                                   asset_id=spec['assets'][asset_index]['id'],occlusion=kind))
                state.update(index=state['index']+1,stage='PREPARE')
                write_json(out/'progress.json',dict(frames=len(frames),expected_frames=6,wall_s=time.monotonic()-started))
                if len(frames)==6:
                    finish()
            except Exception:
                finish(traceback.format_exc())

        handle[0]=u.register_slate_post_tick_callback(tick)
    except Exception:
        finish(traceback.format_exc())


if __name__ == '__main__':
    try:
        engine()
    except Exception:
        out=Path(os.environ['BA_CNH_INSERT_OUTPUT'])
        (out/'startup-failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        import unreal
        unreal.SystemLibrary.quit_editor()
