"""UE Editor deterministic RGB-D capture. Invoke via run_street_experiment.py.

No assets/maps are saved. Sequencer time is fixed throughout each RGB/depth pair.
Only evaluator/ receives collision queries or actor transforms.
"""
import array
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import time
import traceback
import unreal as u

OUT = Path(os.environ['BA_UE_CAPTURE_OUTPUT'])
MODEL = OUT/'model'
TRUTH = OUT/'evaluator'
WIDTH, HEIGHT, FOV = 640, 360, 80.0
DT, DURATION = .2, 8.0
world = u.get_editor_subsystem(u.UnrealEditorSubsystem).get_editor_world()
actors = u.get_editor_subsystem(u.EditorActorSubsystem)
manifest = {'schema_version':'ue-street-rgbd-v1',
            'calibration':{'width':WIDTH,'height':HEIGHT,'horizontal_fov_degrees':FOV,'depth_max_m':100.0},
            'episodes':[]}
truth_rows = []
stage, index = 'setup', 0
started = time.monotonic()
next_action = started+12
captures = []
report = {'status':'RUNNING','capture_mode':'fixed-sequencer-time/editor-render-readback',
          'depth_encoding':'SceneDepth_R_float32_cm_div_100','dt_s':DT}


def save_json(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def npy(path, values):
    header = str({'descr':'<f4','fortran_order':False,'shape':(HEIGHT,WIDTH)})
    header += ' '*((64-(10+len(header)+1)%64)%64)+'\n'
    with path.open('wb') as f:
        f.write(b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header.encode('ascii'))
        data=array.array('f',values)
        if __import__('sys').byteorder!='little': data.byteswap()
        data.tofile(f)


def pose(x,y,z):
    return dict(x=x,y=y,z=z,pitch=0.0,yaw=0.0,roll=0.0)


def make_capture(source, fmt):
    a=actors.spawn_actor_from_class(u.SceneCapture2D,u.Vector(0,0,100000))
    c=a.capture_component2d
    c.capture_every_frame=False
    c.capture_on_movement=False
    c.always_persist_rendering_state=True
    c.fov_angle=FOV
    c.capture_source=source
    c.texture_target=u.RenderingLibrary.create_render_target2d(world,WIDTH,HEIGHT,fmt)
    captures.append(a)
    return c


def finish(error=None):
    u.unregister_slate_post_tick_callback(handle)
    report['status']='FAIL' if error else 'PASS'
    report['elapsed_s']=time.monotonic()-started
    if error: report['error']=error
    save_json(OUT/'capture.json',report)
    for a in captures:
        u.RenderingLibrary.release_render_target2d(a.capture_component2d.texture_target)
        actors.destroy_actor(a)
    # This helper owns its editor process, and never saves the modified scene.
    u.SystemLibrary.quit_editor()


def tick(delta):
    global stage,index,next_action,rgb,depth,calibration_box,episode,episode_index
    try:
        if time.monotonic()-started>900: raise RuntimeError('Capture timeout; partial output retained')
        if time.monotonic()<next_action: return
        if stage=='setup':
            MODEL.mkdir(parents=True)
            TRUTH.mkdir()
            rgb=make_capture(u.SceneCaptureSource.SCS_FINAL_COLOR_LDR,u.TextureRenderTargetFormat.RTF_RGBA8)
            depth=make_capture(u.SceneCaptureSource.SCS_SCENE_DEPTH,u.TextureRenderTargetFormat.RTF_RGBA32F)
            # An isolated, fronto-parallel 10m wall falsifies cm/m, device-Z and radial-depth errors.
            calibration_box=actors.spawn_actor_from_class(u.StaticMeshActor,u.Vector(1005,0,100000))
            calibration_box.static_mesh_component.set_static_mesh(u.load_asset('/Engine/BasicShapes/Cube'))
            calibration_box.set_actor_scale3d(u.Vector(.1,200,200))
            depth.capture_scene()
            stage='calibration'
            next_action=time.monotonic()+2
        elif stage=='calibration':
            depth.capture_scene()
            values=u.RenderingLibrary.read_render_target_raw(world,depth.texture_target,normalize=False)
            measured=[values[(HEIGHT//2)*WIDTH+x].r/100 for x in (WIDTH//4,WIDTH//2,3*WIDTH//4)]
            report['depth_calibration']={'expected_forward_m':10.0,'measured_center_and_offaxis_m':measured}
            assert all(abs(v-10)<.025 for v in measured),str(report['depth_calibration'])
            actors.destroy_actor(calibration_box)
            seq=u.load_asset('/Game/StreetLab/StreetActivity')
            assert u.LevelSequenceEditorBlueprintLibrary.open_level_sequence(seq)
            u.LevelSequenceEditorBlueprintLibrary.pause()
            episode_index=0
            stage='episode'
        elif stage=='episode':
            eid=('walk_a','walk_b')[episode_index]
            y=(-3.5,0.0)[episode_index]
            folder=MODEL/eid
            folder.mkdir()
            plan={'schema_version':'dtr-c1-plan-receipt-v1','coordinate_frame':'ANCHOR_FORWARD_RIGHT',
                  'plan_id':eid+'-straight','session_id':eid,'issued_at_s':0.0,'valid_from_s':0.0,
                  'expires_at_s':DURATION+4,'time_parameterized_waypoints':[
                      {'time_s':round(i*DT,4),'forward_m':round(i*DT*1.2,4),'right_m':0.0}
                      for i in range(int((DURATION+4)/DT)+1)]}
            plan['receipt_sha256']=hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest().upper()
            save_json(folder/'plan.json',plan)  # Before executing any trajectory.
            episode={'episode_id':eid,'route_frame':{'center_xy_m':[26.0,y],'z_origin_m':.27,
                     'forward_xy':[1,0],'right_xy':[0,1]},'plan_path':eid+'/plan.json','frames':[]}
            manifest['episodes'].append(episode)
            index=0
            stage='position'
        elif stage=='position':
            t=round(index*DT,4)
            u.LevelSequenceEditorBlueprintLibrary.set_current_time(round(t*30))
            xyz=u.Vector((26+1.2*t)*100,episode['route_frame']['center_xy_m'][1]*100,187)
            for a in captures: a.set_actor_location(xyz,False,False)
            # Warm temporal history at this exact scene time; no world/actor advancement.
            rgb.capture_scene()
            stage='read'
            next_action=time.monotonic()+.12
        elif stage=='read':
            t=round(index*DT,4)
            rgb.capture_scene()
            depth.capture_scene()
            values=u.RenderingLibrary.read_render_target_raw(world,depth.texture_target,normalize=False)
            stem=episode['episode_id']+'/'+str(index).zfill(4)
            npy(MODEL/(stem+'.npy'),(v.r/100 if math.isfinite(v.r) and 0<v.r<10000 else 0.0 for v in values))
            u.RenderingLibrary.export_render_target(world,rgb.texture_target,str((MODEL/stem).parent),Path(stem).name+'.png')
            assert (MODEL/(stem+'.png')).is_file(),'RGB export missing'
            x,y=26+1.2*t,episode['route_frame']['center_xy_m'][1]
            episode['frames'].append({'sample_index':index,'time_s':t,'rgb_path':stem+'.png',
                 'depth_path':stem+'.npy','camera_transform':pose(x,y,1.87),
                 'wearer_transform':pose(x,y,.27),'command_velocity':{'x':1.2,'y':0.,'z':0.}})
            # Actual engine collision query for a wearer capsule, with ground clearance 2cm.
            start=u.Vector(x*100,y*100,119)
            hit=u.SystemLibrary.capsule_trace_single(world,start,start+u.Vector(.01,0,0),30,90,
                u.TraceTypeQuery.TRACE_TYPE_QUERY1,False,captures,u.DrawDebugTrace.NONE)
            hit_fields=hit.to_tuple() if hit is not None else None
            hit_actor=hit_fields[9] if hit_fields else None
            persons=[]
            for a in actors.get_all_level_actors():
                if isinstance(a,u.SkeletalMeshActor):
                    p=a.get_actor_location()
                    persons.append({'actor':a.get_actor_label(),'x_m':p.x/100,'y_m':p.y/100,'z_m':p.z/100})
            truth_rows.append({'episode_id':episode['episode_id'],'sample_index':index,'time_s':t,
                'wearer_xy_m':[x,y],'capsule_radius_m':.3,'blocking_contact':hit is not None,
                'contact_actor':hit_actor.get_actor_label() if hit_actor else None,'actors':persons})
            index+=1
            if index>round(DURATION/DT):
                episode_index+=1
                stage='episode' if episode_index<2 else 'done'
            else: stage='position'
            if index%10==0: u.log('BA_RGBD_PROGRESS '+episode['episode_id']+' '+str(index))
        elif stage=='done':
            save_json(MODEL/'manifest.json',manifest)
            save_json(TRUTH/'frames.json',truth_rows)
            report['episodes']=len(manifest['episodes'])
            report['frames']=sum(len(e['frames']) for e in manifest['episodes'])
            report['blocking_contact_frames']=sum(r['blocking_contact'] for r in truth_rows)
            finish()
    except Exception:
        error=traceback.format_exc()
        u.log_error(error)
        finish(error)


handle=u.register_slate_post_tick_callback(tick)
