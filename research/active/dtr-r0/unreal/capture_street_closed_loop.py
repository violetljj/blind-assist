"""Run causal, lockstep sensor/action experiments in the actual UE street.

The sensor worker is a separate process and accepts only model/ and navigation.
Scenario geometry and continuous contact truth remain in this UE-side evaluator.
"""
import array
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback
import urllib.request
import unreal as u

sys.path.insert(0,str(Path(__file__).parent))
import street_scenarios as scenarios

OUT=Path(os.environ['BA_UE_LIVE_OUTPUT'])
MODEL=OUT/'model'
EVAL=OUT/'evaluator'
PORT=int(os.environ['BA_UE_LIVE_PORT'])
CASES=json.loads(os.environ.get('BA_UE_LIVE_CASES','[]'))
# The crossing actor approaches at 45 degrees. Keep its center inside the
# sensor frustum so this case tests geometric occlusion, not permanent off-FOV.
W,H,FOV=640,360,100.0
CAMERA_PITCH=-10.0
ANCHOR_X,ANCHOR_Y=26.0,0.0
MAX_TIME=14.0
world=u.get_editor_subsystem(u.UnrealEditorSubsystem).get_editor_world()
api=u.get_editor_subsystem(u.EditorActorSubsystem)
started=time.monotonic()
next_action=started+10
stage='setup'
captures=[]
spawned=[]
catalog=[s for s in scenarios.scenario_catalog() if not CASES or s['id'] in CASES]
jobs=[(s,arm) for s in catalog for arm in s['arms']]
job_index=0
report={'status':'RUNNING','mode':'LIVE_SENSOR_ACTION_LOCKSTEP','completed_episodes':0,
        'total_episodes':len(jobs),'engine_version':u.SystemLibrary.get_engine_version()}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def npy(path,values):
    header=str({'descr':'<f4','fortran_order':False,'shape':(H,W)})
    header+=' '*((64-(10+len(header)+1)%64)%64)+'\n'
    with path.open('wb') as f:
        f.write(b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header.encode())
        data=array.array('f',values)
        if sys.byteorder!='little': data.byteswap()
        data.tofile(f)


def make_capture(source,fmt):
    a=api.spawn_actor_from_class(u.SceneCapture2D,u.Vector(0,0,100000))
    c=a.capture_component2d
    c.capture_every_frame=False
    c.capture_on_movement=False
    c.always_persist_rendering_state=True
    c.fov_angle=FOV
    c.capture_source=source
    c.texture_target=u.RenderingLibrary.create_render_target2d(world,W,H,fmt)
    if source==u.SceneCaptureSource.SCS_FINAL_COLOR_LDR:
        c.texture_target.set_editor_property('target_gamma',2.2)
    pp=max((a for a in api.get_all_level_actors() if isinstance(a,u.PostProcessVolume)
            and a.get_editor_property('unbound')),
           key=lambda a:a.get_editor_property('priority'),default=None)
    if pp:
        c.post_process_settings=pp.settings
        c.post_process_blend_weight=1.0
    settings=c.post_process_settings
    settings.override_dynamic_global_illumination_method=True
    settings.dynamic_global_illumination_method=u.DynamicGlobalIlluminationMethod.LUMEN
    settings.override_reflection_method=True
    settings.reflection_method=u.ReflectionMethod.LUMEN
    c.post_process_settings=settings
    a.set_actor_rotation(u.Rotator(pitch=CAMERA_PITCH),False)
    captures.append(a)
    return c


def spawn_actor(spec):
    if spec['kind']=='pedestrian':
        a=api.spawn_actor_from_class(u.SkeletalMeshActor,u.Vector(0,0,0))
        comp=a.skeletal_mesh_component
        person=human_assets[0]
        comp.set_skeletal_mesh_asset(u.load_asset(person['mesh']))
        # OverrideAnimationData evaluates at delta=0 and refreshes bone transforms.
        # SetPosition alone only changes animation time; it can leave the A-pose
        # rendered. A stopped, zero-rate instance cannot advance during readback.
        comp.override_animation_data(u.load_asset(person['animation']),True,False,0.0,0.0)
        comp.set_update_animation_in_editor(True)
        # Exact authored mesh dimensions are recorded alongside the explicit body proxy.
        scale=175/person['native_height_cm']
        a.set_actor_scale3d(u.Vector(scale,scale,scale))
        a.tags=[u.Name('laboratory_pedestrian')]
    else:
        a=api.spawn_actor_from_class(u.StaticMeshActor,u.Vector(0,0,0))
        comp=a.static_mesh_component
        comp.set_static_mesh(u.load_asset('/Engine/BasicShapes/Cube'))
        hx,hy=spec['half_extents_m']
        a.set_actor_scale3d(u.Vector(2*hx,2*hy,spec['height_m']))
        matname={'occluder':'Brick','barrier':'Wood','low_obstacle':'Wood','tactile_ground':'Tactile'}.get(spec['kind'],'Limestone')
        mat=u.load_asset('/Game/StreetLab/Materials/'+matname)
        if mat: comp.set_material(0,mat)
    a.set_actor_label('Experiment '+spec['id'])
    # The explicit swept proxy is truth; visual assets cannot silently redefine it.
    comp.set_collision_profile_name('NoCollision')
    spawned.append(a)
    return a


def move_visuals(t):
    for s in scenarios.actors_at(spec,t):
        a=visuals[s['id']]
        z=floor+s['base_m']
        if s['kind']=='pedestrian':
            # Orient the human along their observable scripted travel direction.
            nxt=next(v for v in scenarios.actors_at(spec,t+.01) if v['id']==s['id'])
            dx,dy=nxt['x_m']-s['x_m'],nxt['y_m']-s['y_m']
            yaw=math.degrees(math.atan2(dy,dx)) if abs(dx)+abs(dy)>1e-6 else last_yaw.get(s['id'],180.0)
            last_yaw[s['id']]=yaw
            a.set_actor_rotation(u.Rotator(yaw=yaw-90),False)
            anim=u.load_asset(human_assets[0]['animation'])
            position=t%max(.01,anim.sequence_length) if abs(dx)+abs(dy)>1e-6 else 0.0
            # Same-mode OverrideAnimationData does not seek an existing instance.
            # Seek first, then force its synchronous zero-delta evaluation.
            a.skeletal_mesh_component.set_position(position,False)
            a.skeletal_mesh_component.override_animation_data(anim,True,False,position,0.0)
            # Keep the rendered mesh feet on measured ground regardless of FBX root origin.
            loc=a.get_actor_location()
            origin,extent=a.get_actor_bounds(False)
            bottom_offset=(origin.z-extent.z-loc.z)/100
            z-=bottom_offset
        else: z+=s['height_m']/2
        a.set_actor_location(u.Vector((ANCHOR_X+s['x_m'])*100,(ANCHOR_Y+s['y_m'])*100,z*100),False,False)


def plan_for(t,ego,command,issued):
    value={'schema_version':'dtr-c1-plan-receipt-v1','coordinate_frame':'ANCHOR_FORWARD_RIGHT',
           'plan_id':eid+'-'+str(index),'session_id':eid,'issued_at_s':issued,
           'valid_from_s':t,'expires_at_s':t+4.0,'time_parameterized_waypoints':[
               {'time_s':round(t+j*.2,5),'forward_m':ego['x_m']+command['vx_mps']*j*.2,
                'right_m':ego['y_m']+command['vy_mps']*j*.2} for j in range(21)]}
    value['receipt_sha256']=hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest().upper()
    relative=eid+'/plan-'+str(index).zfill(4)+'.json'
    write(MODEL/relative,value)
    return relative


def publish():
    # Only the current causal prefix is visible to the separate model process.
    write(MODEL/'manifest.json',{'schema_version':'ue-live-prefix-v1','calibration':{
        'width':W,'height':H,'horizontal_fov_degrees':FOV,'depth_max_m':100.0},'episodes':[model_episode]})


def finish(error=None):
    u.unregister_slate_post_tick_callback(handle)
    report['status']='FAIL' if error else 'COMPLETE'
    if error: report['error']=error
    report['elapsed_s']=time.monotonic()-started
    write(OUT/'run.json',report)
    for a in captures:
        u.RenderingLibrary.release_render_target2d(a.capture_component2d.texture_target)
    u.SystemLibrary.quit_editor()


def tick(delta):
    global stage,next_action,rgb,depth,floor,human_assets,job_index,spec,arm,eid,visuals,last_yaw
    global index,ego,applied,plan_path,model_episode,episode_result,contacts,warmup_left
    try:
        if time.monotonic()<next_action: return
        if stage=='setup':
            MODEL.mkdir(parents=True,exist_ok=True)
            EVAL.mkdir(parents=True,exist_ok=True)
            human_assets=json.loads((Path(u.Paths.project_dir())/'Saved/lab-visual-upgrade.json').read_text())['humans']
            assert human_assets,'Clothed human assets required'
            # Disable unrelated background motion; experiment actors are driven explicitly.
            for a in api.get_all_level_actors():
                if isinstance(a,(u.SkeletalMeshActor,u.LevelSequenceActor)): api.destroy_actor(a)
            hit=u.SystemLibrary.line_trace_single(world,u.Vector(ANCHOR_X*100,0,80),
                u.Vector(ANCHOR_X*100,0,-100),u.TraceTypeQuery.TRACE_TYPE_QUERY1,False,[],u.DrawDebugTrace.NONE)
            assert hit,'Measured walking surface missing'
            floor=hit.to_tuple()[5].z/100
            assert -.1<floor<.5,'Unexpected walking surface'
            report['surface_height_m']=floor
            report['proxy_checks']=scenarios.validate_catalog()
            assert report['proxy_checks']['passed']
            write(EVAL/'scenarios.json',catalog)
            rgb=make_capture(u.SceneCaptureSource.SCS_FINAL_COLOR_LDR,u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
            depth=make_capture(u.SceneCaptureSource.SCS_SCENE_DEPTH,u.TextureRenderTargetFormat.RTF_RGBA32F)
            stage='episode'
        elif stage=='episode':
            if job_index>=len(jobs): finish();return
            spec,arm=jobs[job_index]
            eid='episode_'+str(job_index).zfill(4)
            complete=EVAL/'episodes'/(eid+'.json')
            if complete.exists() and json.loads(complete.read_text()).get('completed'):
                job_index+=1
                report['completed_episodes']=job_index
                return
            for a in spawned: api.destroy_actor(a)
            spawned.clear()
            visuals={s['id']:spawn_actor(s) for s in spec['actors']}
            last_yaw={}
            checkpoint=EVAL/'checkpoints'/(eid+'.json')
            if checkpoint.exists():
                state=json.loads(checkpoint.read_text())
                index,ego,applied,plan_path=[state[k] for k in ('index','ego','applied','plan_path')]
                model_episode,episode_result=state['model_episode'],state['episode_result']
                contacts=state['contacts']
            else:
                index=0
                ego=dict(spec['ego_start'])
                applied={'vx_mps':spec['ego_speed_mps'],'vy_mps':0.0,'action':'NOMINAL'}
                plan_path=plan_for(0.0,ego,applied,0.0)
                model_episode={'episode_id':eid,'route_frame':{'center_xy_m':[ANCHOR_X,ANCHOR_Y],
                    'z_origin_m':floor,'forward_xy':[1,0],'right_xy':[0,1]},'plan_path':plan_path,'frames':[]}
                episode_result={'episode_id':eid,'scenario_id':spec['id'],'arm':arm,'frames':[],
                    'goal_forward_m':8.0,'goal_reached':False,'completed':False,'physical_actor_bounds':{}}
                contacts=[]
            stage='position'
        elif stage=='position':
            t=round(index*spec['dt_s'],5)
            move_visuals(t)
            for a in captures: a.set_actor_location(u.Vector((ANCHOR_X+ego['x_m'])*100,ego['y_m']*100,(floor+1.6)*100),False,False)
            if index==0:
                for name,a in visuals.items():
                    origin,extent=a.get_actor_bounds(False)
                    episode_result['physical_actor_bounds'][name]={'center_cm':list(origin.to_tuple()),'extent_cm':list(extent.to_tuple())}
            rgb.capture_scene()
            warmup_left=32 if index==0 else 0
            stage='warmup' if warmup_left else 'read'
            next_action=time.monotonic()+.10
        elif stage=='warmup':
            rgb.capture_scene()
            warmup_left-=1
            if warmup_left<=0: stage='read'
        elif stage=='read':
            t=round(index*spec['dt_s'],5)
            stem=eid+'/'+str(index).zfill(4)
            if not ((MODEL/(stem+'.png')).exists() and (MODEL/(stem+'.npy')).exists()):
                rgb.capture_scene();depth.capture_scene()
                values=u.RenderingLibrary.read_render_target_raw(world,depth.texture_target,normalize=False)
                npy(MODEL/(stem+'.npy'),(v.r/100 if math.isfinite(v.r) and 0<v.r<10000 else 0.0 for v in values))
                u.RenderingLibrary.export_render_target(world,rgb.texture_target,str((MODEL/stem).parent),Path(stem).name+'.png')
            pose={'x':ANCHOR_X+ego['x_m'],'y':ego['y_m'],'z':floor,'pitch':0.,'yaw':0.,'roll':0.}
            observation={'sample_index':index,'time_s':t,'rgb_path':stem+'.png','depth_path':stem+'.npy',
                'camera_transform':dict(pose,z=floor+1.6,pitch=CAMERA_PITCH),'wearer_transform':pose,
                'command_velocity':{'x':applied['vx_mps'],'y':applied['vy_mps'],'z':0.},'plan_path':plan_path}
            # A resumed pending step reuses the previous committed prefix, then appends once.
            model_episode['frames']=model_episode['frames'][:index]+[observation]
            publish()
            request=urllib.request.Request('http://127.0.0.1:'+str(PORT),data=json.dumps({
                'episode_id':eid,'sample_index':index,'goal_forward_m':8.0}).encode(),headers={'Content-Type':'application/json'})
            try:
                with urllib.request.urlopen(request,timeout=180) as response: result=json.load(response)
            except urllib.error.HTTPError as exc:
                raise RuntimeError(exc.read().decode()) from exc
            frame={'sample_index':index,'time_s':t,'ego':dict(ego),'contacts_since_previous':contacts,
                   'response':result,'applied_command':dict(applied),'rgb_path':stem+'.png','depth_path':stem+'.npy'}
            episode_result['frames']=episode_result['frames'][:index]+[frame]
            episode_result['goal_reached']=ego['x_m']>=8.0
            if t>=MAX_TIME or episode_result['goal_reached']:
                episode_result['completed']=True
                write(EVAL/'episodes'/(eid+'.json'),episode_result)
                write(MODEL/eid/'episode.json',model_episode)
                job_index+=1
                report['completed_episodes']=job_index
                write(OUT/'run.json',report)
                u.log('BA_LIVE_EPISODE_COMPLETE '+eid+' goal='+str(episode_result['goal_reached']))
                stage='episode'
            else:
                command=result['command'] if arm=='ASSISTED' else {'vx_mps':1.0,'vy_mps':0.0,'action':'NOMINAL'}
                prev=dict(ego)
                ego={'x_m':ego['x_m']+command['vx_mps']*spec['dt_s'],
                     'y_m':ego['y_m']+command['vy_mps']*spec['dt_s']}
                contacts=scenarios.contacts_for_step(spec,t,t+spec['dt_s'],prev,ego)
                index+=1
                applied=command
                plan_path=plan_for(t+spec['dt_s'],ego,command,t)
                write(EVAL/'checkpoints'/(eid+'.json'),{'index':index,'ego':ego,'applied':applied,
                    'plan_path':plan_path,'model_episode':model_episode,'episode_result':episode_result,'contacts':contacts})
                stage='position'
                if index%10==0:
                    report.update(active_episode=eid,active_frame=index,last_activity_utc=time.time())
                    write(OUT/'run.json',report)
                    u.log('BA_LIVE_PROGRESS '+eid+' '+str(index))
    except Exception:
        error=traceback.format_exc()
        u.log_error(error)
        finish(error)


handle=u.register_slate_post_tick_callback(tick)
