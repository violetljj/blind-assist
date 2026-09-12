"""Single rendered RGB camera plus native-collision ToF and hypothetical radar/IMU.

ToF is radial metres; radar uses a hypothetical horizontal-range proxy.
No depth image or second camera is produced. No RF/hardware validation.
"""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import sys
import time
import traceback


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def basis(camera):
    p,y,r=(math.radians(camera.get(k,0.)) for k in ('pitch','yaw','roll'))
    cp,sp,cy,sy,cr,sr=math.cos(p),math.sin(p),math.cos(y),math.sin(y),math.cos(r),math.sin(r)
    return ((cp*cy,cp*sy,sp),(sr*sp*cy-cr*sy,sr*sp*sy+cr*cy,-sr*cp),(-cr*sp*cy-sr*sy,-cr*sp*sy+sr*cy,cr*cp))


def npy(path,values,shape,dtype='<f4'):
    header=str(dict(descr=dtype,fortran_order=False,shape=shape))
    header+=' '*((64-(10+len(header)+1)%64)%64)+'\n'
    payload=array.array('B' if dtype=='|b1' else 'f',values)
    with Path(path).open('xb') as f:
        f.write(b'\x93NUMPY\x01\x00'+struct.pack('<H',len(header))+header.encode());payload.tofile(f)


def engine():
    import unreal as u
    from ue_capture_readiness import CaptureReadiness
    from mz107_sensors import sensors
    out=Path(os.environ['BA_MZ107_OUT']);specpath=Path(os.environ['BA_MZ107_SPEC']);spec=json.loads(specpath.read_text())
    rig=spec['rig'];w,h=rig['width'],rig['height']
    api=u.get_editor_subsystem(u.EditorActorSubsystem);world=u.EditorLoadingAndSavingUtils.new_blank_map(False)
    mesh=u.load_asset('/Engine/BasicShapes/Cube');parent=u.load_asset('/Engine/EngineDebugMaterials/LevelColorationUnlitMaterial')
    assert mesh and parent
    actors=[];targets=[];captures=[];manifest=[];hashes={}
    report=dict(status='RUNNING',authority='SINGLE_RGB_NATIVE_COLLISION_TOF_HYPOTHETICAL_RADAR_IMU_NOT_HARDWARE_OR_RF',source_sha256=sha(__file__),spec_sha256=sha(specpath),engine_version=u.SystemLibrary.get_engine_version(),source_hashes={name:sha(Path(__file__).with_name(name)) for name in ('mz107_sensors.py','mz99_angle_information_capture.py','ue_capture_readiness.py')})
    started=time.monotonic();state=dict(i=0,warm=0,prepared=False,after=time.monotonic()+2)
    readiness=CaptureReadiness(u,timeout=300)

    def cube(center,size,color=.5,material=None,collision=True,owned=None):
        a=api.spawn_actor_from_class(u.StaticMeshActor,u.Vector(*(v*100 for v in center)));actors.append(a)
        if owned is not None:owned.append(a)
        comp=a.static_mesh_component;comp.set_static_mesh(mesh);comp.set_collision_profile_name('BlockAll' if collision else 'NoCollision')
        a.set_actor_scale3d(u.Vector(*size))
        if material:comp.set_material(0,u.load_asset(material))
        else:
            inst=comp.create_dynamic_material_instance(0,parent);inst.set_vector_parameter_value('Color',u.LinearColor(color,color,color,1.))
        return a

    def textured(obj,owned):
        center,size=obj['center_m'],obj['size_m'];base=cube(center,size,.18 if owned is None else .65,obj.get('material'),owned=owned)
        if obj.get('material') or obj.get('texture',True) is False:return base
        # Distinct unlit albedo patches, 1mm ahead of native face; no collision.
        rng=random.Random(obj.get('texture_seed',101));ny,nz=obj.get('texture_grid',[8,8])
        for iy in range(ny):
            for iz in range(nz):
                c=[center[0]-size[0]/2-.001,center[1]+size[1]*((iy+.5)/ny-.5),center[2]+size[2]*((iz+.5)/nz-.5)]
                cube(c,[.001,size[1]/ny*.98,size[2]/nz*.98],rng.uniform(.55,.8),collision=False,owned=owned)
        return base

    # Controlled native scene boundaries; source geometry is explicit in receipt.
    background=spec.get('background',dict(center_m=[10.,0.,1.5],size_m=[.1,16.,8.],texture=False))
    if background:textured(background,None)
    floor=spec.get('floor',dict(center_m=[4.,0.,-.05],size_m=[24.,20.,.1]))
    if floor:cube(floor['center_m'],floor['size_m'],.22)
    report.update(background=background,floor=floor,texture_geometry='unlit albedo tiles 1mm before front face; no collision')
    def capture(source,fmt):
        a=api.spawn_actor_from_class(u.SceneCapture2D,u.Vector(0,0,0));actors.append(a);captures.append(a)
        c=a.capture_component2d;c.capture_every_frame=False;c.capture_on_movement=False;c.always_persist_rendering_state=True;c.fov_angle=rig['hfov_deg'];c.capture_source=source
        c.texture_target=u.RenderingLibrary.create_render_target2d(world,w,h,fmt)
        if source==u.SceneCaptureSource.SCS_FINAL_COLOR_LDR:c.texture_target.target_gamma=2.2
        pp=c.post_process_settings
        for key,value in [('override_auto_exposure_min_brightness',True),('override_auto_exposure_max_brightness',True),('auto_exposure_min_brightness',1.),('auto_exposure_max_brightness',1.),('override_motion_blur_amount',True),('motion_blur_amount',0.)]:setattr(pp,key,value)
        c.post_process_settings=pp
        return a
    left=capture(u.SceneCaptureSource.SCS_FINAL_COLOR_LDR,u.TextureRenderTargetFormat.RTF_RGBA8_SRGB)
    rawrows=[]; evalrows=[]; provrows=[]; sensor_state={}

    def finish(error=None):
        report.update(status='FAIL' if error else 'PASS',frames=len(manifest),seconds=time.monotonic()-started,hashes=hashes)
        if error:report['error']=error
        for name,rows in [('raw.jsonl',rawrows),('evaluator.jsonl',evalrows),('provenance.jsonl',provrows)]:
            (out/name).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows));hashes[name]=sha(out/name)
        write(out/'manifest.json',dict(rig=rig,rgb_camera_count=1,depth_images_produced=False,frames=manifest))
        hashes['manifest.json']=sha(out/'manifest.json')
        write(out/'receipt.json',report)
        for a in captures:u.RenderingLibrary.release_render_target2d(a.capture_component2d.texture_target)
        for a in reversed(actors):api.destroy_actor(a)
        u.unregister_slate_post_tick_callback(handle);u.SystemLibrary.quit_editor()

    def tick(delta):
        if time.monotonic()<state['after']:return
        try:
            frame=spec['frames'][state['i']];cam=frame['camera']
            if not state['prepared']:
                for a in targets:api.destroy_actor(a);actors.remove(a)
                targets.clear()
                appearance=frame.get('appearance','textured')
                if appearance not in ('textured','flat'):raise ValueError('Only textured and flat appearance supported')
                state['native_targets']=[textured(dict(obj,texture=appearance=='textured' and obj.get('texture',True)),targets) for obj in frame.get('objects',[])]
                f,r,v=basis(cam);pos=[cam[k] for k in ('x','y','z')]
                for a,offset in ((left,0),):
                    a.set_actor_location(u.Vector(*((pos[k]+offset*r[k])*100 for k in range(3))),False,True)
                    a.set_actor_rotation(u.Rotator(pitch=cam.get('pitch',0),yaw=cam.get('yaw',0),roll=cam.get('roll',0)),False)
                left.capture_component2d.capture_scene()
                readiness.begin(world,left.capture_component2d);state['prepared']=True;return
            if readiness.state!='READY':
                if not readiness.poll():return
            left.capture_component2d.capture_scene();state['warm']+=1
            if state['warm']<24:return
            folder=out/'frame'/frame['id'];folder.mkdir(parents=True,exist_ok=False)
            u.RenderingLibrary.export_render_target(world,left.capture_component2d.texture_target,str(folder),'rgb.png')
            row,evaluation,provenance=sensors(u,world,frame,state['native_targets'],sensor_state)
            row['rgb_path']='frame/'+frame['id']+'/rgb.png'
            row['rgb_intrinsics']=dict(width=w,height=h,fx=w/(2*math.tan(math.radians(rig['hfov_deg']/2))),fy=w/(2*math.tan(math.radians(rig['hfov_deg']/2))),cx=w/2,cy=h/2)
            rawrows.append(row);evalrows.append(evaluation);provrows.append(provenance)
            paths=dict(rgb=row['rgb_path']);hashes[row['rgb_path']]=sha(folder/'rgb.png')
            manifest.append(dict(id=frame['id'],episode=frame['episode'],time_s=frame['time_s'],paths=paths))
            state.update(i=state['i']+1,warm=0,prepared=False)
            write(out/'progress.json',dict(frames=len(manifest),seconds=time.monotonic()-started))
            if state['i']==len(spec['frames']):finish()
        except Exception:finish(traceback.format_exc())
    handle=u.register_slate_post_tick_callback(tick)


def main():
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[4];art=(root/'artifacts.local').resolve();out=a.output.resolve()
    if not out.is_relative_to(art) or out.exists():raise ValueError('Fresh canonical artifact output required')
    spec=json.loads(a.spec.read_text());assert 0<len(spec['frames'])<=300
    assert len({f['id'] for f in spec['frames']})==len(spec['frames'])
    assert all('/' not in f['id'] and '\\' not in f['id'] and f['id'] not in ('.','..') for f in spec['frames'])
    out.mkdir(parents=True);shutil.copyfile(__file__,out/Path(__file__).name);shutil.copyfile(a.spec,out/'spec.json')
    helper=Path(__file__).with_name('ue_capture_readiness.py');shutil.copyfile(helper,out/helper.name)
    for name in ('mz107_sensors.py','mz99_angle_information_capture.py'):
        shutil.copyfile(Path(__file__).with_name(name),out/name)
    sys.path.insert(0,str(root/'tools'));from ue_native_capture import run_owned
    from run_city_pcg_capture import cache_service_port
    import psutil
    if any((p.info['name'] or '').startswith('UnrealEditor') for p in psutil.process_iter(['name'])):raise RuntimeError('Editor occupied')
    cache=art/'work/mz107-rgb-tof-radar-imu-20260913/ddc';cache.mkdir(parents=True,exist_ok=True);port=cache_service_port(cache)
    env=dict(os.environ,BA_MZ107_OUT=str(out),BA_MZ107_SPEC=str(out/'spec.json'));env['UE-LocalDataCachePath']=str(cache)
    plugin=art/'ue-hlod-b2/package/BlindAssistCapture.uplugin'
    cmd=['F:/epic/UE_5.8/Engine/Binaries/Win64/UnrealEditor.exe',str(art/'unreal/BlindAssistStreetLab/BlindAssistStreetLab.uproject'),'-RenderOffscreen','-unattended','-nosound','-nop4','-NoSplash','-ddc=NoShared','-ini:Engine:[Zen.AutoLaunch]:DesiredPort='+str(port),'-PLUGIN='+str(plugin),'-EnablePlugins=PythonScriptPlugin,BlindAssistCapture','-ExecCmds=py '+(out/Path(__file__).name).as_posix(),'-abslog='+str(out/'editor.log'),'-ini:Engine:[/Script/EngineSettings.GameMapsSettings]:EditorStartupMap=','-ini:EditorPerProjectUserSettings:[/Script/UnrealEd.EditorLoadingSavingSettings]:LoadLevelAtStartup=None']
    write(out/'launch.json',dict(command=cmd,spec_sha256=sha(a.spec),source_sha256=sha(__file__),helper_sha256=sha(helper),plugin_binary_sha256=sha(plugin.parent/'Binaries/Win64/UnrealEditor-BlindAssistCapture.dll'),persistent_cache=dict(path=str(cache),port=port)))
    run_owned(cmd,env,out,600)
    receipt=json.loads((out/'receipt.json').read_text());assert receipt['status']=='PASS',receipt
    assert receipt['frames']==len(spec['frames'])
    for name,digest in receipt['hashes'].items():assert sha(out/name)==digest
    print(json.dumps(receipt))


if os.environ.get('BA_MZ107_SPEC'):
    sys.path.insert(0,str(Path(__file__).parent));engine()
elif __name__=='__main__':main()
