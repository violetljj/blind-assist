"""Fresh native UE collision observations; hypothetical sensors, not RF simulation.

Host: prepare --output PATH; capture --spec PATH --output PATH [--episodes 2].
The engine entry is selected solely by BA_MZ96_SPEC/BA_MZ96_OUT environment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

AUTHORITY = 'UE_NATIVE_COLLISION_GEOMETRY_PLUS_HYPOTHETICAL_RADAR_NOT_RF'
DT = .1
FRAMES = 40


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(seed=96012):
    rng = random.Random(seed)
    order = list(range(128)); rng.shuffle(order)
    split = {k:('train' if j < 80 else 'validation' if j < 96 else 'test') for j,k in enumerate(order)}
    scenes = []
    families = ('static', 'crossing', 'receding', 'outside', 'weakthin', 'multiclutter', 'approach', 'clear')
    for i in range(128):
        family = families[i % len(families)]
        objects = []
        for k in range(0 if family == 'clear' else (3 if family == 'multiclutter' else rng.randint(1,2))):
            x, z = rng.uniform(-1.1,1.1), rng.uniform(2.8,3.8)
            vx, vz = 0., 0.
            if family == 'crossing':
                x = rng.choice([-1,1])*rng.uniform(.7,1.6); vx = -math.copysign(rng.uniform(.3,.9),x)
            elif family == 'receding':
                z = rng.uniform(.8,2.8); vz = rng.uniform(1.,1.6)
            elif family in ('outside','multiclutter'):
                x = rng.choice([-1,1])*rng.uniform(1.,2.2)
            elif family == 'approach':
                vz = -rng.uniform(.2,.8)
            width = rng.uniform(.025,.08) if family == 'weakthin' else rng.uniform(.12,.45)
            objects.append(dict(x=x,z=z,vx=vx,vz=vz,width=width,height=rng.uniform(1.4,2.),reflectivity=rng.uniform(.35,.7) if family=='weakthin' else rng.uniform(.7,1.)))
        scenes.append(dict(episode_id=f'ue96-{i:04d}',split=split[i],family=family,
            sensor_seed=rng.randrange(2**31),wearer_speed=rng.uniform(.5,.9),
            yaw_amplitude=rng.uniform(0.,15.),yaw_omega=rng.uniform(.8,1.8),
            angle_bias=rng.choice([-10.,0.,10.]),imu_bias=rng.uniform(-1.,1.),
            objects=objects))
    # Independent RNG: ghost augmentation does not change existing scene geometry.
    ghosts=random.Random(seed+700001)
    ghost_indices=set(ghosts.sample(range(128),32))
    for i,scene in enumerate(scenes):
        scene['radar_ghost']=(dict(x=ghosts.uniform(-1.3,1.3),z=ghosts.uniform(1.5,4.5),
            authority='GENERATED_RADAR_GHOST_NOT_ENGINE_GEOMETRY') if i in ghost_indices else None)
    return dict(schema='mz96-fresh-ue-native-v1',authority=AUTHORITY,seed=seed,dt=DT,frames=FRAMES,
        truth='HORIZONTAL_POINT_CENTER_CURRENT_OR_1S_ROUTE_WEDGE_NOT_PHYSICAL_BODY_COLLISION',
        sensor='8x8 native ToF rays aggregated to 8 horizontal slots; actor-directed occlusion-tested native Radar hit proxy; Doppler from engine center displacement; iid transient false returns and 32 scene persistent hypothetical ghosts with matched .8 detection/noise',
        split_authority='PRECAPTURE_SCENE_GROUPS_80_16_32_NO_REPLICATED_VARIANTS',scenes=scenes)


def intersection(x,z,vx,vz):
    lo,hi=0.,1.; slope=math.tan(math.radians(12))
    for a,b in ((z-.2,vz),(slope*z-x,slope*vz-vx),(slope*z+x,slope*vz+vx)):
        if abs(b)<1e-12:
            if a<0:return False
        elif b>0:lo=max(lo,-a/b)
        else:hi=min(hi,-a/b)
        if lo>hi:return False
    return z+vz*(lo+hi)/2>.2


def hazard(x,z,vx,vz):
    r=math.hypot(x,z)
    direct=z>.2 and r<3.18 and abs(math.degrees(math.atan2(x,z)))<=12
    return direct or (z>.2 and r<3.6 and intersection(x,z,vx,vz)),direct


def ghost_return(ghost,wearer_speed,t,yaw):
    """Sensor-generator-only static world ghost, no UE geometry or truth support."""
    if ghost is None:return None
    x,z=ghost['x'],ghost['z']-wearer_speed*t
    r=math.hypot(x,z);a=math.degrees(math.atan2(x,z))-yaw
    if abs(a)>60:return None
    return r,a,-wearer_speed*z/max(r,1e-9)


def measure_radar(r,a,v,bias,rng):
    """Identical detection, noise and quantization for real and ghost returns."""
    if rng.random()>=.8:return None
    return (max(.05,round((r+rng.gauss(0,.06))/.05)*.05),
        round((a+rng.gauss(0,2))/10)*10+bias,round((v+rng.gauss(0,.08))/.1)*.1)


def engine_capture():
    import unreal as u
    out=Path(os.environ['BA_MZ96_OUT']); spec_path=Path(os.environ['BA_MZ96_SPEC'])
    spec=json.loads(spec_path.read_text()); count=int(os.environ.get('BA_MZ96_EPISODES','128'))
    api=u.get_editor_subsystem(u.EditorActorSubsystem)
    world=u.EditorLoadingAndSavingUtils.new_blank_map(False)
    mesh=u.load_asset('/Engine/BasicShapes/Cube')
    started=time.monotonic(); actors=[]; total=0
    report=dict(status='RUNNING',authority=AUTHORITY,engine_version=u.SystemLibrary.get_engine_version(),spec_sha256=sha(spec_path),source_sha256=sha(__file__),episodes=count)
    rawfile=(out/'raw.jsonl').open('w',encoding='utf-8')
    evalfile=(out/'evaluator.jsonl').open('w',encoding='utf-8')
    geofile=(out/'native-geometry.jsonl').open('w',encoding='utf-8')

    def emit(f,v):f.write(json.dumps(v,allow_nan=False)+'\n')
    def trace(start,end):
        hit=u.SystemLibrary.line_trace_single(world,start,end,u.TraceTypeQuery.TRACE_TYPE_QUERY1,True,[],u.DrawDebugTrace.NONE)
        if not hit:return None
        fields=hit.to_tuple()
        if not fields[0]:return None
        return fields[5],fields[10]
    try:
        for scene in spec['scenes'][:count]:
            for a in actors:api.destroy_actor(a)
            actors=[]
            for obj in scene['objects']:
                a=api.spawn_actor_from_class(u.StaticMeshActor,u.Vector(obj['z']*100,obj['x']*100,100))
                a.static_mesh_component.set_static_mesh(mesh)
                a.static_mesh_component.set_collision_profile_name('BlockAll')
                a.set_actor_scale3d(u.Vector(obj['width'],obj['width'],obj['height']))
                actors.append(a)
            rng=random.Random(scene['sensor_seed']); prevyaw=0.
            for j in range(FRAMES):
                t=j*DT; camera_z=scene['wearer_speed']*t
                origin=u.Vector(camera_z*100,0,100)
                yaw=scene['yaw_amplitude']*math.sin(scene['yaw_omega']*t)
                bounds=[]
                for a,obj in zip(actors,scene['objects']):
                    a.set_actor_location(u.Vector((obj['z']+obj['vz']*t)*100,(obj['x']+obj['vx']*t)*100,100),False,True)
                    c,e=a.get_actor_bounds(False)
                    # Future engine transform gives evaluator displacement; restored before rays.
                    a.set_actor_location(u.Vector((obj['z']+obj['vz']*(t+DT))*100,(obj['x']+obj['vx']*(t+DT))*100,100),False,True)
                    nxt,_=a.get_actor_bounds(False)
                    a.set_actor_location(c,False,True)
                    bounds.append(dict(x=c.y/100,z=c.x/100-camera_z,vx=(nxt.y-c.y)/(100*DT),vz=(nxt.x-c.x)/(100*DT)-scene['wearer_speed'],center_m=[c.x/100,c.y/100,c.z/100],extent_m=[e.x/100,e.y/100,e.z/100]))
                row=dict(episode_id=scene['episode_id'],time_s=t,tof_packet_received=rng.random()>=.1,
                    tof_range_m=[None]*8,tof_theta_deg=[-22.5+(k+.5)*45/8 for k in range(8)],tof_range_sigma_m=[.04]*8,tof_status=[255]*8,
                    radar_packet_received=True,radar_range_m=[None]*4,radar_velocity=[None]*4,radar_angle=[None]*4,radar_valid=[False]*4,
                    delta_yaw=0. if j==0 else yaw-prevyaw+DT*scene['imu_bias']+rng.gauss(0,.08),delta_pitch=0.,imu_valid=True)
                prevyaw=yaw
                nativehits=0
                for col,az in enumerate(row['tof_theta_deg']):
                    hits=[]
                    for v in range(8):
                        el=-22.5+(v+.5)*45/8; a=math.radians(az+yaw); e=math.radians(el)
                        end=origin+u.Vector(600*math.cos(e)*math.cos(a),600*math.cos(e)*math.sin(a),600*math.sin(e))
                        hit=trace(origin,end)
                        if hit:
                            p,comp=hit; nativehits+=1
                            r=math.sqrt((p.x-origin.x)**2+(p.y-origin.y)**2+(p.z-origin.z)**2)/100
                            refl=next((o['reflectivity'] for act,o in zip(actors,scene['objects']) if act.static_mesh_component==comp),.7)
                            if r<=4:hits.append((r,refl))
                    if row['tof_packet_received'] and hits:
                        r,refl=min(hits); pd=(.9 if r<=2 else .65 if r<=3 else .35)*refl
                        if rng.random()<pd:row['tof_range_m'][col]=max(.05,round((r+rng.gauss(0,.04))/.02)*.02);row['tof_status'][col]=5
                    elif not row['tof_packet_received']:row['tof_status'][col]=0
                radar=[]; radarhits=0
                for a,b in zip(actors,bounds):
                    az=math.degrees(math.atan2(b['x'],b['z']))-yaw
                    if abs(az)>60:continue
                    hit=trace(origin,u.Vector(*(v*100 for v in b['center_m'])))
                    if not hit or hit[1]!=a.static_mesh_component:continue
                    radarhits+=1
                    p,_=hit; r=math.hypot((p.x-origin.x)/100,(p.y-origin.y)/100)
                    rv=(b['x']*b['vx']+b['z']*b['vz'])/max(math.hypot(b['x'],b['z']),1e-9)
                    measured=measure_radar(r,az,rv,scene['angle_bias'],rng)
                    if measured:radar.append(measured)
                ghost=ghost_return(scene.get('radar_ghost'),scene['wearer_speed'],t,yaw)
                ghost_detected=False
                if ghost:
                    measured=measure_radar(*ghost,scene['angle_bias'],rng)
                    if measured:radar.append(measured);ghost_detected=True
                if rng.random()<.15:radar.append((round(rng.uniform(.4,5)/.05)*.05,round(rng.uniform(-60,60)/10)*10,round(rng.uniform(-1,1)/.1)*.1))
                for k,(r,a,v) in enumerate(sorted(radar)[:4]):
                    row['radar_range_m'][k]=r;row['radar_angle'][k]=a;row['radar_velocity'][k]=v;row['radar_valid'][k]=True
                truths=[hazard(b['x'],b['z'],b['vx'],b['vz']) for b in bounds]
                truth=any(v[0] for v in truths);direct=any(v[1] for v in truths)
                emit(rawfile,row)
                emit(evalfile,dict(episode_id=scene['episode_id'],time_s=t,truth=truth,any_direct_truth=direct,future_only_truth=truth and not direct,split=scene['split'],family=scene['family'],persistent_ghost_scene=scene.get('radar_ghost') is not None))
                emit(geofile,dict(episode_id=scene['episode_id'],time_s=t,native_bounds=bounds,tof_native_hits=nativehits,radar_native_hits=radarhits,generated_ghost_detected_before_truncation=ghost_detected,ghost_authority='GENERATED_RADAR_GHOST_NOT_ENGINE_GEOMETRY'))
                total+=1
            rawfile.flush();evalfile.flush();geofile.flush()
            write(out/'progress.json',dict(frames=total,episode=scene['episode_id'],seconds=time.monotonic()-started))
        report.update(status='PASS',frames=total)
    except Exception:report.update(status='FAIL',error=traceback.format_exc(),frames=total)
    finally:
        rawfile.close();evalfile.close();geofile.close()
        for a in actors:api.destroy_actor(a)
        report['seconds']=time.monotonic()-started
        report['hashes']={p.name:sha(p) for p in (out/'raw.jsonl',out/'evaluator.jsonl',out/'native-geometry.jsonl')}
        write(out/'receipt.json',report)
        u.SystemLibrary.quit_editor()


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--output',type=Path,required=True);prep.add_argument('--seed',type=int,default=96012)
    cap=sub.add_parser('capture');cap.add_argument('--spec',type=Path,required=True);cap.add_argument('--output',type=Path,required=True);cap.add_argument('--episodes',type=int,default=128)
    args=p.parse_args();root=Path(__file__).resolve().parents[4];art=(root/'artifacts.local').resolve()
    if not args.output.resolve().is_relative_to(art):raise ValueError('Output outside canonical artifacts')
    if args.command=='prepare':
        args.output.mkdir(parents=True,exist_ok=False);write(args.output/'spec.json',source(args.seed))
        write(args.output/'freeze.json',dict(spec_sha256=sha(args.output/'spec.json'),source_sha256=sha(__file__),split_counts=dict(train=80,validation=16,test=32)))
        return
    if not 1<=args.episodes<=128:raise ValueError('Episode limit')
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,args.output/Path(__file__).name)
    shutil.copyfile(args.spec,args.output/'spec.json')
    sys.path.insert(0,str(root/'tools'));from ue_native_capture import run_owned
    from run_city_pcg_capture import cache_service_port
    import psutil
    if any(p.info['name'].lower().startswith('unrealeditor') for p in psutil.process_iter(['name']) if p.info['name']):
        raise RuntimeError('Existing Unreal editor occupied; preserve other work')
    cache=art/'work/mz96-ue-decision-20260912/ddc';cache.mkdir(parents=True,exist_ok=True)
    port=cache_service_port(cache)
    env=dict(os.environ,BA_MZ96_SPEC=str((args.output/'spec.json').resolve()),BA_MZ96_OUT=str(args.output.resolve()),BA_MZ96_EPISODES=str(args.episodes))
    env['UE-LocalDataCachePath']=str(cache)
    cmd=['F:/epic/UE_5.8/Engine/Binaries/Win64/UnrealEditor.exe',str(art/'unreal/BlindAssistStreetLab/BlindAssistStreetLab.uproject'),'-NullRHI','-unattended','-nosound','-nop4','-NoSplash','-EnablePlugins=PythonScriptPlugin','-ExecCmds=py '+(args.output/Path(__file__).name).resolve().as_posix(),'-abslog='+str((args.output/'editor.log').resolve()),'-ini:Engine:[/Script/EngineSettings.GameMapsSettings]:EditorStartupMap=','-ini:EditorPerProjectUserSettings:[/Script/UnrealEd.EditorLoadingSavingSettings]:LoadLevelAtStartup=None']
    cmd+=['-ddc=NoShared','-ini:Engine:[Zen.AutoLaunch]:DesiredPort='+str(port)]
    write(args.output/'launch.json',dict(command=cmd,spec_sha256=sha(args.spec),source_sha256=sha(__file__),persistent_cache=dict(path=str(cache),zen_port=port)))
    run_owned(cmd,env,args.output,1800)
    receipt=json.loads((args.output/'receipt.json').read_text())
    if receipt['status']!='PASS' or receipt['frames']!=args.episodes*FRAMES:raise RuntimeError(receipt)
    print(json.dumps(receipt))


if os.environ.get('BA_MZ96_SPEC') and 'unreal' in sys.modules:
    engine_capture()
elif __name__=='__main__':
    # Unreal's ExecCmds interpreter may not import unreal until the script does.
    if os.environ.get('BA_MZ96_SPEC'):engine_capture()
    else:main()
