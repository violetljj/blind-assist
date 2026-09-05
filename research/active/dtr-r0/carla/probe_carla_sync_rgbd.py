"""Bounded 720p engineering probe, isolated from all research outcome cohorts."""
import argparse
import hashlib
import json
from pathlib import Path
import queue
import socket
import subprocess
import time

import carla
import numpy as np
import fast_sensor_png


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def pose(transform):
    return [float(getattr(transform.location, k)) for k in ('x','y','z')]+[
        float(getattr(transform.rotation,k)) for k in ('pitch','yaw','roll')]


def listening(port):
    with socket.socket() as connection:
        connection.settimeout(.2)
        return connection.connect_ex(('127.0.0.1',port))==0


def receive(channel, frame):
    deadline=time.monotonic()+20
    while True:
        image=channel.get(timeout=max(.01,deadline-time.monotonic()))
        if image.frame==frame:return image
        if image.frame>frame:raise RuntimeError('sensor_skipped_requested_frame')
        if time.monotonic()>deadline:raise TimeoutError('camera_frame_timeout')


def capture(world, client, root, modes, bases, fast=False):
    sensors={};channels={};records=[]
    setup=time.perf_counter()
    try:
        settings=world.get_settings();settings.synchronous_mode=False
        world.apply_settings(settings)
        for mode in modes:
            blueprint=world.get_blueprint_library().find('sensor.camera.'+mode)
            for key,value in {'image_size_x':'1280','image_size_y':'720','fov':'90','sensor_tick':'0.1'}.items():
                blueprint.set_attribute(key,value)
            channels[mode]=queue.Queue()
            sensors[mode]=world.spawn_actor(blueprint,bases[0])
            sensors[mode].listen(channels[mode].put)
        for mode in modes:
            for _ in range(5):channels[mode].get(timeout=20)
        settings.synchronous_mode=True;world.apply_settings(settings)
        for channel in channels.values():
            while not channel.empty():channel.get_nowait()
        for _ in range(10):
            frame=world.tick(20.)
            for mode in modes:receive(channels[mode],frame)
        setup=time.perf_counter()-setup
        tick_s=write_s=0.;started=time.perf_counter()
        for scene,base in enumerate(bases):
            for sample in range(50):
                transform=carla.Transform(carla.Location(base.location.x+sample*.04,base.location.y,base.location.z),base.rotation)
                before=time.perf_counter()
                responses=client.apply_batch_sync([carla.command.ApplyTransform(s.id,transform) for s in sensors.values()],True)
                if any(r.has_error() for r in responses):raise RuntimeError('camera_transform_failed')
                frame=world.get_snapshot().frame
                images={mode:receive(channels[mode],frame) for mode in modes}
                tick_s+=time.perf_counter()-before
                for mode,image in images.items():
                    if (image.width,image.height)!=(1280,720):raise RuntimeError('wrong_resolution')
                    if max(abs(a-b) for a,b in zip(pose(image.transform),pose(transform)))>1e-3:
                        raise RuntimeError('camera_pose_mismatch')
                    array=np.frombuffer(image.raw_data,dtype=np.uint8).reshape(720,1280,4)
                    if float(array[:,:,:3].std())<=0:raise RuntimeError('blank_sensor_frame')
                    before=time.perf_counter()
                    path=root/f'{scene}-{sample:03d}-{mode}.png'
                    if fast:
                        with path.open('xb') as stream:
                            stream.write(fast_sensor_png.encode_bgra(image.raw_data,1280,720))
                    else:image.save_to_disk(str(path))
                    digest=hashlib.sha256(path.read_bytes()).hexdigest()
                    write_s+=time.perf_counter()-before
                    records.append({'scene':scene,'sample':sample,'mode':mode,'frame':frame,
                        'timestamp':image.timestamp,'pose':pose(image.transform),'path':path.name,
                        'bytes':path.stat().st_size,'sha256':digest,
                        'pixel_sha256':hashlib.sha256(array[:,:,[2,1,0,3]].tobytes()).hexdigest()})
            print(f'{root.name}: scene {scene+1}/2 complete',flush=True)
        elapsed=time.perf_counter()-started
        if len(records)!=100*len(modes):raise RuntimeError('missing_frames')
        return {'setup_and_warmup_s':setup,'capture_s':elapsed,'tick_and_receive_s':tick_s,
                'png_write_and_hash_s':write_s,'records':records}
    finally:
        for sensor in sensors.values():
            try:sensor.stop();sensor.destroy()
            except RuntimeError:pass


def run(args):
    root=args.output.resolve()
    canonical=Path(__file__).resolve().parents[4]/'artifacts.local'
    if not root.is_relative_to(canonical.resolve()):raise ValueError('output_outside_artifacts')
    if any(listening(port) for port in (2000,2001,2002)):raise RuntimeError('CARLA_ports_occupied')
    root.mkdir(parents=True,exist_ok=False)
    schedule=([('serial_rgb',['rgb'],False),('serial_depth',['depth'],False),('sync_rgbd',['rgb','depth'],False)]
              if args.comparison=='sensors' else [('native_sync',['rgb','depth'],False),('fast_sync',['rgb','depth'],True)])
    write(root/'protocol.json',{'kind':'ENGINEERING_DEVELOPMENT_NOT_R1','backend':args.backend,
          'resolution':[1280,720],'scenes':2,'frames_per_scene':50,'dt':.1,
          'order':[s[0] for s in schedule],'comparison':args.comparison,'same_persistent_server':True,
          'encoder_sha256':hashlib.sha256(Path(fast_sensor_png.__file__).read_bytes()).hexdigest(),
          'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'scope':'Camera throughput and integrity only, no avoidance score or long-run reliability claim'})
    server=None;world=None;original=None;result={'status':'FAILED'}
    try:
        with (root/'server.log').open('xb') as log:
            started=time.perf_counter()
            server=subprocess.Popen([str(args.server),'-'+args.backend,'-RenderOffScreen','-nosound',
                '-quality-level=Low','-carla-rpc-port=2000'],cwd=args.server.parent,
                stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            last_error='RPC port not open'
            while time.perf_counter()-started<90:
                if server.poll() is not None:raise RuntimeError('server_exited_before_ready')
                if not listening(2000):time.sleep(.5);continue
                client=carla.Client('127.0.0.1',2000);client.set_timeout(10.)
                try:world=client.get_world();break
                except RuntimeError as error:
                    last_error=str(error);print('READY_WAIT '+last_error,flush=True);time.sleep(.5)
            if world is None:raise TimeoutError('server_ready_timeout: '+last_error)
            ready=time.perf_counter()-started;client.set_timeout(20.)
            original=world.get_settings();settings=world.get_settings()
            settings.synchronous_mode=True;settings.fixed_delta_seconds=.1
            settings.substepping=True;settings.max_substep_delta_time=.01;settings.max_substeps=10
            world.apply_settings(settings)
            world.set_weather(carla.WeatherParameters.ClearNoon)
            spawns=world.get_map().get_spawn_points()
            bases=[carla.Transform(carla.Location(t.location.x,t.location.y,t.location.z+1.7),t.rotation)
                   for t in (spawns[0],spawns[min(10,len(spawns)-1)])]
            write(root/'scenes.json',{'map':world.get_map().name,'camera_start_poses':[pose(t) for t in bases]})
            result={'status':'RUNNING','server_ready_s':ready,'groups':{}}
            for name,modes,fast in schedule:
                target=root/name;target.mkdir()
                print('START '+name,flush=True)
                group=capture(world,client,target,modes,bases,fast)
                write(target/'receipt.json',group);result['groups'][name]=group
            candidate='sync_rgbd' if args.comparison=='sensors' else 'fast_sync'
            paired=result['groups'][candidate]['records']
            for a,b in zip(paired[::2],paired[1::2]):
                if a['frame']!=b['frame'] or a['timestamp']!=b['timestamp'] or a['pose']!=b['pose']:
                    raise RuntimeError('RGB_depth_not_aligned')
            serial=sum(result['groups'][g]['capture_s'] for g in (('serial_rgb','serial_depth') if args.comparison=='sensors' else ('native_sync',)))
            result.update(status='PASS',paired_frames=100,raw_images=400,
                reference_capture_s=serial,candidate_capture_s=result['groups'][candidate]['capture_s'],
                capture_speedup=serial/result['groups'][candidate]['capture_s'])
    except Exception as error:
        result.update(status='FAILED',error=repr(error))
        raise
    finally:
        if world is not None and original is not None:
            try:world.apply_settings(original)
            except RuntimeError:pass
        if server is not None and server.poll() is None:
            subprocess.run(['taskkill','/PID',str(server.pid),'/T','/F'],capture_output=True)
            server.wait(timeout=15)
        deadline=time.monotonic()+15
        while any(listening(port) for port in (2000,2001,2002)) and time.monotonic()<deadline:time.sleep(.25)
        result['ports_released']=not any(listening(port) for port in (2000,2001,2002))
        write(root/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='groups'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=('dx11','dx12'),default='dx12')
    parser.add_argument('--comparison',choices=('sensors','png'),default='sensors')
    run(parser.parse_args())
