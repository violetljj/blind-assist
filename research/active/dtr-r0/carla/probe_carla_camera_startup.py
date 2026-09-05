"""Three cold starts with synchronous pipeline compilation; engineering only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import carla
import fast_sensor_png
from probe_carla_sync_rgbd import capture,listening,pose,write

FLAGS=['-dx12','-RenderOffScreen','-nosound','-quality-level=Low',
       '-carla-rpc-port=2000','-ExecCmds=r.AsyncPipelineCompile 0']


def run(args):
    root=args.output.resolve()
    canonical=Path(__file__).resolve().parents[4]/'artifacts.local'
    if not root.is_relative_to(canonical.resolve()):raise ValueError('Output routing')
    if any(listening(p) for p in (2000,2001,2002)):raise RuntimeError('Ports occupied')
    root.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),Path(capture.__code__.co_filename),Path(fast_sensor_png.__file__)]
    (root/'code').mkdir()
    for p in sources:shutil.copyfile(p,root/'code'/p.name)
    write(root/'protocol.json',{'authority':'ENGINEERING_DEVELOPMENT_ONLY','cold_starts':3,
        'pairs_per_start':100,'resolution':[1280,720],'flags':FLAGS,
        'hypothesis':'Disabling asynchronous pipeline compilation permits repeated camera startup',
        'cvar_application':'Requested by ExecCmds; shipping build provides no independent CVar readback',
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}})
    result={'status':'RUNNING','starts':[]}
    try:
        for attempt in range(3):
            target=root/str(attempt);target.mkdir();server=None;world=None
            item={'start':attempt,'status':'FAILED'}
            started=time.monotonic()
            try:
                with (target/'server.log').open('xb') as log:
                    server=subprocess.Popen([str(args.server),*FLAGS],cwd=args.server.parent,
                        stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    while time.monotonic()-started<90:
                        if server.poll() is not None:raise RuntimeError('Server exited')
                        if listening(2000):
                            client=carla.Client('127.0.0.1',2000);client.set_timeout(3.)
                            try:world=client.get_world();break
                            except RuntimeError:pass
                        time.sleep(.5)
                    if world is None:raise TimeoutError('Readiness')
                    item['ready_s']=time.monotonic()-started
                    settings=world.get_settings();settings.synchronous_mode=False
                    settings.fixed_delta_seconds=.1;settings.substepping=True
                    settings.max_substep_delta_time=.01;settings.max_substeps=10
                    world.apply_settings(settings);world.set_weather(carla.WeatherParameters.ClearNoon)
                    points=world.get_map().get_spawn_points()
                    bases=[carla.Transform(carla.Location(t.location.x,t.location.y,t.location.z+1.7),t.rotation)
                           for t in (points[0],points[min(10,len(points)-1)])]
                    item['poses']=[pose(t) for t in bases]
                    print('CAMERA_START '+str(attempt),flush=True)
                    receipt=capture(world,client,target,['rgb','depth'],bases,fast=True)
                    for a,b in zip(receipt['records'][::2],receipt['records'][1::2]):
                        if any(a[k]!=b[k] for k in ('frame','timestamp','pose')):raise ValueError('Pair mismatch')
                    write(target/'receipt.json',receipt)
                    item.update(status='PASS',capture_s=receipt['capture_s'],images=len(receipt['records']))
            except Exception as error:
                item['error']=repr(error);raise
            finally:
                if server is not None and server.poll() is None:
                    subprocess.run(['taskkill','/PID',str(server.pid),'/T','/F'],capture_output=True)
                    server.wait(timeout=15)
                end=time.monotonic()+15
                while any(listening(p) for p in (2000,2001,2002)) and time.monotonic()<end:time.sleep(.25)
                item['ports_released']=not any(listening(p) for p in (2000,2001,2002))
                item['elapsed_s']=time.monotonic()-started
                result['starts'].append(item);write(target/'outcome.json',item)
                if not item['ports_released']:raise RuntimeError('Ports not released')
            print(json.dumps(item),flush=True)
        result['status']='CAPTURE_PASS_PENDING_INDEPENDENT_PIXELS'
    except Exception as error:
        result.update(status='FAILED',error=repr(error));raise
    finally:write(root/'result.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--server',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
