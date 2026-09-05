"""New Development composite; retain old R1 failure, reuse only complete shards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import carla
from probe_carla_sync_rgbd import listening

HERE=Path(__file__).resolve().parent
GROUPS=('FIT_ONLY','FINAL_A','FINAL_B')
MISSING=(('FINAL_A','depth'),('FINAL_B','wearable'),('FINAL_B','depth'))


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest().upper()
def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2)


def command(args,log,server=None):
    with log.open('x') as output:
        process=subprocess.Popen([str(a) for a in args],stdout=output,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+900
        try:
            while process.poll() is None:
                if server is not None and server.poll() is not None:
                    raise RuntimeError('Owned CARLA server exited during capture')
                if time.monotonic()>deadline:raise TimeoutError('Command deadline')
                time.sleep(.5)
        finally:
            if process.poll() is None:
                process.terminate()
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait(timeout=15)
    if process.returncode:raise RuntimeError('Command failed; retained log '+str(log))


def capture(args,root,group,sensor):
    server=None
    try:
        if any(listening(p) for p in (2000,2001,2002)):raise RuntimeError('CARLA ports occupied')
        with (root/f'{group}-{sensor}-server.log').open('xb') as output:
            flags=['-dx12','-RenderOffScreen','-nosound','-quality-level=Low','-carla-rpc-port=2000']
            if args.startup_probe:flags.append('-ExecCmds=r.AsyncPipelineCompile 0')
            server=subprocess.Popen([str(args.server),*flags],
                cwd=args.server.parent,stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            deadline=time.monotonic()+90;ready=False
            while time.monotonic()<deadline:
                if server.poll() is not None:raise RuntimeError('Server exited before readiness')
                if listening(2000):
                    client=carla.Client('127.0.0.1',2000);client.set_timeout(10.)
                    try:client.get_world();ready=True;break
                    except RuntimeError:pass
                time.sleep(.5)
            if not ready:raise TimeoutError('Server readiness deadline')
            command([sys.executable,HERE/'capture_dtr_carla_c2_fast_png.py','--protocol',root/group/'protocol.json',
                     '--output-root',root/'raw'/group,'--sensor',sensor],root/f'{group}-{sensor}-capture.log',server)
    finally:
        if server is not None and server.poll() is None:
            subprocess.run(['taskkill','/PID',str(server.pid),'/T','/F'],capture_output=True)
            server.wait(timeout=15)
        deadline=time.monotonic()+15
        while any(listening(p) for p in (2000,2001,2002)) and time.monotonic()<deadline:time.sleep(.25)
        if any(listening(p) for p in (2000,2001,2002)):raise RuntimeError('CARLA ports remain')


def run(args):
    root=args.output.resolve();old=args.previous.resolve(strict=True)
    if not root.is_relative_to((HERE.parents[3]/'artifacts.local').resolve()):raise ValueError('Output routing')
    startup=None
    if args.startup_probe:
        probe=args.startup_probe.resolve(strict=True)
        outcome=read(probe/'result.json');pixels=read(probe/'pixel-validation.json')
        if outcome['status']!='CAPTURE_PASS_PENDING_INDEPENDENT_PIXELS' or pixels['status']!='PASS':
            raise ValueError('Camera startup probe not admitted')
        if pixels.get('verified_images')!=600:raise ValueError('Startup pixels incomplete')
        if any(pixels[name+'_sha256'].upper()!=sha(probe/(name+'.json')) for name in ('protocol','result')):
            raise ValueError('Startup pixel receipt binding')
        if len(outcome['starts'])!=3 or any(s['status']!='PASS' or s['images']!=200 or not s['ports_released'] for s in outcome['starts']):
            raise ValueError('Three complete cold starts required')
        from probe_carla_camera_startup import FLAGS
        if read(probe/'protocol.json')['flags']!=FLAGS:raise ValueError('Startup launch profile drift')
        startup={name:{'path':str(probe/name),'sha256':sha(probe/name)} for name in ('protocol.json','result.json','pixel-validation.json')}
    root.mkdir(parents=True,exist_ok=False)
    reused=[]
    for group in GROUPS:
        (root/group).mkdir();(root/'raw'/group/'shards').mkdir(parents=True)
        for filename in ('protocol.json','annex.json'):
            shutil.copyfile(old/group/filename,root/group/filename)
        shutil.copyfile(root/group/'protocol.json',root/'raw'/group/'frozen_protocol.json')
        for sensor in ('instance','wearable','depth','witness'):
            if (group,sensor) in MISSING:continue
            source=(old/'raw'/group/'shards'/sensor).resolve(strict=True)
            result=source/'result.json'
            if read(result)['status']!='DTR_CARLA_C2_RAW_SHARD_CAPTURE_COMPLETE':raise ValueError('Incomplete reuse')
            destination=root/'raw'/group/'shards'/sensor
            for directory,dirs,files in os.walk(source,followlinks=False):
                target=destination/Path(directory).relative_to(source);target.mkdir(parents=True,exist_ok=True)
                for filename in files:os.link(Path(directory)/filename,target/filename)
            reused.append({'group':group,'sensor':sensor,'source':str(source),'result_sha256':sha(result)})
    code=[HERE/name for name in ('run_dtr_fast_composite_source.py','capture_dtr_carla_c2_fast_png.py',
          'capture_dtr_carla_c2_rich_scene.py','fast_sensor_png.py','join_dtr_final_roster_source.py',
          'join_dtr_carla_c2_rich_scene.py','finalize_dtr_final_roster_join.py','audit_dtr_final_roster_source.py',
          'validate_dtr_fast_png_receipts.py')]
    if startup:code.extend(HERE/name for name in ('probe_carla_camera_startup.py','validate_carla_camera_startup.py'))
    plan={'status':'NEW_DEVELOPMENT_COMPOSITE_SOURCE_SEALED','claim':'DEVELOPMENT_COMPOSITE_REUSED_SOURCE_NOT_FRESH_CONFIRMATION',
          'previous_failed_execution':str(old),'previous_terminal_sha256':sha(old/'execution-terminal.json'),
          'reused_complete_shards':reused,'missing_shards':MISSING,'scenario_algorithm_threshold_changes':False,
          'retry_per_missing_shard':False,'old_R1_status_unchanged':True,
          'startup_probe':startup,'async_pipeline_compile_requested':0 if startup else None,
          'method_access':'ONLY_AFTER_ALL_SOURCE_GATES_JOINS_AND_SHARED_DETECTOR_ADMISSION',
          'code_files':[{'path':str(p),'sha256':sha(p)} for p in code]}
    write(root/'execution-authority.json',plan)
    (root/'code-snapshot').mkdir()
    for path in code:shutil.copyfile(path,root/'code-snapshot'/path.name)
    try:
        for group in GROUPS:
            print('VERIFY_SOURCE '+group,flush=True)
            command([args.research_python,HERE/'audit_dtr_final_roster_source.py','--protocol',root/group/'protocol.json',
                     '--root',root/'raw'/group,'--annex',root/group/'annex.json',
                     '--output',root/'raw'/group/'roster-source-gate.json'],root/f'{group}-source-audit.log')
        join_names=('join_dtr_final_roster_source.py','join_dtr_carla_c2_rich_scene.py','finalize_dtr_final_roster_join.py')
        write(root/'join-annex.json',{'code_files':[r for r in plan['code_files'] if Path(r['path']).name in join_names],
            'source_gate_sha256':{g:sha(root/'raw'/g/'roster-source-gate.json') for g in GROUPS}})
        for group,sensor in MISSING:
            for ref in plan['code_files']:
                if sha(Path(ref['path']))!=ref['sha256']:raise ValueError('Execution code drift')
            print('CAPTURE_MISSING '+group+' '+sensor,flush=True)
            capture(args,root,group,sensor)
        command([args.research_python,HERE/'validate_dtr_fast_png_receipts.py','--root',root],root/'fast-png-validation.log')
        for group in GROUPS:
            print('JOIN '+group,flush=True)
            command([args.research_python,HERE/'join_dtr_final_roster_source.py','--root',root/'raw'/group,
                     '--protocol',root/group/'protocol.json','--annex',root/'join-annex.json'],root/f'{group}-join.log')
        write(root/'source-admission.json',{'status':'DEVELOPMENT_COMPOSITE_SOURCE_ADMITTED','claim':plan['claim'],
              'execution_authority_sha256':sha(root/'execution-authority.json'),
              'fast_png_validation_sha256':sha(root/'fast-png-validation.json'),
              'joined_results':{g:sha(root/'raw'/g/'r1-joined-result.json') for g in GROUPS}})
        print('DEVELOPMENT_COMPOSITE_SOURCE_ADMITTED',flush=True)
    except Exception as error:
        write(root/'source-failure.json',{'status':'NOT_EVALUABLE_NEW_COMPOSITE','reason':repr(error),'method_access':False})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','previous','server','research-python'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--startup-probe',type=Path,help='Admitted three-start synchronous-PSO engineering probe')
    run(parser.parse_args())
