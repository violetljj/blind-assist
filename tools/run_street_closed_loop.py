"""Run UE -> isolated live DTR worker -> motion -> UE, with durable step checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',action='append',default=[])
    p.add_argument('--map',choices=('StreetLabV2','StreetLabV3','StreetLabV4'),default='StreetLabV2',
                   help='Choose the actual scene; map identity is recorded for every new run')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--reuse-open-loop',type=Path,help='Reuse verified full controls; execute eight assisted branches only')
    args=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    scripts=repo/'research/active/dtr-r0/unreal'
    if args.reuse_open_loop and args.case:
        p.error('--reuse-open-loop requires the complete catalog, without --case')
    project=repo/'artifacts.local/unreal/BlindAssistStreetLab'
    map_asset='/Game/StreetLab/'+args.map
    map_file=project/'Content/StreetLab'/f'{args.map}.umap'
    if not map_file.is_file(): p.error(f'Build the selected scene first: {map_file}')
    output=args.output.resolve()
    if not output.is_relative_to((repo/'artifacts.local').resolve()): p.error('Output must remain under artifacts.local')
    output.mkdir(parents=True,exist_ok=args.resume)
    sources=['capture_street_closed_loop.py','street_live_server.py','street_live_policy.py','street_scenarios.py','ue_dtr_replay.py','ue_replay_cache.py','reuse_street_open_loop.py']
    identity={'cases':args.case,'sources':{name:sha(scripts/name) for name in sources},
              'map_sha256':sha(map_file)}
    if args.map!='StreetLabV2': identity['map_asset']=map_asset
    if args.map=='StreetLabV4':
        identity['render_config_sha256']=sha(project/'Config/DefaultEngine.ini')
    if args.reuse_open_loop:
        identity['reused_open_loop_run']=str(args.reuse_open_loop.resolve())
    identity_file=output/'identity.json'
    if args.resume:
        if json.loads(identity_file.read_text())!=identity: raise RuntimeError('Resume identity changed; preserve this run and choose a new output')
    else: identity_file.write_text(json.dumps(identity,indent=2))
    if args.map=='StreetLabV4':
        frozen_map=output/'scene_snapshot'/map_file.name
        frozen_map.parent.mkdir(exist_ok=True)
        if not frozen_map.exists(): shutil.copy2(map_file,frozen_map)
        if sha(frozen_map)!=identity['map_sha256']:
            raise RuntimeError('Frozen scene copy differs from the run identity')
    lock=output/'owner.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    model=output/'model'
    model.mkdir(exist_ok=True)
    worker=None
    editor=None
    try:
        if args.reuse_open_loop and not args.resume:
            sys.path.insert(0,str(scripts))
            from reuse_street_open_loop import reuse_open_loop
            reuse_open_loop(args.reuse_open_loop,output,identity)
        sensor=output/'sensor-worker'
        sensor.mkdir(exist_ok=True)
        ready=sensor/'ready.json'
        if ready.exists(): ready.unlink()
        with (sensor/'worker.log').open('a') as log:
            worker=subprocess.Popen([sys.executable,str(scripts/'street_live_server.py'),
                '--model-root',str(model),'--output',str(sensor)],stdout=log,stderr=subprocess.STDOUT)
            begin=time.monotonic()
            while not ready.exists():
                if worker.poll() is not None: raise RuntimeError('DTR worker failed; inspect worker.log')
                if time.monotonic()-begin>180: raise RuntimeError('DTR worker startup timeout')
                time.sleep(.5)
            port=json.loads(ready.read_text())['port']
            if args.reuse_open_loop:
                old_backend=json.loads((args.reuse_open_loop/'sensor-worker/backend.json').read_text())
                if old_backend['model_sha256']!=json.loads(ready.read_text())['model_sha256']:
                    raise RuntimeError('Reused controls require unchanged model weights')
            env=dict(os.environ,BA_UE_LIVE_OUTPUT=str(output),BA_UE_LIVE_PORT=str(port),BA_UE_LIVE_CASES=json.dumps(args.case))
            env['UE-LocalDataCachePath']=str(project/'DerivedDataCache')
            editor=subprocess.Popen([str(args.engine/'Engine/Binaries/Win64/UnrealEditor.exe'),
                str(project/'BlindAssistStreetLab.uproject'),map_asset,
                '-ExecCmds=py '+(scripts/'capture_street_closed_loop.py').as_posix(),
                '-RenderOffscreen','-unattended','-nosound','-nop4','-NoSplash','-ddc=NoShared',
                '-LocalDataCachePath='+str(project/'DerivedDataCache'),'-abslog='+str(output/'capture.log')],env=env)
            (output/'processes.json').write_text(json.dumps({'owner_pid':os.getpid(),'worker_pid':worker.pid,
                    'editor_pid':editor.pid,'port':port},indent=2))
            print('LIVE_LOOP_STARTED',json.loads((output/'processes.json').read_text()),flush=True)
            code=editor.wait(timeout=3600)
            if code: raise RuntimeError(f'UE exited {code}')
        report=json.loads((output/'run.json').read_text())
        print(json.dumps(report,indent=2),flush=True)
        if report['status']!='COMPLETE': raise RuntimeError('Live UE run failed; per-step checkpoints retained')
    except BaseException as exc:
        report_path=output/'run.json'
        report=json.loads(report_path.read_text()) if report_path.exists() else {}
        report.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAIL',
                      launcher_error=str(exc),terminal_utc=time.time(),checkpoints_retained=True)
        report_path.write_text(json.dumps(report,indent=2))
        raise
    finally:
        for process in (editor,worker):
            if process and process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
        lock.unlink(missing_ok=True)
    subprocess.run([sys.executable,str(scripts/'evaluate_street_closed_loop.py'),'--run',str(output)],check=True)


if __name__=='__main__': main()
