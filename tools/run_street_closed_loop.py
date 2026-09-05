"""Run UE -> isolated live DTR worker -> motion -> UE, with durable step checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',action='append',default=[])
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    scripts=repo/'research/active/dtr-r0/unreal'
    project=repo/'artifacts.local/unreal/BlindAssistStreetLab'
    output=args.output.resolve()
    if not output.is_relative_to((repo/'artifacts.local').resolve()): p.error('Output must remain under artifacts.local')
    output.mkdir(parents=True,exist_ok=args.resume)
    sources=['capture_street_closed_loop.py','street_live_server.py','street_live_policy.py','street_scenarios.py','ue_dtr_replay.py','ue_replay_cache.py']
    identity={'cases':args.case,'sources':{name:sha(scripts/name) for name in sources},
              'map_sha256':sha(project/'Content/StreetLab/StreetLabV2.umap')}
    identity_file=output/'identity.json'
    if args.resume:
        if json.loads(identity_file.read_text())!=identity: raise RuntimeError('Resume identity changed; preserve this run and choose a new output')
    else: identity_file.write_text(json.dumps(identity,indent=2))
    lock=output/'owner.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    model=output/'model'
    model.mkdir(exist_ok=True)
    worker=None
    editor=None
    try:
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
            env=dict(os.environ,BA_UE_LIVE_OUTPUT=str(output),BA_UE_LIVE_PORT=str(port),BA_UE_LIVE_CASES=json.dumps(args.case))
            editor=subprocess.Popen([str(args.engine/'Engine/Binaries/Win64/UnrealEditor.exe'),
                str(project/'BlindAssistStreetLab.uproject'),'/Game/StreetLab/StreetLabV2',
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
