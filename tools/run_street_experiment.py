"""Capture the UE street and run the existing DTR stack on isolated RGB-D inputs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True,help='New run directory beneath artifacts.local')
    p.add_argument('--capture-only',action='store_true')
    p.add_argument('--reuse-capture',action='store_true')
    p.add_argument('--weights',type=Path)
    args=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    scripts=repo/'research/active/dtr-r0/unreal'
    project_root=repo/'artifacts.local/unreal/BlindAssistStreetLab'
    out=args.output.resolve()
    if not out.is_relative_to((repo/'artifacts.local').resolve()):
        p.error('--output must be beneath artifacts.local')
    if not args.reuse_capture:
        out.mkdir(parents=True,exist_ok=False)
        env=dict(os.environ,BA_UE_CAPTURE_OUTPUT=str(out))
        command=[str(args.engine/'Engine/Binaries/Win64/UnrealEditor.exe'),
                 str(project_root/'BlindAssistStreetLab.uproject'),
                 '-ExecCmds=py '+(scripts/'capture_street_rgbd.py').as_posix(),
                 '-unattended','-nosound','-nop4','-NoSplash','-RenderOffscreen',
                 '-ddc=NoShared','-LocalDataCachePath='+str(project_root/'DerivedDataCache'),
                 '-abslog='+str(out/'capture.log')]
        process=subprocess.Popen(command,env=env)
        print('CAPTURE_PID:',process.pid,flush=True)
        try:
            code=process.wait(timeout=1200)
            if code: raise RuntimeError(f'UE exited {code}; inspect {out / "capture.log"}')
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
    receipt=json.loads((out/'capture.json').read_text())
    print(json.dumps(receipt,indent=2),flush=True)
    if receipt['status']!='PASS': raise RuntimeError('UE capture failed')
    if args.capture_only: return
    replay=[sys.executable,str(scripts/'ue_dtr_replay.py'),'--model-root',str(out/'model'),
            '--output',str(out/'predictions')]
    if args.weights: replay+=['--weights',str(args.weights)]
    subprocess.run(replay,check=True)
    subprocess.run([sys.executable,str(scripts/'evaluate_street_replay.py'),'--run',str(out)],check=True)


if __name__=='__main__':
    main()
