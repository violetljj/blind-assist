"""Launch isolated metric depth and RGB-D edge checks; require offline validation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();repo=Path(__file__).resolve().parents[1]
    output=args.output.resolve()
    if not output.is_relative_to((repo/'artifacts.local').resolve()):p.error('Artifact routing')
    if output.exists():p.error('Calibration output already exists')
    output.parent.mkdir(parents=True,exist_ok=True)
    scripts=repo/'research/active/dtr-r0/unreal';sys.path.insert(0,str(scripts))
    from street_process_lifecycle import TaskProcessTree
    project=repo/'artifacts.local/unreal/BlindAssistStreetLab'
    env=dict(os.environ,BA_UE_DEPTH_CALIBRATION_OWNED_PROCESS='1',BA_UE_DEPTH_CALIBRATION_OUTPUT=str(output))
    env['UE-LocalDataCachePath']=str(project/'DerivedDataCache')
    process=subprocess.Popen([str(args.engine/'Engine/Binaries/Win64/UnrealEditor.exe'),
        str(project/'BlindAssistStreetLab.uproject'),'/Game/StreetLab/StreetLabV4',
        '-ExecCmds=py '+(scripts/'ue_depth_calibration.py').as_posix(),
        '-RenderOffscreen','-unattended','-nosound','-nop4','-NoSplash','-ddc=NoShared',
        '-abslog='+str(output.with_suffix('.log'))],env=env)
    tree=TaskProcessTree(process,owner=str(output))
    try:
        code=tree.wait(timeout=240)
        if code:raise RuntimeError('Calibration editor failed: '+str(code))
        result=json.loads((output/'result.json').read_text())
        from ue_depth_calibration import verify
        validation=verify(output)
        print(json.dumps({'status':result['status'],'cases':len(result['cases']),
                          'independent_validation':validation['status'],
                          'alignment_cases':validation['alignment_cases_verified'],
                          'max_error_m':max((c['max_absolute_error_m'] for c in result['cases']),default=None)}))
        if validation['status']!='PASS':raise RuntimeError('Calibration did not pass; retained result')
    except Exception as error:
        output.mkdir(exist_ok=True)
        failure=output/'alignment-validation.json'
        if not failure.exists():
            with failure.open('x',encoding='utf-8') as stream:
                json.dump({'status':'FAIL','authority':'ENGINEERING_DEVELOPMENT_ONLY',
                           'reason':str(error),'exception_type':type(error).__name__,
                           'stage':'native_capture_or_verification'},stream,indent=2)
        raise
    finally:
        receipt=tree.cleanup()
        output.mkdir(exist_ok=True)
        (output/'process-release.json').write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
