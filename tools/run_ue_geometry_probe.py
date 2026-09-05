"""Owned UE visual geometry measurement with process-tree cleanup."""
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
    a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    out=a.output.resolve()
    if not out.is_relative_to((repo/'artifacts.local').resolve()) or out.exists():p.error('Need new canonical output')
    out.parent.mkdir(parents=True,exist_ok=True)
    scripts=repo/'research/active/dtr-r0/unreal';sys.path.insert(0,str(scripts))
    from street_process_lifecycle import TaskProcessTree
    project=repo/'artifacts.local/unreal/BlindAssistStreetLab'
    env=dict(os.environ,BA_UE_GEOMETRY_OUTPUT=str(out))
    env['UE-LocalDataCachePath']=str(project/'DerivedDataCache')
    proc=subprocess.Popen([str(a.engine/'Engine/Binaries/Win64/UnrealEditor.exe'),str(project/'BlindAssistStreetLab.uproject'),
        '-ExecCmds=py '+(scripts/'ue_visual_geometry_probe.py').as_posix(),'-RenderOffscreen','-unattended','-nosound','-nop4','-NoSplash',
        '-ddc=NoShared','-abslog='+str(out.with_suffix('.log'))],env=env)
    tree=TaskProcessTree(proc,owner=str(out))
    try:
        code=tree.wait(timeout=180)
        result=json.loads((out/'result.json').read_text())
        if code or result['status']!='PASS':raise RuntimeError('Native geometry probe failed')
        print(json.dumps({'status':result['status'],'samples':len(result['samples'])}))
    finally:
        receipt=tree.cleanup();out.mkdir(exist_ok=True)
        (out/'process-release.json').write_text(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
