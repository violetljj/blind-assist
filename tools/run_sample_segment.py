"""Build or inspect the independent WillowSampleV1 UE sample, with owned cleanup."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('build','polish','materials','inspect','audit','finish','sensors'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(repo/'research/active/dtr-r0/unreal'))
    from street_process_lifecycle import TaskProcessTree
    from run_obstacle_research import engine_root
    out=a.output.resolve()
    if not out.is_relative_to((repo/'artifacts.local').resolve()) or out.exists():
        p.error('Use a new output under artifacts.local')
    out.mkdir(parents=True)
    project=repo/'artifacts.local/unreal/BlindAssistStreetLab'
    env=dict(os.environ,BA_SAMPLE_OUTPUT=str(out),BA_SAMPLE_ACTION=a.action)
    env['UE-LocalDataCachePath']=str(project/'DerivedDataCache')
    proc=subprocess.Popen([str(engine_root()/'Engine/Binaries/Win64/UnrealEditor.exe'),
        str(project/'BlindAssistStreetLab.uproject'),
        '-ExecCmds=py '+(repo/'research/active/dtr-r0/unreal/build_sample_segment.py').as_posix(),
        '-RenderOffscreen','-unattended','-nosound','-nop4','-NoSplash','-ddc=NoShared',
        '-abslog='+str(out/'editor.log')],env=env)
    tree=TaskProcessTree(proc,owner=str(out))
    try:
        code=tree.wait(timeout=900)
        receipt=json.loads((out/'receipt.json').read_text())
        if code or receipt['status']!='PASS':
            raise RuntimeError('Sample build/inspection failed; receipt and editor log retained')
        print(json.dumps({k:receipt.get(k) for k in ('status','map','map_sha256','source_unchanged','geometry_counts')},indent=2))
    finally:
        (out/'process-release.json').write_text(json.dumps(tree.cleanup(),indent=2))


if __name__=='__main__':main()
