"""Sequential owned City captures with durable per-region receipts; no retries."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

def main():
    p=argparse.ArgumentParser()
    for key in ('specs','output','config','plugin'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--regions',nargs='+',required=True);p.add_argument('--timeout',type=float,default=1800.)
    p.add_argument('--labels',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parents[4]
    artifacts=(root/'artifacts.local').resolve()
    assert a.output.resolve().is_relative_to(artifacts) and not a.output.exists()
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'));a.output.mkdir(parents=True)
    results=[]
    try:
        for region in a.regions:
            active=subprocess.run(['pwsh','-NoProfile','-Command',
                "@(Get-CimInstance Win32_Process | Where-Object {$_.Name -like 'UnrealEditor*'}).Count"],capture_output=True,text=True,check=True)
            if int(active.stdout.strip()):raise RuntimeError('Host editor occupied; preserve other work')
            out=a.output/region;start=time.time();print('CAPTURE '+region,flush=True)
            with (a.output/(region+'-console.log')).open('x') as log:
                subprocess.run([sys.executable,'-B',str(root/'tools/run_city_pcg_capture.py'),
                    '--project',cfg['project'],'--engine',cfg['engine'],'--plugin',str(a.plugin),
                    '--spec',str(a.specs/(region+'.json')),'--output',str(out),'--timeout',str(a.timeout)],
                    cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
                subprocess.run([sys.executable,'-B',str(root/'tools/verify_city_pcg_world.py'),
                    '--capture',str(out)],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
                if a.labels:
                    subprocess.run([sys.executable,'-B',str(root/'research/active/dtr-r0/nearfield/body_query_collection_labels.py'),
                        '--captures',str(out),'--output',str(out/'labels')],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
                    labeled=json.loads((out/'labels/result.json').read_text())
                    if not labeled['complete_group_acceptance']:
                        raise RuntimeError(f'{region}: complete-group admission failed; inspect retained labels before continuing')
            results.append(dict(region=region,capture=str(out.resolve()),seconds=time.time()-start,status='PASS'))
            (a.output/'progress.json').write_text(json.dumps(dict(status='RUNNING',regions=results),indent=2))
            print('VERIFIED '+region,flush=True)
        (a.output/'completion.json').write_text(json.dumps(dict(status='PASS',regions=results),indent=2))
    except BaseException as e:
        (a.output/'completion.json').write_text(json.dumps(dict(status='FAIL',regions=results,error=repr(e)),indent=2));raise

if __name__=='__main__':main()
