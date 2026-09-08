"""Task-isolated worker capture using the admitted immutable coverage runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['canary','all']);p.add_argument('--version',default='v3');a=p.parse_args()
    b=Path(os.environ['BLINDASSIST_ARTIFACTS'])/'work/body-query-v1-20260908'
    old=Path(os.environ['BLINDASSIST_ARTIFACTS'])/'work/city-coverage64-20260908'
    runtime=b/'runtime/source';cfg=json.loads((old/'runtime/worker-config.json').read_text())
    spec=json.loads((b/f'source-{a.version}/spec-{a.mode}-v1.json').read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(Path(cfg['map_file']))==spec['map_sha256']
    spec['map_file']=cfg['map_file']
    sp=b/f'spec-{a.mode}-worker-{a.version}.json'
    with sp.open('x') as f:json.dump(spec,f,indent=2)
    out=b/f'capture-{a.mode}-{a.version}'
    for cmd in ([sys.executable,'-B',str(runtime/'tools/run_city_pcg_capture.py'),'--project',cfg['project'],
                 '--plugin',cfg['plugin'],'--engine',cfg['engine'],'--spec',str(sp),'--output',str(out),'--timeout','900'],
                [sys.executable,'-B',str(runtime/'tools/verify_city_pcg_world.py'),'--capture',str(out)]):
        subprocess.run(cmd,cwd=runtime,check=True)
    report=json.loads((out/'world-verification.json').read_text())
    expected={'CLEAR':[0,0],'BODY_ONLY':[1,0],'HEAD_ONLY':[0,1],'BOTH':[1,1]}
    expected.update({k:[0,0] for k in ('LOW','ABOVE','LATERAL_OUT','FAR_OUT')})
    rows=[dict(index=i,group=c['group_id'],role=c['source_role'],family=c['condition']['family'],
               expected=[0,0] if c['floor_check'] else expected[c['variant_id']],actual=r['body_head_visible_targets'])
          for i,(c,r) in enumerate(zip(spec['cases'],report['rows']))]
    admission=dict(status=report['status'],native_intent_matches=sum(r['expected']==r['actual'] for r in rows),
                   count=len(rows),rows=rows,source_spec_sha256=sha(sp),
                   scope='Intent discrepancies retained; native geometry remains authoritative')
    (out/'query-native-admission.json').write_text(json.dumps(admission,indent=2))
    print(json.dumps({k:v for k,v in admission.items() if k!='rows'}))

if __name__=='__main__':main()
