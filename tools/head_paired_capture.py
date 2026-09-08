import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('mode',choices=['probe','canary','all']);a=p.parse_args()
b=Path(os.environ['BLINDASSIST_ARTIFACTS'])/'work/head-paired-20260908'
old=Path(os.environ['BLINDASSIST_ARTIFACTS'])/'work/city-coverage64-20260908'
runtime=old/'runtime/source';own=b/'static-runtime-v2';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if not own.exists():
 source=own/'research/active/dtr-r0/nearfield';source.mkdir(parents=True)
 (own/'tools').mkdir()
 launch=(runtime/'tools/run_city_pcg_capture.py').read_text()
 launch=launch.replace('from run_obstacle_research import engine_root',f'import sys\nsys.path.insert(0, {str(runtime/"tools")!r})\nfrom run_obstacle_research import engine_root')
 launch=launch.replace("source = REPO / 'research/active/dtr-r0/nearfield'","source = Path(__file__).resolve().parents[1] / 'research/active/dtr-r0/nearfield'")
 (own/'tools/run_city_pcg_capture.py').write_text(launch)
 for name in ['city_pcg_capture.py','ue_pair_export.py','ue_capture_readiness.py']:
  data=(runtime/'research/active/dtr-r0/nearfield'/name).read_text()
  if name=='city_pcg_capture.py':
   control=(b/'head_paired_static_control.py').read_text()
   data=data.replace('def prepare(case):',control+'\n\ndef prepare(case):')
   anchor="            for actor in api.get_all_level_actors():\n                for cls, kind in ((u.DirectionalLightComponent,'sun'),(u.SkyLightComponent,'skylight')):"
   assert data.count(anchor)==1
   data=data.replace(anchor,"            report['head_p1_static_control']=freeze_background_wpo(u,api)\n"+anchor)
  (source/name).write_text(data)
config=json.loads((old/'runtime/worker-config.json').read_text())
name='static-probe' if a.mode=='probe' else a.mode
spec=json.loads((b/f'spec-{name}-v1.json').read_text())
spec['map_file']=config['map_file'];assert sha(Path(config['map_file']))==spec['map_sha256']
spec.update(head_p1_protocol_sha256=sha(b/'HEAD_PAIRED_20260908.md'),
 head_p1_static_control_sha256=sha(b/'head_paired_static_control.py'),
 head_p1_source_contract='BACKGROUND_WPO_DISABLED_IN_MEMORY_ALL_VARIANTS_NOT_ORIGINAL_DYNAMIC_RGB_PARITY',
 head_p1_capture_helper_sha256=sha(Path(__file__)),
 head_p1_source_revision='c70e4f530d3883932b4c2ced4161c7f8cbc64489')
sp=b/f'spec-{name}-worker-v2.json'
with sp.open('x') as f:json.dump(spec,f,indent=2)
capture=b/f'capture-{name}-v2'
for command in [
 [sys.executable,'-B',str(own/'tools/run_city_pcg_capture.py'),'--project',config['project'],'--plugin',config['plugin'],'--engine',config['engine'],'--spec',str(sp),'--output',str(capture),'--timeout','900'],
 [sys.executable,'-B',str(runtime/'tools/verify_city_pcg_world.py'),'--capture',str(capture)]]:
 subprocess.run(command,cwd=runtime,check=True)
if a.mode=='probe':
 import numpy as np
 rows=[]
 for i,j in [(0,1),(2,3)]:
  x=np.load(capture/f'evaluator/world_support/{i:04d}.npy');y=np.load(capture/f'evaluator/world_support/{j:04d}.npy')
  rows.append(dict(indices=[i,j],exactly_equal=bool(np.array_equal(x,y)),differences=int((x!=y).sum())))
 result=dict(status='PASS' if all(r['exactly_equal'] for r in rows) else 'FAIL',rows=rows)
 (capture/'repeat-admission.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 if result['status']!='PASS':raise ValueError('Repeat admission failed')
else:
 subprocess.run([sys.executable,'-B',str(b/'make_head_paired.py'),'build','--capture',str(capture),'--output',str(b/f'cache-{a.mode}-v2')],cwd=runtime,check=True)
