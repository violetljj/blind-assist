"""Scoped supervised-train versus heldout-truth admission and explicit RunSpecs."""
import json,sys
from pathlib import Path
import vpp_geometry_prepare_20260923 as reuse
ROOT=Path(__file__).resolve().parents[4];ART=ROOT/'artifacts.local';BASE=ART/'evidence/ba-stereo-adapt-20260923'
PROTOCOL='research/active/dtr-r0/nearfield/STEREO_ADAPT_PROTOCOL_20260923.md'
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)
def setup():
 blocks=[]
 for locator,scopes in [
  ('evidence/ba-stereo-adapt-20260923',[('plan','configuration')]),
  ('evidence/ba-stereo-adapt-20260923-capture-v1',[('.','evaluator')]),
  ('evidence/ba-stereo-adapt-20260923-prepared',[('train_examples','observation'),('eval_observations','observation'),('eval_truth','evaluator'),('receipt.json','configuration')]),
  ('evidence/ba-stereo-adapt-20260923-models',[('.','configuration')]),
  ('evidence/ba-stereo-adapt-20260923-predictions',[('.','evaluator')]),
  ('evidence/ba-stereo-adapt-20260923-evaluation',[('.','evaluator')])]:
  blocks.append(dict(locator=locator,source_family='ue-stereo-adapt-layouts-20260923',evidence_status='development_consumed',evidence=PROTOCOL,
   allowed_inputs=[dict(relative_path=p,role=r) for p,r in scopes]))
 policy=ROOT/'data/ue-reuse-policy.json';old=policy.read_text(encoding='utf-8');current=json.loads(old)
 assert not {b['locator'] for b in blocks}&{b['locator'] for b in current['contracts']}
 insert='\n'+',\n'.join('    '+json.dumps(b,indent=2).replace('\n','\n    ') for b in blocks)+','
 new=old.replace('"contracts": [','"contracts": ['+insert,1);json.loads(new);policy.write_bytes(new.encode())
 write(BASE/'own-policy-contracts.json',blocks)
 write(BASE/'initial-registration.json',[reuse.register(BASE/'plan','source_material'),reuse.register(ART/'unreal/BlindAssistStreetLab/BlindAssistStreetLab.uproject','source_material')])
 print('ADAPT_CAPTURE_ADMISSION_READY')
def execution_specs():
 from hashlib import sha256
 here=Path(__file__).parent
 prepared=ART/'evidence/ba-stereo-adapt-20260923-prepared'
 models=ART/'evidence/ba-stereo-adapt-20260923-models'
 predictions=ART/'evidence/ba-stereo-adapt-20260923-predictions'
 evaluation=ART/'evidence/ba-stereo-adapt-20260923-evaluation'
 config=ART/'evidence/ba-vpp-geometry-20260923/plan/config.json'
 public=ART/'evidence/ba-vpp-geometry-20260923-prepared'
 def inp(alias,path,role):return dict(alias=alias,path=str(path),role=role,purpose='fixed stereo adaptation comparison')
 def save(stage,mode,inputs,command,result):
  write(BASE/(stage+'-run-spec.json'),dict(schema='blindassist-asset-run-v1',id='stereo-adapt-20260923-'+stage,
   route='ue-stereo-adapt',question='Does bounded nearfield disparity adaptation recover supported contact geometry?',
   evaluator=PROTOCOL,evidence_boundary='New layout Development and consumed historical regression, not hardware or final confirmation',
   reuse=dict(mode=mode,query='fixed stereo update supervised adaptation grouped layouts'),inputs=inputs,
   outputs=[dict(alias='result',path=str(result),role='result',required=True)],result_output='result',command=[str(x) for x in command]))
 common=[inp('launch_seal',BASE/'plan/launch-seal.json','configuration'),inp('frontend_config',config,'configuration'),
  inp('vpp_code',ART/'work/vpp-geometry-20260923/vppstereo/vpp_standalone.py','configuration'),
  inp('model',ART/'work/foundation-geometry-20260923/weights/23-51-11','configuration'),
  inp('source',ART/'work/mz103-depth-frontend-20260912/FoundationStereo','configuration')]
 for arm in ('ordinary','balanced'):
  dest=models/arm
  save('train-'+arm,'training',common+[inp('supervised_examples',prepared/'train_examples','observation')],
   [sys.executable,here/'stereo_adapt_run_20260923.py','--stage','train','--arm',arm,'--config',config,
    '--inputs',prepared/'train_examples/examples.json','--output',dest],dest/'receipt.json')
 for arm in ('baseline','ordinary','balanced'):
  dest=predictions/arm
  inputs=common+[inp('eval_observations',prepared/'eval_observations','observation')]
  cmd=[sys.executable,here/'stereo_adapt_run_20260923.py','--stage','infer','--arm',arm,'--config',config,
    '--inputs',prepared/'eval_observations/examples.json','--output',dest]
  if arm!='baseline':
   inputs += [inp('adaptation',models/arm,'configuration')]+[inp('historical_'+n,public/n,'observation') for n in ('rgb','tof','rgb-inputs.json','tof-inputs.json')]
   cmd += ['--updates',models/arm/'final-delta.pt','--historical-rgb',public/'rgb-inputs.json','--historical-tof',public/'tof-inputs.json']
  save('infer-'+arm,'development',inputs,cmd,dest/'receipt.json')
 eval_inputs=[inp('plan',BASE/'plan','configuration'),inp('predictions',predictions,'evaluator'),
  inp('heldout_public',prepared/'eval_observations','observation'),inp('heldout_truth',prepared/'eval_truth','evaluator'),
  inp('prepared_receipt',prepared/'receipt.json','configuration'),inp('public_tof',public/'tof','observation'),
  inp('public_tof_manifest',public/'tof-inputs.json','observation'),
  inp('baseline',ART/'evidence/ba-vpp-geometry-20260923-frontend-v1','evaluator'),
  inp('prior_contact_truth',ART/'evidence/ba-surface-contact-20260923-evaluation-v1','evaluator')]
 eval_inputs += [inp(panel,ART/path,'evaluator') for panel,path in (
  ('mz101','work/mz101-stereo-tof-spatial-20260912/capture-v1'),('mz102','work/mz102-stereo-support-20260912/capture-v1'))]
 save('evaluate','diagnostic',eval_inputs,[sys.executable,here/'stereo_adapt_evaluate_20260923.py','--output',evaluation],evaluation/'summary.json')
 code=[here/name for name in ('STEREO_ADAPT_PROTOCOL_20260923.md','stereo_adapt_run_20260923.py','stereo_adapt_model.py','stereo_adapt_loss.py','stereo_adapt_evaluate_20260923.py')]
 write(BASE/'plan/launch-seal.json',dict(seed=20260923,updates=192,train_iters=8,eval_iters=32,
  code={str(p):sha256(p.read_bytes()).hexdigest() for p in code},
  inputs={str(p):sha256(p.read_bytes()).hexdigest() for p in (config,prepared/'train_examples/examples.json',prepared/'eval_observations/examples.json')}))
 write(BASE/'prepared-registration.json',reuse.register(prepared,'development_consumed'))
 print('STEREO_ADAPT_EXECUTION_SPECS_SEALED')

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='execution-specs':execution_specs()
 else:setup()
