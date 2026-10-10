"""Frozen RGB and six-arm sensor features; never opens reference labels."""
import argparse,csv,json,time,hashlib,subprocess
from pathlib import Path
import numpy as np
from sync_fusion_features_v1 import FEATURES,features_for_depth
from sync_rgb_tof_dataset_v1 import peak_readout,sample
from rgb_body_query_reference_eval import rays,ray_interval

NAMES=FEATURES[:16]
ARMS=('native_perturbed','faro_rho015_ambient1','faro_rho060_ambient1','faro_rho030_ambient3','faro_rho030_ambient1','faro_rho030_ambient10')
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
ROOT=REPO/'artifacts.local/work/sync-fusion-confirm-v3-dev-20261011'
def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def save(p,o):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf8') as f:json.dump(o,f,indent=2,allow_nan=False);f.write('\n')
def protocol():
 p=load(ROOT/'PLAN.json');r=load(ROOT/'protocol_commit.json');assert sha(ROOT/'PLAN.json')==r['plan_sha256'];assert tuple(p['features']['names'])==NAMES
 rel='research/active/dtr-r0/nearfield/SYNC_FUSION_CONFIRM_V3_PROTOCOL_DEV_20261011.json'
 blob=subprocess.check_output(['git','show',r['commit']+':'+rel],cwd=REPO)
 raw=(ROOT/'PLAN.json').read_bytes()
 assert raw.replace(b'\r\n',b'\n')==blob.replace(b'\r\n',b'\n'), 'Committed protocol content changed beyond Git newline normalization'
 assert hashlib.sha256(blob).hexdigest()==r['git_blob_sha256']
 return p
def infer(budget):
 protocol();pub=load(ROOT/'public_roster.json');ad=load(ROOT/'admission_seal.json');assert sha(ROOT/'public_roster.json')==ad['public_roster_sha256'];assert len(pub['rows'])==576 and len(set(str(r['visit_id']) for r in pub['rows']))==18
 from rgb_body_query_residual_target import PUBLIC
 permitted=PUBLIC|{'role','cohort','visit_id','window_id','timestamp_s','frame_id','source_id'}
 roster=ROOT/'rgb_public_roster.json';save(roster,dict(rows=[{k:v for k,v in r.items() if k in permitted} for r in pub['rows']]))
 seal=ROOT/'rgb_observation_seal.json';save(seal,dict(public_roster_sha256=sha(roster),accepted_visit_ids=ad['accepted_visit_ids'],reference_read=False))
 import sync_rgb_tof_v1_rgb_infer as frozen
 frozen.run(argparse.Namespace(repo=REPO,runroot=ROOT/'rgb_inference',roster=roster,plan_seal=seal,budget_s=budget))
 assert load(ROOT/'rgb_inference/dual_resident_terminal.json')['status']=='COMPLETE', 'Preserve incomplete RGB terminal; no success label'
def build(budget):
 start=time.perf_counter();protocol();assert not (ROOT/'feature_manifest.json').exists();rows=[];inputs={};sim=[]
 def bind(p):inputs[str(Path(p).resolve())]=sha(p)
 for p in (ROOT/'PLAN.json',ROOT/'public_roster.json',ROOT/'synthesis_manifest.json',ROOT/'input_coverage_gate.csv',ROOT/'frozen_readout_seal.json'):bind(p)
 public=load(ROOT/'public_roster.json');queries=public['queries'];pub={r['source_id']:r for r in public['rows']};synth=load(ROOT/'synthesis_manifest.json');seal=load(ROOT/'frozen_readout_seal.json')
 with (ROOT/'input_coverage_gate.csv').open(newline='',encoding='utf8') as f:allowed={r['frame_id'] for r in csv.DictReader(f) if r['arm']=='faro_rho030_ambient1' and r['K']=='0' and int(r['joint_pass_zones'])>=52}
 pred={}
 for model in ('dav2','unidepth'):
  pred[model]={}
  for rel in ('predictions.json','sealed_eval/predictions.json'):
   p=ROOT/'rgb_inference'/model/rel;bind(p);data=load(p);assert data['status']=='COMPLETE'
   for r in data['rows']:pred[model][r['source_id']]=r
 for fi,frame in enumerate(synth['frames']):
  if time.perf_counter()-start>=budget-5:raise TimeoutError('Feature command-wall cap')
  r=pub[frame['source_id']];K=np.asarray(r['depth_K']);shape=tuple(r['depth_shape']);rx,ry=rays(K,shape);it=[ray_interval(rx,ry,q) for q in queries];rgb={}
  for model in pred:
   pr=pred[model][frame['source_id']];bind(pr['path'])
   with np.load(pr['path'],allow_pickle=False) as z:dep=z['depth']
   rgb[model]=[features_for_depth(dep,*v) for v in it]
  base={}
  for arm in ('native_perturbed','faro_rho030_ambient1'):
   for k in (0,1):
    obs=next(x for x in frame['arms'] if x['arm']==arm and x['repeat']==k);bind(obs['path']);assert sha(obs['path'])==obs['sha256']
    with np.load(obs['path'],allow_pickle=False) as z:base[arm,k]={n:z[n] for n in z.files}
  for arm in ARMS:
   if arm=='faro_rho030_ambient1' and frame['frame_id'] not in allowed:continue
   obs=[]
   for k in (0,1):
    z=dict(base[arm if arm in ('native_perturbed','faro_rho030_ambient1') else 'faro_rho030_ambient1',k])
    if arm not in ('native_perturbed','faro_rho030_ambient1'):
     ss=.5 if arm=='faro_rho015_ambient1' else 2 if arm=='faro_rho060_ambient1' else 1;aa=3 if arm=='faro_rho030_ambient3' else 10 if arm=='faro_rho030_ambient10' else 1
     hist,n,bg=sample(z['expectation'][None]*ss,z['ambient'][None]*aa,int(z['seed']));z.update(hist=hist[0],counts=n[0],background=bg[0],expectation=z['expectation']*ss,ambient=z['ambient']*aa)
     path=ROOT/'simulated'/frame['source_id']/(arm+f'_k{k}.npz');path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();np.savez_compressed(path,**z)
     sim.append(dict(path=str(path.resolve()),sha256=sha(path),arm=arm,repeat=k,source_id=frame['source_id'],frame_id=frame['frame_id'],seed=int(z['seed']),signal_scale=ss,ambient_scale=aa))
    dep=peak_readout(z['hist'],z['background'],z['coverage'],z['grid_K'],shape,z['grid_pose'],K,shape,z['rgb_pose'],seal)[0] if np.isfinite(z['grid_pose']).all() else np.full(shape,np.inf)
    obs.append([features_for_depth(dep,*v) for v in it])
   for j,q in enumerate(queries):
    s0,v0=obs[0][j];s1,v1=obs[1][j];valid=np.isfinite(s0) and np.isfinite(s1);ts=(s0+s1)/2 if valid else -np.inf;tv=(v0+v1)/2;tv[0]=ts if valid else -10.;tv[1]=float(not valid)
    ds,dv=rgb['dav2'][j];us,uv=rgb['unidepth'][j];band=0 if q['low'][2]==.3 else 1 if q['low'][2]==.8 else 2
    X=np.r_[tv[[0,1,4,3,6,5,7,8,9]],int(np.isfinite(s0))+int(np.isfinite(s1)),dv[:3],uv[:3]]
    for c,name in enumerate(NAMES):
     if name.endswith('_m'):X[c]=np.clip(X[c],-10,10)
    rows.append(dict(X=X,arm=arm,visit_id=str(frame['visit_id']),frame_id=frame['source_id'],query_id=q['name'],band=band,tof_score=ts,rgb_score=us if band==0 else ds,role=frame['role']))
  if fi%64==0:print('features',fi,len(rows),time.perf_counter()-start,flush=True)
 arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]};assert np.isfinite(arrays['X']).all();assert len(set(zip(arrays['arm'],arrays['frame_id'],arrays['query_id'])))==len(rows)
 outputs=[]
 for role in ('cal','eval'):
  mask=arrays['role']==role;path=ROOT/'features'/(role+'.npz');path.parent.mkdir(exist_ok=True);np.savez_compressed(path,**{k:v[mask] for k,v in arrays.items() if k!='role'},feature_names=np.asarray(NAMES));outputs.append(dict(role=role,path=str(path.resolve()),sha256=sha(path),rows=int(mask.sum())))
 deps=[HERE/x for x in ('sync_fusion_confirm_v3_observations.py','sync_fusion_features_v1.py','sync_rgb_tof_dataset_v1.py','rgb_body_query_reference_eval.py','rgb_body_query_query_calibration_probe.py','sync_rgb_tof_v1_rgb_infer.py','rgb_depth_backbone_adapters.py')]
 for p in deps:bind(p)
 save(ROOT/'feature_manifest.json',dict(outputs=outputs,inputs=[dict(path=k,sha256=v) for k,v in inputs.items()],simulations=sim,rows=len(rows),feature_names=list(NAMES),reference_read=False,seconds=time.perf_counter()-start))
 manifest=load(ROOT/'dataset_manifest_v3.json')
 manifest.update(base_data_manifest_sha256=sha(ROOT/'dataset_manifest_v3.json'),six_arms=list(ARMS),derived_pressure_inputs=sim,feature_manifest_sha256=sha(ROOT/'feature_manifest.json'),feature_outputs=outputs)
 save(ROOT/'dataset_manifest_v3_complete.json',manifest)
def main():
 a=argparse.ArgumentParser();a.add_argument('stage',choices=('infer','features'));a.add_argument('--budget-s',type=float,default=300);args=a.parse_args();t=time.perf_counter()
 try:
  {'infer':infer,'features':build}[args.stage](args.budget_s);save(ROOT/(args.stage+'_observation_terminal.json'),dict(status='COMPLETE',command_wall_s=time.perf_counter()-t))
 except Exception as e:
  save(ROOT/(args.stage+'_observation_failure.json'),dict(error=repr(e),command_wall_s=time.perf_counter()-t));raise
if __name__=='__main__':main()
