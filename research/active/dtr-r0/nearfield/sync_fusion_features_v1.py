"""Observation-only synchronized fusion features. Never opens a reference or label file."""
from __future__ import annotations
import argparse,csv,json,time,hashlib
from pathlib import Path
import numpy as np
from sync_rgb_tof_dataset_v1 import peak_readout
from rgb_body_query_reference_eval import rays,ray_interval
from rgb_body_query_query_calibration_probe import nth_score

SOURCES=('native_perturbed','faro_rho030_ambient1')
BANDS=('0.3-0.8m','0.8-1.5m','1.5-3m')
MISSING_MARGIN=-10.
BASE=('margin16_m','margin16_missing','finite_fraction','unknown_fraction','support_pixels','inside_fraction','before_fraction','after_fraction','median_entry_offset_m','median_exit_offset_m')
FULL_FEATURES=tuple('tof_'+x for x in BASE)+('tof_valid_K_count','tof_K_margin_absdiff_m','tof_zone_snr_mean','tof_zone_snr_max','tof_accepted_zone_fraction','tof_zone_coverage_mean')+tuple('dav_'+x for x in BASE)+tuple('uni_'+x for x in BASE)+('band_index','query_x_index','query_y_index','query_z_index','domain_pixels','domain_fraction')

FEATURES=('tof_margin16_m', 'tof_margin16_missing', 'tof_support_pixels', 'tof_unknown_fraction', 'tof_before_fraction', 'tof_inside_fraction', 'tof_after_fraction', 'tof_median_entry_offset_m', 'tof_median_exit_offset_m', 'tof_valid_K_count', 'dav_margin16_m', 'dav_margin16_missing', 'dav_finite_fraction', 'uni_margin16_m', 'uni_margin16_missing', 'uni_finite_fraction', 'band_index', 'query_x_index', 'query_y_index', 'query_z_index')

def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf8')

def features_for_depth(z,entry,exit,domain):
 finite=domain&np.isfinite(z)&(z>0);n=int(domain.sum());nf=int(finite.sum());margin=np.minimum(z[finite]-entry[finite],exit[finite]-z[finite]);score=nth_score(margin);support=int((margin>=0).sum())
 values=[score if np.isfinite(score) else MISSING_MARGIN,float(not np.isfinite(score)),nf/n if n else 0.,1-nf/n if n else 1.,float(support),support/n if n else 0.,float((z[finite]<entry[finite]).sum())/n if n else 0.,float((z[finite]>exit[finite]).sum())/n if n else 0.,float(np.median(z[finite]-entry[finite])) if nf else MISSING_MARGIN,float(np.median(z[finite]-exit[finite])) if nf else MISSING_MARGIN]
 values=np.asarray(values,np.float64)
 return score,values

def quality(hist,background,coverage):
 peak=hist.argmax(-1);height=np.take_along_axis(hist,peak[...,None],-1)[...,0];bg=np.take_along_axis(background,peak[...,None],-1)[...,0];snr=height/np.sqrt(np.maximum(height,0)+2*bg+1)
 return np.array([snr.mean(),snr.max(),((coverage>=.75)&(snr>=3)).mean(),coverage.mean()],np.float64)

def run(args):
 started=time.perf_counter();root=args.root;root.mkdir(parents=True,exist_ok=True)
 plan_path=root/'PLAN.json';assert plan_path.exists(),'Wait for parent PLAN seal';assert sha(plan_path)==args.plan_sha256
 assert tuple(load(plan_path)['features']['names'])==FEATURES
 # This manifest is deliberately public-only; the source dataset manifest contains gate labels.
 roster=load(args.v1/'public_roster.json');public={r['source_id']:r for r in roster['rows']};queries=roster['queries'];assert len(queries)==27
 centers=np.array([(np.asarray(q['low'])+q['high'])/2 for q in queries]);axes=[sorted(set(centers[:,j])) for j in range(3)]
 synth=load(args.v11/'synthesis_manifest.json');seal=load(args.v11/'frozen_readout_seal.json')
 assert seal['tof_peak_snr_min']==3 and seal['tof_zone_coverage_min']==.75 and seal['query_pixel_support_min']==16
 with (args.v11/'input_coverage_gate.csv').open(newline='',encoding='utf8') as f:
  faro_allowed={r['frame_id'] for r in csv.DictReader(f) if r['arm']=='faro_rho030_ambient1' and r['K']=='0' and int(r['joint_pass_zones'])>=52}
 predictions={};inputs=[]
 def record(path):inputs.append(dict(path=str(Path(path).resolve()),sha256=sha(path)))
 for path in (plan_path,args.v1/'public_roster.json',args.v11/'synthesis_manifest.json',args.v11/'input_coverage_gate.csv',args.v11/'frozen_readout_seal.json'):record(path)
 for model in ('dav2','unidepth'):
  indexed={}
  for relative in ('predictions.json','sealed_eval/predictions.json'):
   path=args.v1/'rgb_inference'/model/relative;record(path)
   for r in load(path)['rows']:indexed[r['source_id']]=r
  predictions[model]=indexed
 groups={};admission=[]
 for frame in synth['frames']:
  if time.perf_counter()-started>=args.budget_s:raise TimeoutError('Feature command-wall allocation reached')
  role=frame['role'];assert role in ('train','cal','eval');pub=public[frame['source_id']];K=np.asarray(pub['depth_K']);shape=tuple(pub['depth_shape']);rx,ry=rays(K,shape);intervals=[ray_interval(rx,ry,q) for q in queries]
  rgb={}
  for model in ('dav2','unidepth'):
   row=predictions[model][frame['source_id']];record(row['path'])
   with np.load(row['path'],allow_pickle=False) as f:z=f['depth']
   assert z.shape==shape;rgb[model]=[features_for_depth(z,*interval) for interval in intervals]
  for source in SOURCES:
   admitted=source!='faro_rho030_ambient1' or frame['frame_id'] in faro_allowed
   admission.append(dict(source=source,role=role,visit_id=frame['visit_id'],frame_id=frame['frame_id'],admitted=admitted))
   if not admitted:continue
   observations=[];qualities=[]
   for repeat in (0,1):
    obs=next(a for a in frame['arms'] if a['arm']==source and a['repeat']==repeat);record(obs['path'])
    with np.load(obs['path'],allow_pickle=False) as f:
     if np.isfinite(f['grid_pose']).all():z,_,_,_=peak_readout(f['hist'],f['background'],f['coverage'],f['grid_K'],shape,f['grid_pose'],K,shape,f['rgb_pose'],seal)
     else:z=np.full(shape,np.inf)
     qualities.append(quality(f['hist'],f['background'],f['coverage']))
    observations.append([features_for_depth(z,*interval) for interval in intervals])
   for j,q in enumerate(queries):
    score0,v0=observations[0][j];score1,v1=observations[1][j];both=np.isfinite(score0) and np.isfinite(score1)
    tof=(score0+score1)/2 if both else -np.inf;tv=(v0+v1)/2;tv[0]=tof if both else MISSING_MARGIN;tv[1]=float(not both)
    valid_K=int(np.isfinite(score0))+int(np.isfinite(score1));dav,dv=rgb['dav2'][j];uni,uv=rgb['unidepth'][j];band=0 if q['high'][2]<=.8 else 1 if q['high'][2]<=1.5 else 2;rgbscore=uni if band==0 else dav
    low=np.asarray(q['low']);high=np.asarray(q['high']);domain=intervals[j][2];domain_n=int(domain.sum())
    vector=np.concatenate((tv,[valid_K,abs(score0-score1) if both else 0.],np.mean(qualities,axis=0),dv,uv,[band],[axes[axis].index(centers[j,axis]) for axis in range(3)],[domain_n,domain_n/np.prod(shape)]))
    assert len(vector)==len(FULL_FEATURES) and np.isfinite(vector).all()
    vector=vector[[FULL_FEATURES.index(name) for name in FEATURES]]
    meter_columns=[i for i,name in enumerate(FEATURES) if name.endswith('_m')];vector[meter_columns]=np.clip(vector[meter_columns],-10,10)
    groups.setdefault((source,role),[]).append(dict(X=vector,frame_id=frame['source_id'],sensor_frame_id=frame['frame_id'],source_id=frame['source_id'],query_id=q['name'],query_index=j,visit_id=str(frame['visit_id']),capture=str(frame['capture']),band=BANDS[band],band_index=band,tof_score=tof,rgb_score=rgbscore,or_score=max(tof,rgbscore),and_score=min(tof,rgbscore)))
  save(root/'feature_progress.json',dict(last_frame=frame['frame_id'],elapsed_s=time.perf_counter()-started,rows=sum(map(len,groups.values()))))
 outputs=[]
 for source in SOURCES:
  for role in ('train','cal','eval'):
   rows=groups.get((source,role),[]);folder=root/'features'/source;folder.mkdir(parents=True,exist_ok=True);path=folder/f'{role}.npz'
   if path.exists():raise FileExistsError('Never overwrite sealed features')
   arrays={'X':np.stack([r['X'] for r in rows]) if rows else np.empty((0,len(FEATURES)),np.float64),'feature_names':np.asarray(FEATURES)}
   for key in ('frame_id','sensor_frame_id','source_id','query_id','visit_id','capture','band'):arrays[key]=np.asarray([r[key] for r in rows],dtype=str)
   for key in ('query_index','band_index'):arrays[key]=np.asarray([r[key] for r in rows],np.int16)
   for key in ('tof_score','rgb_score','or_score','and_score'):arrays[key]=np.asarray([r[key] for r in rows],np.float64)
   np.savez_compressed(path,**arrays);outputs.append(dict(source=source,role=role,path=str(path.resolve()),sha256=sha(path),rows=len(rows),frames=len({r['frame_id'] for r in rows}),visits=sorted({r['visit_id'] for r in rows})))
 save(root/'feature_manifest.json',dict(status='COMPLETE',outputs=outputs,feature_names=FEATURES,dimension=len(FEATURES),labels_read=False,reference_paths_read=False,K2_rows_independent=False,observation_only=True,offset_K_average='Arithmetic K0/K1 mean including fixed -10 absent-offset placeholders; valid_K_count means >=16 finite query rays, not median existence',seconds=time.perf_counter()-started))
 save(root/'feature_inputs.json',inputs);save(root/'feature_admission.json',dict(rule='FARO only predeclared K0 joint_pass_zones>=52; native all frames; uniform across roles',rows=admission))
 print(json.dumps(dict(seconds=time.perf_counter()-started,outputs=outputs)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--v1',type=Path,required=True);p.add_argument('--v11',type=Path,required=True);p.add_argument('--plan-sha256',required=True);p.add_argument('--budget-s',type=float,default=400);run(p.parse_args())
