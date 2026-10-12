"""ARKit FARO CNH v1: true grid geometry, visit split, sealed eval inputs only.

Main, rho and ambient arms are diagnostics, never threshold selection. The
cyclic-native arm is isolated and explicitly optimistic. Eval never loads labels.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, shutil, time
from pathlib import Path
from dataclasses import replace
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
from sync_rgb_tof_pilot_dev import (S,F,quiet_electronics,sample,reproject_depth,
    zone_map,rays,ray_interval,save,load,sha,write_csv,summarize,BIN_M)


def pose_at(traj,t):
 """Interpolate official CV camera-to-world poses, no extrapolation."""
 times=traj[:,0];right=int(np.searchsorted(times,t));left=right-1
 if right<len(times) and abs(times[right]-t)<1e-8:left=right
 if left<0 or right>=len(times):raise ValueError('Grid/source pose outside trajectory: no extrapolation')
 indices=[left] if left==right else [left,right]
 rotations=Rotation.from_rotvec(traj[indices,1:4]).as_matrix()
 trans=traj[indices,4:7];c2wrot=rotations.transpose(0,2,1)
 position=-np.einsum('nij,nj->ni',c2wrot,trans)
 pose=np.eye(4)
 if left==right:pose[:3,:3]=c2wrot[0];pose[:3,3]=position[0]
 else:
  alpha=(t-times[left])/(times[right]-times[left]);pose[:3,3]=(1-alpha)*position[0]+alpha*position[1]
  pose[:3,:3]=Slerp(times[indices],Rotation.from_matrix(c2wrot))([t]).as_matrix()[0]
 return pose,dict(left_s=float(times[left]),right_s=float(times[right]),interpolated=left!=right)

def expectation(z,K,rho_value,ambient_factor):
 directions,weights=S.angular_rays(16)
 u=np.rint(K[0,0]*directions[...,0]/directions[...,2]+K[0,2]).astype(int)
 v=np.rint(K[1,1]*directions[...,1]/directions[...,2]+K[1,2]).astype(int)
 inside=(u>=0)&(u<z.shape[1])&(v>=0)&(v<z.shape[0]);axial=z[v.clip(0,z.shape[0]-1),u.clip(0,z.shape[1]-1)]
 valid=inside&np.isfinite(axial)&(axial>0)
 radial=np.where(valid,axial/directions[...,2],np.inf).reshape(1,8,8,16,16)
 rho=np.where(valid,rho_value,0).reshape(1,8,8,16,16);cosine=np.where(valid,directions[...,2],0).reshape(1,8,8,16,16)
 p,_=S.nominal_parameters();quarter=replace(p,signal_counts=p.signal_counts/4,ambient_counts=p.ambient_counts*ambient_factor/4,noise_scale=0.)
 w=weights.reshape(8,8,16,16);fine=np.empty((1,16,16,16));amb=np.empty((1,16,16))
 for qy in range(2):
  for qx in range(2):
   ys,xs=slice(qy*8,(qy+1)*8),slice(qx*8,(qx+1)*8);qw=w[:,:,ys,xs].reshape(8,8,64);fraction=qw.sum(-1)/weights.sum(-1)
   rr=rho[:,:,:,ys,xs].reshape(1,8,8,64)*(4*fraction[None,:,:,None])
   mean,aa=quiet_electronics(radial[:,:,:,ys,xs].reshape(1,8,8,64),rr,cosine[:,:,:,ys,xs].reshape(1,8,8,64),qw,quarter)
   fine[:,qy::2,qx::2]=mean.reshape(1,8,8,16,8).sum(-1);amb[:,qy::2,qx::2]=aa
 return fine.reshape(1,8,2,8,2,16).sum((2,4)),amb.reshape(1,8,2,8,2).sum((2,4)),valid.mean(-1)

def peak_readout(hist,background,coverage,grid_K,grid_shape,grid_pose,rgb_K,rgb_shape,rgb_pose,seal):
 peak=hist.argmax(-1);height=np.take_along_axis(hist,peak[...,None],-1)[...,0]
 bg=np.take_along_axis(background,peak[...,None],-1)[...,0];snr=height/np.sqrt(np.maximum(height,0)+2*bg+1)
 valid=(coverage>=seal['tof_zone_coverage_min'])&(snr>=seal['tof_peak_snr_min'])
 rx,ry,ix,iy,fov=zone_map(grid_K,grid_shape)
 radial=(peak+.5)*BIN_M;z=np.where(fov&valid[iy,ix],radial[iy,ix]/np.sqrt(1+rx*rx+ry*ry),np.inf)
 # 3D points of grid rays move through world into the RGB query-camera frame.
 rgb_z=reproject_depth(z,grid_K,grid_pose,rgb_pose,rgb_K,rgb_shape)
 return rgb_z,snr,radial,valid

def evaluate_queries(z,K,queries,labels,states,meta,minimum):
 rx,ry=rays(K,z.shape);rows=[]
 for j,q in enumerate(queries):
  entry,exit,domain=ray_interval(rx,ry,q);mask=domain&np.isfinite(z)&(z>0)&(z>=entry)&(z<=exit)
  rows.append(dict(**meta,query=q['name'],band=f'{q["low"][2]:g}-{q["high"][2]:g}m',state=states[j]['state'],support=bool(mask.sum()>=minimum),witness=bool((mask&(labels[j]==1)).sum()>=minimum),support_pixels=int(mask.sum()),positive_pixels=int((mask&(labels[j]==1)).sum())))
 return rows

def run(args):
 start=time.perf_counter();root=args.root;manifest=load(args.manifest);seal=load(args.thresholds)
 assert sha(args.thresholds)==sha(args.previous_thresholds),'Threshold source must be byte-identical'
 assert sha(root/'PLAN.json')=='4f87e9875d1c990dc1a6ad1dd2e7606ca3c6476431b1f5f69dcea46f2b4e619e'
 plan=load(root/'PLAN.json')
 assert not (root/'synthesis_manifest.json').exists(),'Never overwrite a previous run'
 queries=manifest['queries'];assert len(queries)==27
 references={r['source_id']:r for r in load(root/'train_cal_reference_roster.json')['rows'] if r['role']!='eval'}
 visits={};outputs=[];qrows=[];integrity=[];previous_elapsed=0.
 if args.resume:
  partial=load(root/'synthesis_partial_manifest.json');assert partial['manifest_sha256']==sha(args.manifest)
  outputs=partial['frames'];previous_elapsed=partial['elapsed_s']
  for frame in outputs:
   for observation in frame['arms']:assert sha(observation['path'])==observation['sha256'],'Changed previous CNH payload'
  with (root/'train_cal_queries_partial.csv').open(newline='',encoding='utf8') as f:qrows=list(csv.DictReader(f))
  for row in qrows:
   for key in ('support','witness'):row[key]=row[key]=='True'
   for key in ('frame','window','repeat','support_pixels','positive_pixels'):row[key]=int(row[key])
 complete_ids={frame['frame_id'] for frame in outputs}
 for accepted_ordinal,entry in enumerate(manifest['entries']):
  for window in sorted({fr['window_id'] for fr in entry['frames']}):
   window_frames=[fr for fr in entry['frames'] if fr['window_id']==window]
   assert len(window_frames)==16
   np.testing.assert_allclose(np.diff([fr['grid_timestamp_s'] for fr in window_frames]),.2,atol=1e-8,rtol=0)
  role=entry['role'];assert role in ('train','cal','eval');visit=str(entry['visit_id']);assert visits.setdefault(visit,role)==role
  traj=np.loadtxt(entry.get('trajectory_path',entry['frames'][0]['trajectory_path']))
  for ordinal,fr in enumerate(entry['frames']):
   fr=dict(fr)
   resume_id=f"{entry['capture']}_{fr['window_id']}_{ordinal:03d}"
   if resume_id in complete_ids:continue
   if role!='eval':fr.update(references[fr['source_id']])
   if time.perf_counter()-start>=args.budget_s:
    write_csv(root/'train_cal_queries_partial.csv',qrows)
    save(root/'synthesis_budget_stop.json',dict(status='BUDGET_STOP',completed_frames=len(outputs),elapsed_s=time.perf_counter()-start))
    raise TimeoutError('CNH v1 CPU command allocation reached')
   grid_t=fr['grid_timestamp_s'];rgb_t=fr['rgb_timestamp_s'];K=np.asarray(fr['depth_K']);shape=tuple(fr['depth_shape'])
   pose_valid=True
   try:grid_pose,gb=pose_at(traj,grid_t);rgb_pose,rb=pose_at(traj,rgb_t)
   except ValueError:
    pose_valid=False;grid_pose=rgb_pose=np.full((4,4),np.nan);gb=rb=dict(unavailable=True)
   geometric=np.full(shape,np.inf);source_valid=bool(fr.get('highres_depth_path')) and pose_valid;sb=None
   if source_valid:
    source_t=fr['highres_timestamp_s'];assert abs(source_t-grid_t)<=.1000001
    try:source_pose,sb=pose_at(traj,source_t)
    except ValueError:source_valid=False;sb=dict(unavailable=True)
   if source_valid:
    sourcez=np.asarray(Image.open(fr['highres_depth_path']),float)/1000
    geometric=reproject_depth(sourcez,np.asarray(fr['highres_K']),source_pose,grid_pose,K,shape)
   # Native arm is separate geometry only. Never open reference_path during eval.
   native=np.asarray(Image.open(fr['native_depth_path']),float)/1000
   native_grid=reproject_depth(native,K,rgb_pose,grid_pose,K,shape) if pose_valid else np.full(shape,np.inf)
   labels=states=None
   if role!='eval':
    with np.load(fr['reference_path'],allow_pickle=False) as f:labels=f['labels']
    states=fr['queries']
   frameid=f"{entry['capture']}_{fr['window_id']}_{ordinal:03d}";target=root/'cnh'/role/frameid;target.mkdir(parents=True,exist_ok=True)
   frameout=dict(capture=entry['capture'],visit_id=visit,role=role,window_id=fr['window_id'],frame_id=frameid,source_id=fr['source_id'],rgb_path=fr['rgb_path'],grid_timestamp_s=grid_t,rgb_timestamp_s=rgb_t,source_available=source_valid,grid_pose_bracket=gb,rgb_pose_bracket=rb,source_pose_bracket=sb,arms=[])
   for arm_spec in plan['arms']:
    arm=arm_spec['name'];rho=arm_spec['rho'];ambient_factor=arm_spec['ambient_scale']
    is_cyclic=arm=='native_circular_upper';geometry=native_grid if is_cyclic else geometric
    expected,ambient,coverage=expectation(geometry,K,rho,ambient_factor)
    for repeat in range(2):
     seed=int(np.random.SeedSequence([20261011,entry.get('accepted_ordinal',accepted_ordinal),int(fr['window_id']),int(fr['window_frame']),repeat]).generate_state(1)[0]);hist,counts,background=sample(expected,ambient,seed)
     path=target/f'{arm}_k{repeat}.npz'
     np.savez_compressed(path,hist=hist[0],ambient=ambient[0],counts=counts[0],background=background[0],expectation=expected[0],coverage=coverage,valid=bool(np.any(coverage>0)),source_available=pose_valid if is_cyclic else source_valid,grid_K=K,grid_pose=grid_pose,rgb_pose=rgb_pose,rgb_K=K,seed=seed)
     frameout['arms'].append(dict(arm=arm,repeat=repeat,path=str(path.resolve()),sha256=sha(path),coverage_mean=float(coverage.mean()),valid=bool(np.any(coverage>0))))
     if role!='eval':
      if pose_valid:rgb_z,snr,radial,accepted=peak_readout(hist[0],background[0],coverage,K,shape,grid_pose,K,shape,rgb_pose,seal)
      else:rgb_z=np.full(shape,np.inf)
      qrows.extend(evaluate_queries(rgb_z,K,queries,labels,states,dict(role=role,scan=str(entry['capture']),frame=ordinal,window=fr['window_id'],arm=arm,repeat=repeat),seal['query_pixel_support_min']))
   outputs.append(frameout);integrity.append(dict(role=role,capture=entry['capture'],frame_id=frameid,source_available=source_valid,registered_coverage=float(np.mean(np.isfinite(geometric)))))
   save(root/'synthesis_progress.json',dict(completed_frames=len(outputs),elapsed_s=time.perf_counter()-start))
   save(root/'synthesis_partial_manifest.json',dict(status='PARTIAL',frames=outputs,manifest_sha256=sha(args.manifest),elapsed_s=previous_elapsed+time.perf_counter()-start,eval_method_metrics_computed=False))
 save(root/'synthesis_manifest.json',dict(status='COMPLETE',frames=outputs,manifest_sha256=sha(args.manifest),threshold_sha256=sha(args.thresholds),elapsed_s=previous_elapsed+time.perf_counter()-start,eval_method_metrics_computed=False))
 integrity=[dict(role=f['role'],capture=f['capture'],frame_id=f['frame_id'],source_available=f['source_available'],subray_coverage_mean=next(a['coverage_mean'] for a in f['arms'] if a['arm']=='faro_rho030_ambient1' and a['repeat']==0)) for f in outputs]
 write_csv(root/'train_cal_queries.csv',qrows);write_csv(root/'input_coverage.csv',integrity)
 for role in ('train','cal'):
  for repeat in (0,1):write_csv(root/f'{role}_k{repeat}_summary.csv',summarize([r for r in qrows if r['role']==role and r['repeat']==repeat]))
 print(json.dumps(dict(frames=len(outputs),train_cal_query_rows=len(qrows),seconds=time.perf_counter()-start)))

def describe_with_rgb(root,manifest_path,dav_manifest,uni_manifest,thresholds,budget_s=120):
 """Train/cal only. Eval prediction files are indexed as metadata but never opened."""
 started=time.perf_counter()
 manifest=load(manifest_path);synthesis=load(root/'synthesis_manifest.json');seal=load(thresholds)
 source={}
 references={r['source_id']:r for r in load(root/'train_cal_reference_roster.json')['rows'] if r['role']!='eval'}
 for entry in manifest['entries']:
  if entry['role']=='eval':continue
  for ordinal,fr in enumerate(entry['frames']):source[f"{entry['capture']}_{fr['window_id']}_{ordinal:03d}"]=dict(fr,**{k:v for k,v in references[fr['source_id']].items() if k not in fr})
 def index(path):
  return {str(r.get('source_id',r.get('rgb_path'))):r for r in load(path)['rows']}
 predictions={'dav':index(dav_manifest),'uni':index(uni_manifest)}
 rows=[]
 for frame in synthesis['frames']:
  if time.perf_counter()-started>=budget_s:raise TimeoutError('Train/cal description allocation reached')
  if frame['role']=='eval':continue
  assert frame['role'] in ('train','cal')
  fr=source[frame['frame_id']];K=np.asarray(fr['depth_K']);shape=tuple(fr['depth_shape']);zz={}
  for arm in ('dav','uni'):
   indexed=predictions[arm];pr=indexed.get(fr['source_id'],indexed.get(fr['rgb_path']));assert pr is not None
   path=Path(pr.get('depth_path',pr.get('path')))
   if path.suffix=='.npy':zz[arm]=np.load(path,allow_pickle=False)
   else:
    with np.load(path,allow_pickle=False) as f:zz[arm]=f['depth']
   assert zz[arm].shape==shape
  with np.load(fr['reference_path'],allow_pickle=False) as f:labels=f['labels']
  rx,ry=rays(K,shape)
  for observation in frame['arms']:
   with np.load(observation['path'],allow_pickle=False) as f:
    if np.isfinite(f['grid_pose']).all():tof,_,_,_=peak_readout(f['hist'],f['background'],f['coverage'],f['grid_K'],shape,f['grid_pose'],K,shape,f['rgb_pose'],seal)
    else:tof=np.full(shape,np.inf)
   for j,q in enumerate(manifest['queries']):
    entry,exit,domain=ray_interval(rx,ry,q);near=q['high'][2]<=.8
    masks={}
    for name,z,cut,active in [('tof',tof,0.,True),('dav',zz['dav'],.24403834342956543,not near),('uni',zz['uni'],.09616100788116455,near)]:
     masks[name]=domain&np.isfinite(z)&(z>0)&(np.minimum(z-entry,exit-z)>=cut)&active
    rgb=masks['uni'] if near else masks['dav'];masks.update(rgb_banded=rgb,OR=masks['tof']|rgb,AND=masks['tof']&rgb)
    for method,mask in masks.items():
     rows.append(dict(source_available=frame['source_available'],role=frame['role'],scan=frame['capture'],frame=frame['frame_id'],window=frame['window_id'],noise_arm=observation['arm'],repeat=observation['repeat'],arm=method,query=q['name'],band=f'{q["low"][2]:g}-{q["high"][2]:g}m',state=fr['queries'][j]['state'],support=bool(mask.sum()>=16),witness=bool((mask&(labels[j]==1)).sum()>=16),support_pixels=int(mask.sum()),positive_pixels=int((mask&(labels[j]==1)).sum())))
 write_csv(root/'train_cal_fusion_queries.csv',rows)
 for role in ('train','cal'):
  for noise_arm in {r['noise_arm'] for r in rows}:
   for repeat in (0,1):write_csv(root/f'{role}_{noise_arm}_k{repeat}_fusion_summary.csv',summarize([r for r in rows if r['role']==role and r['noise_arm']==noise_arm and r['repeat']==repeat]))
 for role in ('train','cal'):
  for noise_arm in {r['noise_arm'] for r in rows}:
   for repeat in (0,1):write_csv(root/f'{role}_{noise_arm}_k{repeat}_fusion_FARO_available_summary.csv',summarize([r for r in rows if r['source_available'] and r['role']==role and r['noise_arm']==noise_arm and r['repeat']==repeat]))
 save(root/'train_cal_description_receipt.json',dict(rows=len(rows),roles=['train','cal'],eval_payloads_loaded=False,threshold_sha256=sha(thresholds)))

def compact_description(root):
 """Merge train+cal tables, preserving independent K denominators and source subset."""
 groups={}
 with (root/'train_cal_fusion_queries.csv').open(newline='',encoding='utf8') as f:
  for row in csv.DictReader(f):
   if row['arm'] not in ('tof','rgb_banded','OR','AND'):continue
   for subset in ['all_grid']+(['FARO_available'] if row['source_available']=='True' else []):
    key=(subset,row['noise_arm'],row['repeat'],row['arm'],row['band'])
    g=groups.setdefault(key,dict(subset=subset,noise_arm=row['noise_arm'],K=int(row['repeat']),method=row['arm'],band=row['band'],queries=0,POS=0,W=0,FREE=0,F=0,UNKNOWN=0,U=0))
    g['queries']+=1
    if row['state']=='POSITIVE':g['POS']+=1;g['W']+=row['witness']=='True'
    elif row['state']=='FREE_ON_SAMPLED_RAYS':g['FREE']+=1;g['F']+=row['support']=='True'
    elif row['state']=='UNKNOWN':g['UNKNOWN']+=1;g['U']+=row['support']=='True'
    else:raise AssertionError(row['state'])
 rows=[groups[k] for k in sorted(groups)];write_csv(root/'train_cal_compact.csv',rows)
 save(root/'train_cal_compact.json',dict(roles=['train','cal'],K_not_pooled=True,rows=rows))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--thresholds',type=Path,required=True);p.add_argument('--previous-thresholds',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--budget-s',type=float,default=160);run(p.parse_args())
