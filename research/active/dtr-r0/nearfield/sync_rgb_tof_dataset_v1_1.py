"""v1.1 static FARO surfaces; isolated empirical native perturbation; no eval labels."""
import argparse,json,time,shutil,csv
from pathlib import Path
import numpy as np
from PIL import Image
import sync_rgb_tof_dataset_v1 as V
from sync_rgb_tof_dataset_v1 import peak_readout,ray_interval,summarize
from sync_rgb_tof_pilot_dev import save,load,sha,write_csv,reproject_depth,rays

ROOT_PLAN='ccf3a6e12040aa32ce4adae7967ce03fefb23b1bff8f358f74841b38ca87bc4a'

def voxel_key(points):
 q=np.floor(points/.02).astype(np.int64)+(1<<20)
 assert np.all((q>=0)&(q<(1<<21)))
 return (q[:,0].astype(np.uint64)<<42)|(q[:,1].astype(np.uint64)<<21)|q[:,2].astype(np.uint64)

def fuse(entry,out,deadline):
 keys=[];sums=[];counts=[];receipts=[];seen=set();traj_cache={};started=time.perf_counter()
 for fr in entry['frames']:
  if time.perf_counter()>deadline:raise TimeoutError('Surface budget')
  sid=fr['source_id'];assert sid not in seen;seen.add(sid)
  tp=fr['trajectory_path'];
  if tp not in traj_cache:traj_cache[tp]=np.loadtxt(tp)
  traj=traj_cache[tp]
  try:pose,br=V.pose_at(traj,fr['timestamp_s'])
  except ValueError:receipts.append(dict(source_id=sid,status='UNKNOWN_POSE_NO_EXTRAPOLATION'));continue
  z=np.asarray(Image.open(fr['depth_path']),float)[::4,::4]/1000;yy,xx=np.nonzero(np.isfinite(z)&(z>0));d=z[yy,xx];K=np.asarray(fr['K'])
  p=np.stack(((xx*4-K[0,2])/K[0,0]*d,(yy*4-K[1,2])/K[1,1]*d,d),1)@pose[:3,:3].T+pose[:3,3]
  if 'target_from_source_world' in fr:
   transform=np.asarray(fr['target_from_source_world']);p=p@transform[:3,:3].T+transform[:3,3]
  key=voxel_key(p);uniq,inv,n=np.unique(key,return_inverse=True,return_counts=True)
  total=np.stack([np.bincount(inv,weights=p[:,j],minlength=len(uniq)) for j in range(3)],1)
  keys.append(uniq);sums.append(total);counts.append(n);receipts.append(dict(source_id=sid,status='USED',points=len(p),voxels=len(uniq),pose_bracket=br))
 if keys:
  key=np.concatenate(keys);uniq,inv,nframes=np.unique(key,return_inverse=True,return_counts=True);pp=np.concatenate(sums);nn=np.concatenate(counts)
  n=np.bincount(inv,weights=nn,minlength=len(uniq));points=np.stack([np.bincount(inv,weights=pp[:,j],minlength=len(uniq))/n for j in range(3)],1)
 else:points=np.empty((0,3));nframes=np.empty(0,int);uniq=np.empty(0,np.uint64)
 out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,points=points,source_frame_count=nframes,voxel_key=uniq)
 receipt=dict(capture=entry['capture'],visit_id=entry['visit_id'],role=entry['role'],path=str(out.resolve()),sha256=sha(out),voxels=len(points),frames_total=len(receipts),frames_used=sum(r['status']=='USED' for r in receipts),seconds=time.perf_counter()-started,frames=receipts)
 save(out.with_suffix('.json'),receipt);return receipt

def render(points,counts,pose,K,shape):
 """Exact fixed 2cm camera-plane disk test at pixel centres, nearest optical-Z."""
 p=(points-pose[:3,3])@pose[:3,:3];valid=p[:,2]>0;p=p[valid];counts=counts[valid]
 z=p[:,2];u=K[0,0]*p[:,0]/z+K[0,2];v=K[1,1]*p[:,1]/z+K[1,2]
 radx=K[0,0]*.02/z;rady=K[1,1]*.02/z
 ok=(u+radx>=0)&(u-radx<shape[1])&(v+rady>=0)&(v-rady<shape[0]);u,v,z,radx,rady,counts=[x[ok] for x in (u,v,z,radx,rady,counts)]
 depth=np.full(np.prod(shape),np.inf);source=np.zeros(np.prod(shape),np.int32)
 # Each group uses the same conservative pixel bounding radius; exact circle below.
 radius=np.ceil(np.maximum(radx,rady)+.5).astype(int);cx=np.rint(u).astype(int);cy=np.rint(v).astype(int)
 for big in np.flatnonzero(radius>32):
  yy,xx=np.indices(shape);inside=((xx-u[big])/radx[big])**2+((yy-v[big])/rady[big])**2<=1
  pixel=np.flatnonzero(inside.ravel());closer=z[big]<depth[pixel];depth[pixel[closer]]=z[big];source[pixel[closer]]=counts[big]
 for radius_value in np.unique(radius[radius<=32]):
  ii=np.flatnonzero(radius==radius_value)
  for dy in range(-radius_value,radius_value+1):
   for dx in range(-radius_value,radius_value+1):
    x=cx[ii]+dx;y=cy[ii]+dy
    hit=(x>=0)&(x<shape[1])&(y>=0)&(y<shape[0])&(((x-u[ii])/radx[ii])**2+((y-v[ii])/rady[ii])**2<=1)
    sel=ii[hit];pixel=y[hit]*shape[1]+x[hit]
    np.minimum.at(depth,pixel,z[sel])
    # Source support follows the nearest surface rather than adding overlapping disks.
    winners=z[sel]<=depth[pixel]+1e-9
    source[pixel[winners]]=counts[sel[winners]]
 return depth.reshape(shape),source.reshape(shape)

def fit_residual(old,root,deadline):
 oldm=load(old/'dataset_manifest.json');refs={f['frame_id']:f for f in load(old/'synthesis_manifest.json')['frames']};pool=[[],[],[]];receipts=[]
 for entry in oldm['entries']:
  if entry['role']=='eval':continue
  traj=np.loadtxt(entry['trajectory_path'])
  for i,fr in enumerate(entry['frames']):
   fid=f"{entry['capture']}_{fr['window_id']}_{i:03d}"
   if not refs[fid]['source_available']:continue
   if time.perf_counter()>deadline:raise TimeoutError('Residual budget')
   source_pose,_=V.pose_at(traj,fr['highres_timestamp_s']);rgb_pose,_=V.pose_at(traj,fr['rgb_timestamp_s']);native=np.asarray(Image.open(fr['native_depth_path']),float)/1000
   faro=reproject_depth(np.asarray(Image.open(fr['highres_depth_path']),float)/1000,np.asarray(fr['highres_K']),source_pose,rgb_pose,np.asarray(fr['depth_K']),tuple(fr['depth_shape']))
   paired=np.isfinite(faro)&(faro>0)&np.isfinite(native)&(native>0);sizes=[]
   for j,(lo,hi) in enumerate(((.3,.8),(.8,1.5),(1.5,3.))):
    mask=paired&(native>=lo)&(native<hi);pool[j].append((faro-native)[mask]);sizes.append(int(mask.sum()))
   receipts.append(dict(frame_id=fid,role=entry['role'],paired_band_pixels=sizes))
 arrays={f'band{j}':np.concatenate(x) if x else np.empty(0) for j,x in enumerate(pool)};np.savez_compressed(root/'residual_pool.npz',**arrays)
 save(root/'residual_fit.json',dict(frames=receipts,eval_frames_used=0,band_counts={k:len(v) for k,v in arrays.items()},signed='registered_FARO_optical_Z minus native_optical_Z',statistics={k:dict(median=float(np.median(v)),p10=float(np.quantile(v,.1)),p90=float(np.quantile(v,.9))) if len(v) else None for k,v in arrays.items()}));return arrays

def perturb(native,pool,seed):
 rng=np.random.default_rng(seed);out=np.full(native.shape,np.inf)
 for j,(lo,hi) in enumerate(((.3,.8),(.8,1.5),(1.5,3.))):
  mask=np.isfinite(native)&(native>=lo)&(native<hi);values=pool[f'band{j}']
  if len(values):out[mask]=native[mask]+rng.choice(values,size=int(mask.sum()),replace=True)
 out[~(out>0)]=np.inf;return out

def synthesize(old,root,surface_receipts,pool,deadline):
 manifest=load(old/'dataset_manifest.json');old_frames={f['frame_id']:f for f in load(old/'synthesis_manifest.json')['frames']};circular_exact=0;surfaces={str(r['capture']):r for r in surface_receipts};outputs=[];checks=[];started=time.perf_counter()
 for entry in manifest['entries']:
  sr=surfaces.get(str(entry['capture']));points=np.empty((0,3));counts=np.empty(0,int)
  if sr:
   with np.load(sr['path']) as f:points=f['points'];counts=f['source_frame_count']
  traj=np.loadtxt(entry['trajectory_path'])
  for ordinal,fr in enumerate(entry['frames']):
   if time.perf_counter()>deadline:raise TimeoutError('Synthesis budget')
   K=np.asarray(fr['depth_K']);shape=tuple(fr['depth_shape']);fid=f"{entry['capture']}_{fr['window_id']}_{ordinal:03d}"
   valid_pose=True
   try:gp,gb=V.pose_at(traj,fr['grid_timestamp_s']);rp,rb=V.pose_at(traj,fr['rgb_timestamp_s'])
   except ValueError:valid_pose=False;gp=rp=np.full((4,4),np.nan);gb=rb=dict(unavailable=True)
   geometry,support=render(points,counts,gp,K,shape) if valid_pose and len(points) else (np.full(shape,np.inf),np.zeros(shape,int))
   native=np.asarray(Image.open(fr['native_depth_path']),float)/1000
   ng=reproject_depth(native,K,rp,gp,K,shape) if valid_pose else np.full(shape,np.inf)
   target=root/'cnh'/entry['role']/fid;target.mkdir(parents=True,exist_ok=True)
   directions,_=V.S.angular_rays(16);uu=np.rint(K[0,0]*directions[...,0]/directions[...,2]+K[0,2]).astype(int);vv=np.rint(K[1,1]*directions[...,1]/directions[...,2]+K[1,2]).astype(int)
   inside=(uu>=0)&(uu<shape[1])&(vv>=0)&(vv<shape[0]);sampled=geometry[vv.clip(0,shape[0]-1),uu.clip(0,shape[1]-1)];hit=inside&np.isfinite(sampled)&(sampled>0)
   near_subray=(hit&(sampled>=.3)&(sampled<.8)).sum(-1);unknown_subray=(~hit).sum(-1)
   geometry_path=target/'input_geometry.npz';np.savez_compressed(geometry_path,depth=geometry,source_frame_count=support,grid_pose=gp,rgb_pose=rp,K=K,near_subray_hits=near_subray,unknown_subrays=unknown_subray)
   frame=dict(capture=entry['capture'],visit_id=entry['visit_id'],role=entry['role'],window_id=fr['window_id'],frame_id=fid,source_id=fr['source_id'],rgb_path=fr['rgb_path'],grid_timestamp_s=fr['grid_timestamp_s'],rgb_timestamp_s=fr['rgb_timestamp_s'],source_available=bool(np.isfinite(geometry).any()),grid_pose_bracket=gb,rgb_pose_bracket=rb,geometry_path=str(geometry_path.resolve()),arms=[])
   specs=[('faro_rho030_ambient1',.3,1),('faro_rho015_ambient1',.15,1),('faro_rho060_ambient1',.6,1),('faro_rho030_ambient3',.3,3),('native_perturbed',.3,1),('native_circular_upper',.3,1)]
   for arm,rho,ambient_factor in specs:
    fixed=ng if arm=='native_circular_upper' else geometry
    ex,amb,cov=V.expectation(fixed,K,rho,ambient_factor)
    for k in range(2):
     seedparts=[20261011,entry['accepted_ordinal'],fr['window_id'],fr['window_frame'],k];seed=int(np.random.SeedSequence(seedparts).generate_state(1)[0])
     if arm=='native_perturbed':
      pseed=int(np.random.SeedSequence([20261011,1101,*seedparts[1:]]).generate_state(1)[0]);nz=perturb(native,pool,pseed);gg=reproject_depth(nz,K,rp,gp,K,shape) if valid_pose else np.full(shape,np.inf);ex,amb,cov=V.expectation(gg,K,rho,ambient_factor)
     hist,n,bg=V.sample(ex,amb,seed)
     if arm=='native_circular_upper':
      original=next(o for o in old_frames[fid]['arms'] if o['arm']==arm and o['repeat']==k)
      with np.load(original['path']) as original_data:
       for name,arr in [('hist',hist[0]),('ambient',amb[0]),('counts',n[0]),('background',bg[0]),('coverage',cov)]:np.testing.assert_array_equal(arr,original_data[name])
      circular_exact+=1
     path=target/f'{arm}_k{k}.npz'
     np.savez_compressed(path,hist=hist[0],ambient=amb[0],counts=n[0],background=bg[0],expectation=ex[0],coverage=cov,valid=bool(np.any(cov>0)),grid_K=K,grid_pose=gp,rgb_pose=rp,rgb_K=K,seed=seed)
     frame['arms'].append(dict(arm=arm,repeat=k,path=str(path.resolve()),sha256=sha(path),coverage_mean=float(cov.mean()),valid=bool(np.any(cov>0))))
     peak=hist[0].argmax(-1);height=np.take_along_axis(hist[0],peak[...,None],-1)[...,0];back=np.take_along_axis(bg[0],peak[...,None],-1)[...,0];snr=height/np.sqrt(np.maximum(height,0)+2*back+1)
     checks.append(dict(frame_id=fid,role=entry['role'],arm=arm,K=k,geometry_pass_zones=int((cov>=.75).sum()),joint_pass_zones=int(((cov>=.75)&(snr>=3)).sum()),valid_grid_frame=bool(valid_pose),geometry_coverage=float(np.isfinite(geometry).mean()),near_subray_hits=int(near_subray.sum()),unknown_subrays=int(unknown_subray.sum()),near_hit_pixels=int(((geometry>=.3)&(geometry<.8)).sum()),unknown_pixels=int((~np.isfinite(geometry)).sum())))
   frame['faro_source_available']=frame['source_available'];frame['common_available']=bool(valid_pose and all(o['valid'] for o in frame['arms']))
   outputs.append(frame);save(root/'synthesis_partial_manifest.json',dict(status='PARTIAL',frames=outputs,eval_method_metrics_computed=False));save(root/'progress.json',dict(stage='synthesis',frames=len(outputs),seconds=time.perf_counter()-started))
 save(root/'synthesis_manifest.json',dict(status='COMPLETE',frames=outputs,eval_query_metrics_computed=False));write_csv(root/'input_coverage_gate.csv',checks);save(root/'circular_exact_parity.json',dict(checked_frames_times_K=circular_exact,status='PASS',arrays=['hist','ambient','counts','background','coverage']))

def geometry_consistency(old,root,deadline):
 old_frames={f['frame_id']:f for f in load(old/'synthesis_manifest.json')['frames']};new_frames={f['frame_id']:f for f in load(root/'synthesis_manifest.json')['frames']};rows=[]
 for entry in load(old/'dataset_manifest.json')['entries']:
  traj=np.loadtxt(entry['trajectory_path'])
  for i,fr in enumerate(entry['frames']):
   fid=f"{entry['capture']}_{fr['window_id']}_{i:03d}"
   if not old_frames[fid]['source_available']:continue
   if time.perf_counter()>deadline:raise TimeoutError('Geometry checks budget')
   source_pose,_=V.pose_at(traj,fr['highres_timestamp_s']);grid_pose,_=V.pose_at(traj,fr['grid_timestamp_s'])
   original=reproject_depth(np.asarray(Image.open(fr['highres_depth_path']),float)/1000,np.asarray(fr['highres_K']),source_pose,grid_pose,np.asarray(fr['depth_K']),tuple(fr['depth_shape']))
   with np.load(new_frames[fid]['geometry_path']) as f:reconstructed=f['depth']
   paired=np.isfinite(original)&np.isfinite(reconstructed)&(original>0)&(reconstructed>0)
   for lo,hi in ((.3,.8),(.8,1.5),(1.5,3.)):
    valid=paired&(original>=lo)&(original<hi);err=reconstructed[valid]-original[valid]
    rows.append(dict(frame_id=fid,role=entry['role'],band=f'{lo:g}-{hi:g}m',paired_pixels=int(valid.sum()),original_valid_pixels=int((np.isfinite(original)&(original>=lo)&(original<hi)).sum()),median_signed_m=float(np.median(err)) if len(err) else None,median_abs_m=float(np.median(abs(err))) if len(err) else None,p90_abs_m=float(np.quantile(abs(err),.9)) if len(err) else None))
 write_csv(root/'original51_geometry_consistency.csv',rows);save(root/'original51_geometry_consistency.json',dict(frames=len({r['frame_id'] for r in rows}),scope='Input reconstruction self-consistency includes contributing source frames; not independent accuracy or eval method metric',rows=rows))

def describe_v11(root,manifest_path,dav_manifest,uni_manifest,thresholds,budget_s=120):
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
     rows.append(dict(source_available=frame['common_available'],role=frame['role'],scan=frame['capture'],frame=frame['frame_id'],window=frame['window_id'],noise_arm=observation['arm'],repeat=observation['repeat'],arm=method,query=q['name'],band=f'{q["low"][2]:g}-{q["high"][2]:g}m',state=fr['queries'][j]['state'],support=bool(mask.sum()>=16),witness=bool((mask&(labels[j]==1)).sum()>=16),support_pixels=int(mask.sum()),positive_pixels=int((mask&(labels[j]==1)).sum())))
 write_csv(root/'train_cal_fusion_queries.csv',rows)
 for role in ('train','cal'):
  for noise_arm in {r['noise_arm'] for r in rows}:
   for repeat in (0,1):write_csv(root/f'{role}_{noise_arm}_k{repeat}_fusion_summary.csv',summarize([r for r in rows if r['role']==role and r['noise_arm']==noise_arm and r['repeat']==repeat]))
 for role in ('train','cal'):
  for noise_arm in {r['noise_arm'] for r in rows}:
   for repeat in (0,1):write_csv(root/f'{role}_{noise_arm}_k{repeat}_fusion_common_input_summary.csv',summarize([r for r in rows if r['source_available'] and r['role']==role and r['noise_arm']==noise_arm and r['repeat']==repeat]))
 save(root/'train_cal_description_receipt.json',dict(rows=len(rows),roles=['train','cal'],eval_payloads_loaded=False,threshold_sha256=sha(thresholds)))


def compact_v11(root):
 V.compact_description(root)
 obj=load(root/'train_cal_compact.json')
 for row in obj['rows']:
  if row['subset']=='FARO_available':row['subset']='all_arm_common'
 save(root/'train_cal_compact.json',obj);write_csv(root/'train_cal_compact.csv',obj['rows'])

def near_reference_diagnostic(old,root):
 frames={f['frame_id']:f for f in load(root/'synthesis_manifest.json')['frames']};rows=[]
 for entry in load(old/'dataset_manifest.json')['entries']:
  if entry['role']=='eval':continue
  for i,fr in enumerate(entry['frames']):
   fid=f"{entry['capture']}_{fr['window_id']}_{i:03d}"
   with np.load(frames[fid]['geometry_path']) as g:
    if not np.isfinite(g['grid_pose']).all():rows.append(dict(frame_id=fid,role=entry['role'],pose_valid=False,native_near_subrays=0,FARO_hit_on_native_near=0));continue
    native=np.asarray(Image.open(fr['native_depth_path']),float)/1000;K=g['K'];ng=reproject_depth(native,K,g['rgb_pose'],g['grid_pose'],K,native.shape);geometry=g['depth']
   directions,_=V.S.angular_rays(16);u=np.rint(K[0,0]*directions[...,0]/directions[...,2]+K[0,2]).astype(int);v=np.rint(K[1,1]*directions[...,1]/directions[...,2]+K[1,2]).astype(int)
   inside=(u>=0)&(u<native.shape[1])&(v>=0)&(v<native.shape[0]);u=u.clip(0,native.shape[1]-1);v=v.clip(0,native.shape[0]-1)
   near=inside&np.isfinite(ng[v,u])&(ng[v,u]>=.3)&(ng[v,u]<.8);hit=np.isfinite(geometry[v,u])&(geometry[v,u]>0)
   rows.append(dict(frame_id=fid,role=entry['role'],pose_valid=True,native_near_subrays=int(near.sum()),FARO_hit_on_native_near=int((near&hit).sum())))
 save(root/'train_cal_native_defined_near_coverage.json',dict(scope='Native-defined optical-Z near-band diagnostic; finite geometry hit coverage, not independent truth or accuracy; no eval native payload used',native_near_subrays=sum(r['native_near_subrays'] for r in rows),FARO_hit_on_native_near=sum(r['FARO_hit_on_native_near'] for r in rows),rows=rows))

def coverage_gate_summary(root):
 with (root/'input_coverage_gate.csv').open(newline='',encoding='utf8') as f:rows=list(csv.DictReader(f))
 out=[]
 for arm in sorted({x['arm'] for x in rows}):
  for k in ('0','1'):
   rr=[x for x in rows if x['arm']==arm and x['K']==k]
   out.append(dict(arm=arm,K=int(k),grid_frames=len(rr),geometry_only_pass_frames=sum(int(x['geometry_pass_zones'])>=52 for x in rr),joint_pass_frames=sum(int(x['joint_pass_zones'])>=52 for x in rr)))
 manifest=load(root/'synthesis_manifest.json');primary=next(x for x in out if x['arm']=='faro_rho030_ambient1' and x['K']==0)
 save(root/'coverage_gate_summary.json',dict(required_frames=308,required_zones_per_frame=52,denominator_frames=384,primary_pass=primary['joint_pass_frames']>=308,arms=out,FARO_source_available_frames=sum(f['source_available'] for f in manifest['frames']),all_arm_common_frames=sum(f['common_available'] for f in manifest['frames'])))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--budget-s',type=float,default=900);a=p.parse_args();started=time.perf_counter();deadline=started+a.budget_s
 assert sha(a.root/'PLAN.json')==ROOT_PLAN
 src=load(a.source);receipts=[]
 targets=load(a.old/'dataset_manifest.json')['entries'];source_index={str(e['capture']):e for e in src['entries']}
 for target in targets:
  entry=dict(source_index[str(target['capture'])]);entry['frames']=list(entry['frames'])
  if entry.get('transform_path'):
   target_T=np.load(entry['transform_path'])
   for extra in src['entries']:
    if str(extra['visit_id'])==str(entry['visit_id']) and str(extra['capture'])!=str(entry['capture']) and extra.get('transform_path'):
     transform=target_T@np.linalg.inv(np.load(extra['transform_path']))
     entry['frames'].extend(dict(fr,target_from_source_world=transform.tolist()) for fr in extra['frames'])
  out=a.root/'surfaces'/f"{entry['capture']}.npz"
  if out.exists():
   receipt=load(out.with_suffix('.json'));assert sha(out)==receipt['sha256']
   assert {r['source_id'] for r in receipt['frames']}=={fr['source_id'] for fr in entry['frames']},'Cached surface omitted source frames'
  else:receipt=fuse(entry,out,deadline)
  receipts.append(receipt);save(a.root/'surface_manifest.json',dict(entries=receipts));print('surface',entry['capture'],receipt['voxels'],receipt['seconds'],flush=True)
 pool=fit_residual(a.old,a.root,deadline);synthesize(a.old,a.root,receipts,pool,deadline);geometry_consistency(a.old,a.root,deadline)
 save(a.root/'geometry_compute.json',dict(seconds=time.perf_counter()-started));print('COMPLETE',time.perf_counter()-started)
if __name__=='__main__':main()
