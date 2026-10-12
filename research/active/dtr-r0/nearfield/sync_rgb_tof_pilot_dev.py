"""Consumed Development FARO registered-depth CNH feasibility, label-blind renderer."""
from __future__ import annotations
import argparse, json, time, hashlib, csv
from pathlib import Path
from dataclasses import replace, asdict
from collections import defaultdict
import numpy as np
from PIL import Image
from cnh_pose_expected_only import quiet_electronics, S, F
from cnh_displacement_ceiling_render import sample
from rgb_body_query_reference_eval import rays, ray_interval

EDGE=np.tan(np.deg2rad(22.5))
BIN_M=8*F.RAW_BIN_M
DAV_CUT=.24403834342956543
UNI_CUT=.09616100788116455

def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf8')
def load(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_csv(p,rows):
 if rows:
  with Path(p).open('w',newline='',encoding='utf8') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def pincam(p):
 w,h,fx,fy,cx,cy=np.loadtxt(p);return np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]),(int(h),int(w))
def render_registered_depth(z,K,seed=20261010):
 """First-visible nearest-pixel intersection; optical Z→radial, no reference input.
 Piecewise frontoparallel depth pixels imply cosine=ray_z; unknown retains angular mass.
 """
 directions,weights=S.angular_rays(16)
 u=np.rint(K[0,0]*directions[...,0]/directions[...,2]+K[0,2]).astype(int)
 v=np.rint(K[1,1]*directions[...,1]/directions[...,2]+K[1,2]).astype(int)
 inside=(u>=0)&(u<z.shape[1])&(v>=0)&(v<z.shape[0])
 axial=z[v.clip(0,z.shape[0]-1),u.clip(0,z.shape[1]-1)]
 valid=inside&np.isfinite(axial)&(axial>0)
 radial=np.where(valid,axial/directions[...,2],np.inf).reshape(1,8,8,16,16)
 rho=np.where(valid,.3,0).reshape(1,8,8,16,16)
 cosine=np.where(valid,directions[...,2],0).reshape(1,8,8,16,16)
 p,_=S.nominal_parameters();quarter=replace(p,signal_counts=p.signal_counts/4,ambient_counts=p.ambient_counts/4,noise_scale=0.)
 w=weights.reshape(8,8,16,16);fine=np.empty((1,16,16,16));amb=np.empty((1,16,16))
 for qy in range(2):
  for qx in range(2):
   ys,xs=slice(qy*8,(qy+1)*8),slice(qx*8,(qx+1)*8)
   qw=w[:,:,ys,xs].reshape(8,8,64);fraction=qw.sum(-1)/weights.sum(-1)
   rr=rho[:,:,:,ys,xs].reshape(1,8,8,64)*(4*fraction[None,:,:,None])
   mean,aa=quiet_electronics(radial[:,:,:,ys,xs].reshape(1,8,8,64),rr,cosine[:,:,:,ys,xs].reshape(1,8,8,64),qw,quarter)
   fine[:,qy::2,qx::2]=mean.reshape(1,8,8,16,8).sum(-1);amb[:,qy::2,qx::2]=aa
 expectation=fine.reshape(1,8,2,8,2,16).sum((2,4));ambient=amb.reshape(1,8,2,8,2).sum((2,4))
 hist,counts,background=sample(expectation,ambient,seed)
 peak=hist[0].argmax(-1);height=np.take_along_axis(hist[0],peak[...,None],axis=-1)[...,0]
 # Observable background estimate; no expected signal or native labels in peak gate.
 bg=np.take_along_axis(background[0],peak[...,None],axis=-1)[...,0]
 snr=height/np.sqrt(np.maximum(height,0)+2*bg+1)
 return dict(hist=hist[0],ambient=ambient[0],expectation=expectation[0],counts=counts[0],background=background[0],coverage=valid.mean(-1),weighted_coverage=(valid*weights).sum(-1)/weights.sum(-1),peak_range=(peak+.5)*BIN_M,snr=snr)
def camera_pose(traj,timestamp):
 from scipy.spatial.transform import Rotation
 row=traj[int(abs(traj[:,0]-timestamp).argmin())]
 ext=np.eye(4);ext[:3,:3]=Rotation.from_rotvec(row[1:4]).as_matrix();ext[:3,3]=row[4:7]
 return np.linalg.inv(ext),float(row[0]-timestamp)
def reproject_depth(z,K,source_pose,target_pose,target_K,target_shape):
 yy,xx=np.nonzero(np.isfinite(z)&(z>0));dd=z[yy,xx]
 xyz=np.stack(((xx-K[0,2])/K[0,0]*dd,(yy-K[1,2])/K[1,1]*dd,dd),axis=1)
 transform=np.linalg.inv(target_pose)@source_pose
 points=xyz@transform[:3,:3].T+transform[:3,3]
 positive=points[:,2]>0;points=points[positive]
 u=np.rint(target_K[0,0]*points[:,0]/points[:,2]+target_K[0,2]).astype(int)
 v=np.rint(target_K[1,1]*points[:,1]/points[:,2]+target_K[1,2]).astype(int)
 ok=(u>=0)&(u<target_shape[1])&(v>=0)&(v<target_shape[0])
 result=np.full(np.prod(target_shape),np.inf)
 np.minimum.at(result,v[ok]*target_shape[1]+u[ok],points[ok,2])
 return result.reshape(target_shape)

def zone_map(K,shape):
 rx,ry=rays(K,shape);valid=(abs(rx)<=EDGE)&(abs(ry)<=EDGE)
 ix=np.floor((rx+EDGE)/(2*EDGE)*8).astype(int).clip(0,7);iy=np.floor((ry+EDGE)/(2*EDGE)*8).astype(int).clip(0,7)
 return rx,ry,ix,iy,valid

def summarize(rows):
 out=[]
 for arm in sorted({r['arm'] for r in rows}):
  for band in ('0.3-0.8m','0.8-1.5m','1.5-3m'):
   rr=[r for r in rows if r['arm']==arm and r['band']==band];pos=[r for r in rr if r['state']=='POSITIVE'];free=[r for r in rr if r['state']=='FREE_ON_SAMPLED_RAYS'];unk=[r for r in rr if r['state']=='UNKNOWN']
   assert len(pos)+len(free)+len(unk)==len(rr)
   out.append(dict(arm=arm,band=band,queries=len(rr),POS=len(pos),W=sum(r['witness'] for r in pos),FREE=len(free),F=sum(r['support'] for r in free),UNKNOWN=len(unk),U=sum(r['support'] for r in unk)))
 return out

def sanity_summary(zones,frames,out):
 rows=[]
 for lo,hi in ((.3,.8),(.8,1.5),(1.5,3),(3,100)):
  for accepted in (False,True):
   rr=[r for r in zones if r['native_radial_median'] is not None and lo<=r['native_radial_median']<hi and (r['accepted'] or not accepted)]
   errors=np.array([r['peak_radial_error'] for r in rr]);cov=np.array([r['coverage'] for r in rr])
   rows.append(dict(band=f'{lo:g}-{hi:g}m',subset='accepted' if accepted else 'all',zones=len(rr),median_absolute_error_m=float(np.median(abs(errors))) if len(rr) else None,p90_absolute_error_m=float(np.quantile(abs(errors),.9)) if len(rr) else None,median_signed_error_m=float(np.median(errors)) if len(rr) else None,mean_coverage=float(cov.mean()) if len(rr) else None))
 masks=[]
 for fr in frames:
  with np.load(fr['sample']) as z:
   native=z['native_z'];registered=z['registered_z'];_,_,_,_,fov=zone_map(z['target_K'],native.shape)
   near=fov&np.isfinite(native)&(native>=.3)&(native<.8)
   masks.append(dict(scan=fr['scan'],frame=fr['frame'],tof_fov_pixels=int(fov.sum()),native_valid_fov_pixels=int((fov&np.isfinite(native)&(native>0)).sum()),native_near_fov_pixels=int(near.sum()),geometry_missing_on_native_near_pixels=int((near&~(np.isfinite(registered)&(registered>0))).sum())))
 K=np.array([[220.,0.,127.5],[0.,220.,95.5],[0.,0.,1.]])
 plane=render_registered_depth(np.full((192,256),2.),K,20261010)
 save(out/'sanity.json',dict(error_by_native_radial_band=rows,near_missing=masks,all_zone_mean_coverage=float(np.mean([r['coverage'] for r in zones])),accepted_zones=sum(r['accepted'] for r in zones),zone_count=len(zones),nominal_frontoparallel_2m_plane=dict(peak_snr_median=float(np.median(plane['snr'])),ambient_median=float(np.median(plane['ambient'])),peak_range_median=float(np.median(plane['peak_range']))),pilot_peak_snr_median=float(np.median([r['peak_snr'] for r in zones])),pilot_ambient_median=float(np.median([r['ambient'] for r in zones]))))
 write_csv(out/'peak_errors_by_band.csv',rows)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[4]);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--budget-s',type=float,default=180);a=ap.parse_args();start=time.perf_counter()
 root=a.repo/'artifacts.local/work';prior=root/'rgb-near-readout-scale-dev-20261010';out=a.output;out.mkdir(parents=True,exist_ok=True)
 if (out/'results.json').exists():raise FileExistsError('Preserve completed run')
 plan=load(out.parent/'implementation_seal.json');assert plan['tof_peak_snr_min']==3 and plan['tof_zone_coverage_min']==.75
 pred={arm:{(r['scan'],r['frame']):r for r in load(prior/model/'predictions.json')['rows']} for arm,model in [('dav','dav2'),('uni','unidepth')]}
 allrows=[];zones=[];frames=[];pairing=[];inputs=[]
 for scan in ('41069021','41069048'):
  mp=prior/'new-eval-sensor'/('new_arkit_'+scan)/'dataset_manifest.json';m=load(mp);inputs.append(dict(path=str(mp),sha256=sha(mp)))
  src=out.parent/'sources'/scan;traj=np.loadtxt(src/'lowres_wide.traj');inputs.append(dict(path=str(src/'lowres_wide.traj'),sha256=sha(src/'lowres_wide.traj')));paths=sorted((src/'highres_depth').glob('*.png'));times=np.array([float(p.stem.split('_')[-1]) for p in paths]);t0=m['rows'][0]['timestamp_s']
  for ref in m['rows']:
   if time.perf_counter()-start>a.budget_s:raise TimeoutError('Allocated command wall exceeded')
   ts=ref['timestamp_s'];idx=int(abs(times-ts).argmin());dt=float(times[idx]-ts);target=t0+round((ts-t0)/.2)*.2
   pairing.append(dict(scan=scan,frame=ref['frame'],rgb_timestamp=ts,nominal_5Hz_target=target,rgb_to_grid_delta_s=ts-target,geometry_timestamp=float(times[idx]),geometry_to_rgb_delta_s=dt,accepted=abs(dt)<=.1000001))
   if abs(dt)>.1000001:continue
   path=paths[idx];kp=src/'wide_intrinsics'/(path.stem+'.pincam');K,shape=pincam(kp)
   z=np.asarray(Image.open(path),dtype=np.float64)/1000.;assert z.shape==shape
   source_pose,source_pose_dt=camera_pose(traj,times[idx]);target_pose,target_pose_dt=camera_pose(traj,ts)
   source_rx,source_ry=rays(K,z.shape);source_fov=bool(source_rx.min()<=-EDGE and source_rx.max()>=EDGE and source_ry.min()<=-EDGE and source_ry.max()>=EDGE)
   registered=reproject_depth(z,K,source_pose,target_pose,np.asarray(ref['depth_K']),ref['depth_shape'])
   data=render_registered_depth(registered,np.asarray(ref['depth_K']),20261010+int(scan)+ref['frame']);npz=out/(path.stem+'_cnh.npz');
   with np.load(ref['reference_path'],allow_pickle=False) as reference_file:native_saved=reference_file['depth']
   np.savez_compressed(npz,**data,geometry_K=K,registered_z=registered,native_z=native_saved,target_K=ref['depth_K'],source_pose=source_pose,target_pose=target_pose,seed=20261010+int(scan)+ref['frame'])
   inputs.extend(dict(path=str(p),sha256=sha(p)) for p in (path,kp,Path(ref['reference_path'])))
   with np.load(ref['reference_path'],allow_pickle=False) as f:
    labels=f['labels'];native=f['depth'] if 'depth' in f.files else None
   rx,ry,ix,iy,fov=zone_map(ref['depth_K'],ref['depth_shape']);good=(data['coverage']>=.75)&(data['snr']>=3)
   tofvalid=fov&good[iy,ix];tofz=data['peak_range'][iy,ix]/np.sqrt(1+rx*rx+ry*ry)
   zz={}
   for arm in ('dav','uni'):
    pp=Path(pred[arm][(ref['scan'],ref['frame'])]['path']);inputs.append(dict(path=str(pp),sha256=sha(pp)))
    if pp.suffix=='.npy':zz[arm]=np.load(pp,allow_pickle=False)
    else:
     with np.load(pp,allow_pickle=False) as f:zz[arm]=f['depth']
   # Native LiDAR z is in retained source PNG, never passed to synthesis.
   if native is None:
    candidates=list((prior/'source').rglob(ref['source_id']+'.png')) if (prior/'source').exists() else []
    candidates=[p for p in candidates if 'depth' in str(p).lower()]
    if len(candidates)==1:native=np.asarray(Image.open(candidates[0]),float)/1000
   for y in range(8):
    for x in range(8):
     mask=fov&(ix==x)&(iy==y);nz=np.array([]) if native is None else native[mask&(native>0)&np.isfinite(native)]
     median=None if not len(nz) else float(np.median(nz));radnative=None if not len(nz) else float(np.median((native*np.sqrt(1+rx*rx+ry*ry))[mask&(native>0)&np.isfinite(native)]))
     zones.append(dict(scan=scan,frame=ref['frame'],zone_y=y,zone_x=x,coverage=float(data['coverage'][y,x]),peak_snr=float(data['snr'][y,x]),ambient=float(data['ambient'][y,x]),accepted=bool(good[y,x]),peak_range=float(data['peak_range'][y,x]),native_z_median=median,native_radial_median=radnative,peak_radial_error=None if radnative is None else float(data['peak_range'][y,x]-radnative)))
   # Preserve old public finite DepthPro domain used by the RGB reference evaluation.
   dpindex=load(prior/'depthpro/predictions.json')['rows'];dpr=next(r for r in dpindex if (r['scan'],r['frame'])==(ref['scan'],ref['frame']))
   inputs.append(dict(path=dpr['path'],sha256=sha(dpr['path'])))
   with np.load(dpr['path'],allow_pickle=False) as f:dp=f['depth']
   public=np.isfinite(dp)&(dp>0)
   for j,q in enumerate(m['queries']):
    entry,exit,domain=ray_interval(rx,ry,q);near=q['high'][2]<=.8
    base={}
    for arm,zval,valid,cut,active in [('tof',tofz,tofvalid,0.,True),('dav',zz['dav'],np.isfinite(zz['dav'])&(zz['dav']>0),DAV_CUT,not near),('uni',zz['uni'],np.isfinite(zz['uni'])&(zz['uni']>0),UNI_CUT,near)]:
     base[arm]=public&domain&valid&(np.minimum(zval-entry,exit-zval)>=cut)&active
    rgb=base['uni'] if near else base['dav']
    arms=dict(base,rgb_banded=rgb,tof_OR_rgb_banded=base['tof']|rgb,tof_AND_rgb_banded=base['tof']&rgb,tof_OR_dav=base['tof']|base['dav'],tof_AND_dav=base['tof']&base['dav'],tof_OR_uni=base['tof']|base['uni'],tof_AND_uni=base['tof']&base['uni'])
    for arm,mask in arms.items():
     allrows.append(dict(scan=scan,frame=ref['frame'],query=q['name'],band=f'{q["low"][2]:g}-{q["high"][2]:g}m',state=ref['queries'][j]['state'],arm=arm,support=bool(mask.sum()>=16),witness=bool((mask&(labels[j]==1)).sum()>=16),support_pixels=int(mask.sum()),positive_pixels=int((mask&(labels[j]==1)).sum())))
   frames.append(dict(scan=scan,frame=ref['frame'],source=str(path),sample=str(npz),geometry_delta_s=dt,source_pose_delta_s=source_pose_dt,target_pose_delta_s=target_pose_dt,highres_fov_covered=source_fov,coverage_mean=float(data['coverage'].mean()),fov_covered=bool(rx.min()<=-EDGE and rx.max()>=EDGE and ry.min()<=-EDGE and ry.max()>=EDGE)))
 sanity_summary(zones,frames,out)
 write_csv(out/'query_table.csv',allrows);write_csv(out/'zones.csv',zones);write_csv(out/'pairing.csv',pairing);write_csv(out/'summary.csv',summarize(allrows));save(out/'inputs.json',inputs)
 save(out/'results.json',dict(status='COMPLETE',seconds=time.perf_counter()-start,frames=frames,summary=summarize(allrows),pairs=pairing,bin_width_m=BIN_M,nominal_parameters=asdict(S.nominal_parameters()[0]),limitations=['FARO independent scanner but official low-high filtering induces selection correlation','Piecewise frontoparallel depth surface and cosine proxy; constant rho=.3','Sparse consumed anchors; matching <=.1s then static FARO surface reprojected through nearest trajectory poses; pose timestamp residual retained; no continuous synchronized performance claim','Query labels are native-LiDAR references, not contact events; 16 pixels not 16 independent ToF measurements']))
 print(json.dumps(dict(frames=len(frames),queries=len(allrows),seconds=time.perf_counter()-start)))
if __name__=='__main__':main()
