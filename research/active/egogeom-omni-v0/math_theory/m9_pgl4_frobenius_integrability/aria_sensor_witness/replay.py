"""Observe two rectified Aria walks with KLT and metric poses, without depth products."""
import os
for k in ('MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[k]='1'

from collections import Counter,deque
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE=Path(__file__).resolve().parent
MATH=HERE.parents[1]
ROOT=MATH.parents[3]
ART=ROOT/'artifacts.local'
RAW=ART/'downloads/aria-gen2-pilot-20261117'
PACK=ART/'work/egogeom-omni-v0/zeroshot_eval/aria_g2_20clips'
OUT=ART/'work/egogeom-omni-v0/math_theory'/HERE.parent.name/HERE.name
M2=MATH/'m2_lie_group_scale_iekf/rgb_triangulation'
sys.path.insert(0,str(M2))
from metric_scale import MetricScale,View,camera_rays


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def inventory():
    recorded=json.loads((RAW/'RECEIPT.json').read_text())
    files={}
    for name,entry in recorded['files'].items():
        path=RAW/name
        if path.stat().st_size!=entry['bytes']:
            raise ValueError('ZIP byte count mismatch')
        with zipfile.ZipFile(path) as f:
            members=[n for n in f.namelist() if not n.endswith('/')]
        types=Counter('rectified grayscale' if 'rectified_images/' in n and n.endswith('.png') else
                      'FoundationStereo depth' if n.endswith('.png') else Path(n).name for n in members)
        suspects=[n for n in members if any(s in n.lower() for s in ('point_cloud','semidense','observation',
                              'points.csv','.vrs','.lidar','disparity','pointcloud'))]
        files[name]=dict(members=len(members),contents=dict(types),point_or_sensor_members=suspects,
                         directory_name_sha256=hashlib.sha256('\n'.join(members).encode()).hexdigest(),
                         bytes=entry['bytes'],receipt_sha1=entry['sha1'])
    return files


def recorded_poses(path,timestamps,closed):
    """Last pose stamp <= exposure. One streaming scan, keep at most two CSV rows."""
    prefix='world' if closed else 'odometry'
    tkey=['t'+k+'_'+prefix+'_device' for k in 'xyz']
    qkey=['q'+k+'_'+prefix+'_device' for k in 'xyzw']
    uid='graph_uid' if closed else 'session_uid'
    poses=[]
    with path.open(newline='') as f:
        it=iter(csv.DictReader(f))
        previous=None; nxt=next(it,None)
        for ts in timestamps:
            while nxt is not None and int(nxt['tracking_timestamp_us'])*1000<=ts:
                previous=nxt; nxt=next(it,None)
            if previous is None:
                poses.append(None)
                continue
            age=(int(ts)-1000*int(previous['tracking_timestamp_us']))*1.e-6
            if age>5 or float(previous['quality_score'])<.3:
                poses.append(None)
            else:
                pos=np.asarray([float(previous[k]) for k in tkey])
                rot=Rotation.from_quat([float(previous[k]) for k in qkey]).as_matrix()
                poses.append((rot,pos,previous[uid],age))
    return poses


def calibrate(params,closed):
    # First recorded rectified-camera/device relative rigid pose. Reuse a single
    # extrinsic forever; later refined poses only audit its residual.
    estimates=[]
    for d,c in zip(params,closed):
        if c is None:
            estimates.append(None)
            continue
        qc=d['T_world_camera']
        rc=Rotation.from_quat(qc['QuaternionXYZW']).as_matrix()
        tc=np.asarray(qc['Translation'])
        rd,td,_,_=c
        estimates.append((rd.T@rc,rd.T@(tc-td)))
    if estimates[0] is None:
        raise ValueError('Cannot anchor fixed rectification extrinsic on first exposure')
    r0,t0=estimates[0]
    rot=[np.rad2deg(Rotation.from_matrix(r0.T@r).magnitude()) for r,t in estimates if r is not None] if all(v is not None for v in estimates) else [
        np.rad2deg(Rotation.from_matrix(r0.T@v[0]).magnitude()) for v in estimates if v is not None]
    trans=[np.linalg.norm(v[1]-t0) for v in estimates if v is not None]
    return r0,t0,dict(first_index=0,valid=len(trans),angle_deg_p50_p95_max=np.percentile(rot,[50,95,100]).tolist(),
                      translation_m_p50_p95_max=np.percentile(trans,[50,95,100]).tolist())


def corridor(points,fw,right):
    if not len(points):
        return 0,False,0
    f=points[:,:2]@fw; x=points[:,:2]@right
    # Fixed nominal eye height. Do not derive floor from official stereo reference.
    h=points[:,2]+1.55
    cand=(np.abs(x)<.30)&(f>=.30)&(f<=4.)&(h>=.15)&(h<=1.9)
    vox=np.unique(np.floor(np.column_stack([x[cand],h[cand],f[cand]])/.05).astype(int),axis=0)
    z=np.sort((vox[:,2]+.5)*.05) if len(vox) else np.empty(0)
    peak=int((np.searchsorted(z,z+.30,side='right')-np.arange(len(z))).max()) if len(z) else 0
    # A clearance proxy requires measured rays traversing each 20 cm lateral x
    # two-height tile in seven range sections. Unoccupied/unobserved bins are unknown.
    sections=[]
    for zc in (.5,1.,1.5,2.,2.5,3.,3.5):
        beyond=f>zc+.30
        xx=x[beyond]*zc/f[beyond]
        hh=1.55+points[beyond,2]*zc/f[beyond]
        keep=(np.abs(xx)<.3)&(hh>=.15)&(hh<1.9)
        tile=np.floor((xx[keep]+.3)/.2).astype(int)+3*(hh[keep]>=1.025)
        sections.append(len(np.unique(tile)))
    return peak,all(k==6 for k in sections),sum(sections)


def arm_step(history,gray,K,R,t,uid,engine):
    if history and history[-1][0]!=uid:
        history.clear()
    look=R[:2,2].copy()
    if history:
        delta=(t-history[-min(len(history),2)][1].t)[:2]
        if np.linalg.norm(delta)>=.15:
            look=delta
    fw=look/max(np.linalg.norm(look),1.e-8)
    right=np.array([fw[1],-fw[0]])
    cur=View(gray,K,R,t)
    refs=[(abs(d-.32),v) for _,v in history if .12<=(d:=np.linalg.norm(v.t-t))<=1.15 and
          (v.R*R).sum()>2.45]
    history.append((uid,cur))
    matches=engine._matches(min(refs,key=lambda a:a[0])[1],cur) if refs else None
    if matches is None:
        return 0,False,0,0
    uv,z,_=matches
    points=camera_rays(uv,K)*z[:,None]@R.T
    peak,clear,tiles=corridor(points,fw,right)
    return peak,clear,tiles,len(points)


def predict(walk,names):
    src=RAW/walk/'x'
    param_path=src/'depth/pinhole_camera_parameters.json'
    traj=json.loads(param_path.read_text())
    params=traj[:len(names)*150*6:6]
    timestamps=[int(d['frameTimestampNs']) for d in params]
    closed=recorded_poses(src/'closed_loop_trajectory.csv',timestamps,True)
    op=recorded_poses(src/'open_loop_trajectory.csv',timestamps,False)
    extR,extT,extrinsic=calibrate(params,closed)
    meta={'calibration_drift':extrinsic,'parameters_sha256':sha(param_path),
          'open_trajectory_sha256':sha(src/'open_loop_trajectory.csv'),'closed_trajectory_sha256':sha(src/'closed_loop_trajectory.csv'),
          'missing_open_loop':sum(s is None for s in op),'open_asof_age_ms_p50_p95_max':np.percentile(
              [s[3] for s in op if s is not None],[50,95,100]).tolist(),'clips':{}}
    engine=MetricScale()
    for clip,name in enumerate(names):
        if int(name.stem.rsplit('c',1)[1])!=clip:
            raise ValueError('unexpected packed clip order')
        predictions={k:[] for k in ('provided','open')}
        histories={k:deque(maxlen=10) for k in predictions}
        hashstream=hashlib.sha256()
        with np.load(name) as pack:
            pack_R,pack_t=pack['poses_R'],pack['poses_t']
        for j in range(150):
            idx=clip*150+j
            item=params[idx]
            intr=item['camera']['Parameters']
            kp=np.array([[intr[0]/2,0.,(intr[2]+.5)/2-.5],
                         [0.,intr[1]/2,(intr[3]+.5)/2-.5],[0.,0.,1.]])
            img=src/'depth/rectified_images'/('image_%08d.png'%item['index'])
            data=img.read_bytes()
            hashstream.update(hashlib.sha256(data).digest())
            gr=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_GRAYSCALE)
            if gr.shape!=(512,512):
                raise ValueError(str(img))
            gr=cv2.equalizeHist(cv2.resize(gr,(256,256),interpolation=cv2.INTER_AREA))
            r=Rotation.from_quat(item['T_world_camera']['QuaternionXYZW']).as_matrix()
            t=np.array(item['T_world_camera']['Translation'])
            if not np.allclose(pack_R[j],r,atol=1.e-6,rtol=0) or not np.allclose(pack_t[j],t,atol=2.e-5,rtol=0):
                raise ValueError('camera frame and package pose mismatch')
            poses={'provided':(r,t,'provided')}
            if op[idx] is not None:
                rd,td,sid,_=op[idx]
                poses['open']=(rd@extR,rd@extT+td,sid)
            for k,rows in predictions.items():
                start=time.perf_counter()
                if k in poses:
                    values=arm_step(histories[k],gr,kp,*poses[k],engine)
                else:
                    histories[k].clear()
                    values=(0,False,0,0)
                rows.append((*values,1000*(time.perf_counter()-start)))
        target=OUT/(name.stem+'.npz')
        np.savez_compressed(target,**{k:np.array(a) for k,a in predictions.items()})
        meta['clips'][name.stem]={'input_frame_content_hash_sequence':hashstream.hexdigest(),
                                 'prediction_sha256':sha(target),'first_timestamp_ns':timestamps[clip*150],
                                 'last_timestamp_ns':timestamps[clip*150+149]}
        print(name.stem,flush=True)
    return meta


def main():
    cv2.setNumThreads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    meta={'local_archive_inventory':inventory(),'code_sha256':{p.name:sha(p) for p in (Path(__file__),M2/'metric_scale.py')},'walks':{}}
    for walk,n in [('walk_0',9),('walk_1',10)]:
        paths=sorted(PACK.glob('aria_'+walk+'_c*.npz'))
        if len(paths)!=n:
            raise ValueError('expected 19 clips')
        meta['walks'][walk]=predict(walk,paths)
    (OUT/'observation_inventory.json').write_text(json.dumps(meta,indent=2,allow_nan=False),encoding='utf8')


if __name__=='__main__':
    main()
