"""Consumed phone Development: thin candidate motion observability, never safety.

The inference stage reads fixed autonomous ridge candidates and RGB/IMU only.
Weak labels are loaded only by --evaluate after all feature files are complete.
Nominal IMU derotation is an exploratory readout: exposure timing and absolute
camera/IMU alignment are unverified. No DAV metric depth, alarm or veto is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from atom_camera import AtomCamera, ROOT

OUT = ROOT / 'artifacts.local/work/thin-information-dev-20261014/approach'
SOURCE = ROOT / 'artifacts.local/work/thin-local-geometry-dev-20261013/camera_chain_frames.jsonl'
RAW = ROOT / 'artifacts.local/work/rgb-real-feasibility-20261012/raw'
KEYS = ('073554', '073635', '073740', '073821')
CONFIG = dict(max_pair_gap_s=.5, fb_px=1.5, min_points=4,
              min_line_span=.25, point_distance_px=20, scale_mad_max=.08,
              corridor_radius_px=5, max_corners=20, seeds=9,
              imu_gap_max_us=30000, timing_offsets_us=[-100000, 0, 100000])


def read(path):
    return [json.loads(q) for q in path.read_text(encoding='utf-8-sig').splitlines() if q.strip()]


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parts(c):
    p = np.array([[c['x1'], c['y1']], [c['x2'], c['y2']]], float)
    u = p[1] - p[0]; length = np.linalg.norm(u); u /= max(length, 1e-8)
    return p, p.mean(0), u, float(length)


def project(points, rotation, k):
    p = np.c_[points, np.ones(len(points))] @ np.linalg.inv(k).T @ rotation.T @ k.T
    return p[:, :2] / p[:, 2:3]


class Imu:
    def __init__(self, folder, mounting, rct):
        records = [r for r in read(folder/'imu.index.jsonl') if r.get('valid_frame')]
        self.records = records
        self.t = np.array([r['device_timestamp_us'] for r in records], float)
        self.w = np.deg2rad(np.array([r['gyro_dps'] for r in records], float)) @ (rct @ mounting).T
        self.boots = {r['boot_id'] for r in records}

    def between(self, a, b, boot, offset=0):
        if boot not in self.boots or len(self.boots) != 1:
            return None, dict(status='BOOT_MISMATCH')
        a += offset; b += offset
        if not a < b or a < self.t[0] or b > self.t[-1]:
            return None, dict(status='IMU_CLOCK_RANGE_UNKNOWN')
        lo = max(0, np.searchsorted(self.t, a)-1); hi = min(len(self.t), np.searchsorted(self.t, b)+1)
        tt = np.r_[a, self.t[lo:hi][(self.t[lo:hi]>a)&(self.t[lo:hi]<b)], b]
        support = self.t[lo:hi]
        if len(support)<2 or np.diff(support).max()>CONFIG['imu_gap_max_us']:
            return None, dict(status='IMU_GAP_UNKNOWN', max_gap_us=float(np.diff(support).max()) if len(support)>1 else None)
        angular = np.stack([np.interp(tt, self.t, self.w[:,d]) for d in range(3)], -1)
        rotation = np.eye(3)
        for dt, wa, wb in zip(np.diff(tt)/1e6, angular[:-1], angular[1:]):
            # A static world bearing moves oppositely to the sensor rotation.
            rotation = cv2.Rodrigues(-(wa+wb)*.5*dt)[0] @ rotation
        angle = float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(rotation)[0])))
        return rotation, dict(status='NOMINAL', samples=len(tt), angle_deg=angle,
                             exposure_timing='UNKNOWN', independent_extrinsics='UNKNOWN')


def seeds(gray, c):
    p, m, u, length = parts(c); h,w = gray.shape
    lo = np.maximum(0, np.floor(p.min(0)-8)).astype(int)
    hi = np.minimum([w,h], np.ceil(p.max(0)+9)).astype(int)
    yy,xx = np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
    n = np.array([-u[1],u[0]])
    mask = (np.abs((xx-m[0])*n[0]+(yy-m[1])*n[1])<=CONFIG['corridor_radius_px']).astype('uint8')*255
    corners = cv2.goodFeaturesToTrack(gray[lo[1]:hi[1],lo[0]:hi[0]],CONFIG['max_corners'],.01,3,mask=mask)
    a = np.empty((0,2),np.float32) if corners is None else corners[:,0]+lo
    explicit = p[0] + np.linspace(.05,.95,CONFIG['seeds'])[:,None]*(p[1]-p[0])
    return np.r_[a, explicit].astype(np.float32)


def batched_flow(a, b, candidates):
    groups=[]; pts=[]
    for c in candidates:
        x=seeds(a,c); groups.append((len(pts),len(pts)+len(x))); pts.extend(x)
    if not pts: return []
    x=np.array(pts,np.float32).reshape(-1,1,2)
    y,s1,e1=cv2.calcOpticalFlowPyrLK(a,b,x,None,winSize=(15,15),maxLevel=3)
    back,s2,_=cv2.calcOpticalFlowPyrLK(b,a,y,None,winSize=(15,15),maxLevel=3)
    x=x[:,0]; y=y[:,0]; fb=np.linalg.norm(back[:,0]-x,axis=1)
    good=(s1[:,0]>0)&(s2[:,0]>0)&np.isfinite(y).all(1)&(fb<=CONFIG['fb_px'])
    good&=(y[:,0]>=0)&(y[:,0]<b.shape[1])&(y[:,1]>=0)&(y[:,1]<b.shape[0])
    return [(x[lo:hi][good[lo:hi]],y[lo:hi][good[lo:hi]],
             dict(seeded=hi-lo,fb_points=int(good[lo:hi].sum()),
                  median_fb_px=float(np.median(fb[lo:hi][good[lo:hi]])) if good[lo:hi].any() else None))
            for lo,hi in groups]


def flow_readout(x,y,c,k,dt):
    if len(x)<CONFIG['min_points']:
        return dict(status='UNKNOWN_FEW_POINTS')
    _,_,u,length=parts(c)
    span=float(np.ptp(x@u)/max(length,1))
    shift=np.median(y-x,0)
    angular_a=np.arctan((x[:,0]-k[0,2])/k[0,0])
    angular_b=np.arctan((y[:,0]-k[0,2])/k[0,0])
    pairs=np.triu_indices(len(x),1)
    dx=np.linalg.norm(x[pairs[0]]-x[pairs[1]],axis=1)
    dy=np.linalg.norm(y[pairs[0]]-y[pairs[1]],axis=1)
    keep=(dx>=CONFIG['point_distance_px'])&(dy>0)
    logs=np.log(dy[keep]/dx[keep])
    gain=float(np.median(logs)) if len(logs) else None
    mad=float(np.median(np.abs(logs-gain))) if len(logs) else None
    quality=span>=CONFIG['min_line_span'] and len(logs)>=6 and mad<=CONFIG['scale_mad_max']
    result=dict(status='AVAILABLE' if quality else 'UNKNOWN_SCALE_SUPPORT',points=len(x),line_span=span,
                scale_pairs=int(len(logs)),pair_logscale=gain,logscale_mad=mad,
                shift_px=shift.tolist(),horizontal_deg_s=float(np.rad2deg(np.median(angular_b-angular_a))/dt),
                residual_motion_px_s=float(np.median(np.linalg.norm(y-x,axis=1))/dt),
                coherent_scale=bool(quality),scale_log_per_s=gain/dt if quality else None,
                support_scope='local image points within 5px of fixed candidate; object purity UNKNOWN')
    return result


def width_probe(gray,c):
    p,m,u,length=parts(c); n=np.array([-u[1],u[0]])
    centers=p[0]+np.linspace(.1,.9,16)[:,None]*(p[1]-p[0])
    offsets=np.arange(-16,17,dtype=float)
    samples=centers[:,None,:]+offsets[None,:,None]*n
    valid=(samples[:,:,0]>=1)&(samples[:,:,0]<gray.shape[1]-2)&(samples[:,:,1]>=1)&(samples[:,:,1]<gray.shape[0]-2)
    prof=cv2.remap(gray,samples[:,:,0].astype('float32'),samples[:,:,1].astype('float32'),cv2.INTER_LINEAR).astype(float)
    grad=np.diff(prof,axis=1); widths=[]; contrasts=[]
    for ok,g in zip(valid.all(1),grad):
        if not ok:continue
        best=None
        for left in range(7,17):
            for right in range(max(16,left+2),min(26,left+19)):
                if g[left]*g[right]>=0:continue
                score=min(abs(g[left]),abs(g[right]))
                if best is None or score>best[0]:best=(score,right-left)
        if best and best[0]>=8:contrasts.append(best[0]);widths.append(best[1])
    if not widths:return dict(status='UNKNOWN_EDGE_PAIR',samples=0,width_px=None,mad_px=None)
    med=float(np.median(widths));mad=float(np.median(np.abs(np.array(widths)-med)))
    return dict(status='EDGE_PAIR_PROXY' if len(widths)>=8 and mad<=2 else 'UNKNOWN_WIDTH_STABILITY',
                samples=len(widths),width_px=med,mad_px=mad,median_edge_gradient=float(np.median(contrasts)),
                scope='appearance edge-pair proxy; may be background; not physical width')


def feature_stage():
    start=time.perf_counter(); cv2.setNumThreads(2); OUT.mkdir(parents=True,exist_ok=True)
    cam=AtomCamera(); k=cam.K_upright
    mountpath=ROOT/'artifacts.local/hardware-bringup/atom-imu-v1/imu-mounting-operator-pilot.json'
    extpath=ROOT/'artifacts.local/work/atom-camera-calibration-20261012/camera_tof_extrinsic.json'
    mount=np.array(json.loads(mountpath.read_text(encoding='utf8'))['tof_from_imu_rotation'])
    rct=np.array(json.loads(extpath.read_text(encoding='utf8'))['rotation'])
    inputrows=[r for r in read(SOURCE) if r['key'] in KEYS]
    summary=dict(config=CONFIG,backend='TASK_NOT_GPU_SUITABLE: sparse CPU OpenCV LK, tiny IMU rotations',
                 gpu_s=0,complete=False,sessions={},source_sha256=sha(SOURCE),
                 calibration_sha256=sha(cam_path := ROOT/'artifacts.local/work/atom-camera-calibration-20261012/calibration.json'),
                 mounting_sha256=sha(mountpath),extrinsic_sha256=sha(extpath),
                 evidence='consumed real Development; label-blind features; no metric depth or alarm')
    old=None; prev=None; prevgray=None; prevergb=None
    with (OUT/'features.jsonl').open('w',encoding='utf8') as out:
        for row in inputrows:
            if time.perf_counter()-start>600:raise TimeoutError('600s feature cap')
            key=row['key']
            if key!=old:
                folder=RAW/row['session'];rgb=read(folder/'rgb.jsonl');imu=Imu(folder,mount,rct)
                summary['sessions'][key]=Counter(); old=key; prev=None;prevgray=None;prevergb=None
            item=rgb[row['rgb_index']]
            gray=cv2.cvtColor(cam.undistort_upright(cv2.imread(str(folder/item['file']))),cv2.COLOR_BGR2GRAY)
            receipt_dt=(row['rgb_t_ns']-prev['rgb_t_ns'])/1e9 if prev else None
            device_dt=(item['device_capture_timestamp_us']-prevergb['device_capture_timestamp_us'])/1e6 if prev else None
            paired=prev is not None and 0<receipt_dt<=.5 and 0<device_dt<=.5 and item['boot_id']==prevergb['boot_id']
            rotation=None; imuq=dict(status='UNKNOWN_NO_PAIR')
            if paired:
                rotation,imuq=imu.between(prevergb['device_capture_timestamp_us'],item['device_capture_timestamp_us'],item['boot_id'])
            cs=[];usedprev=[];idx=[]
            prevmap={} if not paired else {c['track_id']:c for c in prev['candidates']}
            for j,c in enumerate(row['candidates']):
                p,m,u,length=parts(c)
                f=dict(candidate_index=j,rank=c['rank'],track_id=c['track_id'],hits=c['hits'],stable=c['stable'],
                       segment=p.reshape(-1).tolist(),horizontal_angle_deg=float(np.rad2deg(np.arctan((m[0]-k[0,2])/k[0,0]))),
                       segment_angle_length_deg=float(np.rad2deg(np.arctan(length/k[1,1]))),
                       width=width_probe(gray,c),motion=dict(status='UNKNOWN_NO_ADJACENT_TRACK'))
                cs.append(f)
                if c['track_id'] in prevmap:usedprev.append(prevmap[c['track_id']]);idx.append(j)
            flows=batched_flow(prevgray,gray,usedprev) if paired else []
            for j,c,(x,y,q) in zip(idx,usedprev,flows):
                raw=flow_readout(x,y,c,k,device_dt)
                corrected=flow_readout(x,project(y,rotation.T,k),c,k,device_dt) if rotation is not None and len(y) else dict(status='UNKNOWN_IMU')
                offsetarms={}
                if rotation is not None and len(y):
                    for offset in (-100000,100000):
                        rr,qq=imu.between(prevergb['device_capture_timestamp_us'],item['device_capture_timestamp_us'],item['boot_id'],offset)
                        offsetarms[str(offset)]=flow_readout(x,project(y,rr.T,k),c,k,device_dt) if rr is not None else qq
                cs[j]['motion']=dict(status='PAIRED',previous_i=prev['i'],device_dt_s=device_dt,receipt_dt_s=receipt_dt,
                                     point_quality=q,raw=raw,nominal_imu=corrected,timing_sensitivity=offsetarms,
                                     opposite_gyro_parity=flow_readout(x,project(y,rotation,k),c,k,device_dt) if rotation is not None and len(y) else dict(status='UNKNOWN_IMU'))
            result=dict(key=key,session=row['session'],i=row['i'],rgb_index=row['rgb_index'],
                        receipt_t_ns=row['rgb_t_ns'],camera_capture_us=item['device_capture_timestamp_us'],boot_id=item['boot_id'],
                        paired=paired,imu=imuq,candidates=cs)
            out.write(json.dumps(result,allow_nan=False)+'\n')
            summary['sessions'][key].update(frames=1,candidates=len(cs),adjacent_pairs=int(paired),nominal_imu_pairs=int(rotation is not None),
                                             duplicate_or_gap=int(prev is not None and not paired),track_pairs=len(flows),
                                             scale_available=sum(f['motion'].get('raw',{}).get('coherent_scale',False) for f in cs))
            prev=row;prevgray=gray;prevergb=item
    summary.update(complete=True,wall_s=time.perf_counter()-start,annotation_read=False)
    save('feature_receipt.json',summary);print(json.dumps(summary,indent=2))


def mechanical_control():
    """Known static RGB pixels warped by rotation/scale/translation; not real turns."""
    start=time.perf_counter();cam=AtomCamera();k=cam.K_upright
    rows=[r for r in read(SOURCE) if r['key'] in KEYS]
    examples=[]
    # First/middle/last available frame of each session, selected without labels.
    for key in KEYS:
        fr=[r for r in rows if r['key']==key and len(r['candidates'])>0]
        for j in (0,len(fr)//2,len(fr)-1): examples.append(fr[j])
    allrows=[];panels=[]
    for r in examples:
        folder=RAW/r['session'];rgb=read(folder/'rgb.jsonl')
        im=cam.undistort_upright(cv2.imread(str(folder/rgb[r['rgb_index']]['file'])))
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);size=(gray.shape[1],gray.shape[0])
        variants={}
        for axis,di in (('pitch',0),('yaw',1),('roll',2)):
            for sign in (-1,1):
                v=np.zeros(3);v[di]=sign*3
                rotation=cv2.Rodrigues(np.deg2rad(v))[0]
                variants[f'{axis}_{sign:+d}']=(cv2.warpPerspective(gray,k@rotation@np.linalg.inv(k),size),rotation)
        variants.update({
            'pure_translation':(cv2.warpAffine(gray,np.array([[1.,0.,-12.],[0.,1.,0.]]),size),np.eye(3)),
            'pure_expansion':(cv2.warpAffine(gray,np.array([[1.04,0.,-.04*k[0,2]],[0.,1.04,-.04*k[1,2]]]),size),np.eye(3))})
        for variant,(b,rr) in variants.items():
            for j,(c,(x,y,q)) in enumerate(zip(r['candidates'],batched_flow(gray,b,r['candidates']))):
                raw=flow_readout(x,y,c,k,.2)
                corrected=flow_readout(x,project(y,rr.T,k),c,k,.2) if len(y) else dict(status='UNKNOWN_FEW_POINTS')
                allrows.append(dict(key=r['key'],i=r['i'],candidate_index=j,variant=variant,point_quality=q,raw=raw,corrected=corrected))
        if not panels:
            for name in ('yaw_+1','pitch_-1','pure_expansion'):
                b,rr=variants[name]
                tile=cv2.cvtColor(b,cv2.COLOR_GRAY2BGR)
                cv2.putText(tile,name,(8,25),0,.65,(0,255,255),2);panels.append(tile)
    # The display transpose is an improper coordinate transform. Angular
    # velocity is an axial vector: det(Q)*Q*w, not Q*w. This algebra fixture
    # protects the parity contract but cannot identify the real IMU-to-raw axis.
    q=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,1.]])
    fixtures=[]
    for di in range(3):
        for sign in (-1,1):
            w=np.zeros(3);w[di]=sign*np.deg2rad(3)
            expected=q@cv2.Rodrigues(-w)[0]@q.T
            proper=cv2.Rodrigues(-(np.linalg.det(q)*q@w))[0]
            naive=cv2.Rodrigues(-(q@w))[0]
            fixtures.append(dict(raw_axis=di,sign=sign,axial_vector_max_matrix_error=float(np.max(np.abs(expected-proper))),
                                 naive_vector_rotation_error_deg=float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(expected.T@naive)[0])))))
    result=dict(evidence='controlled image warp of consumed RGB; no real motion/no exposure or gyro calibration validation',
                frames=len(examples),candidate_feature_rows=len(allrows),variants={},wall_s=time.perf_counter()-start)
    result['transpose_axis_fixture']=dict(q=q.tolist(),det_q=float(np.linalg.det(q)),cases=fixtures,
        actual_phone_gyro_mapping='UNKNOWN: existing fitted camera<-ToF and operator mounting used nominally; opposite gyro parity is a sensitivity arm, neither newly calibrated')
    for v in variants:
        rr=[r for r in allrows if r['variant']==v]
        paired=[r for r in rr if r['raw'].get('coherent_scale') and r['corrected'].get('coherent_scale')]
        result['variants'][v]=dict(candidate_pairs=len(rr),paired_coherent_scale=len(paired),
            raw_motion_px_s=float(np.median([r['raw']['residual_motion_px_s'] for r in paired])) if paired else None,
            corrected_motion_px_s=float(np.median([r['corrected']['residual_motion_px_s'] for r in paired])) if paired else None,
            raw_logscale_per_s=float(np.median([r['raw']['scale_log_per_s'] for r in paired])) if paired else None,
            corrected_logscale_per_s=float(np.median([r['corrected']['scale_log_per_s'] for r in paired])) if paired else None)
    (OUT/'mechanical_rows.jsonl').write_text('\n'.join(json.dumps(r) for r in allrows)+'\n')
    cv2.imwrite(str(OUT/'mechanical_controls.jpg'),np.hstack(panels));save('mechanical_summary.json',result)
    print(json.dumps(result,indent=2))


def audit_sources():
    """Inspect existing metadata of explicitly consumed phone batches only."""
    batches={
        'morning_20261012':ROOT/'artifacts.local/work/phone-imu-capture-20261012/raw',
        'afternoon_20261012':ROOT/'artifacts.local/work/phone-new-capture-20261012/raw',
        'consumed_20261013':ROOT/'artifacts.local/work/frozen-eval-capture-20261013/raw',
        'consumed_20261013b':ROOT/'artifacts.local/work/frozen-eval-capture-20261013b/raw',
        'consumed_20261013c':ROOT/'artifacts.local/work/frozen-eval-capture-20261013c/raw'}
    rows=[]
    for name,root in batches.items():
        if not root.exists():continue
        for folder in sorted(root.iterdir()):
            p=folder/'manifest.json'
            if not p.exists():continue
            m=json.loads(p.read_text(encoding='utf8'));imu=[r for r in read(folder/'imu.index.jsonl') if r.get('valid_frame')]
            rgb=read(folder/'rgb.jsonl')
            vals=np.array([r['gyro_dps'] for r in imu])
            rows.append(dict(batch=name,session=folder.name,rgb=len(rgb),imu=len(imu),
                             note=m.get('scene_note'),gyro_p95_dps=float(np.percentile(np.linalg.norm(vals,axis=1),95)) if len(vals) else None,
                             rgb_device_timestamp_records=sum(bool(r.get('device_capture_timestamp_us')) for r in rgb)))
    records=[]
    for filename in ('FROZEN_EVAL_LABELS_20261013.json','FROZEN_EVAL_LABELS_20261013B.json','FROZEN_EVAL_LABELS_20261013C.json'):
        p=ROOT/'research/active/hardware-bringup'/filename
        if p.exists(): records.append(dict(path=str(p),sha256=sha(p),labels=json.loads(p.read_text(encoding='utf8'))))
    result=dict(sessions=rows,existing_scene_labels=records,
                true_stationary_turn='NOT_EVALUABLE: no explicit stationary-turn class or measured zero translation in inspected scene metadata',
                reading_scope='metadata of previously consumed batches; no newly frozen eval or inference')
    save('source_audit.json',result)
    print(json.dumps(dict(sessions=len(rows),true_stationary_turn=result['true_stationary_turn'],max_gyro_p95=max(r['gyro_p95_dps'] or 0 for r in rows)),indent=2))


def real_rotation_control():
    """Official-pose oracle derotation on an already-consumed ADT window.

    This has small measured camera translation, not proven body stationarity.
    No thin-target truth or proprietary ADT image is exported.
    """
    start=time.perf_counter();cv2.setNumThreads(2)
    auditpath=OUT.parent/'integration/rotation_audit.json'
    audit=json.loads(auditpath.read_text(encoding='utf8'));choice=audit['selected'][0]
    folder=ROOT/Path(choice['frames_npz']).parent
    first=choice['index_start_inclusive'];last=choice['index_end_inclusive'];sl=slice(first,last+1)
    with np.load(folder/'frames.npz') as z:
        images=z['rgb'][sl].copy();rcc=z['R_cc'][sl].copy();tt=z['t_ns'][sl].copy()
    with np.load(folder/'poses.npz') as z:
        pos=z['pos'][sl].copy();fwd=z['fwd'][sl].copy()
    meta=json.loads((folder/'meta.json').read_text(encoding='utf8'))
    fx,fy,cx,cy=meta['projection_params'];k=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
    rsc=[]
    for f,r in zip(fwd,rcc):
        y=np.array([0.,-1.,0.]);a=np.stack([np.cross(y,f),y,f]);rsc.append(a.T@r)
    rows=[]
    for j in range(len(images)-1):
        a=cv2.cvtColor(images[j],cv2.COLOR_RGB2GRAY);b=cv2.cvtColor(images[j+1],cv2.COLOR_RGB2GRAY)
        x=cv2.goodFeaturesToTrack(a,400,.01,8)
        y,s1,_=cv2.calcOpticalFlowPyrLK(a,b,x,None,winSize=(21,21),maxLevel=4)
        back,s2,_=cv2.calcOpticalFlowPyrLK(b,a,y,None,winSize=(21,21),maxLevel=4)
        fb=np.linalg.norm(back[:,0]-x[:,0],axis=1)
        keep=(s1[:,0]>0)&(s2[:,0]>0)&(fb<=1.5)&np.isfinite(y[:,0]).all(1)
        px=x[:,0][keep];py=y[:,0][keep]
        rotation=rsc[j+1].T@rsc[j]
        corrected=project(py,rotation.T,k)
        raw=np.linalg.norm(py-px,axis=1);after=np.linalg.norm(corrected-px,axis=1)
        dt=(tt[j+1]-tt[j])/1e9
        rows.append(dict(sequence=choice['sequence'],previous_index=first+j,index=first+j+1,dt_s=dt,
            camera_translation_m=float(np.linalg.norm(pos[j+1]-pos[j])),
            rotation_deg=float(np.rad2deg(np.linalg.norm(cv2.Rodrigues(rotation)[0]))),
            corners=len(x),fb_points=int(keep.sum()),raw_median_px=float(np.median(raw)),
            corrected_median_px=float(np.median(after)),raw_p90_px=float(np.percentile(raw,90)),
            corrected_p90_px=float(np.percentile(after,90)),
            raw_median_px_s=float(np.median(raw)/dt),corrected_median_px_s=float(np.median(after)/dt)))
    result=dict(sequence=choice,pose_contract=audit['pose_contract'],paired_frames=len(rows),rows=rows,
        raw_median_pair_motion_px_s=float(np.median([r['raw_median_px_s'] for r in rows])),
        corrected_median_pair_motion_px_s=float(np.median([r['corrected_median_px_s'] for r in rows])),
        audit_sha256=sha(auditpath),source_poses_sha256=sha(folder/'poses.npz'),
        evidence='real already-consumed ADT rotation-dominated background; official-pose oracle only; no isolated rod',
        wall_s=time.perf_counter()-start,image_exported=False)
    save('real_rotation_summary.json',result)
    print(json.dumps(result,indent=2))


def track_stage():
    """Aggregate label-blind autonomous observed tracks; no missing-point fill."""
    start=time.perf_counter();groups={}
    for row in read(OUT/'features.jsonl'):
        for c in row['candidates']:
            groups.setdefault((row['key'],c['track_id']),[]).append(dict(i=row['i'],camera_capture_us=row['camera_capture_us'],
                receipt_t_ns=row['receipt_t_ns'],candidate_index=c['candidate_index'],segment=c['segment'],
                horizontal_angle_deg=c['horizontal_angle_deg'],hits=c['hits'],stable=c['stable']))
    tracks=[]
    for (key,tid),obs in groups.items():
        chains=[];chain=[];seen=set()
        for o in obs:
            t=o['camera_capture_us']
            if t in seen:continue
            seen.add(t)
            if chain and (not 0<t-chain[-1]['camera_capture_us']<=500000 or not 0<o['receipt_t_ns']-chain[-1]['receipt_t_ns']<=500000000):
                chains.append(chain);chain=[]
            chain.append(o)
        if chain:chains.append(chain)
        for ci,chain in enumerate(chains):
            aa=[o['horizontal_angle_deg'] for o in chain]
            tracks.append(dict(key=key,track_id=tid,chain=ci,observed_images=len(chain),
                duration_s=(chain[-1]['camera_capture_us']-chain[0]['camera_capture_us'])/1e6,
                angle_span_deg=max(aa)-min(aa),first_to_last_angle_deg=aa[-1]-aa[0],
                observations=chain,source='fixed existing autonomous camera-chain; advancing RGB only; no gap interpolation'))
    (OUT/'track_chains.jsonl').write_text('\n'.join(json.dumps(t) for t in tracks)+'\n',encoding='utf8')
    summary={}
    for key in KEYS:
        rr=[t for t in tracks if t['key']==key]
        summary[key]=dict(chains=len(rr),chains_two_or_more=sum(t['observed_images']>=2 for t in rr),
                         duration_p50_s=float(np.median([t['duration_s'] for t in rr])),
                         longest_s=max(t['duration_s'] for t in rr),label_read=False)
    save('track_summary.json',dict(sessions=summary,wall_s=time.perf_counter()-start,labels_read=False))
    print(json.dumps(summary,indent=2))


def track_audit():
    """Post-result visual audit; selection never enters features or tracking."""
    cam=AtomCamera();frames={(r['key'],r['i']):r for r in read(SOURCE)}
    tracks=read(OUT/'track_chains.jsonl');tiles=[];selected=[]
    for key,tid,chain in (('073635',1,0),('073821',2,0),('073821',31,0)):
        t=next(t for t in tracks if (t['key'],t['track_id'],t['chain'])==(key,tid,chain))
        obs=t['observations'];group=[]
        for j in (0,len(obs)//2,len(obs)-1):
            o=obs[j];r=frames[key,o['i']];folder=RAW/r['session'];rgb=read(folder/'rgb.jsonl')
            im=cam.undistort_upright(cv2.imread(str(folder/rgb[r['rgb_index']]['file'])))
            a,b=np.array(o['segment']).astype(int).reshape(2,2);cv2.line(im,tuple(a),tuple(b),(0,255,0),2)
            banner=np.zeros((45,480,3),np.uint8)
            cv2.putText(banner,f'{key} tid={tid} i={o["i"]} angle={o["horizontal_angle_deg"]:.1f}',(4,25),0,.6,(255,255,255),1)
            group.append(np.vstack((banner,im)));selected.append(dict(key=key,track_id=tid,chain=chain,observation=o))
        tiles.append(np.hstack(group))
    cv2.imwrite(str(OUT/'autonomous_track_audit.jpg'),np.vstack(tiles));save('autonomous_track_audit.json',dict(selected=selected,scope='post-result visual audit only'))


def evaluate():
    from thin_structure_evaluate_dev_20261013 import matches
    featurepath=OUT/'features.jsonl'
    assert json.loads((OUT/'feature_receipt.json').read_text(encoding='utf8'))['complete']
    labels_path=ROOT/'artifacts.local/work/thin-structure-dev-20261013/annotations.json'
    labels=json.loads(labels_path.read_text(encoding='utf8'));frames={(r['key'],r['i']):r for r in read(featurepath)}
    samples=[];panels={};cam=AtomCamera();trackrows=read(OUT/'track_chains.jsonl') if (OUT/'track_chains.jsonl').exists() else []
    tracklookup={(t['key'],t['track_id'],o['camera_capture_us']):t for t in trackrows for o in t['observations']}
    for s in labels['samples']:
        if s['session'] not in KEYS:continue
        f=frames[s['session'],s['i']]
        hits=[c for c in f['candidates'] if s['segment'] and matches(s['segment'],dict(zip(('x1','y1','x2','y2'),c['segment'])))]
        # A fixed rank chooses one matched candidate for readable descriptive stats.
        chosen=min(hits,key=lambda c:c['rank']) if hits else None
        matched_tracks=[dict(track_id=t['track_id'],chain=t['chain'],observed_images=t['observed_images'],duration_s=t['duration_s'],
                        angle_span_deg=t['angle_span_deg'],first_to_last_angle_deg=t['first_to_last_angle_deg'])
                        for c in hits if (t:=tracklookup.get((s['session'],c['track_id'],f['camera_capture_us']))) is not None]
        samples.append(dict(key=s['session'],i=s['i'],label_status=s['label_status'],segment=s['segment'],matched_tracks=matched_tracks,
                            matched=len(hits),selected_lowest_rank=chosen,
                            support_counts=dict(motion_pairs=sum(c['motion']['status']=='PAIRED' for c in hits),
                                raw_coherent_scale=sum(c['motion'].get('raw',{}).get('coherent_scale',False) for c in hits),
                                nominal_imu_coherent_scale=sum(c['motion'].get('nominal_imu',{}).get('coherent_scale',False) for c in hits))))
        im=cv2.imread(str(ROOT/f'artifacts.local/work/thin-structure-dev-20261013/raw_{s["session"]}_{s["i"]:04d}.jpg'))
        if s['segment']:
            a,b=np.asarray(s['segment']).reshape(2,2);cv2.line(im,tuple(a),tuple(b),(255,0,255),2)
        if chosen:
            a,b=np.asarray(chosen['segment']).astype(int).reshape(2,2);cv2.line(im,tuple(a),tuple(b),(0,255,0),2)
        banner=np.zeros((65,480,3),np.uint8)
        motion=chosen['motion'] if chosen else {}
        raw=motion.get('raw',{}); corrected=motion.get('nominal_imu',{})
        cv2.putText(banner,f'{s["session"]} i={s["i"]} match={len(hits)}',(5,22),0,.55,(255,255,255),1)
        def fmt(v):return 'UNK' if v is None else f'{v:.2f}'
        cv2.putText(banner,'x deg/s raw/imu '+fmt(raw.get('horizontal_deg_s'))+'/'+fmt(corrected.get('horizontal_deg_s')),(5,43),0,.5,(255,255,255),1)
        cv2.putText(banner,'scale/s '+fmt(raw.get('scale_log_per_s'))+'/'+fmt(corrected.get('scale_log_per_s')),(5,61),0,.5,(255,255,255),1)
        panels.setdefault(s['session'],[]).append(np.vstack((banner,im)))
    summary={}
    for key in KEYS:
        rr=[r for r in samples if r['key']==key and r['label_status']=='visible_segment']
        both=[r for r in rr if r['selected_lowest_rank'] and r['selected_lowest_rank']['motion'].get('raw',{}).get('coherent_scale') and r['selected_lowest_rank']['motion'].get('nominal_imu',{}).get('coherent_scale')]
        stats={}
        for arm in ('raw','nominal_imu'):
            for field in ('horizontal_deg_s','scale_log_per_s','residual_motion_px_s'):
                values=[r['selected_lowest_rank']['motion'][arm][field] for r in both]
                stats[arm+'_'+field]=dict(n=len(values),median=float(np.median(values)) if values else None,values=values)
        summary[key]=dict(visible_samples=len(rr),localized=sum(r['matched']>0 for r in rr),
                          any_matched_pair=sum(r['support_counts']['motion_pairs']>0 for r in rr),
                          any_matched_raw_scale=sum(r['support_counts']['raw_coherent_scale']>0 for r in rr),
                          any_matched_imu_scale=sum(r['support_counts']['nominal_imu_coherent_scale']>0 for r in rr),
                          lowest_rank_paired_coherent_scale=len(both),stats=stats)
        pp=panels[key]
        cv2.imwrite(str(OUT/f'target_readout_{key}.jpg'),np.vstack((np.hstack(pp[:3]),np.hstack(pp[3:]))))
    # Oracle geometry is a separate visible-target upper bound, never tracking.
    oracle={}
    for key in KEYS:
        rr=[r for r in samples if r['key']==key and r['segment']]
        mx=[];ll=[];ts=[]
        for r in rr:
            p=np.asarray(r['segment']).reshape(2,2);mx.append(p[:,0].mean());ll.append(np.linalg.norm(p[1]-p[0]))
            ts.append(frames[key,r['i']]['camera_capture_us']/1e6)
        if len(rr)>1:
            elapsed=ts[-1]-ts[0]
            oracle[key]=dict(weak_visible_segments=len(rr),interval_s=elapsed,
                horizontal_span_deg=float(np.rad2deg(np.arctan((max(mx)-cam.K_upright[0,2])/cam.K_upright[0,0])-np.arctan((min(mx)-cam.K_upright[0,2])/cam.K_upright[0,0]))),
                first_to_last_center_change_px=float(mx[-1]-mx[0]),
                visible_length_ratio=float(ll[-1]/ll[0]),
                note='oracle weak annotation geometry; visible crop length changes, not the same physical endpoints, not TTC')
    result=dict(annotation_sha256=sha(labels_path),features_sha256=sha(featurepath),summary=summary,samples=samples,
                oracle_visible_geometry=oracle,three_class_classification='NOT_EVALUABLE',
                limits='assistant weak localization, only 4 real clips; no verified real stationary turn, camera exposure timing/extrinsics unknown; local corners may lie on background')
    trackeval={}
    for key in KEYS:
        rr=[r for r in samples if r['key']==key and r['segment']]
        ts={};matching={}
        for r in rr:
            for t in r['matched_tracks']:
                ident=(t['track_id'],t['chain']);ts[ident]=t;matching.setdefault(ident,set()).add(frames[key,r['i']]['camera_capture_us'])
        targettracks=[dict(t,weak_samples_matched=len(matching[ident])) for ident,t in ts.items()]
        trackeval[key]=dict(visible_samples=len(rr),samples_with_target_chain=sum(bool(r['matched_tracks']) for r in rr),
            target_matched_chains=len(targettracks),repeated_weak_match_chains=sum(t['weak_samples_matched']>=2 for t in targettracks),
            matched_chains=targettracks,
            observation_scope='single-sample target match is not object-pure ground truth for the full autonomous chain')
    result['autonomous_track_evaluation']=trackeval
    save('evaluation.json',result);print(json.dumps(dict(summary=summary,oracle_visible_geometry=oracle),indent=2))


def validate():
    start=time.perf_counter()
    original={(r['key'],r['i']):r for r in read(SOURCE) if r['key'] in KEYS}
    features=read(OUT/'features.jsonl');count=0
    assert len(features)==len(original)==278
    for r in features:
        old=original[r['key'],r['i']]
        assert len(old['candidates'])==len(r['candidates'])
        for c,f in zip(old['candidates'],r['candidates']):
            assert f['segment']==[c[k] for k in ('x1','y1','x2','y2')]
            assert f['track_id']==c['track_id'] and f['rank']==c['rank'] and f['hits']==c['hits']
            if f['motion']['status']=='PAIRED':
                assert 0<f['motion']['device_dt_s']<=.5 and 0<f['motion']['receipt_dt_s']<=.5
            count+=1
    mech=json.loads((OUT/'mechanical_summary.json').read_text(encoding='utf8'))
    for name,r in mech['variants'].items():
        if name.startswith(('yaw','pitch','roll')):
            assert r['corrected_motion_px_s']<1.0,(name,r)
    assert abs(mech['variants']['pure_translation']['corrected_motion_px_s']-60)<.1
    assert abs(mech['variants']['pure_expansion']['corrected_logscale_per_s']-math.log(1.04)/.2)<.01
    assert all(r['axial_vector_max_matrix_error']<1e-12 for r in mech['transpose_axis_fixture']['cases'])
    for t in read(OUT/'track_chains.jsonl'):
        obs=t['observations']
        assert len({o['camera_capture_us'] for o in obs})==len(obs)
        assert all(0<b['camera_capture_us']-a['camera_capture_us']<=500000 and 0<b['receipt_t_ns']-a['receipt_t_ns']<=500000000 for a,b in zip(obs,obs[1:]))
    evaluation=json.loads((OUT/'evaluation.json').read_text(encoding='utf8'))
    assert evaluation['features_sha256']==sha(OUT/'features.jsonl')
    hashes={p.name:sha(p) for p in OUT.glob('*.json*') if p.name!='verification.json'}
    result=dict(status='PASS',original_frames=len(original),unchanged_candidates=count,
                exact_candidate_geometry_rank_track_identity=True,independent_advancing_rgb_chains=True,
                signed_rotation_and_translation_expansion_mechanics=True,
                source_code_sha256=sha(Path(__file__)),original_candidates_sha256=sha(SOURCE),
                feature_sha256=sha(OUT/'features.jsonl'),payload_hashes=hashes,
                wall_s=time.perf_counter()-start)
    save('verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['features','mechanical','audit','evaluate','rotation','tracks','track-audit','validate']);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    {'features':feature_stage,'mechanical':mechanical_control,'audit':audit_sources,'evaluate':evaluate,'rotation':real_rotation_control,'tracks':track_stage,'track-audit':track_audit,'validate':validate}[args.stage]()
