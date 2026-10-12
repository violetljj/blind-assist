"""R2 causal nominal-observable gate. Selfcheck must precede any risk curves."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
import cnh_tristate_dev as R1

ROOT=R1.ROOT
OUT=R1.WORK/'cnh-tristate-dev-r2-20261006'
RANGE=float(np.sqrt(2.1**2+.29**2+.9**2))
FRAMES=R1.FRAMES


def deadline():
    if time.time()>=R1.read(OUT/'PLAN.json')['deadline_unix']:
        raise TimeoutError('R2 45minute wall budget; retain completed evidence')


def direction(poses,f):
    if f<5 or not np.isfinite(poses[max(0,f-5):f+1]).all(): return None
    d=(poses[f,:3,3]-poses[f-5,:3,3])[[0,2]]
    return d/np.linalg.norm(d) if np.linalg.norm(d)>=.05 else None


def nominal(poses,h):
    d=direction(poses,h)
    if d is None: return None
    p=np.eye(4);p[:3,3]=poses[h,:3,3]
    p[:3,:3]=R1.rotation(np.rad2deg(np.arctan2(d[0],d[1])))@R1.rotation(-10,'x')
    return p


def points(poses,f,d):
    """Inherited world XZ lattice and lateral sentinels, current-height slices."""
    initial=poses[0,[0,2],2].copy();initial/=np.linalg.norm(initial)
    basis=np.stack(([initial[1],-initial[0]],initial),axis=1)
    anchor=poses[0,[0,2],3];current=poses[f,[0,2],3];right=np.array([d[1],-d[0]])
    corners=np.array([current+a*right+b*d for a in (-.29,.29) for b in (.9,2.1)])
    ij=(corners-anchor)@basis/.05;lo=np.floor(ij.min(0)).astype(int)-1;hi=np.ceil(ij.max(0)).astype(int)+1
    x,z=np.meshgrid(np.arange(lo[0],hi[0]+1),np.arange(lo[1],hi[1]+1))
    xz=np.stack((x.ravel(),z.ravel()),axis=1)*.05@basis.T+anchor
    rel=xz-current;keep=(np.abs(rel@right)<=.29+1e-10)&(rel@d>=.9-1e-10)&(rel@d<=2.1+1e-10)
    sent=np.array([current+a*right+b*d for a in (-.29,.29) for b in np.arange(.9,2.1001,.05)])
    xz=np.concatenate((xz[keep],sent))
    ys=np.unique(np.concatenate([np.r_[np.arange(lo,hi+1e-10,.05),hi] for lo,hi in R1.HEIGHTS]))+poses[f,1,3]
    pp=np.zeros((len(xz),len(ys),3));pp[:,:,0]=xz[:,0,None];pp[:,:,2]=xz[:,1,None];pp[:,:,1]=ys[None,:]
    return pp.reshape(-1,3)


def coverage(poses,f,angles=(0,)):
    d=direction(poses,f)
    if d is None: return dict(valid=False,passed=False,points=0,masked=0,missing=0,min_margin=None)
    pp=points(poses,f,d);mask=np.zeros(len(pp),bool);fresh=np.zeros(len(pp),bool)
    for h in range(f-3,f+1):
        p=nominal(poses,h)
        if p is not None: mask|=R1.fov(pp,p,RANGE)
        for a in angles: fresh|=R1.fov(pp,poses[h]@R1.extrinsic(a),RANGE)
    missing=mask&~fresh
    return dict(valid=True,passed=bool(mask.any() and not missing.any()),points=len(pp),masked=int(mask.sum()),missing=int(missing.sum()),
        min_margin=None,missing_points=pp[missing],query_points=pp,mask=mask)


def metadata(row):
    u,c,b=row['unit'],row['config'],row['batch']
    if b<98000: return R1.original_motion(u,c)
    op=(R1.AUG if b==98000 else R1.CONT)/f'observations/{"calibration" if b==98000 else "evaluation"}/unit{u}.npz'
    with np.load(op) as z:
        return z['sensor_center'],z['travel'],z['noisy_center'][z['configs'].tolist().index(c)]


def freeze(started):
    OUT.mkdir(parents=True,exist_ok=True)
    R1.save(OUT/'PLAN.json',dict(task='CNH_TRISTATE_DEV_R2_20261006',phase='EXPLORE consumed synthetic Development, descriptive',
        started_unix=started,deadline_unix=started+2700,maximum_wall_seconds=2700,r1_commit='a66e0954781eb64e4c15f970fee00c10461745e6',
        revision='User corrects incompatible R1 range/volume/validity; R1 preserved, NOT_COMPARABLE risk labels2.5->2.1m',
        query=dict(forward_m=[.9,2.1],lateral_m=[-.29,.29],heights=R1.HEIGHTS,range_m=RANGE,range_formula='sqrt(2.1^2+.29^2+.90^2)',
            heights_origin='current estimated sensor origin, not original time0 height',world_lattice_m=.05,sentinels='Exact lateral+/-.29 boundaries, forward every.05m'),
        nominal='For each past h in[f-3,f], same estimated origin; yaw0 relative to last1s estimated displacement at h; pitch-10;h<5 skipped. Mask is union of nominalFOVs on the current worldquery. No truth input.',
        gate='Direction available, poses finite; nonempty mask and all mask points fresh under actual branchFOV union; empty mask unavailable',
        validity_assumption='Simulation has no tracking failure; finiteSE3 and available direction treated valid. Real deployment needs measured tracking/gravity quality.',
        alarms=dict(threshold=R1.THRESHOLD,query='Unchanged original M3',variant_A='No hold',variant_B='Any prior obstacle output in last2.6s forbids clear; otherwise unknown, never veto current alarm',
            history='Existing model outputs start frame3; no earlier alarm score invented'),
        thresholds='Reuse R1 exact20 common label-blind score quantiles; no threshold fit to R2 labels',
        label=dict(forward_m=[0,2.1],lateral_m=[-.30,.30],heights=R1.HEIGHTS,contact='Closed physical volume, current true travel; no clear_all/future path',
            person_front='Saved travel origin proxy; anatomical offset unavailable',graze='width.40 shell minus maincontact'),
        volume='Independent weighted voxel-center quadrature, not sentinel counts; by HEIGHT and forwardbins. Mask exclusions called cane-scope boundary, no evidence cane detects all excluded obstacles.',
        selfcheck=dict(selection='true|headyaw relative travel|<3deg,mode!=2,frame>=5; evaluation geometry used only to select this diagnostic',
            minimum_single_pass_fraction=.9,on_below='Stop; report reason; no onlinefull/riskcurves/truthlabel replay or retuning'),
        budget_scope='CPU only; existing scores/metadata/boxes; no render/inference/train/GPU; no binary scientific verdict',
        sources={str(p.relative_to(ROOT)):R1.sha(p) for p in [Path(__file__),Path(R1.__file__),R1.OUT/'online.npz',R1.OUT/'rows.json',R1.OUT/'result.json']}))
    (OUT/'PLAN.sha256').write_text(R1.sha(OUT/'PLAN.json')+'\n')
    (OUT/'source').mkdir();(OUT/'source/cnh_tristate_dev_r2.py').write_bytes(Path(__file__).read_bytes())


def selfcheck():
    tick=time.monotonic();rows=R1.read(R1.OUT/'rows.json');records=[];examples=[]
    for i,row in enumerate(rows):
        if row['mode']==2:continue
        deadline();sensor,travel,noisy=metadata(row)
        yaw=np.rad2deg(np.arctan2(sensor[:,0,2],sensor[:,2,2])-np.arctan2(travel[:,0,2],travel[:,2,2]))
        yaw=(yaw+180)%360-180
        for f in FRAMES:
            if f<5 or abs(yaw[f])>=3:continue
            c=coverage(noisy,int(f));records.append(dict(**row,frame=int(f),actual_head_deg=float(yaw[f]),
                valid=c['valid'],passed=c['passed'],points=c['points'],masked=c['masked'],missing=c['missing']))
            if not c['passed'] and len(examples)<12:
                examples.append(dict(**records[-1],missing_points=c['missing_points'][:8].tolist(),
                    estimated_yaw_deg=float(np.rad2deg(np.arctan2(noisy[f,0,2],noisy[f,2,2]))),
                    nominal_direction_deg=float(np.rad2deg(np.arctan2(direction(noisy,int(f))[0],direction(noisy,int(f))[1])))))
        if (i+1)%480==0:print('selfcheck sequences',i+1,'selected',len(records),round(time.monotonic()-tick,1),'s',flush=True)
    count=sum(r['passed'] for r in records);n=len(records);rate=count/n if n else None
    result=dict(status='CONTRACT_SELF_CHECK_COMPLETE',passed_frames=count,selected_frames=n,pass_fraction=rate,
        contract_consistent=bool(n and rate>=.9),action='CONTINUE' if n and rate>=.9 else 'STOP_CONTRACT_SELF_CHECK',
        scientific_binary_verdict='NOT_APPLICABLE',frames={str(f):dict(n=sum(r['frame']==f for r in records),passed=sum(r['frame']==f and r['passed'] for r in records)) for f in (7,8)},
        batch={str(b):dict(n=sum(r['batch']==b for r in records),passed=sum(r['batch']==b and r['passed'] for r in records)) for b in R1.score_sources()},
        missing_points_sum=sum(r['missing'] for r in records),examples=examples,seconds=time.monotonic()-tick,
        provenance=dict(plan_sha256=R1.sha(OUT/'PLAN.json'),r1_online_sha256=R1.sha(R1.OUT/'online.npz'),source_sha256=R1.sha(__file__)))
    R1.save(OUT/'selfcheck_records.json',records);R1.save(OUT/'selfcheck.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='examples'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','selfcheck']);p.add_argument('--started-unix',type=float,default=time.time());a=p.parse_args()
    if a.stage=='freeze':freeze(a.started_unix)
    else:selfcheck()
