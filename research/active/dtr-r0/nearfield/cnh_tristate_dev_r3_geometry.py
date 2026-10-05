"""R3 geometry-only tolerance calibration. Never reads scenes, scores or labels."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation
import cnh_tristate_dev as R1
import cnh_tristate_dev_r2 as R2

OUT=R1.WORK/'cnh-tristate-dev-r3-20261006'
RANGE=R2.RANGE
MARGINS=np.arange(0.,20.0001,.5)


def prepared(poses,f,pp=None):
    """One transform per historical pose; reuse across core margins/branches."""
    d=R2.direction(poses,f)
    valid=d is not None and np.isfinite(poses[max(0,f-3):f+1]).all()
    if not valid:
        pp=np.empty((0,3)) if pp is None else np.asarray(pp,float)
        return dict(valid=False,points=pp,world=pp,
                    nominal_local=np.empty((0,len(pp),3)),actual_local=np.empty((0,len(pp),3)))
    pp=R2.points(poses,f,d) if pp is None else np.asarray(pp,float)
    nominal=[];actual=[]
    for h in range(max(0,f-3),f+1):
        p=R2.nominal(poses,h)
        if p is not None:nominal.append((pp-p[:3,3])@p[:3,:3])
        actual.append((pp-poses[h,:3,3])@poses[h,:3,:3])
    return dict(valid=True,points=pp,world=pp,nominal_local=np.array(nominal),
                actual_local=np.array(actual))


def capacity(local):
    """Largest horizontal AND vertical angular erosion admitting each point."""
    if local.shape[0]==0:return np.full(local.shape[1],-np.inf)
    angles=np.rad2deg(np.arctan2(np.abs(local[...,:2]),local[...,2,None]))
    cap=22.5-angles.max(-1)
    cap[(local[...,2]<=0)|(np.linalg.norm(local,axis=-1)>RANGE+R1.EPS)]=-np.inf
    return cap.max(0)


def _fresh(local,angles):
    seen=np.zeros(local.shape[1],bool)
    for a in angles:
        loc=local@R1.extrinsic(a)[:3,:3]
        z=loc[...,2]
        seen|=((z>0)&(np.abs(loc[...,0])<=R1.EDGE*z+R1.EPS)&
               (np.abs(loc[...,1])<=R1.EDGE*z+R1.EPS)&
               (np.linalg.norm(loc,axis=-1)<=RANGE+R1.EPS)).any(0)
    return seen


def gates(prep,margins,angles=(0,-15,15)):
    margins=np.atleast_1d(margins).astype(float)
    cp=capacity(prep['nominal_local'])
    mask=cp[None,:]+R1.EPS>=margins[:,None]
    freshs=np.stack((_fresh(prep['actual_local'],[angles[0]]),
                      _fresh(prep['actual_local'],angles[1:])))
    missing=(mask[:,None,:]&~freshs[None,:,:]).sum(-1)
    counts=mask.sum(-1)
    return dict(passed=prep['valid']&(counts[:,None]>0)&(missing==0),
                mask_count=counts,missing_count=missing,mask=mask,fresh=freshs,
                capacity=cp)


def core_mask(points,poses,f,m):
    return capacity(prepared(poses,f,points)['nominal_local'])+R1.EPS>=m


def fresh(points,poses,f,angles):
    return _fresh(prepared(poses,f,points)['actual_local'],angles)


def frozen_noisy():
    source=R1.WORK/'cnh-track-a-v5-20260928/data/source'
    if str(source) not in sys.path:sys.path.insert(0,str(source))
    from cnh_track_a_readout import noisy_poses
    return noisy_poses


def noise_pose(true_poses,seed,model='original'):
    if model=='original':return frozen_noisy()(true_poses,seed,dt=.2)
    if model!='gravity':raise ValueError(model)
    # E_grav contract: exact original RNG consumption; local zero-mean noise,
    # world-Y left bias; no X/Z constant bias or additional filtering.
    rng=np.random.default_rng(seed)
    scale=rng.uniform(-.2,.2);bias=rng.choice([-1.,1.],3)*np.deg2rad(1.)
    out=np.asarray(true_poses).copy()
    for i in range(1,len(out)):
        delta=np.linalg.inv(true_poses[i-1])@true_poses[i]
        delta[:3,3]*=1+scale+rng.normal(0,.02)
        innovation=rng.normal(0,np.deg2rad(.1),3)
        out[i,:3,3]=out[i-1,:3,3]+out[i-1,:3,:3]@delta[:3,3]
        left=Rotation.from_rotvec([0.,bias[1]*.2,0.]).as_matrix()
        out[i,:3,:3]=left@out[i-1,:3,:3]@delta[:3,:3]@Rotation.from_rotvec(innovation).as_matrix()
    return out


def synthetic(seed,model='original'):
    poses=np.repeat(np.eye(4)[None],16,axis=0)
    poses[:,:3,:3]=R1.rotation(-10,'x')
    poses[:,2,3]=np.arange(16)*.16-2.4
    return poses,noise_pose(poses,seed,model)


def quadrature(poses,f):
    """5cm clipped cells: true integration weights, never sentinel counts."""
    d=R2.direction(poses,f)
    if d is None:return None
    right=np.array([d[1],-d[0]])
    def edges(lo,hi):return np.unique(np.r_[lo,np.arange(lo+.05,hi-1e-10,.05),hi])
    xe=edges(-.29,.29);xx=(xe[:-1]+xe[1:])/2;wx=np.diff(xe)
    zz=[];wz=[];bins=[]
    for b,(lo,hi) in enumerate(zip([.9,1.2,1.5,1.8],[1.2,1.5,1.8,2.1])):
        ee=edges(lo,hi);zz.extend((ee[:-1]+ee[1:])/2);wz.extend(np.diff(ee));bins.extend([b]*(len(ee)-1))
    ys=[];wy=[];qs=[]
    for q,(lo,hi) in enumerate(R1.HEIGHTS):
        ee=edges(lo,hi);ys.extend((ee[:-1]+ee[1:])/2);wy.extend(np.diff(ee));qs.extend([q]*(len(ee)-1))
    x,z,y=np.meshgrid(xx,zz,ys,indexing='ij')
    rel=np.stack((x.ravel(),z.ravel()),1)
    xz=poses[f,[0,2],3]+rel[:,0,None]*right+rel[:,1,None]*d
    pp=np.column_stack((xz[:,0],y.ravel()+poses[f,1,3],xz[:,1]))
    weight=(wx[:,None,None]*np.asarray(wz)[None,:,None]*np.asarray(wy)[None,None,:]).ravel()
    group=np.broadcast_to(np.asarray(bins)[None,:,None]*2+np.asarray(qs)[None,None,:],x.shape).ravel()
    return pp,weight,group


def _volume_arrays(poses,f,margins):
    quad=quadrature(poses,f)
    if quad is None:return np.zeros((len(margins),8)),np.zeros(8)
    pp,ww,gg=quad;cap=capacity(prepared(poses,f,pp)['nominal_local'])
    total=np.bincount(gg,weights=ww,minlength=8)
    core=np.zeros((len(margins),8))
    for g in range(8):
        chosen=gg==g;cc=cap[chosen];weight=ww[chosen];order=np.argsort(cc)
        sums=np.r_[0.,np.cumsum(weight[order])]
        idx=np.searchsorted(cc[order],np.asarray(margins)-R1.EPS,side='left')
        core[:,g]=sums[-1]-sums[idx]
    return core,total


def volume(poses,f,m):
    core,total=_volume_arrays(poses,f,[m]);core=core[0]
    return dict(core_m3=float(core.sum()),total_m3=float(total.sum()),fraction=float(core.sum()/total.sum()),
                core_by_distance_height_m3=core.reshape(4,2).tolist(),
                total_by_distance_height_m3=total.reshape(4,2).tolist(),
                fraction_by_distance_height=(core/total).reshape(4,2).tolist(),
                method='5cm clipped voxel-center quadrature; boundary discretization, no sentinel volume')


def calibrate():
    plan=R1.read(OUT/'PLAN.json');n=512;tick=time.monotonic()
    passall=np.zeros((2,n,11,len(MARGINS)),bool)
    maskcount=np.zeros(passall.shape,np.int32)
    coreall=np.zeros((2,len(MARGINS),8));totalall=np.zeros((2,8))
    receipts={}
    for mi,model in enumerate(('original','gravity')):
        for replica in range(n):
            if time.time()>=plan['deadline_unix']:raise TimeoutError('R3 shared deadline')
            seed=np.random.SeedSequence([2026100631,replica])
            true,poses=synthetic(seed,model)
            for fi,f in enumerate(range(5,16)):
                prep=prepared(poses,f);cap=capacity(prep['nominal_local'])
                actual=_fresh(prep['actual_local'],[0])
                counts=len(cap)-np.searchsorted(np.sort(cap),MARGINS-R1.EPS,side='left')
                max_missing=float(cap[~actual].max(initial=-np.inf))
                passall[mi,replica,fi]=(counts>0)&(MARGINS-R1.EPS>max_missing)
                maskcount[mi,replica,fi]=counts
                core,total=_volume_arrays(poses,f,MARGINS)
                coreall[mi]+=core;totalall[mi]+=total
            if replica%64==0:print(model,replica,'elapsed',round(time.monotonic()-tick,1),flush=True)
        rate=passall[mi].mean((0,1));feasible=np.flatnonzero((rate>=.99)&(maskcount[mi]>0).all((0,1)))
        selected=int(feasible[0]) if len(feasible) else None
        receipts[model]=dict(margin_deg=None if selected is None else float(MARGINS[selected]),
            passed=None if selected is None else int(passall[mi,:,:,selected].sum()),denominator=n*11,
            pass_fraction=None if selected is None else float(rate[selected]),
            core_fraction=None if selected is None else float(coreall[mi,selected].sum()/totalall[mi].sum()),
            core_by_distance_height_m3=None if selected is None else (coreall[mi,selected]/(n*11)).reshape(4,2).tolist(),
            total_by_distance_height_m3=(totalall[mi]/(n*11)).reshape(4,2).tolist(),
            fraction_by_distance_height=None if selected is None else (coreall[mi,selected]/totalall[mi]).reshape(4,2).tolist(),
            curve=[dict(margin_deg=float(m),passed=int(passall[mi,:,:,j].sum()),pass_fraction=float(rate[j]),
                core_fraction=float(coreall[mi,j].sum()/totalall[mi].sum()),minimum_mask_points=int(maskcount[mi,:,:,j].min())) for j,m in enumerate(MARGINS)])
    primary='gravity' if receipts['original']['core_fraction'] is not None and receipts['original']['core_fraction']<.5 else 'original'
    result=dict(models=receipts,primary_model=primary,margin_deg=receipts[primary]['margin_deg'],
        selected_model=primary,selected_m=receipts[primary]['margin_deg'],
        repetitions=n,frames=list(range(5,16)),seed='SeedSequence([2026100631,replica])',
        pooling='5632 correlated frames from512 geometry-only trajectories; calibration, not independent certificate',
        elapsed_seconds=time.monotonic()-tick,
        volume_method='Mean512x11 weighted5cm clipped-cell midpoint quadrature; approximate angular boundary',
        source_sha256=R1.sha(Path(__file__)))
    R1.save(OUT/'calibration.json',result)
    np.savez_compressed(OUT/'calibration.npz',margins=MARGINS,passed=passall,mask_count=maskcount,
                        core_volume_sums=coreall,total_volume_sums=totalall)
    (OUT/'source').mkdir(exist_ok=True)
    (OUT/'source/cnh_tristate_dev_r3_geometry.py').write_bytes(Path(__file__).read_bytes())
    print({k:{a:v for a,v in r.items() if a!='curve'} for k,r in receipts.items()},flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['calibrate'])
    parser.parse_args();calibrate()
