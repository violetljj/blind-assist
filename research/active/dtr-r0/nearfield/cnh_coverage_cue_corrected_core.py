"""Frozen, CPU-only synthetic coverage-cue closed loops.

Corrected twelve-cell replay. E_oracle reads true current pose by explicit
upper-reference contract; E_grav reads estimated pose. Future remains evaluator-only.
No sensor return, image, histogram, neural network or obstacle input is used.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import cnh_coverage_cue_geometry as G

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-coverage-cue-corrected-20261005'
DT, SPEED, C = .2, .8, 12


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temporary=p.with_suffix(p.suffix+'.tmp')
    with temporary.open('w',encoding='utf8',newline='\n') as f:
        json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    temporary.replace(p)


def configurations():
    result=[dict(name=m,method=m,p=0,L=0.,T=0.,diagnostic=False) for m in ('none','dual','zero')]
    for method in ('A','B','v1'):
        for p in ([0] if method=='A' else [0,1,2,3,5]):
            for L in [.3,.6,1.]:
                for T in [.2,.4]:
                    result.append(dict(name=f'{method}_p{p}_L{L}_T{T}',method=method,p=p,L=L,T=T,diagnostic=False))
            result.append(dict(name=f'{method}_p{p}_L0_T0',method=method,p=p,L=0.,T=0.,diagnostic=True))
    assert len(result)==80
    return result


def cell_centers(plan):
    points=[];side=[];height=[]
    for x in plan['cells']['x_centers_m']:
        for q,values in enumerate(plan['cells']['height_centers_m'].values()):
            for y in values:
                points.append([x,y,0.]);side.append(0 if x<0 else 1);height.append(q)
    return np.asarray(points),np.asarray(side),np.asarray(height)


def exogenous(sigma,tau,replica,n,seed):
    rng=np.random.default_rng([seed,int(sigma),int(tau*10),replica])
    innovation=rng.normal(0,sigma*np.sqrt(1-np.exp(-2*DT/tau)),n)
    holds=[];scans=[]
    for rate,dest,is_scan in [(.08,holds,False),(.04,scans,True)]:
        clock=0.
        while True:
            clock+=rng.exponential(1/rate)
            if clock>=n*DT:break
            begin=int(np.ceil(clock/DT))
            if is_scan:
                dest.append(dict(start=begin,end=begin+45,sign=int(rng.choice([-1,1]))))
            else:
                dest.append(dict(start=begin,end=begin+int(np.ceil(rng.uniform(1,6)/DT)),
                                 amplitude=float(rng.uniform(10,20))*int(rng.choice([-1,1]))))
    hv=np.zeros((len(holds),n));sv=np.zeros((len(scans),n))
    for j,e in enumerate(holds):hv[j,e['start']:min(n,e['end'])]=e['amplitude']
    # 0->-20 (1.5s), -20->+20 (3s), +20->0 (1.5s): total6s.
    # Retain PLAN speed/endpoints; its9s arithmetic error is disclosed.
    for j,e in enumerate(scans):
        phase=(np.arange(n)-e['start'])*DT
        value=np.where(phase<1.5,-phase*40/3,np.where(phase<4.5,-20+(phase-1.5)*40/3,20-(phase-4.5)*40/3))
        active=(phase>=0)&(phase<6.)
        sv[j,active]=value[active]*e['sign']
        e['end']=e['start']+30
    return innovation,holds,scans,hv,sv


def noise(replica,sigma,tau,n):
    rng=np.random.default_rng([2026100507,int(sigma),int(tau*10),replica])
    scale=rng.uniform(-.2,.2)
    bias=rng.choice([-1.,1.],3)*np.deg2rad(1.)
    translation=np.empty(n);rotation=np.empty((n,3));translation[0]=1.;rotation[0]=0
    for t in range(1,n):
        translation[t]=1+scale+rng.normal(0,.02)
        rotation[t]=bias*DT+rng.normal(0,np.deg2rad(.1),3)
    return translation,G.rodrigues(rotation),dict(scale=float(scale),bias_deg_s=np.rad2deg(bias).tolist())


def true_visible(yaw,points,ranges):
    """Evaluator only, world station ranges and actual yaw; no occlusion."""
    world=np.broadcast_to(points,(len(yaw),len(ranges),C,3)).copy()
    world[...,2]=np.asarray(ranges)[None,:,None]
    rotation=G.rotation_yaw_pitch(yaw)
    local=np.einsum('pqci,pij->pqcj',world,rotation)
    return G.visible_spheres(local)


def estimate_angles(rotations,positions,t):
    velocity=(positions[:,t]-positions[:,t-5])/1.
    velocity[:,1]=0.
    norm=np.linalg.norm(velocity,axis=1)
    direction=np.divide(velocity,norm[:,None],out=np.zeros_like(velocity),where=norm[:,None]>.05)
    averaged=rotations[:,t-5:t+1,:,2].mean(1);instant=rotations[:,t,:,2].copy()
    averaged[:,1]=0;instant[:,1]=0
    valid=(norm>=.05)&(np.linalg.norm(averaged,axis=1)>=.05)&(np.linalg.norm(instant,axis=1)>=.05)
    def angle(head):
        return np.rad2deg(np.arctan2(direction[:,2]*head[:,0]-direction[:,0]*head[:,2],(direction*head).sum(1)))
    return angle(averaged),angle(instant),velocity,direction,valid


class VirtualTracker:
    """Causal virtual stations, fixed after creation in estimated world gauge.

    Each station is created when it enters the two-second prediction horizon.
    Future station time is a public routing key, not evaluator truth. Estimated
    front ranges and current velocity determine predicted deadline eligibility.
    """
    def __init__(self,ids,points,side):
        self.ids=ids;self.points=points;self.side=side
        self.world=None;self.count=None

    def create(self,position,velocity,direction,q):
        right=np.stack([direction[:,2],np.zeros(len(direction)),-direction[:,0]],1)
        center=position[:,None]+velocity[:,None]*(DT*np.asarray(q)[None,:,None])+.97*direction[:,None]
        return center[:,:,None]+self.points[None,None,:,0,None]*right[:,None,None]+self.points[None,None,:,1,None]*np.array([0.,1.,0.])

    def flags(self,rotations,positions,velocity,direction,valid,t):
        ids=self.ids;position=positions[ids,t];rotation=rotations[ids,t]
        velocity=velocity[ids];direction=direction[ids]
        if self.world is None:
            self.world=self.create(position,velocity,direction,np.arange(11))
            self.count=np.zeros((len(ids),11,C),np.int8)
        else:
            self.world=np.concatenate([self.world[:,1:],self.create(position,velocity,direction,[10])],axis=1)
            self.count=np.concatenate([self.count[:,1:],np.zeros((len(ids),1,C),np.int8)],axis=1)
        relative=self.world-position[:,None,None]
        front=(relative*direction[:,None,None]).sum(-1)
        local=np.einsum('pqci,pij->pqcj',relative,rotation)
        in_window=(front>=.9)&(front<=1.4)
        observed=np.zeros(in_window.shape,bool)
        observed[in_window]=G.visible_spheres(local[in_window])
        self.count+=observed
        speed=np.maximum(np.linalg.norm(velocity,axis=1),.05)
        first=np.maximum(1,np.ceil((front-1.4)/(speed[:,None,None]*DT)-1e-10)).astype(int)
        last=np.floor((front-.9)/(speed[:,None,None]*DT)+1e-10).astype(int)
        # Enumerate the entire H=2s horizon. A slow estimated horizontal
        # velocity can put more than four samples inside [.9,1.4]m.
        steps=first[...,None]+np.arange(10)
        keep=(steps<=last[...,None])&(steps<=10)
        eligible=(front>=.9)&(front-speed[:,None,None]*2<=.9)
        # A positive side certificate needs no more rays: even if EVERY
        # remaining possible exposure is visible, this cell cannot reach3.
        certain=(self.count+keep.sum(-1)<3)&eligible
        result=np.stack([certain[...,self.side==s].any((1,2)) for s in (0,1)],axis=1)
        active=np.flatnonzero(~result.all(1))
        if not len(active):
            result[~valid[ids]]=False
            return result
        future=relative[:,:,:,None]-velocity[:,None,None,None]*(DT*steps[...,None])
        local=np.einsum('pqcfi,pij->pqcfj',future[active],rotation[active])
        relevant=keep[active]&eligible[active,...,None]
        predicted=np.zeros(relevant.shape,bool)
        predicted[relevant]=G.visible_spheres(local[relevant])
        future_count=predicted.sum(-1)
        bad=(self.count[active]+future_count<3)&eligible[active]
        result[active]=np.stack([bad[...,self.side==s].any((1,2)) for s in (0,1)],axis=1)
        result[~valid[ids]]=False
        return result


def fork_unnecessary(cues,actual_yaw,ou_states,cancel_states,innovation,holds,scans,hv,sv,points,side,zero_counts,tau):
    """Evaluator fork: true future with no new cue after the event, H=2sec."""
    if not cues:return np.empty((0,2),bool),np.empty((0,2),bool)
    nc=len(cues);future=np.zeros((nc,11))
    starts=np.array([e['start'] for e in holds]);scan_starts=np.array([e['start'] for e in scans])
    times=np.array([e['frame'] for e in cues]);configs=np.array([e['config'] for e in cues])
    states=ou_states[configs,times].copy();canceled=cancel_states[configs,times]
    future[:,0]=actual_yaw[configs,times]
    for step in range(1,11):
        tt=np.minimum(times+step,len(innovation)-1)
        states=np.exp(-DT/tau)*states+innovation[tt]
        h=(hv[:,tt].T*(starts[None]>canceled[:,None])).sum(1)
        s=(sv[:,tt].T*(scan_starts[None]>canceled[:,None])).sum(1)
        future[:,step]=states+h+s
    # q0 deadline is .0875s after cue; q0..9 lie in the fixed2s horizon.
    q=np.arange(10)[:,None];exposure=q+np.array([-2,-1,0])[None]
    yaw=np.empty((nc,10,3))
    for j in range(10):
        for f in range(3):
            offset=int(exposure[j,f])
            yaw[:,j,f]=actual_yaw[configs,np.maximum(times+offset,0)] if offset<=0 else future[:,offset]
    world=np.broadcast_to(points,(nc,10,3,C,3)).copy();world[...,2]=np.array([1.29,1.13,.97])[None,None,:,None]
    local=np.einsum('nefci,nefij->nefcj',world,G.rotation_yaw_pitch(yaw))
    counts=G.visible_spheres(local).sum(2)
    bad=counts<3
    unnecessary=np.stack([~bad[...,side==s].any((1,2)) for s in (0,1)],1)
    restorable=zero_counts>=3
    restorable_unnecessary=np.stack([~bad[...,((side==s)&restorable)].any((1,2)) for s in (0,1)],1)
    return unnecessary,restorable_unnecessary


def simulate(job):
    model,sigma,tau,replica=job;plan=read(OUT/'PLAN.json')
    dest=OUT/model/'streams'/f'sigma{sigma}_tau{tau}_K{replica}.json'
    if dest.exists():return read(dest)
    if time.time()>=plan['deadline_unix']-300:return dict(status='NOT_RUN',sigma=sigma,tau=tau,replica=replica)
    tick=time.monotonic();cfg=configurations();P=len(cfg);N=600
    points,side,height=cell_centers(plan)
    innovation,holds,scans,hv,sv=exogenous(sigma,tau,replica,N,plan['head_motion']['seed'])
    trans_noise,rot_noise,noise_receipt=noise(replica,sigma,tau,N)
    # Preserve original RNG consumption and local zero-mean innovations.
    rng=np.random.default_rng([2026100507,int(sigma),int(tau*10),replica])
    rng.uniform(-.2,.2);bias=rng.choice([-1.,1.],3)*np.deg2rad(1.)
    bias_y=bias[1];step_noise=np.zeros((N,3))
    for tt in range(1,N):
        rng.normal(0,.02);step_noise[tt]=rng.normal(0,np.deg2rad(.1),3)
    noise_receipt.update(model=model,gravity_bias_deg_s=float(np.rad2deg(bias_y)),
                         convention='world-Y left bias; original local step noise unchanged')
    hs=np.array([e['start'] for e in holds]);ss=np.array([e['start'] for e in scans])
    actual=np.zeros((P,N));rot=np.empty((P,N,3,3));pos=np.zeros((P,N,3))
    ou=np.zeros(P);ou_ledger=np.zeros((P,N));cancel=np.full(P,-1,int);cancel_ledger=np.zeros((P,N),int)
    pending=np.full(P,-1,int);turn_start=np.full(P,-1,int);turn_angle=np.zeros(P);turn_frames=np.zeros(P,int);cool_until=np.zeros(P,int)
    consecutive=np.zeros((P,2),int);viscount=np.zeros((P,N,C),np.int8)
    warmup_unknown=np.zeros(P,int);cues=[];hold_realized=np.zeros((P,len(holds)),bool)
    v1=np.array([j for j,c in enumerate(cfg) if c['method']=='v1'])
    tracker=VirtualTracker(v1,points,side)
    L=np.array([c['L'] for c in cfg]);T=np.array([c['T'] for c in cfg]);p=np.array([c['p'] for c in cfg])
    methods=np.array([c['method'] for c in cfg]);cue_policy=np.isin(methods,['A','B','v1'])
    true_previous=None
    for t in range(N):
        if t%50==0 and time.time()>=plan['deadline_unix']-300:
            partial=dest.with_suffix('.partial.npz');partial.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(partial,true_yaw=actual[:,:t],estimated_rotation=rot[:,:t],estimated_position=pos[:,:t],coverage_counts=viscount[:,:t])
            return dict(status='NOT_RUN_COMPLETE_SEQUENCE',sigma=sigma,tau=tau,replica=replica,completed_frames=t)
        ou=np.exp(-DT/tau)*ou+innovation[t]
        hmask=(hs[None]>cancel[:,None]);smask=(ss[None]>cancel[:,None])
        h=(hv[:,t][None]*hmask).sum(1);s=(sv[:,t][None]*smask).sum(1)
        base=ou+h+s
        starting=(pending>=0)&(t>=pending)&(turn_start<0)
        turn_start[starting]=t;turn_angle[starting]=base[starting]
        turn_frames[starting]=np.ceil(T[starting]/DT-1e-10).astype(int)
        turning=turn_start>=0
        fraction=np.divide(t-turn_start,np.maximum(turn_frames,1),out=np.zeros(P),where=turning)
        actual[:,t]=np.where(turning,turn_angle*(1-np.minimum(fraction,1)),base)
        complete=turning&((t-turn_start)>=turn_frames)
        actual[complete,t]=0;ou[complete]=0;cancel[complete]=t
        pending[complete]=-1;turn_start[complete]=-1;cool_until[complete]=t+5
        actual[methods=='zero',t]=0
        active_holds=(hv[:,t]!=0)[None]&hmask&(~turning[:,None])
        hold_realized|=active_holds
        true_rot=G.rotation_yaw_pitch(actual[:,t])
        if t==0:
            rot[:,t]=true_rot
        else:
            delta=np.swapaxes(true_previous,1,2)@true_rot
            local_translation=true_previous[:,2,:]*(SPEED*DT*trans_noise[t])
            pos[:,t]=pos[:,t-1]+np.einsum('pij,pj->pi',rot[:,t-1],local_translation)
            if model=='E_grav':
                # Constant bias is a LEFT/world increment around gravity Y.
                increment=G.rodrigues(step_noise[t])
                gravity=G.rodrigues(np.array([0.,bias_y*DT,0.]))
                rot[:,t]=gravity@rot[:,t-1]@delta@increment
            else:
                rot[:,t]=rot[:,t-1]@delta@rot_noise[t]
        if model=='E_oracle':
            rot[:,t]=true_rot
            pos[:,t]=np.array([0.,0.,t*SPEED*DT])
        true_previous=true_rot
        # The independent true observer accumulates actual visibility for cells.
        visible=true_visible(actual[:,t],points,[.97,1.13,1.29])
        dual=np.flatnonzero(methods=='dual')[0]
        visible[dual]=true_visible(np.array([actual[dual,t]-15]),points,[.97,1.13,1.29])[0]|true_visible(np.array([actual[dual,t]+15]),points,[.97,1.13,1.29])[0]
        for q in range(3):
            if t+q<N:viscount[:,t+q]+=visible[:,q]
        ou_ledger[:,t]=ou;cancel_ledger[:,t]=cancel
        if t<5:
            warmup_unknown+=cue_policy;continue
        a,b,velocity,direction,valid=estimate_angles(rot,pos,t)
        flags=np.zeros((P,2),bool)
        for m,angle,threshold in [('A',a,8.),('B',b,plan['policies']['B']['theta_deg'])]:
            ids=np.flatnonzero(methods==m)
            flags[ids,0]=angle[ids]>=threshold;flags[ids,1]=angle[ids]<=-threshold
        flags[v1]=tracker.flags(rot,pos,velocity,direction,valid,t)
        flags[~valid]=False;warmup_unknown+=(~valid)&cue_policy
        available=cue_policy&(pending<0)&(turn_start<0)&(t>=cool_until)
        consecutive=np.where(flags&available[:,None],consecutive+1,0)
        fired=(consecutive>=p[:,None]+1)&available[:,None]
        for j in np.flatnonzero(fired.any(1)):
            cues.append(dict(config=int(j),frame=t,sides=fired[j].tolist()))
            wait=max(1,int(np.ceil(L[j]/DT-1e-10)))
            pending[j]=t+wait
            consecutive[j]=0
    zero_counts=viscount[2,50].copy()
    excess,restorable_excess=fork_unnecessary(cues,actual,ou_ledger,cancel_ledger,innovation,holds,scans,hv,sv,points,side,zero_counts,tau)
    evalmask=(np.arange(N)*DT>=10)&(np.arange(N)*DT<118)
    minutes=float(evalmask.sum()*DT/60)
    rows=[]
    for j,c in enumerate(cfg):
        events=[(i,e) for i,e in enumerate(cues) if e['config']==j and evalmask[e['frame']]]
        sides_num=np.zeros(2,int);excess_num=np.zeros(2,int);rest_num=np.zeros(2,int)
        for i,e in events:
            sides_num+=e['sides'];excess_num+=np.array(e['sides'])&excess[i];rest_num+=np.array(e['sides'])&restorable_excess[i]
        cells={}
        for s in (0,1):
            for q in (0,1):
                keep=(side==s)&(height==q);arrivals=viscount[j,evalmask][:,keep]
                cells[('LEFT' if s==0 else 'RIGHT')+'/'+('HEAD' if q==0 else 'BODY')]=dict(arrivals=int(arrivals.size),
                    insufficient_k3=int((arrivals<3).sum()),insufficient_k2=int((arrivals<2).sum()),
                    k3_per_min=float((arrivals<3).sum()/minutes),k2_per_min=float((arrivals<2).sum()/minutes))
        delays=[];censored=0;canceled=0;scheduled=0
        for he,e in enumerate(holds):
            if not(50<=e['start']<590):continue
            scheduled+=1
            if not hold_realized[j,he]:canceled+=1;continue
            tagged=0 if e['amplitude']>0 else 1
            matches=[ce['frame'] for _,ce in events if e['start']<=ce['frame']<e['end'] and ce['sides'][tagged]]
            if matches:delays.append((min(matches)-e['start'])*DT)
            else:censored+=1
        true_position=np.array([0.,0.,(N-1)*SPEED*DT])
        error_rotation=np.swapaxes(G.rotation_yaw_pitch(actual[:,N-1]),1,2)@rot[:,N-1]
        trace=np.trace(error_rotation[j]);rotation_error=float(np.rad2deg(np.arccos(np.clip((trace-1)/2,-1,1))))
        rows.append(dict(**c,cells=cells,minutes=minutes,auditory_cues=len(events),auditory_cues_per_min=len(events)/minutes,
            side_cues=sides_num.tolist(),side_cues_per_min=(sides_num/minutes).tolist(),unnecessary_side_cues=excess_num.tolist(),unnecessary_side_cues_per_min=(excess_num/minutes).tolist(),
            restorable_unnecessary_side_cues=rest_num.tolist(),restorable_unnecessary_side_cues_per_min=(rest_num/minutes).tolist(),
            direction_unavailable_or_warmup_frames=int(warmup_unknown[j]),
            hold_delay=dict(scheduled=scheduled,realized=scheduled-canceled,canceled_by_loop=canceled,detected=len(delays),censored=censored,delays_s=delays),
            final_estimator_rotation_error_deg=rotation_error,final_estimator_translation_error_m=float(np.linalg.norm(pos[j,-1]-true_position))))
    ledger=dest.with_suffix('.npz');ledger.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(ledger,true_yaw=actual,estimated_rotation=rot,estimated_position=pos,coverage_counts=viscount,
        cue_config=np.array([e['config'] for e in cues]),cue_frame=np.array([e['frame'] for e in cues]),cue_sides=np.array([e['sides'] for e in cues],bool),
        unnecessary=excess,restorable_unnecessary=restorable_excess,OU_innovations=innovation,
        hold_values=hv,scan_values=sv,cancel_frames=cancel_ledger,ou_states=ou_ledger)
    result=dict(status='COMPLETE',model=model,sigma=sigma,tau=tau,replica=replica,rows=rows,holds=holds,scans=scans,noise=noise_receipt,
        zero_counts=zero_counts.tolist(),seconds=time.monotonic()-tick,ledger_sha256=sha(ledger),plan_sha256=sha(OUT/'PLAN.json'))
    save(dest,result)
    print('STREAM COMPLETE',model,sigma,tau,replica,round(result['seconds'],2),flush=True)
    return result
