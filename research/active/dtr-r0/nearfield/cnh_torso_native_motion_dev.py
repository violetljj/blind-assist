"""Same-clip native motion for frozen-M3 Development, with a torso-aligned sensor proxy.

Future pelvis1.5m enters source sampling, scene placement, exact query and geometry.
It never enters a causal estimator; scene construction itself is future-conditioned.
Every causal estimator retains its original native 60Hz clip state. No speed scaling.
"""
import argparse
import json
import time

import numpy as np
import cnh_torso_bias_dev as T
import cnh_torso_gait_bias_dev as G
import cnh_torso_bias_replay_dev as B

OUT=T.OUT/'native_motion'
ARMS=('exact','e1','torso','corrected','corrected_gait')
SEED=2026100747
GROUPS=('cold','onset','slowturn','straight')


def estimates(x,initial,gait):
    lab=T.labels(x); yaw=T.torso_yaw(x); ids=np.arange(len(x))
    old,b0,u0=T.causal_correct(x[:,0],yaw,initial)
    new,b1,u1,_=G.causal_correct(x[:,0],yaw,gait)
    p=x[:,0,:2]; distance=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    future=np.searchsorted(distance,distance+1.5,side='left')
    return dict(pelvis=p,head=x[:,6,:2],exact=lab['truth'],
        e1=T.B.yaw(x[:,6,:2]-x[np.maximum(ids-60,0),6,:2]),torso=yaw,
        corrected=old,corrected_gait=new,turn_group=lab['turn_group'],
        support=(future<len(x))&(lab['speed']>=.3),original_valid=lab['valid'],
        bias_initial=b0,bias_gait=b1,updated_initial=u0,updated_gait=u1,
        seen_initial=np.cumsum(u0)>0,seen_gait=np.cumsum(u1)>0,onsets=lab['onsets'])


def window_rows(d,ci,pid):
    """Evaluation strata select sources; future labels never enter estimator updates."""
    support=d['support']; c=np.r_[0,np.cumsum(support)]
    starts=np.flatnonzero(c[192:]-c[:-192]==192)
    valid=set(map(int,starts)); role='calibration' if pid in T.DEV else 'evaluation'
    rows=[]
    def add(start,group):
        if start in valid: rows.append(dict(clip_index=ci,start=int(start),pid=pid,
            role=role,requested_group=group))
    add(0,0)
    for onset in d['onsets']: add(max(0,onset-72),1)
    for start in starts[::12]:
        # Window construction anchor frame10; actual event deadline is regrouped after geometry.
        group=int(d['turn_group'][int(start)+120])
        if group==1:add(int(start),2)
        elif group==0:add(int(start),3)
    return rows


def build():
    tick=time.monotonic(); OUT.mkdir(parents=True,exist_ok=True)
    initial=B.A.read(T.OUT/'result_angles.json')['config']
    gait=B.A.read(G.OUT/'result_angles.json')['config']
    parent=B.A.read(T.OUT/'PLAN.json'); files=parent['files']
    plan=dict(task='CNH_TORSO_NATIVE_MOTION_DEV_20261007',lane='POSTHOC EXPLORE',
        goal='Repair future/current and source/simulator trajectory mismatch, preserving fixed candidates and cold state',
        budget=dict(bank_cpu_seconds=600,replay_setup_render_geometry_inference_wall_seconds=1800,analysis_cpu_seconds=600),
        units=B.A.UNITS,arms=ARMS,seed=SEED,initial_config=initial,gait_config=gait,
        adjustments='Implementation repairs only; no parameter selection, extra units, threshold regrouping, or budget expansion',
        stop='Stop on cumulative stage budget or complete144 units; preserve partial/failure outputs; no incomplete cohort superiority conclusion',
        decision_check='Unknown native event/control counts until geometry. One event minimal effect; saturated/empty strata not strict-win. Overall gains disappearing lowers gait candidate priority; local negatives trigger traces, direction research stays priority.',
        source='All1025 consumed clips; P01..05 calibration sources, P06..10 evaluation sources, nominal identities unknown',
        source_sampling='config%4 cold/onset/slowturn/straight; only contiguous192frame future-target-present walking windows; native stride12 ->16frames5Hz. Cold starts0; onset startmax(0,onset72framesback); others frame120 label with starts decimated12. Actual deadline groups recomputed, not quota interpreted as events.',
        target='Future first pelvis point >=1.5m accumulated native horizontal path. Same native truth, pelvis origin for exact query and all-object geometry. FullSE3 inv(estimatedTravel)@sensor for each arm; translation retained.',
        sensor='Torso-aligned yaw surrogate because head yaw unavailable; head joint horizontal position, constant y0,pitch-10. Native metres, no speed scaling. World rigidly reexpressed with final target yaw forward and final pelvis at origin; mirrored source sign shared every arm.',
        estimator='Original60Hz fullclip causal states, no reset at sampledwindow; clip starts0, no future truth passed to causal updates. E1 native head chord t-60 with clipped cold prefix, ideal published positions, not noisy deployable PDR.',
        event='Existing all-box surface queries and0.9m front-coordinate crossing on actual native target frame, not swept curved-route collision or BlindWays obstacles',
        primary='E1 smallest whole-tie feasible <=2.5% evaluation all13-control FA count; each arm matches actual count, preserve residual. Report timely excluding startup plus full-before, all13/start/main costs, actual turn/cold and Pxx heterogeneity; descriptive not deployment calibration.',
        uncertainty='No significance claim or sim-unit-only CI: overlapping native windows/clips/Pxx, model/calibration/selection uncertainty unresolved',
        deliverable='Native bank, paired rendered observations/raw scores/poses/query and geometry ledger, focused tests and scoped docs/commit/push; M3 frozen',
        provenance=dict(parent_plan_sha256=T.sha(T.OUT/'PLAN.json'),initial_result_sha256=T.sha(T.OUT/'result_angles.json'),gait_result_sha256=T.sha(G.OUT/'result_angles.json'),source_sha256=T.sha(__file__)),files=files)
    B.save(OUT/'PLAN.json',plan)
    packed={}; rows=[]
    with np.load(T.OUT/'series.npz',allow_pickle=False) as z:old={k:z[k] for k in ('e1','torso','corrected')}
    with np.load(G.OUT/'series.npz',allow_pickle=False) as z:new=z['corrected_gait']
    maxdiff=0.
    for ci,r in enumerate(files):
        if time.monotonic()-tick>=600:raise TimeoutError('Native bank CPU600s')
        p=T.B.SRC/r['name']; assert T.sha(p)==r['sha256']; x=np.load(p)
        d=estimates(x,initial,gait); rows.extend(window_rows(d,ci,r['pid']))
        sl=slice(ci*600,(ci+1)*600)
        for a in ('e1','torso','corrected','corrected_gait'):
            stored=new[sl] if a=='corrected_gait' else old[a][sl]
            diff=np.abs(T.B.wrap(T.B.wrap(d[a]-d['exact'])-stored))
            maxdiff=max(maxdiff,float(diff.max()))
            if diff.max()>1e-9:raise ValueError('Frozen native estimator changed: '+a)
        for k,v in d.items():
            if k!='onsets':packed.setdefault(k,[]).append(v)
    arrays={k:np.stack(v) for k,v in packed.items()};arrays.update(pid=np.array([r['pid'] for r in files]),clip=np.array([r['name'] for r in files]))
    counts={role:{g:sum(r['role']==role and r['requested_group']==gi for r in rows) for gi,g in enumerate(GROUPS)} for role in ('calibration','evaluation')}
    if any(v==0 for groups in counts.values() for v in groups.values()):raise ValueError('A source stratum is empty')
    B.atomic_npz(OUT/'bank.npz',**arrays);B.save(OUT/'windows.json',rows)
    B.save(OUT/'bank_result.json',dict(status='COMPLETE',seconds=time.monotonic()-tick,windows=len(rows),counts=counts,
        frozen_error_max_wrapped_diff=maxdiff,bank_sha256=T.sha(OUT/'bank.npz'),windows_sha256=T.sha(OUT/'windows.json')))
    print('NATIVE_BANK',counts,'seconds',time.monotonic()-tick,flush=True)


def load_bank():
    with np.load(OUT/'bank.npz',allow_pickle=False) as z:bank={k:z[k] for k in z.files}
    return bank,B.A.read(OUT/'windows.json')


def draw(bank,windows,unit,config):
    role='calibration' if unit<99000 else 'evaluation'; group=config%4
    ix=[i for i,w in enumerate(windows) if w['role']==role and w['requested_group']==group]
    rng=np.random.default_rng([SEED,unit,config]);wi=int(ix[rng.integers(len(ix))])
    return dict(windows[wi],window=wi),1. if rng.random()<.5 else -1.


def poses(bank,row,sign):
    """Complete query SE3; exact reprojects sensor into the same frame used for truth."""
    ci=int(row['clip_index']); f=int(row['start'])+12*np.arange(16)
    if not bank['support'][ci,int(row['start']):int(row['start'])+192].all():raise ValueError('Incomplete window')
    phi=float(bank['exact'][ci,f[-1]]);rad=np.radians(phi)
    u=np.array([np.cos(rad),np.sin(rad)]);l=np.array([-np.sin(rad),np.cos(rad)])
    anchor=bank['pelvis'][ci,f[-1]]
    def position(p):
        xy=p-anchor;return np.stack([sign*(xy@l),np.zeros(16),xy@u],1)
    def rotation(yaw):
        r=np.repeat(np.eye(4)[None],16,0);v=np.radians(sign*T.B.wrap(yaw-phi));c,s=np.cos(v),np.sin(v)
        r[:,0,0]=c;r[:,0,2]=s;r[:,2,0]=-s;r[:,2,2]=c
        return r
    sensor=rotation(bank['torso'][ci,f]);pitch=np.eye(4);r=np.radians(-10);pitch[1:3,1:3]=[[np.cos(r),-np.sin(r)],[np.sin(r),np.cos(r)]]
    sensor=sensor@pitch;sensor[:,:3,3]=position(bank['head'][ci,f])
    travel=rotation(bank['exact'][ci,f]);travel[:,:3,3]=position(bank['pelvis'][ci,f])
    queries={}
    for a in ARMS:
        estimated=rotation(bank[a][ci,f]);estimated[:,:3,3]=travel[:,:3,3]
        queries[a]=np.linalg.inv(estimated)@sensor
    meta=dict(native_frame_index=f,clip_index=ci,pid=str(bank['pid'][ci]),clip=str(bank['clip'][ci]),
        requested_group=int(row['requested_group']),turn_group=bank['turn_group'][ci,f],
        updated=bank['updated_gait'][ci,f],seen=bank['seen_gait'][ci,f],bias_deg=bank['bias_gait'][ci,f]*sign,
        heading_error=np.stack([sign*T.B.wrap(bank[a][ci,f]-bank['exact'][ci,f]) for a in ARMS]))
    return dict(sensor=sensor,travel=travel,queries=queries,**meta)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['build']);parser.parse_args();build()
