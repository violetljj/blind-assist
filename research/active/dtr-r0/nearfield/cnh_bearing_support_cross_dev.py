"""Frozen cached sparse support diagnostic, separated from evaluator geometry.

Finite-bin samples are standardized evidence, never depth/occupancy probabilities.
All-window first EMA main notices are selected without contact/deadline access.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
from pathlib import Path
import time
import numpy as np
from scipy import ndimage
import cnh_ema_bearing_desmooth_dev as D

O, B, S, E, Y = D.O, D.B, D.S, D.E, D.Y
OUT = B.A.ROOT/'artifacts.local/work/cnh-bearing-support-cross-dev-20261008'
WIDTH = .3002784
EDGE = np.tan(np.pi/8)
LOW = np.array([-.3, -.2, .3])
HIGH = np.array([.3, .9, 3.])
EDGES = [np.r_[np.arange(a,b-1e-8,d),b] for a,b,d in zip(LOW,HIGH,(.1,.2,.2))]
SHAPE = tuple(len(a)-1 for a in EDGES)
INDEX = np.array(list(itertools.product(*(range(n) for n in SHAPE))))
CELL_LOW = np.stack([EDGES[a][INDEX[:,a]] for a in range(3)],-1)
CELL_HIGH = np.stack([EDGES[a][INDEX[:,a]+1] for a in range(3)],-1)
BITS = np.array(list(itertools.product((0,1),repeat=3)))
CORNERS = CELL_LOW[:,None,:]+BITS[None,:,:]*(CELL_HIGH-CELL_LOW)[:,None,:]
CENTRES = (CELL_LOW+CELL_HIGH)/2
SAMPLES = np.array(list(itertools.product((0.,.5,1.),repeat=3)))


def rx(deg):
    a=np.radians(deg);c,s=np.cos(a),np.sin(a)
    return np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])


def extrinsic(deg):
    a=np.radians(deg);c,s=np.cos(a),np.sin(a)
    value=np.eye(4);value[:3,:3]=rx(10)@np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])@rx(-10)
    return value


def bin_samples(indices):
    """27 angular/radial endpoint+midpoint samples per finite CNH bin."""
    idx=np.asarray(indices)
    # Cached layout is angular Y, angular X, radial (projection flat index).
    angular=-EDGE+2*EDGE/8*(idx[:,None,[1,0]]+SAMPLES[None,:,:2])
    direction=np.concatenate((angular,np.ones((*angular.shape[:2],1))),-1)
    direction/=np.linalg.norm(direction,axis=-1,keepdims=True)
    radius=WIDTH*(idx[:,None,2]+SAMPLES[None,:,2])
    return direction*radius[...,None]


def cell_masks(query):
    """Conservative finite-cell horizontal interval, not centroid-only labels."""
    horizontal=np.eye(4);horizontal[:3,:3]=rx(-10)
    transform=horizontal@np.linalg.inv(query)
    points=CORNERS@transform[:3,:3].T+transform[:3,3]
    angle=np.degrees(np.arctan2(points[...,0],points[...,2]))
    forward=(points[...,2]>0).all(-1); partly=(points[...,2]>0).any(-1)&~forward
    lo=angle.min(-1);hi=angle.max(-1)
    masks=np.zeros((len(INDEX),3),bool)
    masks[:,0]=forward&(lo < -10-1e-10)
    masks[:,1]=forward&(hi > -10+1e-10)&(lo < 10-1e-10)
    masks[:,2]=forward&(hi > 10+1e-10)
    masks[partly]=True
    return masks,transform


def support(z, noisy, query, frame, branches):
    """Observed sparse geometry only; no truth, boxes, contacts or deadlines."""
    weights=np.zeros(len(INDEX));covered=np.zeros(len(INDEX),bool)
    for i in range(max(0,frame-7),frame+1):
        base=query@np.linalg.inv(noisy[frame])@noisy[i]
        for branch in branches:
            transform=base@extrinsic((0.,-15.,15.)[branch])
            # Whole-cell certificate requires all corners in ONE exposure.
            local=(CORNERS-transform[:3,3])@transform[:3,:3]
            with np.errstate(divide='ignore',invalid='ignore'):
                inside=(local[...,2]>0)&(np.abs(local[...,0]/local[...,2])<=EDGE)&(np.abs(local[...,1]/local[...,2])<=EDGE)&(np.linalg.norm(local,axis=-1)<=16*WIDTH)
            covered|=inside.all(-1)
            indices=np.argwhere(z[branch,i]>3.)
            if not len(indices):continue
            p=bin_samples(indices)@transform[:3,:3].T+transform[:3,3]
            w=np.repeat((z[branch,i][tuple(indices.T)].astype(float)-3.)/27,27)
            p=p.reshape(-1,3)
            keep=((p>=LOW)&(p<HIGH)).all(-1)
            p,w=p[keep],w[keep]
            if not len(p):continue
            ijk=tuple(np.searchsorted(EDGES[a],p[:,a],side='right')-1 for a in range(3))
            np.add.at(weights,np.ravel_multi_index(ijk,SHAPE),w)
    masks,head=cell_masks(query)
    coverage=np.array([bool(masks[:,j].any() and covered[masks[:,j]].all()) for j in range(3)])
    fraction=np.array([float(covered[masks[:,j]].mean()) if masks[:,j].any() else np.nan for j in range(3)])
    labels,count=ndimage.label((weights>0).reshape(SHAPE),structure=np.ones((3,3,3)))
    flat=labels.ravel();clusters=[]
    for j in range(1,count+1):
        member=flat==j;mask=masks[member].any(0)
        if not mask.any():continue
        centre=np.average(CENTRES[member],axis=0,weights=weights[member])
        p=head[:3,:3]@centre+head[:3,3]
        bearing=float(np.degrees(np.arctan2(p[0],p[2])))
        clusters.append(dict(mask=sum(1<<k for k in np.flatnonzero(mask)),weight=float(weights[member].sum()),bearing=bearing))
    union=0
    for c in clusters:union|=c['mask']
    sector_weight=(weights[:,None]*masks).sum(0)
    dominant=None if not clusters else max(clusters,key=lambda c:c['weight'])
    distance=np.nan if dominant is None else min(abs(dominant['bearing']-b) for b in (-10.,10.))
    state=0 if not clusters else (1 if len(clusters)==1 and union in (1,2,4) else (2 if len(clusters)==1 else 3))
    return dict(mask=union,cluster_count=len(clusters),state=state,coverage=coverage,
                coverage_fraction=fraction,sector_weight=sector_weight,boundary_distance=distance,
                cluster_masks=np.array([c['mask'] for c in clusters],np.int8))


def choose(old, rec):
    """Positive evidence switch/set; absence veto requires full nominal coverage."""
    selected=1<<old
    veto=-1 if not rec['mask']&selected and rec['coverage'][old] else old
    switch=int(np.log2(rec['mask'])) if rec['state']==1 else old
    output_set=rec['mask'] if rec['mask'] else selected
    return veto,switch,output_set


def freeze():
    paths=[Path(__file__),Path(O.__file__),Path(__file__).with_name('cnh_bearing_error_anatomy_dev.py'),
           Path(__file__).with_name('cnh_cvr_projection.py'),OUT/'field_audit.json',S.OUT/'ledger.npz',O.OUT/'ledger.npz',S.OUT/'result.json',
           B.A.ROOT/'artifacts.local/work/cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy']
    plan=dict(task='CNH_BEARING_SUPPORT_CROSS_DEV_20261008',lane='EXPLORE consumed Development',
        goal='Observation/evaluator cross-table chooses cached spatial, association or set follow-up by absolute repair/loss counts',
        authorization='User repeated 推进 and supplied corrected roundtable conclusion; bounded cached diagnostic, no new inference',
        budgets_wall_seconds=dict(metadata_audit=120,cpu_analysis=900,focused_verification=120),
        cohort='96units/3840windows/229contacts/384clear, five yaw conditions; single and dual separate. Missing unit NOT_EVALUABLE, retain229; incomplete cohort cannot rank.',
        observation='All-window FIRST EMA notice output>=2, no contact/deadline reads. Fixed EMA timing/threshold/all alarms; first not advanced after abstention. First input=output+3.',
        normalization='Recompute cached float16 z=(hist-bias)/sqrt(max(16ambient+max(bias,0),1e-9)); signed hist, no extra ambient subtraction. z>3, weights(z-3)/27; one frozen descriptive threshold, no probabilistic occupancy claim.',
        geometry='History at most8 inputs through noisy SE3 and saved EMA query; branches0 single,1+2 dual. 27 endpoint/midpoint samples per finite angular/radial bin, crop shared body/head box x[-.3,.3] y[-.2,.9] z[.3,3]. Fixed .1/.2/.2m cells, final partial cells; 26-neighbour positive-cell clusters, no min-size tuning. No ground/truth mask. Finite cell corner sector masks, partial behind conservatively all3. Support union/multicluster is not object association.',
        coverage='Nominal whole Cartesian-cell certificate: all8 corners within one exposure cone/outer radial ball; OR across exposures. Full sector only if every touching query-domain cell certified. Fraction descriptive, never percentage veto. Noise/pose/visibility errors mean certificate is not reliable sensing or safety proof. Broad near-head query may preclude full coverage: then absence is UNKNOWN and cannot veto; this is a known capability gap, not negative L3 evidence.',
        candidates='VETO remove direction only when selected sector has no sampled support AND full nominal sector coverage; retain obstacle reminder. SWITCH one positive cluster wholly within one sector -> that sector, otherwise keep L2; incomplete coverage does not imply unseen alternatives absent. SET report observed positive-sector union or original singleton if none, no truth pruning; one compound notice, report extra label count, audio cost unknown. DISAGREE dual raw branches original smooth perquery then maxheight, argmax; suppress direction on differing labels. None change EMA alarms.',
        boundary_yaw='Dominant support-cluster centroid angular distance to +/-10deg (finite-cell mask separate); actual max wrapped noisy yaw span of same max5 scoring outputs plus input+3. Compare span>=distance only descriptive, no causal alignment verdict.',
        evaluator='Join only after candidate.npz persisted: immutable target legal mask, runner-up legal diagnostic only, nonexclusive competitor/empty/adjacent/closer flags. Competitor does not legitimate target error. SET clean iff nonempty set subset of legal truth, unique iff legal truth singleton; intersection not enough.',
        routes=dict(spatial='Per sensor four challenges: SWITCH >=3 clean rescues in any challenge, <=1 clean loss each, nonnegative clean and unique nets each; or VETO >=3 wrong removals in any, zero correct loss each and <=10 new direction abstentions each. Dual DISAGREE judged with same veto limits. No per-cell strict improvement; no eight-cell sum.',
            association='Per sensor >=3 wrong events in any challenge with selected competitor AND at least one observed cluster sector compatible with target legal set. This is truth-conditioned diagnostic opportunity, not attainable rescue or object attribution. Follow-up only, zero output changes/loss/abstention by construction; actual association method must later meet explicit paired loss criteria.',
            sets='Per sensor >=3 strictly supported set rescues in any challenge, <=1 clean loss each, nonnegative clean/unique nets each, <=10 newly wrong sets each, zero new abstention; extra labels reported separately without assigning user cost.'),
        decision_check='One event=1/229=.437pp; three events=1.31pp is an investment preference, not statistical significance or safety standard. Original L2 wrong challenge single20/15/22/23,dual22/6/26/23: headroom exists, not every cell must gain3. Allow1 paired correct loss intentionally for positive switch/set; abstention loss0. Paired96 source clusters, reused Development. Full coverage may be impossible for broad sector; do not relax after result.',
        adjustable='Optimize runtime/correct implementation before truth; no threshold/grid/history/cohort/route changes. First-unit cost projection reused; cap failure preserves partial, no full-cohort algorithm ranking. No GPU, render, masking inference, new query, k tuning, phone/user study or baseline promotion.',
        deliverables='Metadata audit, saved observations, paired events/cross-tables, per-config routes/cost, focused checks, Chinese report/current/RUNS/scoped commit and push',
        hashes={str(p.relative_to(B.A.ROOT)):B.A.sha(p) for p in paths})
    B.save(OUT/'PLAN.json',plan)
    return plan


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    plan=freeze();start=time.monotonic();status='FAILED';done=[]
    def check():
        if time.monotonic()-start>=900:raise TimeoutError('900 wall seconds CPU analysis cap')
    try:
        audit=B.A.read(OUT/'field_audit.json')
        assert audit['evaluability']['paired_units_evaluable']==96
        audited={rec['unit']:rec for rec in audit['units']}
        assert tuple(B.A.ANGLES)==(0.,-15.,15.)
        with np.load(S.OUT/'ledger.npz') as f:uid=f['unit'];cfg=f['config'];score=f['score']
        with np.load(O.OUT/'ledger.npz') as f:alarms={n+'/'+s:f[n+'/'+s+'/alarm'] for n in Y.NAMES for s in ('single','dual')}
        # No evaluator metadata enters this phase.
        shape=(5,2,3840);obs={k:np.full(shape,v,dtype=dtype) for k,v,dtype in
            [('first',-1,np.int8),('old',-1,np.int8),('runner',-1,np.int8),('mask',0,np.int8),('state',0,np.int8),
             ('cluster_count',0,np.int16),('veto',-1,np.int8),('switch',-1,np.int8),('set',0,np.int8),
             ('disagree',False,bool),('boundary_distance',np.nan,float),('yaw_span',np.nan,float)]}
        obs['coverage']=np.zeros(shape+(3,),bool);obs['coverage_fraction']=np.full(shape+(3,),np.nan)
        obs['sector_weight']=np.zeros(shape+(3,));obs['cluster_sector_union']=np.zeros(shape,np.int8)
        costs=[]; hashes={}; evaluated=0
        bias=np.load(B.A.ROOT/'artifacts.local/work/cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy').astype(np.float32)
        for u in Y.UNITS:
            check();tick=time.monotonic();rows=np.flatnonzero(uid==u);np.testing.assert_array_equal(cfg[rows],np.arange(40))
            native=Y.M.OUT/'units'/f'unit{u}.npz';physical=Y.OUT/'units'/f'unit{u}.npz'
            paths=[native,physical,E.OUT/'units'/f'unit{u}.npz',S.OUT/'units'/f'unit{u}.npz']
            hashes[str(u)]={str(p.relative_to(B.A.ROOT)):B.A.sha(p) for p in paths}
            assert hashes[str(u)][str(native.relative_to(B.A.ROOT))]==audited[u]['native']['sha256']
            assert hashes[str(u)][str(physical.relative_to(B.A.ROOT))]==audited[u]['physical']['sha256']
            with np.load(native) as n,np.load(physical) as p,np.load(paths[2]) as e,np.load(paths[3]) as raw:
                for ni,name in enumerate(Y.NAMES):
                    src=n if name=='zero' else p;prefix='' if name=='zero' else name+'/'
                    hist=src[prefix+'hist'];ambient=src[prefix+'ambient'];noisy=src[prefix+'noisy']
                    z=((hist.astype(np.float32)-bias)/np.sqrt(np.maximum(16*ambient.astype(np.float32)[...,None]+np.maximum(bias,0),1e-9))).astype(np.float16)
                    np.testing.assert_array_equal(z,src[prefix+'z'])
                    query=e[name+'/ema_query'];yaw=D.yaw_change(noisy)
                    branch_score=B.R.smooth(raw[name+'/raw']).max(-1)
                    branch_labels=branch_score.argmax(0)
                    for si,sensor in enumerate(('single','dual')):
                        alarm=alarms[name+'/'+sensor][rows].copy();alarm[:,:2]=False
                        first=np.where(alarm.any(-1),alarm.argmax(-1),-1)
                        obs['first'][ni,si,rows]=first
                        for c in np.flatnonzero(first>=0):
                            check();r=rows[c];f=int(first[c]);old=int(score[ni,si,r,f].argmax())
                            rec=support(z[:,c],noisy[c],query[c,f+3],f+3,(0,) if si==0 else (1,2))
                            veto,switch,output_set=choose(old,rec)
                            for k in ('mask','state','cluster_count','coverage','coverage_fraction','sector_weight','boundary_distance'):
                                obs[k][ni,si,r]=rec[k]
                            for k,val in [('old',old),('runner',int(np.argsort(-score[ni,si,r,f],kind='stable')[1])),
                                          ('veto',veto),('switch',switch),('set',output_set),('yaw_span',yaw[c,f]),
                                          ('disagree',bool(branch_labels[1,c,f]!=branch_labels[2,c,f]) if si==1 else False)]:
                                obs[k][ni,si,r]=val
                            obs['cluster_sector_union'][ni,si,r]=rec['mask'];evaluated+=1
            B.atomic_npz(OUT/'units'/f'unit{u}.npz',unit=uid[rows],config=cfg[rows],**{k:v[:,:,rows] for k,v in obs.items()})
            costs.append(time.monotonic()-tick);done.append(u)
            if len(done)==1:
                projection=dict(first_unit_seconds=costs[0],projected96_seconds=costs[0]*96,setup_seconds=tick-start,
                    includes='All-window first-main support for all5conditions/single+dual, normalized parity, hashes/read and branch descriptors; first reused')
                B.save(OUT/'cost_projection.json',projection);print('FIRST_UNIT_COST',projection,flush=True)
                if costs[0]*96>900:raise TimeoutError('First unit projects beyond frozen900s; preserve cost evidence, no full-cohort verdict')
            if len(done)%12==0:print('CANDIDATES',len(done),'seconds',round(time.monotonic()-start,2),flush=True)
        check();B.atomic_npz(OUT/'candidate.npz',unit=uid,config=cfg,**obs)
        evaluate(obs,uid,cfg,check)
        B.save(OUT/'input_hashes.json',hashes)
        status='COMPLETE';print('SUPPORT_CROSS_COMPLETE',evaluated,round(time.monotonic()-start,3),flush=True)
    finally:
        B.save(OUT/f'runtime-{time.time_ns()}.json',dict(status=status,seconds=time.monotonic()-start,budget_seconds=900,
            completed_units=done,backend='CPU numpy/scipy; no inference'))


def route(cells,sensor):
    four=[cells[n+'/'+sensor] for n in Y.NAMES[1:]]
    out={}
    for arm in ('veto','disagree'):
        ps=[c['paired'][arm] for c in four]
        out[arm]=any(p['wrong_removed']>=3 for p in ps) and all(p['clean_lost']==0 and p['new_abstain']<=10 for p in ps)
    ps=[c['paired']['switch'] for c in four]
    out['switch']=any(p['clean_rescued']>=3 for p in ps) and all(p['clean_lost']<=1 and p['clean_diff']>=0 and p['unique_diff']>=0 for p in ps)
    ps=[c['paired']['set'] for c in four]
    out['set']=any(p['clean_rescued']>=3 for p in ps) and all(p['clean_lost']<=1 and p['clean_diff']>=0 and p['unique_diff']>=0 and p['new_wrong']<=10 and p['new_abstain']==0 for p in ps)
    out['association']=any(c['association_opportunity']>=3 for c in four)
    return out


def evaluate(obs,uid,cfg,check):
    import cnh_bearing_error_anatomy_dev as A
    with np.load(S.OUT/'ledger.npz') as f:truth=f['truth'];target=f['target_box']
    with np.load(O.OUT/'ledger.npz') as f:contact=f['contact'];deadline=f['deadline'];old={k:f[k] for k in f.files if k.endswith('/first') or k.endswith('/L2/clean') or k.endswith('/L2/unique')}
    assert int(contact.sum())==229
    boxes={};poses={}
    for u in Y.UNITS:
        check()
        with np.load(Y.M.OUT/'units'/f'unit{u}.npz') as n,np.load(Y.OUT/'units'/f'unit{u}.npz') as p:
            boxes[u]=[json.loads(str(s)) for s in n['boxes_json']]
            for name in Y.NAMES:poses[u,name]=n['sensor'] if name=='zero' else p[name+'/sensor']
    cells={};events=[];ledger=dict(unit=uid,config=cfg,contact=contact,deadline=deadline)
    for ni,name in enumerate(Y.NAMES):
        for si,sensor in enumerate(('single','dual')):
            check();key=name+'/'+sensor;first=old[key+'/first'];rr=np.flatnonzero(first>=0)
            np.testing.assert_array_equal(obs['first'][ni,si,rr],first[rr])
            legal=truth[ni,rr,first[rr]];legal_bits=(legal*np.array([1,2,4])).sum(-1)
            arms={};outcomes={}
            for arm in ('old','veto','switch','set','disagree'):
                values=obs['set'][ni,si,rr] if arm=='set' else obs['old' if arm=='disagree' else arm][ni,si,rr]
                mask=values if arm=='set' else np.where(values<0,0,1<<np.maximum(values,0))
                if arm=='disagree':mask=np.where(obs['disagree'][ni,si,rr],0,mask)
                clean=(mask!=0)&((mask&legal_bits)==mask);unique=clean&(legal.sum(-1)==1)
                e={m:np.zeros(len(uid),bool) for m in ('clean','unique','wrong','abstain')}
                for m,val in [('clean',clean),('unique',unique),('wrong',(mask!=0)&~clean),('abstain',mask==0)]:e[m][rr]=val
                for m,val in e.items():ledger[key+'/'+arm+'/'+m]=val
                outcomes[arm]=e;arms[arm]={m:int(v.sum()) for m,v in e.items()}
                assert arms[arm]['clean']+arms[arm]['wrong']+arms[arm]['abstain']==len(rr)
            np.testing.assert_array_equal(outcomes['old']['clean'],old[key+'/L2/clean'])
            np.testing.assert_array_equal(outcomes['old']['unique'],old[key+'/L2/unique'])
            paired={}
            for arm,e in outcomes.items():
                if arm=='old':continue
                base=outcomes['old']
                paired[arm]=dict(clean_rescued=int((e['clean']&~base['clean']).sum()),clean_lost=int((~e['clean']&base['clean']).sum()),
                    clean_diff=arms[arm]['clean']-arms['old']['clean'],unique_diff=arms[arm]['unique']-arms['old']['unique'],
                    wrong_removed=int((base['wrong']&~e['wrong']).sum()),new_wrong=int((~base['wrong']&e['wrong']).sum()),
                    new_abstain=int((e['abstain']&~base['abstain']).sum()))
            flags={};wrong_flags={};cross={};association=0;runner_legal=0;boundary_yaw=0;extra_labels=0
            for r in np.flatnonzero(contact):
                f=int(first[r]);timely=f>=0
                event=dict(name=name,sensor=sensor,unit=int(uid[r]),config=int(cfg[r]),first_output=f,timely=timely,
                           status='EVALUABLE' if timely else 'NOT_TIMELY')
                if timely:
                    chosen=int(obs['old'][ni,si,r]);rec=A.describe(boxes[int(uid[r])][int(cfg[r])],int(target[r]),poses[int(uid[r]),name][cfg[r],f+3],chosen,truth[ni,r,f])
                    wrong=bool(outcomes['old']['wrong'][r]);mask=int(obs['mask'][ni,si,r]);legal_mask=int((truth[ni,r,f]*np.array([1,2,4])).sum())
                    assoc=wrong and rec['selected_has_competitor'] is True and bool(mask&legal_mask)
                    association+=int(assoc)
                    runner_ok=bool(truth[ni,r,f,int(obs['runner'][ni,si,r])]);runner_legal+=int(wrong and runner_ok)
                    boundary=bool(np.isfinite(obs['boundary_distance'][ni,si,r]) and obs['yaw_span'][ni,si,r]>=obs['boundary_distance'][ni,si,r]);boundary_yaw+=int(wrong and boundary)
                    for flag,val in rec.items():
                        if val is True:
                            flags[flag]=flags.get(flag,0)+1
                            if wrong:wrong_flags[flag]=wrong_flags.get(flag,0)+1
                    state=int(obs['state'][ni,si,r]);covered=bool(obs['coverage'][ni,si,r,chosen]);selected_support=bool(mask&(1<<chosen))
                    crosskey=f'state{state}/coverage{int(covered)}/support{int(selected_support)}/wrong{int(wrong)}'
                    cross[crosskey]=cross.get(crosskey,0)+1
                    extra_labels+=int(obs['set'][ni,si,r]).bit_count()-1
                    event.update(rec);event.update(old=chosen,runner=int(obs['runner'][ni,si,r]),runner_legal=runner_ok,
                        mask=mask,state=state,cluster_count=int(obs['cluster_count'][ni,si,r]),selected_coverage=covered,
                        selected_coverage_fraction=float(obs['coverage_fraction'][ni,si,r,chosen]),selected_support=selected_support,
                        runner_support=bool(mask&(1<<int(obs['runner'][ni,si,r]))),dual_disagree=bool(obs['disagree'][ni,si,r]),
                        boundary_distance=float(obs['boundary_distance'][ni,si,r]),yaw_span=float(obs['yaw_span'][ni,si,r]),
                        yaw_ge_boundary=boundary,association_oracle_opportunity=assoc,
                        **{arm+'_'+m:bool(v[m][r]) for arm,v in outcomes.items() for m in ('clean','unique','wrong','abstain')})
                events.append(event)
            cells[key]=dict(events=229,timely=len(rr),not_timely=229-len(rr),not_evaluable=0,arms=arms,paired=paired,
                flags=flags,wrong_flags=wrong_flags,cross_table=cross,association_opportunity=association,wrong_runner_legal=runner_legal,
                wrong_yaw_ge_boundary=boundary_yaw,extra_set_labels=extra_labels,
                all_window_firsts=int((obs['first'][ni,si]>=0).sum()))
    B.atomic_npz(OUT/'ledger.npz',**ledger)
    keys=list(dict.fromkeys(k for e in events for k in e))
    with (OUT/'events.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(events)
    decisions={sensor:route(cells,sensor) for sensor in ('single','dual')}
    B.save(OUT/'result.json',dict(status='COMPLETE',units=96,events=229,metrics=cells,decisions=decisions,
        candidate_sha256=B.A.sha(OUT/'candidate.npz'),ledger_sha256=B.A.sha(OUT/'ledger.npz'),
        limits='Consumed synthetic Development, noisy estimated geometry and privileged original query origin; sampled bin evidence/nominal FOV not dense depth or real coverage. Association oracle not actual rescue. No ground mask/dynamic pitch-roll test, new inference, phone or user evidence.'))
    print('DECISIONS',decisions,flush=True)
    for key,cell in cells.items():print(key,cell['paired'],'association',cell['association_opportunity'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');args=parser.parse_args()
    if args.freeze:OUT.mkdir(parents=True,exist_ok=True);freeze()
    else:run()
