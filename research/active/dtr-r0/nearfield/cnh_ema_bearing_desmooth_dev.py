"""Remove only extra output-score smoothing at frozen EMA notice times."""
from __future__ import annotations
import csv
from pathlib import Path
import time
import numpy as np
import cnh_ema_bearing_opportunity_dev as O

B, S, E, Y = O.B, O.S, O.E, O.Y
OUT = B.A.ROOT/'artifacts.local/work/cnh-ema-bearing-desmooth-dev-20261008'


def fuse(raw):
    """query,branch,config,output,height -> sensor,config,output,query."""
    value=np.asarray(raw,float).max(-1)
    return np.stack((value[:,0],value[:,1:].max(1))).transpose(0,2,3,1)


def yaw_change(noisy):
    yaw=np.degrees(np.arctan2(noisy[:,:,0,2],noisy[:,:,2,2]))[:,3:]
    result=np.zeros_like(yaw)
    for f in range(13):
        delta=(yaw[:,f,None]-yaw[:,max(0,f-4):f+1]+180.)%360.-180.
        result[:,f]=np.abs(delta).max(-1)
    return result


def descriptors(score):
    labels=score.argmax(-1).astype(np.int8)
    sorted_score=np.sort(score,axis=-1)
    margin=sorted_score[...,-1]-sorted_score[...,-2]
    stable=np.zeros_like(labels,bool)
    for f in range(13):
        stable[:,f]=(labels[:,max(0,f-4):f+1]==labels[:,f,None]).all(-1)
    return labels,margin,stable


def paired(new,old,contact):
    return dict(rescued=int((contact & new & ~old).sum()),lost=int((contact & ~new & old).sum()),
        diff=int(new[contact].sum()-old[contact].sum()))


def decision(clean,unique):
    if len(clean)!=4 or len(unique)!=4: raise ValueError('Four challenges per metric required')
    return 'PRIORITIZE_DESMOOTH_FOLLOWUP' if all(x>=0 for x in clean+unique) and any(x>0 for x in clean) else 'RETAIN_ORIGINAL_L2'


def strata(first,contact,values,old,new):
    rr=np.flatnonzero(contact & (first>=0))
    out={}
    for group,masks in values.items():
        rows=[]
        for label,mask in masks:
            members=np.zeros(len(contact),bool); members[rr]=mask[rr,first[rr]]
            pair=paired(new['clean'],old['clean'],members)
            rows.append(dict(label=label,timely=int(members.sum()),
                old_clean=int(old['clean'][members].sum()),k1_clean=int(new['clean'][members].sum()),
                old_unique=int(old['unique'][members].sum()),k1_unique=int(new['unique'][members].sum()),
                rescued=pair['rescued'],lost=pair['lost']))
        assert sum(r['timely'] for r in rows)==len(rr)
        out[group]=rows
    return out


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic(); completed=[]; status='FAILED'; error=None
    attempts=OUT/'attempts'; attempts.mkdir(exist_ok=True)
    spent=sum(B.A.read(p)['seconds'] for p in attempts.glob('analysis*.json'))
    def check():
        if spent+time.monotonic()-started>=180.:raise TimeoutError('Cumulative CPU analysis180s cap')
    try:
        check()
        paths=[Path(__file__),Path(O.__file__),Y.N.HERE/'cnh_tristate_dev.py',S.OUT/'ledger.npz',
            S.OUT/'result.json',O.OUT/'ledger.npz',O.OUT/'result.json',
            O.OUT/'roundtable/final-result.json']
        plan=dict(task='CNH_EMA_BEARING_DESMOOTH_DEV_20261008',lane='EXPLORE consumed Development',
            authorization='User 推进 following ready-reviewed room20261008081541-f66c',
            goal='Compare removal of extra output-score smoothing; inspect association with head-yaw changes, not prove causality or deployment cost',
            budgets_wall_seconds=dict(cpu_analysis=180.,focused_verification=90.),
            candidate='Only k=1: existing current raw output, max heights and selected branches then argmax query; LEFT-first exact ties; no abstention threshold',
            baseline='Reproduce original L2 per-query/branch/height causal2^k weighted max5 outputs then fuse. L1 reference unchanged.',
            frozen='96units/3840windows/229contacts/384clear; all5yaw conditions, all13 outputs; all EMA notices/thresholds/startup retained from sector same-notice budget; original deadlines and all-contact footprint evaluator unchanged',
            separation='Compute and save all-window candidate labels/descriptors before reading contact/deadline/truth. Evaluator first outputs2..deadline inclusive; input=output+3; wrong/abstain never move first.',
            strata=dict(yaw='max abs wrap(current noisy yaw - each noisy yaw in outputs max(0,f-4)..f); includes startup history; <=2deg,(2,10],>10',
                old_margin='Original fused L2 top1-top2 raw logit difference: =0,(0,.1],(.1,.5],>.5; descriptive, not confidence/threshold',
                k1_stability='All k1 argmax labels in same max5 output window equal current label; stable/changing; descriptive not object association'),
            decision='Per sensor four challenges: k1-minus-L2 clean and unique all>=0 and at least one clean>0 -> PRIORITIZE_DESMOOTH_FOLLOWUP, otherwise RETAIN_ORIGINAL_L2 including all-zero/mixed/totalregression. No automatic replacement or statistical noninferiority; does not veto other temporal/spatial mechanisms.',
            decision_check='One event=1/229 (~.437pp); paired units retain96source clusters and229events, cells not independent. Known L2 clean unsaturated in four challenges (wrong single20/15/22/23, dual22/6/26/23). Unique only non-decrease required; no per-cell strict-win. No eight-cell sum or relaxed tolerance.',
            adjustable='Implementation correction only; no k2/k3, tuning, newquery/inference/render/confirmation, margin or yaw-selected policy',
            stop='Cap,missing source,changed lineage or unresolved alignment: preserve completed units, no incomplete-cohort ranking; no cap expansion',
            deliverables='All-window predictions, first-event paired/stratified ledger, exact-tie counts, cost, focused verification, report/current/run log/master push',
            limits='k1 means single output, retains up to8 observations and5 model ensemble; cache check is not increment inference or phone latency. L3 NOT_RUN; M3/phone/480unit identity unchanged.',
            hashes={str(p.relative_to(B.A.ROOT)):B.A.sha(p) for p in paths})
        if (OUT/'PLAN.json').exists():
            if B.A.read(OUT/'PLAN.json')!=B.json_value(plan):raise ValueError('Frozen desmooth plan changed')
        else:B.save(OUT/'PLAN.json',plan)
        with np.load(S.OUT/'ledger.npz') as z:
            uid=z['unit']; cfg=z['config']; old_score=z['score']
        assert old_score.shape==(5,2,3840,13,3)
        raw_score=np.empty_like(old_score); yaws=np.empty((5,3840,13)); raw_hashes={}; noisy_hashes={}; costs=[]
        parent=B.A.read(S.OUT/'result.json')['input_units_sha256']
        physical_parent=B.A.read(Y.OUT/'result.json')['provenance']['unit_sha256']
        native_parent=B.A.read(Y.M.OUT/'replay_result.json')['provenance']['unit_sha256']
        for u in Y.UNITS:
            check(); t=time.monotonic(); rows=np.flatnonzero(uid==u)
            np.testing.assert_array_equal(cfg[rows],np.arange(40))
            dest=OUT/'units'/f'unit{u}.npz'; raw_path=S.OUT/'units'/f'unit{u}.npz'
            native_path=Y.M.OUT/'units'/f'unit{u}.npz'; physical_path=Y.OUT/'units'/f'unit{u}.npz'
            raw_hashes[str(u)]=B.A.sha(raw_path)
            noisy_hashes[str(u)]=dict(native=B.A.sha(native_path),physical=B.A.sha(physical_path))
            if raw_hashes[str(u)]!=parent[str(u)]:raise ValueError('Sector raw lineage changed')
            if noisy_hashes[str(u)]!=dict(native=native_parent[str(u)],physical=physical_parent[str(u)]):raise ValueError('Noisy pose lineage changed')
            saved={}
            with np.load(raw_path) as z,np.load(native_path) as n,np.load(physical_path) as p:
                for ni,name in enumerate(Y.NAMES):
                    raw=z[name+'/raw']; assert raw.shape==(3,3,40,13,2) and np.isfinite(raw).all()
                    current=fuse(raw); original=fuse(B.R.smooth(raw))
                    np.testing.assert_array_equal(original,old_score[ni][:,rows])
                    span=yaw_change(n['noisy'] if name=='zero' else p[name+'/noisy'])
                    raw_score[ni][:,rows]=current; yaws[ni,rows]=span
                    saved[name+'/raw_score']=current; saved[name+'/yaw_change']=span
            B.atomic_npz(dest,**saved); completed.append(u); costs.append(time.monotonic()-t)
            if len(completed)==1:
                B.save(OUT/'cost_projection.json',dict(first_unit_seconds=costs[0],projected96_seconds=costs[0]*96,
                    setup_seconds=t-started,includes='Hash/read source, all5conditions/fusion/parity/yaw and candidate unit save; first reused; no inference'))
        candidates=dict(unit=uid,config=cfg,raw_score=raw_score,old_score=old_score,yaw_change=yaws)
        for ni,name in enumerate(Y.NAMES):
            for si,sensor in enumerate(('single','dual')):
                for arm,data in (('k1',raw_score),('L2',old_score)):
                    labels,margin,stable=descriptors(data[ni,si]);key=name+'/'+sensor+'/'+arm
                    candidates[key+'/labels']=labels; candidates[key+'/margin']=margin; candidates[key+'/stable']=stable
        check();B.atomic_npz(OUT/'candidate.npz',**candidates)
        # Join evaluator-only outcomes only after full candidate persistence.
        with np.load(S.OUT/'ledger.npz') as z: truth=z['truth']
        with np.load(O.OUT/'ledger.npz') as z:
            old={k:z[k] for k in z.files}
        contact=old['contact'];deadline=old['deadline']
        np.testing.assert_array_equal(uid,old['unit']);np.testing.assert_array_equal(cfg,old['config'])
        with np.load(E.OUT/'ledger.npz') as z: ema_score=z['score'][:,:,4];control=z['control']
        points=B.A.read(S.OUT/'result.json');metrics={};ledger=dict(unit=uid,config=cfg,contact=contact,control=control,deadline=deadline);events=[]
        for ni,name in enumerate(Y.NAMES):
            for si,sensor in enumerate(('single','dual')):
                check();key=name+'/'+sensor; theta=points['metrics'][key]['ema']['working_point']['threshold']
                alarm=ema_score[ni,si]>=theta;first=O.first_notice(alarm,contact,deadline)
                np.testing.assert_array_equal(alarm,old[key+'/alarm']);np.testing.assert_array_equal(first,old[key+'/first'])
                ledger[key+'/alarm']=alarm;ledger[key+'/first']=first
                outcomes={};arms={}
                for arm in ('k1','L2','L1'):
                    labels=old[key+'/L1/labels'] if arm=='L1' else candidates[key+'/'+arm+'/labels']
                    rec,e=O.evaluate(labels,first,truth[ni],contact); arms[arm]=rec;outcomes[arm]=e
                    ledger[key+'/'+arm+'/labels']=labels
                    for m,value in e.items():ledger[key+'/'+arm+'/'+m]=value
                    if arm!='k1':
                        for m,value in e.items():np.testing.assert_array_equal(value,old[key+'/'+arm+'/'+m])
                pairs={other:{m:paired(outcomes['k1'][m],outcomes[other][m],contact) for m in ('clean','unique','wrong')} for other in ('L2','L1')}
                span=yaws[ni];margin=candidates[key+'/L2/margin'];stable=candidates[key+'/k1/stable']
                groups=dict(yaw=[('<=2',span<=2.),('(2,10]',(span>2.)&(span<=10.)),('>10',span>10.)],
                    old_margin=[('=0',margin==0.),('(0,.1]',(margin>0.)&(margin<=.1)),('(.1,.5]',(margin>.1)&(margin<=.5)),('>.5',margin>.5)],
                    k1_stability=[('stable',stable),('changing',~stable)])
                rr=np.flatnonzero(first>=0);ff=first[rr]
                ties={a:int((np.sum(candidatescore[ni,si,rr,ff]==candidatescore[ni,si,rr,ff].max(-1)[:,None],-1)>1).sum()) for a,candidatescore in (('k1',raw_score),('L2',old_score))}
                cell=dict(events=229,total_notices=int(alarm.sum()),threshold=theta,arms=arms,
                    k1_against_L2=pairs['L2'],k1_against_L1=pairs['L1'],first_exact_max_ties=ties,
                    strata=strata(first,contact,groups,outcomes['L2'],outcomes['k1']))
                metrics[key]=cell
                for row in np.flatnonzero(contact):
                    f=int(first[row]);timely=f>=0
                    events.append(dict(name=name,sensor=sensor,unit=int(uid[row]),config=int(cfg[row]),
                        first_output=f,first_input_frame=f+3 if timely else -1,deadline=int(deadline[row]),
                        yaw_change=None if not timely else float(span[row,f]),old_margin=None if not timely else float(margin[row,f]),
                        k1_margin=None if not timely else float(candidates[key+'/k1/margin'][row,f]),
                        k1_stable=None if not timely else bool(stable[row,f]),timely=timely,
                        **{a+'_label':-1 if not timely else int(ledger[key+'/'+a+'/labels'][row,f]) for a in ('L1','L2','k1')},
                        **{a+'_'+m:bool(outcomes[a][m][row]) for a in ('L1','L2','k1') for m in ('clean','unique','wrong','abstain')}))
        decisions={}
        for sensor in ('single','dual'):
            clean=[metrics[n+'/'+sensor]['k1_against_L2']['clean']['diff'] for n in Y.NAMES[1:]]
            unique=[metrics[n+'/'+sensor]['k1_against_L2']['unique']['diff'] for n in Y.NAMES[1:]]
            decisions[sensor]=dict(diffs_clean=clean,diffs_unique=unique,decision=decision(clean,unique))
        check();B.atomic_npz(OUT/'ledger.npz',**ledger)
        with (OUT/'events.csv').open('x',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(events[0]));w.writeheader();w.writerows(events)
        B.save(OUT/'result.json',dict(status='COMPLETE',units=96,events=229,controls=384,metrics=metrics,decisions=decisions,
            raw_unit_hashes=raw_hashes,noisy_unit_hashes=noisy_hashes,unit_cpu_seconds=sum(costs),
            candidate_sha256=B.A.sha(OUT/'candidate.npz'),ledger_sha256=B.A.sha(OUT/'ledger.npz'),
            limits=plan['limits'],source_sha256=B.A.sha(Path(__file__))))
        status='COMPLETE';print('DESMOOTH_COMPLETE',decisions,flush=True)
        for key,cell in metrics.items():print(key, {a:(r['first_clean'],r['first_unique'],r['first_wrong']) for a,r in cell['arms'].items()},cell['k1_against_L2'],flush=True)
    except BaseException as exc:
        error=repr(exc);raise
    finally:
        B.save(attempts/f'analysis{time.time_ns()}.json',dict(status=status,error=error,seconds=time.monotonic()-started,
            previous_seconds=spent,budget_seconds=180.,completed_units=completed,backend='CPU numpy only, zero inference'))


if __name__=='__main__':run()
