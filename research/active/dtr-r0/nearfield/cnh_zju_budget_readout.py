"""Frozen global RGB scale versus observed ToF spatial readouts at far budgets.

Calibration thresholds are deployable conditional rules; eval-retuned thresholds
are explicitly oracle ranking diagnostics and cannot establish calibration gain.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time

import h5py
import numpy as np
import cnh_zju_rank_localization as grid
import cnh_zju_distance_anchor as anchor

ROOT=grid.ROOT
OUT=ROOT/'artifacts.local/work/cnh-zju-budget-readout-20261002'
RUN_ID='CNH_ZJU_BUDGET_READOUT_20261002'
ARMS=('raw','tof_idw4','mono','global','depthor')
BUDGETS=(.1,.2)
SEED=2026100228
read=grid.read
save=grid.save
sha=grid.sha


def idw4(points,centres,values):
    if not len(centres):return np.full(len(points),np.nan)
    ds=((points[:,None,:]-centres[None,:,:])**2).sum(axis=2)
    # Stable source-zone order resolves equally distant centres without GT.
    order=np.argsort(ds,axis=1,kind='stable')[:,:min(4,len(centres))]
    d=np.take_along_axis(ds,order,axis=1);v=values[order]
    exact=d==0;has=exact.any(axis=1)
    weights=1/np.maximum(d,1e-12)
    result=(weights*v).sum(axis=1)/weights.sum(axis=1)
    result[has]=(exact*v).sum(axis=1)[has]/exact.sum(axis=1)[has]
    return result


def observed(boxes,mask,returns,mono,depthor):
    nodes=[grid.public_nodes(box,mono.shape)[0] for box in boxes]
    valid=mask&np.isfinite(returns)&(returns>.001)&(returns<10)
    values=[mono[n[:,0],n[:,1]].astype(float) for n in nodes]
    med=np.array([np.median(v[np.isfinite(v)&(v>0)]) if np.any(np.isfinite(v)&(v>0)) else np.nan for v in values])
    eligible=valid&np.isfinite(med)&(med>0)
    scale=float(np.median(returns[eligible]/med[eligible])) if eligible.any() else None
    h,w=mono.shape;centres=[];distances=[]
    for k,box in enumerate(boxes):
        y0,x0,y1,x1=map(int,box);y0,y1=np.clip([y0,y1],0,h);x0,x1=np.clip([x0,x1],0,w)
        if valid[k] and y1>y0 and x1>x0:
            centres.append([(y0+y1-1)/2,(x0+x1-1)/2]);distances.append(returns[k])
    centres=np.asarray(centres).reshape(-1,2);distances=np.asarray(distances)
    rows=[]
    for k,(n,m) in enumerate(zip(nodes,values)):
        m=m.copy();m[~np.isfinite(m)|(m<=0)]=np.nan
        scores=dict(raw=np.full(len(n),returns[k] if valid[k] else np.nan),
            tof_idw4=idw4(n,centres,distances),mono=m,
            global_=m*scale if scale is not None else np.full(len(n),np.nan),depthor=depthor[n[:,0],n[:,1]].astype(float))
        scores['global']=scores.pop('global_')
        rows.append(dict(zone=k,nodes=n,sensor_valid=bool(valid[k]),scores=scores))
    return rows,dict(scale=scale,idw_anchors=len(centres))


def frame(row):
    for a,b in (('input','input_sha256'),('prediction','prediction_sha256'),('depthor','depthor_sha256')):assert sha(row[a])==row[b]
    with h5py.File(row['input'],'r') as f:boxes=f['fr'][:];mask=f['mask'][:];returns=f['hist_data'][:,0]
    with np.load(row['prediction']) as f:mono=f['depth']
    with np.load(row['depthor']) as f:depthor=f['depth']
    zones,info=observed(boxes,mask,returns,mono,depthor)
    with h5py.File(row['input'],'r') as f:gt=f['depth'][:]
    result=dict(id=row['id'],index=row['index'],scene=row['scene'],role=row['role'],info=info,zone_count=64,empty=0,invalid_sensor_zones=0)
    chunks={k:[] for k in ('sensor','known','near','main','far','mixed','thin',*ARMS)}
    for z in zones:
        n=z['nodes'];d=gt[n[:,0],n[:,1]];known=np.isfinite(d)&(d>.001)&(d<10)
        near=known&(d<2.1);far=known&(d>=2.1)
        result['empty']+=len(n)==0;result['invalid_sensor_zones']+=not z['sensor_valid']
        chunks['sensor'].append(np.full(len(n),z['sensor_valid']));chunks['known'].append(known);chunks['near'].append(near)
        chunks['main'].append(near&(d>=1.2));chunks['far'].append(far)
        chunks['mixed'].append(np.full(len(n),near.any() and far.any()))
        chunks['thin'].append(np.full(len(n),1<=near.sum()<=8 and far.any()))
        for a in ARMS:chunks[a].append(z['scores'][a])
    result.update({k:np.concatenate(v) for k,v in chunks.items()})
    return result


def select_threshold(scores,far,budget):
    count=int(far.sum())
    if not count:return dict(defined=False,threshold=None,all_finite=False,n_far=0,allowed=0)
    eligible=far&np.isfinite(scores)&(scores>0)
    values=np.sort(scores[eligible]);allowed=int(np.floor(budget*count+1e-12))
    if len(values)<=allowed:return dict(defined=True,threshold=None,all_finite=True,n_far=count,allowed=allowed)
    return dict(defined=True,threshold=float(values[allowed]),all_finite=False,n_far=count,allowed=allowed)


def calls(scores,rule):
    valid=np.isfinite(scores)&(scores>0)
    if not rule['defined']:return np.zeros(len(scores),bool)
    return valid if rule['all_finite'] else valid&(scores<rule['threshold'])


def merge(frames):
    return {k:np.concatenate([f[k] for f in frames]) for k in ('sensor','known','near','main','far','mixed','thin',*ARMS)}


def counts(data,arm,rule,group='sensor'):
    use=data['sensor'] if group=='sensor' else data['sensor']&data[group]
    p=calls(data[arm],rule);valid=np.isfinite(data[arm])&(data[arm]>0)
    near=use&data['near'];main=use&data['main'];far=use&data['far'];unknown=use&~data['known']
    c=dict(points=int(use.sum()),near=int(near.sum()),main=int(main.sum()),far=int(far.sum()),reference_unknown=int(unknown.sum()),
        tp=int((near&p).sum()),main_tp=int((main&p).sum()),fp=int((far&p).sum()),
        missing_near=int((near&~valid).sum()),missing_far=int((far&~valid).sum()),unknown_reference_calls=int((unknown&p).sum()))
    c['fn']=c['near']-c['tp'];c['main_fn']=c['main']-c['main_tp'];c['tn_nontrigger']=c['far']-c['fp']
    c['closer']=c['near']-c['main'];c['closer_tp']=c['tp']-c['main_tp']
    c['closer_recall']=c['closer_tp']/c['closer'] if c['closer'] else None
    c.update(recall=c['tp']/c['near'] if c['near'] else None,main_recall=c['main_tp']/c['main'] if c['main'] else None,
        far_fpr=c['fp']/c['far'] if c['far'] else None)
    return c


def paired(data,base,rules):
    near=data['sensor']&data['near'];main=data['sensor']&data['main'];far=data['sensor']&data['far']
    a=calls(data['global'],rules['global']);b=calls(data[base],rules[base])
    return dict(main_rescued=int((main&a&~b).sum()),main_lost=int((main&~a&b).sum()),
        closer_rescued=int((near&~main&a&~b).sum()),closer_lost=int((near&~main&~a&b).sum()),
        allnear_rescued=int((near&a&~b).sum()),allnear_lost=int((near&~a&b).sum()),
        far_added=int((far&a&~b).sum()),far_removed=int((far&~a&b).sum()))


def prepare():
    assert not (OUT/'PLAN.json').exists();old=read(anchor.OUT/'PLAN.json')
    OUT.mkdir(parents=True,exist_ok=True)
    line=f'| 2026-10-02 | {RUN_ID} | PRE_RUN; consumed ZJU160 same4cal4eval; raw/ToF IDW4/mono/frozen frame-global/DEPTHOR; no local-anchor tuning; actual public clipped centre IDW4 1/d²; strict distance-score thresholds at cal10/20%far>=2.1 budgets, ties excluded; cal selects raw/IDW4 by main1.2-2.1 recall | NOT_RUN; eval cal-frozen thresholds+actual far cost/allnear/main/closer/thin/mixed, global paired loss/rescue; separate eval-refit oracle budgets diagnostic only | Both budgets require global main +3pp vs calbestToF, <1.2m recall noninferior, actual far<=bestToF AND nominal budget, >=3/4scene main gain; else GLOBAL_READOUT_NOT_ESTABLISHED. Mono/DEPTHOR retained, no universalbest/bodyalarm/freshconfirmation claim; no same-cache readout tuning | `artifacts.local/work/cnh-zju-budget-readout-20261002/REPORT.md` |\n'
    body=grid.RUNS.read_text(encoding='utf-8');assert RUN_ID not in body
    grid.RUNS.write_text(body.rstrip()+'\n'+line,encoding='utf-8');(OUT/'prerun-row.txt').write_text(line,encoding='utf-8')
    save(OUT/'PLAN.json',dict(run_id=RUN_ID,inputs=old['inputs'],roles=old['roles'],source_sha256=sha(__file__),
        dependencies={str(x):sha(x) for x in (anchor.OUT/'PLAN.json',anchor.OUT/'result.json',grid.__file__,anchor.__file__)},
        rules=dict(arms=ARMS,budgets=BUDGETS,selection='cal only strongest ToF raw/IDW4: main recall max, actual far min, raw first tie',
            scores='Frozen global median(return/median public-grid mono) positive scale; IDW4 over centres of clipped sensor-valid rectangles, stable source order and exact-centre averaging; no GT',
            main='Published sensor_valid point domain, 1.2<=GT<2.1; allnear<2.1 retained; far>=2.1; finite .001<GT<10; mixed/thin auxiliary only',
            budget='strict score<threshold, maximal deterministic threshold without exceeding floor(B*Nfar), ties excluded together; UNKNOWN no trigger retained in positive miss denominator',
            gate='Both budgets: eval main global-bestToF>=.03, global closer<1.2 recall>=bestToF, global far<=bestToF and<=budget, >=3/4scene main positive gain. Undefined => NOT_EVALUABLE; else false=>GLOBAL_READOUT_NOT_ESTABLISHED; pass=>CALIBRATED_RGB_SUPPORT_DEV. tolerance1e-12',
            oracle='Repeat budget threshold selection on eval only as optimistic ranking diagnostic, never deployed or gate; compare also eval-best ToF to avoid weak selected baseline',
            bootstrap='1000 paired whole eval scene multinomial draws, seed2026100228, fixed cal rule and selected ToF',
            limits='4eval consumed scenes; repeated/overlap nodes not independent; no physical echo/body calibration; far pixels not clear-body false alarms; no training/inference/download; local stopped'),preregistration=line))
    print('PREPARED160',flush=True)


def run():
    p=read(OUT/'PLAN.json');assert sha(__file__)==p['source_sha256'] and not (OUT/'result.json').exists()
    for path,digest in p['dependencies'].items():assert sha(path)==digest
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:frames=list(pool.map(frame,p['inputs']))
    cal=merge([f for f in frames if f['role']=='cal']);evaluation=merge([f for f in frames if f['role']=='eval'])
    eval_scenes=[s for s,v in p['roles'].items() if v=='eval'];by_scene={s:merge([f for f in frames if f['scene']==s]) for s in p['roles']}
    budgets={};draws=np.random.default_rng(SEED).multinomial(4,np.full(4,.25),1000)
    for budget in BUDGETS:
        rules={a:select_threshold(cal[a],cal['sensor']&cal['far'],budget) for a in ARMS}
        c={a:counts(cal,a,rules[a]) for a in ARMS}
        def pick(records):
            eligible=[a for a in ('raw','tof_idw4') if records[a]['main_recall'] is not None and records[a]['far_fpr'] is not None]
            return max(eligible,key=lambda a:(records[a]['main_recall'],-records[a]['far_fpr'],-ARMS.index(a))) if eligible else None
        best=pick(c)
        e={g:{a:counts(evaluation,a,rules[a],g) for a in ARMS} for g in ('sensor','mixed','thin')}
        scenes={s:{a:counts(data,a,rules[a]) for a in ARMS} for s,data in by_scene.items()}
        estimable=best is not None and all(e['sensor'][a][k] is not None for a in ('global',best) for k in ('main_recall','closer_recall','far_fpr'))
        wins=[s for s in eval_scenes if best is not None and scenes[s]['global']['main_recall'] is not None and scenes[s][best]['main_recall'] is not None and scenes[s]['global']['main_recall']>scenes[s][best]['main_recall']+1e-12]
        gates=dict(main_gain=estimable and e['sensor']['global']['main_recall']-e['sensor'][best]['main_recall']>=.03-1e-12,
            closer_retained=estimable and e['sensor']['global']['closer_recall']>=e['sensor'][best]['closer_recall']-1e-12,
            matched_cost=estimable and e['sensor']['global']['far_fpr']<=min(budget,e['sensor'][best]['far_fpr'])+1e-12,scene_wins=len(wins)>=3)
        oracle_rules={a:select_threshold(evaluation[a],evaluation['sensor']&evaluation['far'],budget) for a in ARMS}
        oracle={a:counts(evaluation,a,oracle_rules[a]) for a in ARMS}
        intervals={}
        for a in ('raw','tof_idw4','mono','depthor'):
            vals={}
            for num,den in (('main_tp','main'),('tp','near'),('fp','far')):
                numerator=np.array([scenes[s]['global'][num]-scenes[s][a][num] for s in eval_scenes]);denominator=np.array([scenes[s]['global'][den] for s in eval_scenes])
                d=draws@denominator;ok=d>0;delta=100*(draws@numerator)[ok]/d[ok]
                vals[num]=dict(ci95_pp=np.percentile(delta,[2.5,97.5]).tolist() if len(delta) else None,valid=int(ok.sum()))
            intervals[a]=vals
        budgets[str(budget)]=dict(rules=rules,cal=c,chosen_tof=best,eval=e,scenes=scenes,gates=gates,estimable=estimable,scene_wins=wins,
            paired={a:paired(evaluation,a,rules) for a in ('raw','tof_idw4','mono','depthor')},bootstrap=intervals,
            oracle_rules=oracle_rules,oracle_eval=oracle,oracle_best_tof=pick(oracle))
    estimable=all(v['estimable'] for v in budgets.values());passed=estimable and all(all(v['gates'].values()) for v in budgets.values())
    verdict=('CALIBRATED_RGB_SUPPORT_DEV' if passed else 'GLOBAL_READOUT_NOT_ESTABLISHED') if estimable else 'NOT_EVALUABLE'
    ledger=[]
    for f in frames:
        v={k:f[k] for k in ('id','index','scene','role','info','zone_count','empty','invalid_sensor_zones')}
        v['counts']={b:{g:{a:counts(f,a,row['rules'][a],g) for a in ARMS} for g in ('sensor','mixed','thin')} for b,row in budgets.items()}
        ledger.append(v)
    save(OUT/'frame-ledger.json',ledger)
    save(OUT/'result.json',dict(verdict=verdict,budgets=budgets,seconds=time.perf_counter()-start,
        all_zones=sum(f['zone_count'] for f in frames),empty_zones=sum(f['empty'] for f in frames),invalid_sensor_zones=sum(f['invalid_sensor_zones'] for f in frames)))
    print(json.dumps(dict(verdict=verdict,gates={b:v['gates'] for b,v in budgets.items()})),flush=True)


def selftest():
    scores=np.array([1.,1.,2.,3.,np.nan]);far=np.ones(5,bool)
    r=select_threshold(scores,far,.2);assert r['threshold']==1 and calls(scores,r).sum()==0
    r=select_threshold(scores,far,.4);assert r['threshold']==2 and calls(scores,r).sum()==2
    r=select_threshold(scores,far,1.);assert r['all_finite'] and calls(scores,r).sum()==4
    assert not select_threshold(scores,np.zeros(5,bool),.1)['defined']
    points=np.array([[0.,0.],[0.,1.]]);centres=np.array([[0.,0.],[0.,2.]])
    assert np.allclose(idw4(points,centres,np.array([1.,3.])),[1.,2.])
    assert np.isnan(idw4(points,np.empty((0,2)),np.empty(0))).all()
    print('PASS_TIED_BUDGET_UNKNOWN_IDW')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('selftest','prepare','run'))
    globals()[parser.parse_args().action]()
