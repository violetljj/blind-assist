"""Observed ToF-centre affine inverse-depth calibration; no GT fitting."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import numpy as np
import h5py
import cnh_zju_relative_reorder as relative

prior=relative.stages.prior
base=prior.base
ROOT=prior.ROOT
OUT=ROOT/'artifacts.local/work/cnh-zju-inverse-affine-20261003'
RUN_ID='CNH_ZJU_INVERSE_AFFINE_20261003'
ARMS=('raw','tof_idw4','depthor','inverse_affine','shuffle_affine')
GROUPS=('sensor','mixed','thin','interpolation','extrapolation')
KEYS=('sensor','known','near','main','far','mixed','thin','interpolation','extrapolation',*ARMS)
read,save,sha=prior.read,prior.save,prior.sha


def prepare():
    assert not (OUT/'PLAN.json').exists()
    old=read(relative.OUT/'PLAN.json');seal=read(relative.OUT/'prediction-seal.json')
    assert seal['plan_sha256']==sha(relative.OUT/'PLAN.json')
    rows=[dict(r,relative_sha256=seal['files'][r['relative']],affine=str(OUT/'predictions'/f'{i:03d}-affine.npz')) for i,r in enumerate(old['rows'])]
    deps={str(p):sha(p) for p in (Path(__file__),Path(prior.__file__),Path(base.__file__),Path(prior.grid.__file__),
        relative.OUT/'PLAN.json',relative.OUT/'prediction-seal.json',prior.OLD/'upstream/src/utils/dataloader.py')}
    OUT.mkdir(parents=True,exist_ok=True)
    row=f'| 2026-10-03 | {RUN_ID} | PRE_RUN; same300 consumed frames; fixed RGBinverse+observedToFcentres TheilSen2parameter 1/d=a*q+b, no GT fit/train/GPU/download; paired shuffledanchor distances, raw/IDW4/DEPTHOR | NOT_RUN; original140cal7scene/160eval8scene, pooled primary and worst retained cal10/20far thresholds eacharm; all main/closer/far/UNKNOWN/missing and interpolation-extrapolation paired | Both primary budgets: main>=calbestToF+3pp AND DEPTHOR+3pp, closer>=both, actualfar<=min(both,budget), all8scene budget, >=6/8scene main aboveboth; no known near/far missing vs DEPTHOR; candidate main>shuffle; pass INVERSE_AFFINE_COMPONENT_DEV else STOP_FIXED_INVERSE_AFFINE. Centre-return association assumed, no physicalreturn/bodyclear/newconfirmation claim | `artifacts.local/work/cnh-zju-inverse-affine-20261003/REPORT.md` |\n'
    runs=prior.grid.RUNS;body=runs.read_text(encoding='utf-8');assert RUN_ID not in body
    runs.write_text(body.rstrip()+'\n'+row,encoding='utf-8');(OUT/'prerun-row.txt').write_text(row,encoding='utf-8')
    save(OUT/'PLAN.json',dict(run_id=RUN_ID,rows=rows,dependencies=deps,case=old['case'],preregistration=row,
        rules=dict(arms=ARMS,budgets=[.1,.2],policies=['pooled','worst_scene'],primary='pooled',
            fit='Native fr centre trunc((start+end)/2) then clip0..480/640, float32 sensor distance as official sparse adapter; mask true. Repeated centre last mask-true assignment wins; then finite.001<d<10. q finite including0. TheilSen all distinct-q pair slopes median, intercept median(1/d-a*q). >=2distinctq, finite a>0 and finite b; else allframe missing.',
            prediction='1/(a*q+b), finitepositive denominator only, no clipping; finite >10m retained. No fallback; NaN prediction = missing, not clear; source GT never accessed while forming maps.',
            shuffle='Observed distances permuted across same eligible centres, PCG64 SHA256(run|id), fit exact same estimator, all fit-failure counts retained',
            comparison='All5arms same nodes/sensor-valid domain. IDW4 follows original clipped-fr-centre interface; its centres differ from legacy sparse adapter. Thresholds calibration-only, strict score<threshold, whole ties excluded, no GT scale.',
            gate='Both pooled budgets: main >= chosen(raw,IDW4)+.03 and DEPTHOR+.03, closer >=both, actual far <=min(chosen,DEPTHOR,budget), every8scene far<=budget and >=6/8scene main aboveboth; missingnear+far<=DEPTHOR and candidate main>shuffle. Undefined primary denominator => NOT_EVALUABLE. tol1e-12. All arms and worst diagnostic retained, cannot rescue primary.',
            subgroup='q within anchor minmax interpolation, outside extrapolation. If no anchors both false, no drop from primary. GT strata mixed/thin evaluator only.',
            limits='Consumed Development; centre q and representative ToF distance are not known same-surface returns; no confirmed range-to-Z/extrinsics/bodytruth; pixel far != bodyclear. Residual small is fit self-consistency only. Different shuffle failures not pure causal attribution. No same-cohort estimator retuning')))
    print('PREPARED300',flush=True)


def verify(p):
    for path,digest in p['dependencies'].items():assert sha(path)==digest,path


def anchors(boxes,mask,returns,inverse):
    centres=np.clip(np.trunc((boxes[:,:2]+boxes[:,2:])/2).astype(int),[0,0],[480,640])
    locations={}
    for k in np.flatnonzero(mask):
        y,x=centres[k]
        assert y<480 and x<640,'Official sparse active centre outside array; preserve execution failure'
        locations[(int(y),int(x))]=float(np.float32(returns[k]))
    positions=[];q=[];d=[]
    for pos,value in locations.items():
        iv=float(inverse[pos])
        if np.isfinite(iv) and np.isfinite(value) and .001<value<10:
            positions.append(pos);q.append(iv);d.append(value)
    return np.asarray(q),np.asarray(d),positions,len(locations)


def fit(q,d):
    info=dict(anchors=len(q),distinct_q=len(np.unique(q)),q_min=float(q.min()) if len(q) else None,q_max=float(q.max()) if len(q) else None,
        d_min=float(d.min()) if len(d) else None,d_max=float(d.max()) if len(d) else None,a=None,b=None,defined=False,reason=None,residual_inverse_median=None)
    if info['distinct_q']<2:info['reason']='LESS_THAN_TWO_DISTINCT_Q';return info
    i,j=np.triu_indices(len(q),1);keep=q[i]!=q[j]
    slopes=(1/d[j[keep]]-1/d[i[keep]])/(q[j[keep]]-q[i[keep]])
    a=float(np.median(slopes));b=float(np.median(1/d-a*q))
    if not np.isfinite(a) or not np.isfinite(b):info['reason']='NONFINITE_FIT';return info
    info.update(a=a,b=b,residual_inverse_median=float(np.median(np.abs(1/d-(a*q+b)))))
    if a<=0:info['reason']='NONPOSITIVE_SLOPE';return info
    info.update(defined=True,reason='OK');return info


def metric(q,info):
    value=np.full(q.shape,np.nan,dtype=np.float64)
    if info['defined']:
        inverse=info['a']*q.astype(np.float64)+info['b'];valid=np.isfinite(inverse)&(inverse>0)
        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):value[valid]=1/inverse[valid]
        value[~np.isfinite(value)|(value<=0)]=np.nan
    return value


def predict():
    p=read(OUT/'PLAN.json');verify(p);assert not (OUT/'prediction-seal.json').exists()
    (OUT/'predictions').mkdir(exist_ok=True);ledger=[];start=time.perf_counter()
    for i,r in enumerate(p['rows']):
        assert sha(r['input'])==r['input_sha256'] and sha(r['relative'])==r['relative_sha256']
        target=Path(r['affine']);assert not target.exists()
        with h5py.File(r['input'],'r') as f:boxes,mask,returns=f['fr'][:],f['mask'][:],f['hist_data'][:,0]
        with np.load(r['relative']) as f:inverse=f['inverse']
        assert np.isfinite(inverse).all() and inverse.shape==(480,640)
        q,d,positions,centres=anchors(boxes,mask,returns,inverse)
        generator=np.random.default_rng(int.from_bytes(__import__('hashlib').sha256((RUN_ID+'|'+r['id']).encode()).digest()[:8],'little'))
        infos=dict(inverse_affine=fit(q,d),shuffle_affine=fit(q,d[generator.permutation(len(d))]))
        maps={a:metric(inverse,info) for a,info in infos.items()}
        np.savez_compressed(target,**maps)
        ledger.append(dict(id=r['id'],scene=r['scene'],role=r['role'],infos=infos,legacy_unique_centres=centres,eligible_positions=positions,
            missing_pixels={a:int((~np.isfinite(v)).sum()) for a,v in maps.items()},greater10_pixels={a:int((v>10).sum()) for a,v in maps.items()}))
        if (i+1)%50==0:print(f'OBSERVED_FIT {i+1}/300',flush=True)
    save(OUT/'fit-ledger.json',ledger)
    save(OUT/'prediction-seal.json',dict(plan_sha256=sha(OUT/'PLAN.json'),fit_ledger_sha256=sha(OUT/'fit-ledger.json'),files={r['affine']:sha(r['affine']) for r in p['rows']}))
    save(OUT/'runtime.json',dict(seconds=time.perf_counter()-start,backend='CPU scalar regression and cached-array I/O, TASK_NOT_GPU_SUITABLE; no new network inference'))
    print('SEALED300',flush=True)


def frame(r,info,p,seal):
    assert sha(r['affine'])==seal['files'][r['affine']] and sha(r['relative'])==r['relative_sha256']
    # Prior forms raw/IDW observations before reading reference; candidate maps are already sealed.
    data=prior.frame(r)
    with np.load(r['affine']) as f:maps={a:f[a] for a in ('inverse_affine','shuffle_affine')}
    with np.load(r['relative']) as f:inverse=f['inverse']
    with h5py.File(r['input'],'r') as f:boxes=f['fr'][:]
    nodes=[prior.grid.public_nodes(b)[0] for b in boxes]
    for arm,v in maps.items():data[arm]=np.concatenate([v[n[:,0],n[:,1]] for n in nodes])
    q=np.concatenate([inverse[n[:,0],n[:,1]] for n in nodes]);f=info['infos']['inverse_affine']
    data['interpolation']=(q>=f['q_min'])&(q<=f['q_max']) if f['anchors'] else np.zeros(len(q),bool)
    data['extrapolation']=~data['interpolation'] if f['anchors'] else np.zeros(len(q),bool)
    return data


def merge(fs):return {k:np.concatenate([f[k] for f in fs]) for k in KEYS}


def paired(data,rules,other,group='sensor'):
    use=data['sensor'] if group=='sensor' else data['sensor']&data[group]
    a=base.calls(data['inverse_affine'],rules['inverse_affine']);b=base.calls(data[other],rules[other])
    masks=dict(main=data['main'],closer=data['near']&~data['main'],far=data['far'],unknown=~data['known'])
    return {k:dict(added=int((use&m&a&~b).sum()),removed=int((use&m&~a&b).sum())) for k,m in masks.items()}


def evaluate():
    p=read(OUT/'PLAN.json');verify(p);assert not (OUT/'result.json').exists()
    seal=read(OUT/'prediction-seal.json');assert seal['plan_sha256']==sha(OUT/'PLAN.json') and seal['fit_ledger_sha256']==sha(OUT/'fit-ledger.json')
    fits=read(OUT/'fit-ledger.json');assert len(fits)==300
    with ThreadPoolExecutor(max_workers=4) as pool:frames=list(pool.map(lambda x:frame(*x,p,seal),zip(p['rows'],fits)))
    cal=merge([f for f in frames if f['role']=='cal']);ev=merge([f for f in frames if f['role']=='eval'])
    scenes={s:merge([f for f in frames if f['scene']==s]) for s in sorted({f['scene'] for f in frames})}
    cs=sorted({f['scene'] for f in frames if f['role']=='cal'});es=sorted({f['scene'] for f in frames if f['role']=='eval'})
    outputs={}
    for budget in (.1,.2):
        sr={s:{a:base.select_threshold(scenes[s][a],scenes[s]['sensor']&scenes[s]['far'],budget) for a in ARMS} for s in cs}
        policies=dict(pooled={a:base.select_threshold(cal[a],cal['sensor']&cal['far'],budget) for a in ARMS},
            worst_scene={a:prior.worst([sr[s][a] for s in cs]) for a in ARMS})
        out={}
        for policy,rules in policies.items():
            c={a:base.counts(cal,a,rules[a]) for a in ARMS};e={a:base.counts(ev,a,rules[a]) for a in ARMS}
            sc={s:{a:base.counts(d,a,rules[a]) for a in ARMS} for s,d in scenes.items()}
            chosen=max(ARMS[:2],key=lambda a:(c[a]['main_recall'],-c[a]['far_fpr'],-ARMS.index(a)))
            candidate=e['inverse_affine'];comparators=[e[a] for a in (chosen,'depthor')]
            estimable=all(candidate[k] is not None for k in ('main_recall','closer_recall','far_fpr')) and all(sc[s]['inverse_affine']['far_fpr'] is not None for s in es)
            wins=[s for s in es if all(sc[s]['inverse_affine']['main_recall'] is not None and sc[s][a]['main_recall'] is not None and sc[s]['inverse_affine']['main_recall']>sc[s][a]['main_recall']+1e-12 for a in (chosen,'depthor'))]
            compliant=[s for s in es if sc[s]['inverse_affine']['far_fpr'] is not None and sc[s]['inverse_affine']['far_fpr']<=budget+1e-12]
            gates=dict(main=estimable and all(candidate['main_recall']>=v['main_recall']+.03-1e-12 for v in comparators),
                closer=estimable and all(candidate['closer_recall']>=v['closer_recall']-1e-12 for v in comparators),
                matched_cost=estimable and candidate['far_fpr']<=min(budget,*[v['far_fpr'] for v in comparators])+1e-12,
                coverage=candidate['missing_near']+candidate['missing_far']<=e['depthor']['missing_near']+e['depthor']['missing_far'],
                all_scene_cost=len(compliant)==8,scene_wins=len(wins)>=6,
                shuffle=estimable and candidate['main_recall']>e['shuffle_affine']['main_recall']+1e-12)
            out[policy]=dict(rules=rules,cal=c,eval=e,scenes=sc,chosen_tof=chosen,estimable=estimable,gates=gates,scene_wins=wins,cost_compliant=compliant,
                subgroups={g:{a:base.counts(ev,a,rules[a],g) for a in ARMS} for g in GROUPS},
                paired={g:{a:paired(ev,rules,a,g) for a in ('raw','tof_idw4','depthor','shuffle_affine')} for g in GROUPS})
        outputs[str(budget)]=dict(cal_scene_rules=sr,policies=out)
    principal=[v['policies']['pooled'] for v in outputs.values()]
    verdict=('INVERSE_AFFINE_COMPONENT_DEV' if all(all(v['gates'].values()) for v in principal) else 'STOP_FIXED_INVERSE_AFFINE') if all(v['estimable'] for v in principal) else 'NOT_EVALUABLE'
    save(OUT/'frame-ledger.json',[dict(id=f['id'],scene=f['scene'],role=f['role'],zones=f['zones'],empty=f['empty'],invalid_sensor=f['invalid_sensor'],
        counts={b:{policy:{g:{a:base.counts(f,a,v['rules'][a],g) for a in ARMS} for g in GROUPS} for policy,v in o['policies'].items()} for b,o in outputs.items()}) for f in frames])
    case=next(f for f in frames if f['id']==p['case'])
    save(OUT/'result.json',dict(verdict=verdict,budgets=outputs,cal_scenes=cs,eval_scenes=es,
        case={b:{policy:{a:base.counts(case,a,v['rules'][a]) for a in ARMS} for policy,v in o['policies'].items()} for b,o in outputs.items()},
        fit_failures={role:{a:sum(not f['infos'][a]['defined'] for f in fits if f['role']==role) for a in ('inverse_affine','shuffle_affine')} for role in ('cal','eval')}))
    print(verdict,{b:o['policies']['pooled']['gates'] for b,o in outputs.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','predict','evaluate'))
    globals()[parser.parse_args().action]()
