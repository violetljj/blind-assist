"""Fixed posthoc v2 evidence/prior audit. Original evaluator and outputs untouched."""
import csv, hashlib, json, time, warnings
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
WORK=REPO/'artifacts.local/work';V2=WORK/'sync-fusion-confirm-v2-dev-20261011'
ROOT=WORK/'sync-fusion-evidence-audit-dev-20261011'
PLAN=HERE/'SYNC_FUSION_EVIDENCE_AUDIT_PLAN_DEV_20261011.json'
BANDS=('0.3-0.8m','0.8-1.5m','1.5-3m');SOURCES=('native_perturbed','faro_rho030_ambient1')
GROUPS=('both','tof_only','rgb_only','neither');SEEDS=(955,956,957)

def load(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def csvwrite(p,rows):
    with Path(p).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def labels(path):
    d={}
    for r in load(path)['rows']:
        for q in r['queries']:
            key=(r['source_id'],q['name']);assert key not in d
            d[key]={'POSITIVE':1,'FREE_ON_SAMPLED_RAYS':0,'UNKNOWN':-1}[q['state']]
    return d
def ids(d):return list(zip(map(str,d['frame_id']),map(str,d['query_id'])))
def evidence(d):
    t=np.isfinite(d['tof_score'])&(d['tof_score']>=0);r=np.isfinite(d['rgb_score'])&(d['rgb_score']>=0)
    return t,r,np.where(t,np.where(r,'both','tof_only'),np.where(r,'rgb_only','neither'))
def ties(scores,y):
    assert not np.isnan(scores).any() and not np.isposinf(scores).any()
    out=[dict(threshold=None,threshold_kind='positive_infinity',W=0,F=0,U=0)]
    ii=np.flatnonzero(np.isfinite(scores));ii=ii[np.argsort(-scores[ii],kind='stable')]
    ends=np.r_[np.where(scores[ii][1:]!=scores[ii][:-1])[0]+1,len(ii)] if len(ii) else []
    cum={k:np.cumsum(y[ii]==v) for k,v in [('W',1),('F',0),('U',-1)]}
    for end in ends:out.append(dict(threshold=float(scores[ii[end-1]]),threshold_kind='finite',**{k:int(v[end-1]) for k,v in cum.items()}))
    return out
def select(table,y,target):
    nf=int((y==0).sum());np_=int((y==1).sum());nu=int((y==-1).sum())
    if not nf or not np_:chosen=table[0];status='NOT_CALIBRATABLE' if not nf else 'NO_CAL_POS'
    else:
        chosen=max((r for r in table if r['F']<=int(np.floor(target*nf))),key=lambda r:(r['W'],-r['F'],float('inf') if r['threshold'] is None else r['threshold']));status='ADOPTED'
    return dict(chosen,target=target,status=status,POS=np_,FREE=nf,UNKNOWN=nu)
def ci(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    return dict(lower=float(np.quantile(x,.025)) if len(x) else None,upper=float(np.quantile(x,.975)) if len(x) else None,valid_replicates=len(x),replicates=2000)

def run():
    start=time.perf_counter();assert not any((ROOT/n).exists() for n in ('summary.json','cal_seal.json','models')), 'Preserve existing outputs; no implicit rerun'
    ROOT.mkdir(parents=True,exist_ok=True);plan=load(PLAN);assert plan['phase']=='POSTHOC_CONSUMED_DEVELOPMENT_AUDIT'
    source_before={};dependencies={};manifest=load(V2/'feature_manifest.json');trainseal=load(V2/'train/train_seal.json');oldcal=load(V2/'cal/cal_seal.json')
    def bind(p,expected=None):
        p=str(Path(p).resolve());h=sha(p)
        if expected:assert h==expected,p
        source_before[p]=h;dependencies[p]=h
    for p in (PLAN,Path(__file__),V2/'feature_manifest.json',V2/'training_reference_roster.json',V2/'cal_reference_roster.json',V2/'train/train_seal.json',V2/'cal/cal_seal.json',V2/'eval/per_query.jsonl',V2/'eval/eval_open.json'):bind(p)
    for e in oldcal['model_entries']:bind(e['path'],e['sha256'])
    oldrows=[json.loads(l) for l in (V2/'eval/per_query.jsonl').open(encoding='utf8')];oldby={(r['source'],r['frame_id'],r['query_id']):r for r in oldrows};assert len(oldby)==len(oldrows)==12150
    ytrainby=labels(V2/'training_reference_roster.json');ycalby=labels(V2/'cal_reference_roster.json')
    entries={(e['source'],e['role']):e for e in manifest['outputs']};data={};ensembles={};models=[];cal_scores={};cells=[]
    def check():
        if time.perf_counter()-start>450:raise TimeoutError('Main command allocation')
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    with threadpool_limits(limits=1):
        for source in SOURCES:
            for role in ('train','cal','eval'):
                e=entries[(source,role)];bind(e['path'],e['sha256'])
                with np.load(e['path'],allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
                assert len(set(ids(d)))==len(d['X']) and np.isfinite(d['X']).all();data[source,role]=d
            tr=data[source,'train'];ca=data[source,'cal'];names=list(map(str,tr['feature_names']));assert names==trainseal['feature_contracts'][source] and len(names)==20
            parts=[]
            for e in trainseal['feature_entries']:
                if e['source']==source:
                    bind(e['path'],e['sha256'])
                    with np.load(e['path']) as z:parts.append({k:z[k].copy() for k in z.files})
            for k in ('X','frame_id','query_id','visit_id','tof_score','rgb_score'):np.testing.assert_equal(tr[k],np.concatenate([p[k] for p in parts],axis=0))
            assert set(map(str,tr['visit_id']))<=set(trainseal['visit_ids'])
            yt=np.array([ytrainby[k] for k in ids(tr)],np.int8);yc=np.array([ycalby[k] for k in ids(ca)],np.int8);known=yt>=0
            assert not(set(map(str,tr['visit_id']))&set(map(str,ca['visit_id'])))
            prior_indices=[names.index(n) for n in plan['models']['prior_features']];a_indices=[i for i,n in enumerate(names) if n not in plan['models']['prior_features']];assert len(a_indices)==16
            for method,columns in [('prior',prior_indices),('a',a_indices)]:
                ensemble=[]
                for seed in SEEDS:
                    check();m=make_pipeline(StandardScaler(),LogisticRegression(C=1,penalty='l2',solver='lbfgs',max_iter=1000,class_weight=None,random_state=seed))
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always');m.fit(tr['X'][known][:,columns],yt[known])
                    assert all('converge' not in str(w.message).lower() for w in caught)
                    path=ROOT/'models'/source/f'{method}_{seed}.joblib';path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(m,path)
                    models.append(dict(source=source,method=method,seed=seed,path=str(path.resolve()),sha256=sha(path),features=[names[i] for i in columns],known_train_rows=int(known.sum()),warnings=[str(w.message) for w in caught]));ensemble.append(m)
                ensembles[source,method]=(ensemble,columns)
            scores={method:np.mean([m.predict_proba(ca['X'][:,cols])[:,1] for m in ens],axis=0,dtype=np.float64) for method,(ens,cols) in [(m,ensembles[source,m]) for m in ('prior','a')]}
            t,r,_=evidence(ca);scores['b']=np.where(t|r,scores['a'],-np.inf);cal_scores[source]=scores
            np.savez_compressed(ROOT/f'{source}_cal_scores.npz',**scores,label=yc,frame_id=ca['frame_id'],query_id=ca['query_id'],band=ca['band'])
            for band in BANDS:
                take=ca['band']==band
                for method in ('prior','a','b'):
                    table=ties(scores[method][take],yc[take]);path=ROOT/'cal_ties'/f'{source}_{method}_{band}.json';save(path,table)
                    cells.append(dict(source=source,method=method,band=band,**select(table,yc[take],plan['models']['targets'][band]),ties_path=str(path.resolve()),ties_sha256=sha(path)))
        save(ROOT/'cal_seal.json',dict(phase='POSTHOC_CAL_SEALED',plan_sha256=sha(PLAN),model_entries=models,cells=cells,original_eval_already_consumed=True,cal_uses_eval_labels=False,train_visits=trainseal['visit_ids'],cal_visits=oldcal['cal_visit_ids']))
        records=[]
        for source in SOURCES:
            ev=data[source,'eval'];t,r,groups=evidence(ev);scores={method:np.mean([m.predict_proba(ev['X'][:,cols])[:,1] for m in ens],axis=0,dtype=np.float64) for method,(ens,cols) in [(m,ensembles[source,m]) for m in ('prior','a')]};scores['b']=np.where(t|r,scores['a'],-np.inf)
            np.savez_compressed(ROOT/f'{source}_eval_scores.npz',**scores,frame_id=ev['frame_id'],query_id=ev['query_id'],band=ev['band'])
            for i,key in enumerate(ids(ev)):
                check();old=oldby[(source,*key)];band=str(ev['band'][i]);orig={m:next(v for k,v in old['predictions'].items() if '/'+m+'/' in k) for m in ('logit','tof','rgb')}
                for m in ('tof','rgb'):
                    cached_score=old['scores'][m];feature_score=ev[m+'_score'][i];assert (cached_score is None and np.isneginf(feature_score)) or cached_score==feature_score
                wp='both' if orig['tof'] and orig['rgb'] else 'tof_only' if orig['tof'] else 'rgb_only' if orig['rgb'] else 'neither'
                row=dict(source=source,frame_id=key[0],query_id=key[1],visit_id=str(ev['visit_id'][i]),band=band,label=old['label'],tof_minimum=bool(t[i]),rgb_minimum=bool(r[i]),evidence_group=str(groups[i]),workpoint_group=wp,original_logit=orig['logit'],original_tof=orig['tof'],original_rgb=orig['rgb'],raw_tof_score=float(ev['tof_score'][i]) if np.isfinite(ev['tof_score'][i]) else None,raw_rgb_score=float(ev['rgb_score'][i]) if np.isfinite(ev['rgb_score'][i]) else None,tof_support_pixels_Kmean=float(ev['X'][i,2]),tof_valid_K_count=int(ev['X'][i,9]),inactive_rgb_minimum=bool(ev['X'][i,10 if band==BANDS[0] else 13]>=0 and ev['X'][i,11 if band==BANDS[0] else 14]==0))
                for method in ('prior','a','b'):
                    cell=next(c for c in cells if c['source']==source and c['band']==band and c['method']==method);cut=cell['threshold'] if cell['threshold_kind']=='finite' else np.inf
                    row[method+'_score']=float(scores[method][i]) if np.isfinite(scores[method][i]) else None;row[method+'_prediction']=bool(scores[method][i]>=cut)
                records.append(row)
    visits=sorted({r['visit_id'] for r in records});assert len(visits)==12
    vi={v:i for i,v in enumerate(visits)};draw=np.random.default_rng(20261011).integers(0,12,size=(2000,12))
    metrics=[];paired=[];decompositions=[]
    for source in SOURCES:
        for band in BANDS:
            rr=[r for r in records if r['source']==source and r['band']==band];y=np.array([r['label'] for r in rr]);v=np.array([vi[r['visit_id']] for r in rr]);preds={m:np.array([r['original_'+m] for r in rr],bool) for m in ('logit','tof','rgb')};preds.update({m:np.array([r[m+'_prediction'] for r in rr],bool) for m in ('prior','a','b')})
            stats={m:[int((p&(y==j)).sum()) for j in (1,0,-1)] for m,p in preds.items()};best=max(('tof','rgb'),key=lambda m:(stats[m][0],-stats[m][1],m=='rgb'));bp=preds[best];npos=int((y==1).sum());nf=int((y==0).sum());nu=int((y==-1).sum())
            bd=np.array([int(((v==j)&(y==1)).sum()) for j in range(12)])[draw].sum(1)
            for method,p in preds.items():
                W,F,U=stats[method];metrics.append(dict(source=source,band=band,method=method,W=W,POS=npos,F=F,FREE=nf,U=U,UNKNOWN=nu,W_rate=W/npos if npos else None,F_rate=F/nf if nf else None))
                if method in ('logit','prior','a','b'):
                    rescue=p&~bp&(y==1);loss=~p&bp&(y==1);net=rescue.astype(int)-loss.astype(int);bnet=np.array([int(net[v==j].sum()) for j in range(12)])[draw].sum(1);rates=np.divide(bnet,bd,out=np.full(2000,np.nan),where=bd>0)
                    paired.append(dict(source=source,band=band,method=method,best_single=best,best_single_W=stats[best][0],rescue=int(rescue.sum()),loss=int(loss.sum()),net=int(net.sum()),net_rate=int(net.sum())/npos if npos else None,net_CI=ci(bnet),net_rate_CI=ci(rates),FREE_added=int((p&~bp&(y==0)).sum()),FREE_removed=int((~p&bp&(y==0)).sum())))
            for category_field in ('evidence_group','workpoint_group'):
                pp=preds['logit'];rescue=pp&~bp&(y==1);loss=~pp&bp&(y==1);groups=np.array([r[category_field] for r in rr]);parts=[]
                for g in GROUPS:
                    take=groups==g;parts.append(dict(group=g,rows=int(take.sum()),POS=int((take&(y==1)).sum()),FREE=int((take&(y==0)).sum()),UNKNOWN=int((take&(y==-1)).sum()),W=int((take&pp&(y==1)).sum()),F=int((take&pp&(y==0)).sum()),U=int((take&pp&(y==-1)).sum()),rescue=int((take&rescue).sum()),loss=int((take&loss).sum()),net=int((take&rescue).sum()-(take&loss).sum())))
                assert sum(p['W'] for p in parts)==stats['logit'][0];decompositions.append(dict(source=source,band=band,category=category_field,best_single=best,parts=parts,evidence_net=sum(p['net'] for p in parts if p['group']!='neither'),neither_net=parts[-1]['net']))
    primary=next(p for p in paired if p['source']==SOURCES[0] and p['band']==BANDS[1] and p['method']=='logit');assert (primary['rescue'],primary['loss'],primary['net'])==(407,22,385)
    bm=next(p for p in paired if p['source']==SOURCES[0] and p['band']==BANDS[1] and p['method']=='b');lower=bm['net_rate_CI']['lower'];conclusion='去先验后融合收益仍成立（事后）' if lower is not None and lower>0 else '去显式先验并门控后未证实融合收益（事后）'
    csvwrite(ROOT/'per_query_evidence.csv',records)
    with (ROOT/'per_query_evidence.jsonl').open('x',encoding='utf8') as f:
        for r in records:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    csvwrite(ROOT/'metrics.csv',metrics);save(ROOT/'paired.json',paired);save(ROOT/'decomposition.json',decompositions)
    for p,h in source_before.items():assert sha(p)==h, 'Original input mutated: '+p
    save(ROOT/'input_seal.json',dict(files=[dict(path=p,sha256=h) for p,h in dependencies.items()],original_files_unchanged=True))
    save(ROOT/'summary.json',dict(status='COMPLETE',phase='POSTHOC_ONLY',plan_sha256=sha(PLAN),source_sha256=sha(__file__),metrics=metrics,paired=paired,decompositions=decompositions,conclusion=conclusion,eval_visits=visits,rows=len(records),CPU_command_wall_s=time.perf_counter()-start,GPU_s=0))
    print(json.dumps(dict(status='COMPLETE',rows=len(records),middle_b=bm,conclusion=conclusion,seconds=time.perf_counter()-start),ensure_ascii=False))

if __name__=='__main__':run()
