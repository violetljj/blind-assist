"""One-shot sealed evaluation, query decisions and four-visit cluster intervals.

No eval input is opened without an explicit --open-eval and a verified cal seal.
The immutable row cache is the only supported input to later audit/aggregation.
"""
from pathlib import Path
import argparse, csv, hashlib, json, time, traceback
from datetime import datetime, timezone
import numpy as np

METHODS=('hgb','logit','tof','rgb','or','and')
STATES=('POSITIVE','FREE_ON_SAMPLED_RAYS','UNKNOWN')

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def immutable(p,value):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def write_csv(p,rows):
    if not rows:return
    with Path(p).open('x',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def target_key(x):return format(float(x),'.12g')
def cell_key(c):return (str(c['source']),str(c['band']),target_key(c['target']),str(c['method']))
def threshold(c):
    if c.get('status')!='ADOPTED':return None
    if c.get('threshold_kind')=='positive_infinity':return float('inf')
    if c.get('threshold') is None:raise ValueError('Unspecified adopted threshold')
    return float(c['threshold'])
def ci(x):
    a=np.asarray(x,float);a=a[np.isfinite(a)]
    return dict(lower=float(np.quantile(a,.025)) if len(a) else None,upper=float(np.quantile(a,.975)) if len(a) else None,valid_replicates=len(a),total_replicates=2000)
def divide(a,b):return np.divide(a,b,out=np.full(np.shape(a),np.nan,float),where=np.asarray(b)>0)

def aggregate(rows,cells,visits,seed):
    """All resamples retain whole visits, including visits with no FARO rows."""
    if len(visits)!=4 or len(set(visits))!=4:raise ValueError('Exactly four fixed eval visits required')
    rng=np.random.default_rng(seed);draw=rng.integers(0,4,size=(2000,4));vi={v:i for i,v in enumerate(visits)}
    summaries=[];paired=[];best=[];success={};bycell={cell_key(c):c for c in cells}
    groups=sorted({k[:3] for k in bycell})
    for source,band,targ in groups:
        rr=[r for r in rows if r['source']==source and r['band']==band]
        y=np.array([r['state'] for r in rr]);v=np.array([vi[r['visit_id']] for r in rr],int)
        den=np.array([(y==s).sum() for s in STATES]);pred={};stats={};bootden=np.array([[sum((v==j)&(y==s)) for s in STATES] for j in range(4)],float)[draw].sum(1)
        for method in METHODS:
            c=bycell.get((source,band,targ,method));cid=c.get('cell_id') if c else None
            if not c or c.get('status')!='ADOPTED':
                summaries.append(dict(source=source,band=band,target=float(targ),method=method,status=c.get('status','MISSING_CAL_CELL') if c else 'MISSING_CAL_CELL',W=None,POS=int(den[0]),F=None,FREE=int(den[1]),U=None,UNKNOWN=int(den[2]),W_rate=None,F_rate=None,W_CI=None,F_CI=None));continue
            p=np.array([r['predictions'][cid] for r in rr],bool);pred[method]=p;n=np.array([sum(p&(y==s)) for s in STATES]);stats[method]=n
            bc=np.array([[sum((v==j)&p&(y==s)) for s in STATES] for j in range(4)],float)[draw].sum(1)
            summaries.append(dict(source=source,band=band,target=float(targ),method=method,status='EVALUATED' if den[0]>0 and den[1]>0 else 'NOT_EVALUABLE_ZERO_DENOMINATOR',W=int(n[0]),POS=int(den[0]),F=int(n[1]),FREE=int(den[1]),U=int(n[2]),UNKNOWN=int(den[2]),W_rate=float(n[0]/den[0]) if den[0] else None,F_rate=float(n[1]/den[1]) if den[1] else None,W_CI=ci(divide(bc[:,0],bootden[:,0])),F_CI=ci(divide(bc[:,1],bootden[:,1]))))
        if not {'tof','rgb'}.issubset(stats):continue
        # Eval comparator identity is fixed before bootstrap, never reselected in replicas.
        chosen=max(('tof','rgb'),key=lambda m:(stats[m][0],-stats[m][1],m=='rgb'))
        best.append(dict(source=source,band=band,target=float(targ),method=chosen,rule='W largest; tie F smallest; tie RGB; identity fixed in bootstrap'))
        for method in METHODS:
            if method not in pred:continue
            for comparator in (chosen,'or'):
                if comparator not in pred:continue
                rescue=(y=='POSITIVE')&pred[method]&~pred[comparator];loss=(y=='POSITIVE')&~pred[method]&pred[comparator]
                d=np.array([sum(rescue&(v==j))-sum(loss&(v==j)) for j in range(4)],float)[draw].sum(1)
                additions={}
                for label,prefix in [('FREE_ON_SAMPLED_RAYS','FREE'),('UNKNOWN','UNKNOWN')]:
                    additions[prefix+'_added']=int(((y==label)&pred[method]&~pred[comparator]).sum())
                    additions[prefix+'_removed']=int(((y==label)&~pred[method]&pred[comparator]).sum())
                paired.append(dict(source=source,band=band,target=float(targ),method=method,comparator=comparator,comparator_role='best_single' if comparator==chosen else 'OR',rescue=int(rescue.sum()),loss=int(loss.sum()),delta_W=int(rescue.sum()-loss.sum()),POS=int(den[0]),delta_W_rate=float((rescue.sum()-loss.sum())/den[0]) if den[0] else None,delta_W_rate_CI=ci(divide(d,bootden[:,0])),**additions))
            if method in ('hgb','logit'):
                success[(source,band,targ,method)]=None if den[0]==0 or den[1]==0 else bool(stats[method][0]>max(stats['tof'][0],stats['rgb'][0]) and stats[method][1]/den[1]<=float(targ)+1e-12)
    conclusions=[];sources=sorted({k[0] for k in success})
    band_targets={k[1]:float(k[2]) for k in success}
    for method in ('hgb','logit'):
        source_bands={s:sorted(k[1] for k,val in success.items() if k[0]==s and k[3]==method and val) for s in sources}
        shared=sorted(set.intersection(*(set(b) for b in source_bands.values()))) if len(source_bands)==2 else []
        native=[b for s,bb in source_bands.items() if 'native' in s for b in bb]
        result='BOTH_SOURCES_AT_LEAST_TWO_SAME_BANDS' if len(shared)>=2 else 'ONLY_SEMICIRCULAR_SOURCE_SUPPORT' if len(native)>=2 else 'NO_TWO_SOURCE_BENEFIT_ESTABLISHED'
        conclusions.append(dict(method=method,band_targets=band_targets,source_success_bands=source_bands,shared_success_bands=shared,result=result))
    return dict(metrics=summaries,paired=paired,best_single=best,conclusions=conclusions,bootstrap=dict(unit='visit_cluster',visits=visits,replicates=2000,seed=seed,zero_denominator='NaN excluded; valid counts reported',comparator_identity='fixed full-eval identity, no replica selection'),witness_contract='query-level predicted positive AND query POSITIVE; not historical pixel-overlap witness')

def reference_index(path):
    rows=load(path)['rows'];out={}
    for r in rows:
        if r.get('role',r.get('split'))!='eval':continue
        fid=r['source_id'] # Feature contract: frame_id is the native source_id.
        for j,q in enumerate(r['queries']):
            for key in (str(j),str(q['name'])):out[(fid,key)]=q['state']
    return out

def run(root,cal_path,reference_path,feature_seal_path,visits,seed,allow_open):
    start=time.perf_counter();root=root.resolve();out=root/'eval';out.mkdir(exist_ok=True)
    if not allow_open:raise ValueError('Explicit OPEN authorization required')
    if (out/'eval_open.json').exists():raise FileExistsError('Eval was already opened; no reopening or rerun')
    seal=load(cal_path);cells=seal['cells'];models=seal['model_entries'];plan=root/'PLAN.json'
    if seal.get('plan_sha256')!=sha(plan):raise ValueError('Plan seal mismatch')
    if len({cell_key(c) for c in cells})!=len(cells):raise ValueError('Duplicate calibration cells')
    for c in cells:
        if 'cell_id' not in c:raise ValueError('Each sealed calibration cell needs cell_id')
        threshold(c)
    if not isinstance(seal.get('feature_contracts'),dict):raise ValueError('Missing sealed feature contracts')
    feature_seal=load(feature_seal_path)
    if feature_seal.get('status')!='COMPLETE':raise ValueError('Feature extraction not sealed complete')
    eval_features=[e for e in feature_seal['outputs'] if e['role']=='eval']
    for source in sorted({c['source'] for c in cells}):
        ee=[e for e in eval_features if e['source']==source]
        if len(ee)!=1 or sha(ee[0]['path'])!=ee[0]['sha256']:raise ValueError('Eval feature seal mismatch')
        if source not in seal['feature_contracts']:raise ValueError('Source missing sealed feature order')
    for m in models:
        if sha(m['path'])!=m['sha256']:raise ValueError('Model seal mismatch')
    immutable(out/'eval_open.json',dict(opened_utc=datetime.now(timezone.utc).isoformat(),cal_seal_sha256=sha(cal_path),feature_seal_sha256=sha(feature_seal_path),plan_sha256=sha(plan),reference_path=str(reference_path),visit_ids=visits,one_shot=True))
    cached=[];status='FAILED_NO_REOPEN';error=None
    try:
        import joblib
        refs=reference_index(reference_path)
        with (out/'per_query.jsonl').open('x',encoding='utf-8') as cache:
            for source in sorted({c['source'] for c in cells}):
                if time.perf_counter()-start>300:raise TimeoutError('Eval compute allocation')
                fp=Path(next(e['path'] for e in eval_features if e['source']==source))
                with np.load(fp,allow_pickle=False) as z:
                    X=z['X'];scores={m:np.array(z[m+'_score'],float) for m in ('tof','rgb','or','and')}
                    names=list(map(str,z['feature_names']));wanted=seal['feature_contracts'][source]
                    if names!=wanted:raise ValueError('Feature order mismatch')
                    for method in ('hgb','logit'):
                        mm=[m for m in models if m['source']==source and m['method']==method]
                        if len(mm)!=3:raise ValueError('Exactly three sealed model seeds required')
                        scores[method]=np.mean([joblib.load(m['path']).predict_proba(X)[:,1] for m in mm],axis=0)
                    if not all(np.all(np.isfinite(scores[m])|np.isneginf(scores[m])) for m in METHODS):raise ValueError('Unexpected NaN or +inf score')
                    sc=[c for c in cells if c['source']==source and c['status']=='ADOPTED']
                    for i in range(len(X)):
                        fid=str(z['frame_id'][i]);qid=str(z['query_id'][i]);band=str(z['band'][i]);visit=str(z['visit_id'][i])
                        if visit not in visits:raise ValueError('Eval visit outside frozen set')
                        predictions={c['cell_id']:bool(scores[c['method']][i]>=threshold(c)) for c in sc if str(c['band'])==band}
                        row=dict(source=source,frame_id=fid,query_id=qid,band=band,visit_id=visit,state=refs[(fid,qid)],scores={m:float(scores[m][i]) if np.isfinite(scores[m][i]) else None for m in METHODS},predictions=predictions)
                        cache.write(json.dumps(row,allow_nan=False)+'\n');cached.append(row)
                    cache.flush()
                immutable(out/f'{source}_input_receipt.json',dict(features_sha256=sha(fp),rows=sum(r['source']==source for r in cached)))
        summary=aggregate(cached,cells,visits,seed);immutable(out/'summary.json',summary)
        write_csv(out/'metrics.csv',[{**r,'W_CI':json.dumps(r['W_CI']),'F_CI':json.dumps(r['F_CI'])} for r in summary['metrics']])
        write_csv(out/'paired.csv',[{**r,'delta_W_rate_CI':json.dumps(r['delta_W_rate_CI'])} for r in summary['paired']]);status='COMPLETE'
    except Exception as e:error=dict(error=repr(e),traceback=traceback.format_exc())
    finally:
        immutable(out/'eval_terminal.json',dict(status=status,error=error,rows_cached=len(cached),command_wall_s=time.perf_counter()-start,GPU_s=0,cache_sha256=sha(out/'per_query.jsonl') if (out/'per_query.jsonl').exists() else None,reopen_permitted=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cal-seal',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--feature-seal',type=Path,required=True);p.add_argument('--visits',nargs=4,required=True);p.add_argument('--bootstrap-seed',type=int,required=True);p.add_argument('--open-eval',action='store_true');a=p.parse_args();run(a.root,a.cal_seal,a.reference,a.feature_seal,a.visits,a.bootstrap_seed,a.open_eval)
