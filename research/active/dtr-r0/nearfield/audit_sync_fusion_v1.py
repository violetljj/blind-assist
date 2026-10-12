"""Independent small-contract checks; never opens eval references before OPEN.

Synthetic checks exercise scientific tie/denominator rules, not model accuracy.
The pre-eval mode reads observation-only arrays and train/cal seals exclusively.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def synthetic():
    from sync_fusion_v1_train_cal import enumerate_ties, select
    from sync_fusion_features_v1 import features_for_depth
    rng=np.random.default_rng(73411)
    cases=0
    for n in (0,1,17,100):
        for _ in range(15):
            scores=rng.choice(np.array([-np.inf,-1.,0.,.1,.5,1.]),n)
            y=rng.choice(np.array([-1,0,1]),n)
            got=enumerate_ties(scores,y)
            expected=[]
            for threshold in [np.inf]+sorted(set(scores[np.isfinite(scores)]),reverse=True):
                p=scores>=threshold
                expected.append((threshold,int(sum(p&(y==1))),int(sum(p&(y==0))),int(sum(p&(y==-1)))))
            assert len(got)==len(expected)
            for row,(threshold,w,f,u) in zip(got,expected):
                actual=np.inf if row['threshold_kind']=='positive_infinity' else row['threshold']
                assert actual==threshold and (row['W'],row['F'],row['U'])==(w,f,u)
            for target in (.02,.05,.10):
                selected=select(got,y,target)
                if not sum(y==0):
                    assert selected['status']=='NOT_CALIBRATABLE'
                    assert selected['threshold_kind']=='positive_infinity'
                elif not sum(y==1):
                    assert selected['status']=='NO_CAL_POS'
                    assert selected['threshold_kind']=='positive_infinity'
                else:
                    legal=[r for r in expected if r[2]<=target*sum(y==0)+1e-12]
                    want=max(legal,key=lambda r:(r[1],-r[2],r[0]))
                    actual=np.inf if selected['threshold_kind']=='positive_infinity' else selected['threshold']
                    assert (actual,selected['W'],selected['F'],selected['U'])==want
            cases+=1
    z=np.full((4,4),np.inf);entry=np.ones_like(z);exit=entry+1;domain=np.ones_like(z,dtype=bool)
    score,x=features_for_depth(z,entry,exit,domain)
    assert np.isneginf(score) and x[0]==-10 and x[1]==1 and x[3]==1
    z[:]=1.5
    score,x=features_for_depth(z,entry,exit,domain)
    assert score==.5 and x[1]==0 and x[4]==16
    z.flat[0]=np.inf
    score,x=features_for_depth(z,entry,exit,domain)
    assert np.isneginf(score) and x[1]==1 and x[4]==15
    return dict(status='PASS',tie_cases=cases,targets_per_case=3,
                missing_tests=3,eval_reference_reads=0,eval_model_score_reads=0)


def preeval(root,cal_path,refs_path,v1,v11):
    import csv, joblib
    from threadpoolctl import threadpool_limits
    plan=load(root/'PLAN.json');fm=load(root/'feature_manifest.json');seal=load(cal_path)
    assert seal['plan_sha256']==sha(root/'PLAN.json')
    assert not (root/'eval'/'eval_open.json').exists()
    public=load(v1/'public_roster.json')['rows'];public={r['source_id']:r for r in public}
    synth=load(v11/'synthesis_manifest.json')['frames'];sensor={r['source_id']:r['frame_id'] for r in synth}
    with (v11/'input_coverage_gate.csv').open(newline='',encoding='utf8') as f:
        allowed={r['frame_id'] for r in csv.DictReader(f) if r['arm']=='faro_rho030_ambient1' and r['K']=='0' and int(r['joint_pass_zones'])>=52}
    data={};rolevis={};counts=[]
    for e in fm['outputs']:
        assert sha(e['path'])==e['sha256']
        with np.load(e['path'],allow_pickle=False) as z:d={k:z[k] for k in z.files}
        assert not set(d)&{'y','state','label','labels','reference'}
        names=d['feature_names'].tolist();assert names==plan['features']['names']
        assert np.isfinite(d['X']).all() and d['X'].shape==(len(d['frame_id']),20)
        ids=list(zip(d['frame_id'].tolist(),d['query_id'].tolist()));assert len(ids)==len(set(ids))
        expect={s for s,r in public.items() if r['role']==e['role'] and (e['source']=='native_perturbed' or sensor[s] in allowed)}
        assert set(d['frame_id'])==expect and len(ids)==len(expect)*27
        for s in expect:assert sum(d['frame_id']==s)==27
        assert np.array_equal(d['or_score'],np.maximum(d['tof_score'],d['rgb_score']))
        assert np.array_equal(d['and_score'],np.minimum(d['tof_score'],d['rgb_score']))
        x=d['X'];assert np.array_equal(x[:,0],np.clip(d['tof_score'],-10,10))
        assert np.array_equal(x[:,1],~np.isfinite(d['tof_score']))
        assert np.array_equal(x[:,1]==0,x[:,9]==2)
        for row,s in enumerate(d['frame_id']):assert str(public[s]['visit_id'])==d['visit_id'][row]
        rolevis.setdefault(e['role'],set()).update(d['visit_id']);data[e['source'],e['role']]=d
        counts.append({k:e[k] for k in ('source','role','rows','frames')})
    assert [len(rolevis[r]) for r in ('train','cal','eval')]==[6,2,4]
    assert not(rolevis['train']&rolevis['cal'] or rolevis['train']&rolevis['eval'] or rolevis['cal']&rolevis['eval'])
    refs=load(refs_path)['rows'];assert all(r['role'] in ('train','cal') for r in refs)
    yi={};state={'POSITIVE':1,'POS':1,'FREE_ON_SAMPLED_RAYS':0,'FREE':0,'UNKNOWN':-1}
    for r in refs:
        for q in r['queries']:yi[r['source_id'],q['name']]=state[q['state']]
    scores={};models=0
    with threadpool_limits(limits=1):
        for source in plan['inputs']['sources']:
            td=data[source,'train'];cd=data[source,'cal']
            yt=np.array([yi[s,q] for s,q in zip(td['frame_id'],td['query_id'])]);known=yt>=0
            scores[source]={m:cd[m+'_score'] for m in ('tof','rgb','or','and')}
            for method in ('hgb','logit'):
                entries=[e for e in seal['model_entries'] if e['source']==source and e['method']==method]
                assert sorted(e['seed'] for e in entries)==[955,956,957];pred=[]
                for e in entries:
                    assert sha(e['path'])==e['sha256'];m=joblib.load(e['path']);models+=1
                    if method=='hgb':
                        for k,v in plan['models']['HGB'].items():
                            if k!='class_weight':assert m.get_params()[k]==v
                    else:
                        scaler,logit=list(m.named_steps.values())
                        assert np.allclose(scaler.mean_,td['X'][known].mean(0),atol=1e-12)
                        assert scaler.n_samples_seen_==sum(known)
                        assert logit.C==1 and logit.solver=='lbfgs' and logit.max_iter==1000
                    pred.append(m.predict_proba(cd['X'])[:,1])
                scores[source][method]=np.mean(pred,axis=0)
    assert len(seal['cells'])==36
    for c in seal['cells']:
        d=data[c['source'],'cal'];take=d['band']==c['band']
        y=np.array([yi[s,q] for s,q in zip(d['frame_id'],d['query_id'])])[take]
        sc=scores[c['source']][c['method']][take];target=plan['calibration']['band_targets'][c['band']]
        assert c['target']==target
        pos=int(sum(y==1));free=int(sum(y==0));unk=int(sum(y==-1))
        assert (pos,free,unk)==(c['POS_denominator'],c['FREE_denominator'],c['UNKNOWN_denominator'])
        cand=[]
        for t in [np.inf]+sorted(set(sc[np.isfinite(sc)]),reverse=True):
            pr=sc>=t;w=int(sum(pr&(y==1)));f=int(sum(pr&(y==0)));u=int(sum(pr&(y==-1)))
            if f<=np.floor(target*free):cand.append((w,-f,t,u))
        want=max(cand) if free and pos else (0,0,np.inf,0)
        actual=np.inf if c['threshold_kind']=='positive_infinity' else c['threshold']
        assert (c['W'],-c['F'],actual,c['U'])==want
        assert c['status']==('NOT_CALIBRATABLE' if not free else 'NO_CAL_POS' if not pos else 'ADOPTED')
        assert sha(c['all_ties_path'])==c['all_ties_sha256']
    return dict(status='PASS',feature_tables=counts,models_verified=models,cal_cells_recomputed=36,
                visits={r:sorted(v) for r,v in rolevis.items()},cal_seal_sha256=sha(cal_path),
                feature_manifest_sha256=sha(root/'feature_manifest.json'),plan_sha256=sha(root/'PLAN.json'),
                eval_observation_features_read=True,eval_reference_reads=0,eval_model_score_reads=0)


def postevaluation(root):
    """Audit immutable cached decisions only; never reopen labels or run models."""
    opened=load(root/'eval/eval_open.json');terminal=load(root/'eval/eval_terminal.json')
    assert terminal['status']=='COMPLETE'
    path=root/'eval/per_query.jsonl';assert sha(path)==terminal['cache_sha256']
    rows=[json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
    seal=load(root/'calibration/cal_seal.json');summary=load(root/'eval/summary.json')
    assert sha(root/'calibration/cal_seal.json')==opened['cal_seal_sha256']
    for model in seal['model_entries']:assert sha(model['path'])==model['sha256']
    assert sha(root/'feature_manifest.json')==opened['feature_seal_sha256']
    assert len({(r['source'],r['frame_id'],r['query_id']) for r in rows})==len(rows)
    visits=opened['visit_ids'];idx={v:i for i,v in enumerate(visits)}
    draws=np.random.default_rng(20261011).integers(0,4,(2000,4))
    multiplicities=np.stack([(draws==j).sum(1) for j in range(4)],axis=1)
    state_names=('POSITIVE','FREE_ON_SAMPLED_RAYS','UNKNOWN')
    bycell={c['cell_id']:c for c in seal['cells']};checked=0
    for r in rows:
        for cid,pred in r['predictions'].items():
            c=bycell[cid];assert c['source']==r['source'] and c['band']==r['band'] and c['status']=='ADOPTED'
            s=r['scores'][c['method']];s=-np.inf if s is None else s
            t=np.inf if c['threshold_kind']=='positive_infinity' else c['threshold']
            assert pred==bool(s>=t);checked+=1
    def interval(numer,denom):
        n=multiplicities@numer;d=multiplicities@denom;valid=d>0;x=n[valid]/d[valid]
        return dict(lower=float(np.percentile(x,2.5)) if len(x) else None,
                    upper=float(np.percentile(x,97.5)) if len(x) else None,
                    valid_replicates=len(x),total_replicates=2000)
    def cluster(r,values):
        out=np.zeros(4,dtype=int)
        for item,value in zip(r,values):out[idx[item['visit_id']]]+=int(value)
        return out
    def same(a,b):
        assert set(a)==set(b)
        for k in a:
            assert (a[k] is None and b[k] is None) or (a[k] is not None and b[k] is not None and np.isclose(a[k],b[k],atol=1e-12,rtol=1e-12)),(k,a[k],b[k])
    metrics={};preds={};grouped={}
    for m in summary['metrics']:
        source,band,method=m['source'],m['band'],m['method'];rr=[r for r in rows if r['source']==source and r['band']==band]
        grouped[source,band]=rr;y=np.array([r['state'] for r in rr]);den=[int(sum(y==s)) for s in state_names]
        assert den==[m['POS'],m['FREE'],m['UNKNOWN']]
        c=next(c for c in seal['cells'] if (c['source'],c['band'],c['method'])==(source,band,method))
        if c['status']!='ADOPTED':
            assert m['status']==c['status'] and all(m[k] is None for k in ('W','F','U','W_CI','F_CI'));continue
        p=np.array([r['predictions'][c['cell_id']] for r in rr]);n=[int(sum(p&(y==s))) for s in state_names]
        assert n==[m['W'],m['F'],m['U']]
        for i,key in ((0,'W'),(1,'F')):
            expect=n[i]/den[i] if den[i] else None
            assert m[key+'_rate']==expect
            same(interval(cluster(rr,p&(y==state_names[i])),cluster(rr,y==state_names[i])),m[key+'_CI'])
        metrics[source,band,method]=m;preds[source,band,method]=p
    best={}
    for b in summary['best_single']:
        s,band=b['source'],b['band'];chosen=max(('tof','rgb'),key=lambda k:(metrics[s,band,k]['W'],-metrics[s,band,k]['F'],k=='rgb'))
        assert b['method']==chosen;best[s,band]=chosen
    for pair in summary['paired']:
        s,b,m,c=pair['source'],pair['band'],pair['method'],pair['comparator']
        assert c==(best[s,b] if pair['comparator_role']=='best_single' else 'or')
        rr=grouped[s,b];y=np.array([r['state'] for r in rr]);p=preds[s,b,m];q=preds[s,b,c]
        added=p&~q;removed=~p&q;rescue=int(sum(added&(y==state_names[0])));loss=int(sum(removed&(y==state_names[0])))
        assert (pair['rescue'],pair['loss'],pair['delta_W'])==(rescue,loss,rescue-loss)
        for label,prefix in zip(state_names[1:],('FREE','UNKNOWN')):
            assert pair[prefix+'_added']==sum(added&(y==label)) and pair[prefix+'_removed']==sum(removed&(y==label))
        delta=cluster(rr,added&(y==state_names[0]))-cluster(rr,removed&(y==state_names[0]))
        same(interval(delta,cluster(rr,y==state_names[0])),pair['delta_W_rate_CI'])
    for conclusion in summary['conclusions']:
        method=conclusion['method'];success={s:[] for s in ('native_perturbed','faro_rho030_ambient1')}
        for s,b,m in metrics:
            if m!=method:continue
            v=metrics[s,b,m]
            if v['POS'] and v['FREE'] and v['W']>max(metrics[s,b,'tof']['W'],metrics[s,b,'rgb']['W']) and v['F']/v['FREE']<=v['target']+1e-12:success[s].append(b)
        success={s:sorted(b) for s,b in success.items()};shared=sorted(set(success['native_perturbed'])&set(success['faro_rho030_ambient1']))
        assert conclusion['source_success_bands']==success and conclusion['shared_success_bands']==shared
        result='BOTH_SOURCES_AT_LEAST_TWO_SAME_BANDS' if len(shared)>=2 else 'ONLY_SEMICIRCULAR_SOURCE_SUPPORT' if len(success['native_perturbed'])>=2 else 'NO_TWO_SOURCE_BENEFIT_ESTABLISHED'
        assert conclusion['result']==result
    return dict(status='PASS',cached_rows=len(rows),query_decisions_checked=checked,
                metrics_checked=len(summary['metrics']),paired_rows_checked=len(summary['paired']),
                bootstrap_replicates=2000,unit='original four visit clusters',
                cache_sha256=sha(path),summary_sha256=sha(root/'eval/summary.json'),
                eval_reference_reopens=0,model_reruns=0,conclusions=summary['conclusions'])


def report_review(root,report):
    text=report.read_text(encoding='utf8');summary=load(root/'eval/summary.json');seal=load(root/'calibration/cal_seal.json')
    label=lambda s:'半循环native' if s=='native_perturbed' else '过门FARO'
    ci=lambda c:'NA' if c is None or c['lower'] is None else f"[{100*c['lower']:.1f}, {100*c['upper']:.1f}]"
    count=0
    for m in summary['metrics']:
        ratios=[f"{m[k] if m[k] is not None else 'NA'}/{m[d]}" for k,d in [('W','POS'),('F','FREE'),('U','UNKNOWN')]]
        row='| '+' | '.join([label(m['source']),m['method'],m['band'],*ratios,ci(m['W_CI']),ci(m['F_CI'])])+' |'
        assert row in text,row;count+=1
    paired=0
    for m in summary['paired']:
        if m['method'] not in ('hgb','logit'):continue
        row='| '+' | '.join([label(m['source']),m['method'],m['band'],f"{m['comparator']} ({m['comparator_role']})",f"{m['rescue']}/{m['loss']}",f"{m['delta_W']:+d}",f"{m['FREE_added']}/{m['FREE_removed']}",ci(m['delta_W_rate_CI'])])+' |'
        assert row in text,row;paired+=1
    for c in seal['cells']:
        t='+∞' if c['threshold_kind']=='positive_infinity' else str(c['threshold'])
        row='| '+' | '.join([label(c['source']),c['method'],c['band'],t,str(c['W']),str(c['F']),c['status']])+' |'
        assert row in text,row
    return dict(status='PASS',metric_rows=count,paired_rows=paired,calibration_rows=len(seal['cells']),
                report_sha256=sha(report),report_path=str(report),scope='text tables and stated evidence boundaries')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--receipt',default='independent_contract_audit.json');p.add_argument('--cal-seal',type=Path)
    p.add_argument('--references',type=Path);p.add_argument('--v1',type=Path);p.add_argument('--v11',type=Path)
    p.add_argument('--posteval',action='store_true');p.add_argument('--report',type=Path);a=p.parse_args()
    start=time.perf_counter()
    result=report_review(a.root,a.report) if a.report else postevaluation(a.root) if a.posteval else preeval(a.root,a.cal_seal,a.references,a.v1,a.v11) if a.cal_seal else synthetic()
    result['seconds']=time.perf_counter()-start
    a.root.mkdir(parents=True,exist_ok=True)
    result['audit_source_sha256']=sha(__file__)
    with (a.root/a.receipt).open('x',encoding='utf8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result))


if __name__=='__main__':main()
