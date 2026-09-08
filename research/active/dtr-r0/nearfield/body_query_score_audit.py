"""Consumed-score oracle ROC and frozen-DEV-threshold paired error audit.

No fitting, inference, threshold deployment, or fresh confirmation. Small arrays
run on CPU. FP budgets mean <= budget; exact-achievable FP points are separate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SCOPE='CONSUMED_DEVELOPMENT_ORACLE_DIAGNOSTIC_ONLY_NO_DEPLOYED_THRESHOLD_CHANGE'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def write(p,value):
    Path(p).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def roc(scores,truth,budgets):
    scores=np.asarray(scores,dtype=np.float64);truth=np.asarray(truth)
    pos=truth==1;neg=truth==0
    if not pos.any() or not neg.any():raise ValueError('Both known classes required')
    thresholds=np.r_[np.nextafter(scores.max(),np.inf),np.unique(scores)[::-1]]
    points=[]
    for t in thresholds:
        p=scores>=t
        points.append(dict(threshold=float(t),TP=int((p&pos).sum()),FP=int((p&neg).sum())))
    def best(rows):
        if not rows:return None
        return max(rows,key=lambda r:(r['TP'],-r['FP'],r['threshold'])).copy()
    envelope=[]
    for b in budgets:
        row=best([r for r in points if r['FP']<=b])
        envelope.append(dict(FP_budget=b,**row,recall=row['TP']/int(pos.sum()),
                             achieved_FPR=row['FP']/int(neg.sum()),
                             exact_FP_point=best([r for r in points if r['FP']==b])))
    delta=scores[pos,None]-scores[neg]
    auc=float(((delta>0)+.5*(delta==0)).mean())
    # Independent tied-score ROC integration must match Mann-Whitney AUC.
    tpr=np.array([r['TP']/pos.sum() for r in points]);fpr=np.array([r['FP']/neg.sum() for r in points])
    auc_roc=float(np.sum((tpr[1:]+tpr[:-1])*.5*np.diff(fpr)))
    if not np.isclose(auc,auc_roc,atol=1e-12,rtol=0):raise ValueError('AUC implementations disagree')
    return dict(positives=int(pos.sum()),negatives=int(neg.sum()),AUC=auc,AUC_ROC_check=auc_roc,
        envelope=envelope,threshold_points=points,
        semantics='Maximum TP at <=FP_budget; exact_FP_point null when score ties skip that FP count')

def paired(scores,truth,records,thresholds):
    a=scores['A']>=thresholds['A'];b=scores['B']>=thresholds['B']
    categories={k:dict(count=0,sample_indices=[],by_family_condition={}) for k in
                ['both_hit','A_only_hit','B_only_hit','both_miss','both_false_alarm',
                 'A_only_false_alarm','B_only_false_alarm','both_correct_clear']}
    samples=[]
    for i,r in enumerate(records):
        if truth[i]==1:
            category='both_hit' if a[i] and b[i] else 'A_only_hit' if a[i] else 'B_only_hit' if b[i] else 'both_miss'
        else:
            category='both_false_alarm' if a[i] and b[i] else 'A_only_false_alarm' if a[i] else 'B_only_false_alarm' if b[i] else 'both_correct_clear'
        target=categories[category];target['count']+=1;target['sample_indices'].append(r['sample_index'])
        target['by_family_condition'].setdefault(r['family'],{}).setdefault(r['condition'],[]).append(r['sample_index'])
        samples.append(dict(sample_index=r['sample_index'],name=r['name'],group_id=r['group_id'],
            family=r['family'],condition=r['condition'],truth=int(truth[i]),category=category,
            A_score=float(scores['A'][i]),B_score=float(scores['B'][i]),A_alert=bool(a[i]),B_alert=bool(b[i])))
    assert sum(r['count'] for r in categories.values())==len(records)
    return dict(thresholds=thresholds,categories=categories,samples=samples)

def run(cache,model_run,output):
    cache=cache.resolve();model_run=model_run.resolve();output=output.resolve()
    if output.exists():raise FileExistsError('Fresh audit output required')
    inputs={}
    def bind(p):inputs[str(p)]=sha(p);return p
    manifest=read(bind(cache/'manifest.json'));selection=read(bind(model_run/'selection.json'))
    result=dict(scope=SCOPE,backend='CPU',backend_reason='TASK_NOT_GPU_SUITABLE_SMALL_ARRAY_METRICS',roles={})
    if manifest['status']!='PASS':raise ValueError('Accepted cache required')
    for role in ('train','dev','eval'):
        folder='supervision' if role=='train' else 'evaluator'
        record=read(bind(cache/folder/f'{role}.json'))
        path=(cache/record['near']['path']).resolve()
        if not path.is_relative_to(cache) or sha(path)!=record['near']['sha256']:raise ValueError('Truth path/hash mismatch')
        truth=np.load(bind(path),allow_pickle=False);records=record['records']
        ids=manifest['partitions'][role]['sample_indices']
        if ids!=record['sample_indices'] or ids!=[r['sample_index'] for r in records]:raise ValueError('Row identity mismatch')
        if truth.shape!=(len(records),2) or not np.isin(truth,[0,1]).all():raise ValueError('Known binary truth required')
        scores={}
        for arm in ('A','B'):
            with np.load(bind(model_run/f'{arm}-{role}.npz'),allow_pickle=False) as data:scores[arm]=data['near'].astype(np.float64)
            if scores[arm].shape!=truth.shape or not np.isfinite(scores[arm]).all() or not ((scores[arm]>=0)&(scores[arm]<=1)).all():raise ValueError('Invalid scores')
        result['roles'][role]=dict(frames=len(records),heads={})
        for h,head in enumerate(('BODY','HEAD')):
            budgets=list(range(25)) if role in ('dev','eval') else [0,1,2]
            thresholds={arm:float(selection[arm]['thresholds'][h]) for arm in ('A','B')}
            if not all(np.isfinite(list(thresholds.values()))):raise ValueError('Invalid thresholds')
            result['roles'][role]['heads'][head]=dict(
                arms={arm:roc(scores[arm][:,h],truth[:,h],budgets) for arm in ('A','B')},
                paired=paired({arm:scores[arm][:,h] for arm in ('A','B')},truth[:,h],records,thresholds))
    if any(sha(p)!=digest for p,digest in inputs.items()):raise ValueError('Input changed during audit')
    output.mkdir(parents=True)
    write(output/'score-audit.json',result)
    receipt=dict(status='PASS',scope=SCOPE,training_runs=0,inference_runs=0,source_sha256=sha(__file__),
        inputs=inputs,output_sha256=sha(output/'score-audit.json'),checks=['cache truth SHA','sample row identity','finite scores','AUC pairwise equals tied ROC integration','paired denominator closure','inputs unchanged'])
    write(output/'receipt.json',receipt)
    print(json.dumps(dict(status='PASS',receipt_sha256=sha(output/'receipt.json'),output=str(output))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('cache','model-run','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.cache,a.model_run,a.output)
