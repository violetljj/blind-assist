"""Zero-fit, post-hoc audit of visible-count evidence; never changes thresholds."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def load(cache,entry):
    path=(cache/entry['path']).resolve()
    if not path.is_relative_to(cache.resolve()) or sha(path)!=entry['sha256']:
        raise ValueError('Cache array path/hash mismatch')
    return np.load(path,allow_pickle=False)


def stats(prob,labels):
    p=1-prob[...,0];y=labels>0;hit=p>=.5
    ce=-np.log(np.take_along_axis(prob,labels[...,None],axis=-1)[...,0].clip(1e-30))
    pos=int(y.sum());neg=int((~y).sum())
    tp=int((hit&y).sum());fp=int((hit&~y).sum())
    positive_loss=float(ce[y].mean()) if pos else None
    empty_loss=float(ce[~y].mean()) if neg else None
    auc=float(((p[y,None]>p[~y]).astype(float)+.5*(p[y,None]==p[~y])).mean()) if pos and neg else None
    return dict(cells=int(y.size),positive=pos,empty=neg,TP=tp,FP=fp,FN=pos-tp,TN=neg-fp,
        nonempty_recall=tp/pos if pos else None,empty_FPR=fp/neg if neg else None,
        nonempty_AUC=auc,count_accuracy=float((prob.argmax(-1)==labels).mean()),
        all_empty_count_accuracy=neg/y.size,
        positive_exact_count_accuracy=float((prob.argmax(-1)[y]==labels[y]).mean()) if pos else None,
        class_counts=np.bincount(labels.reshape(-1),minlength=4).tolist(),
        mean_CE=float(ce.mean()),positive_mean_CE=positive_loss,empty_mean_CE=empty_loss,
        positive_CE_fraction=float(ce[y].sum()/ce.sum()) if ce.sum() else None,
        balanced_CE_descriptive_only=(positive_loss+empty_loss)/2 if pos and neg else None)


def audit(cache,run,out):
    if out.exists():raise FileExistsError(out)
    receipt=read(run/'receipt.json');selection=read(run/'selection.json')
    if receipt['status']!='PASS' or receipt['result_sha256']!=sha(run/'result.json'):
        raise ValueError('Completed source receipt required')
    result=dict(scope='Post-hoc consumed TRAIN/DEV/EVAL diagnostic; no fitting, promotion, or threshold change',
        signal_rule='Nonempty query signal means1-P(count0)>=.5. Categories describe outputs, not causal transport or loss attribution.',
        missing_head_categories='Correct-cell signal below saved near threshold; else wrong same-head cell; else wrong other-head cell; else no target-head signal. Another correctly active head is not a misplaced signal. Categories are exclusive, order matters.',
        backend='CPU',backend_reason='TASK_NOT_GPU_SUITABLE_SMALL_CACHED_ARRAYS',roles={},sources={})
    for role in ('train','dev','eval'):
        metadata=cache/('supervision/train.json' if role=='train' else f'evaluator/{role}.json')
        rec=read(metadata);labels=load(cache,rec['counts']);near=load(cache,rec['near'])
        if not np.isin(labels,[0,1,2,3]).all():raise ValueError('Unknown count labels cannot be negative truth')
        result['sources'][str(metadata)]=sha(metadata);result['roles'][role]={}
        for arm in ('A','B'):
            source=run/f'{arm}-{role}.npz';pred=np.load(source,allow_pickle=False)
            result['sources'][str(source)]=sha(source)
            q=pred['counts'].astype(float);n=pred['near'].astype(float)
            assert q.shape==(*labels.shape,4) and np.isfinite(q).all() and np.allclose(q.sum(-1),1,atol=1e-6)
            summary=dict(all=stats(q,labels),heads={},strata={})
            for h,head in enumerate(('BODY','HEAD')):
                cols=np.arange(h*6,(h+1)*6)
                summary['heads'][head]=dict(all=stats(q[:,cols],labels[:,cols]),ranges={
                    name:stats(q[:,cols[i:i+3]],labels[:,cols[i:i+3]]) for i,name in ((0,'near'),(3,'far'))})
            for field in ('family','condition'):
                summary['strata'][field]={}
                for value in dict.fromkeys(r[field] for r in rec['records']):
                    ids=[i for i,r in enumerate(rec['records']) if r[field]==value]
                    summary['strata'][field][value]=stats(q[ids],labels[ids])
            active=1-q[:,:,0]>=.5;positive=labels>0
            misses=[]
            for i,r in enumerate(rec['records']):
                for h,head in enumerate(('BODY','HEAD')):
                    threshold=selection[arm]['thresholds'][h]
                    if near[i,h]!=1 or n[i,h]>=threshold:continue
                    cols=np.arange(h*6,(h+1)*6);other=np.arange((1-h)*6,(2-h)*6)
                    correct=cols[positive[i,cols]];on=cols[active[i,cols]&positive[i,cols]]
                    wrong=cols[active[i,cols]&~positive[i,cols]]
                    wrong_other=other[active[i,other]&~positive[i,other]]
                    category=('correct_cell_signal_below_near_threshold' if len(on) else
                        'wrong_same_head_cell_signal' if len(wrong) else
                        'wrong_other_head_cell_signal' if len(wrong_other) else 'no_target_head_signal')
                    ranges=sorted({int((c%6)//3) for c in correct})
                    item=dict(sample_index=r['sample_index'],group=r['group_id'],family=r['family'],condition=r['condition'],
                        head=head,near_score=float(n[i,h]),threshold=float(threshold),category=category,
                        truth_query_ids=correct.tolist(),truth_ranges=ranges,
                        correct_active_query_ids=on.tolist(),wrong_same_head_ids=wrong.tolist(),wrong_other_head_ids=wrong_other.tolist(),
                        max_correct_cell_nonempty=float((1-q[i,correct,0]).max()),
                        truth_counts=labels[i].tolist(),nonempty_scores=(1-q[i,:,0]).tolist())
                    misses.append(item)
            summary['near_misses']=misses
            summary['miss_categories']={head:dict(Counter(r['category'] for r in misses if r['head']==head)) for head in ('BODY','HEAD')}
            result['roles'][role][arm]=summary
    result['sources'][str(run/'selection.json')]=sha(run/'selection.json')
    out.mkdir(parents=True)
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'receipt.json').write_text(json.dumps(dict(status='PASS',result_sha256=sha(out/'result.json'),
        code_sha256=sha(Path(__file__)),source_receipt_sha256=sha(run/'receipt.json'),training_steps=0),indent=2),encoding='utf-8')
    print(json.dumps({role:{arm:{'all':v['all'],'miss_categories':v['miss_categories']} for arm,v in rows.items()} for role,rows in result['roles'].items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('cache','run','output'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();audit(a.cache,a.run,a.output)
