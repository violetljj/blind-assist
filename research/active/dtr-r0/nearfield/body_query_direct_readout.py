"""Zero-fit direct readout; fixed FOV mask only, no geometry truth in scoring."""
from pathlib import Path
import argparse,time
import numpy as np
from body_query_data import read,write,sha,truth,fresh_output
from body_query_model import fixed_projection
from body_query_linear_probe import auc
from city_dev_selection import select_threshold

METHODS=('max','mean','top3','count')


def aggregate(logits,valid,threshold):
    p=1/(1+np.exp(-np.clip(logits.astype(float),-700,700)))
    masked=np.where(valid[None],p,-np.inf)
    return dict(max=masked.max(-1),mean=np.where(valid[None],p,0).sum(-1)/valid.sum(-1),
        top3=np.sort(masked,axis=-1)[...,-3:].mean(-1),
        count=((logits>=threshold)&valid[None]).sum(-1)/27.)


def evaluate(scores,query,gt,records,thresholds):
    y=gt['near'];pred=scores>=thresholds
    def rows(ids):
        result={}
        for h,name in enumerate(('BODY','HEAD')):
            x=scores[ids,h];positive=y[ids,h]==1;negative=y[ids,h]==0;p=pred[ids,h]
            candidates=np.r_[np.nextafter(x.max(),np.inf),np.unique(x)]
            oracle=max(int(((x>=t)&positive).sum()) for t in candidates if int(((x>=t)&negative).sum())<=2)
            result[name]=dict(TP=int((p&positive).sum()),FP=int((p&negative).sum()),positive=int(positive.sum()),negative=int(negative.sum()),AUC=auc(x,positive),oracle_TP_FP2=oracle)
        return result
    groups=list(dict.fromkeys(r['group_id'] for r in records));correct=0;pairs=[]
    for group in groups:
        ids=[i for i,r in enumerate(records) if r['group_id']==group]
        correct+=int((pred[ids]==y[ids]).all());by={records[i]['condition']:i for i in ids}
        for pos,neg in [('HEAD_ONLY','CLEAR'),('BOTH','BODY_ONLY')]:
            if pos in by and neg in by:
                i,j=by[pos],by[neg]
                pairs.append(dict(group=group,contrast=pos+'-'+neg,frame_delta=float(scores[i,1]-scores[j,1]),
                    query_delta=(query[i,6:]-query[j,6:]).tolist() if query is not None else None))
    conditions={c:rows([i for i,r in enumerate(records) if r['condition']==c]) for c in dict.fromkeys(r['condition'] for r in records)}
    result=dict(heads=rows(np.arange(len(y))),groups_correct=correct,groups_total=len(groups),conditions=conditions,pairs=pairs,
        positive_pair_deltas=sum(p['frame_delta']>1e-12 for p in pairs),zero_pair_deltas=sum(abs(p['frame_delta'])<=1e-12 for p in pairs))
    if query is not None:
        result['query_AUC']={name:auc(query[:,lo:hi].ravel(),(gt['counts'][:,lo:hi]>0).ravel()) for name,lo,hi in [('all',0,12),('BODY',0,6),('HEAD',6,12),('HEAD_near',6,9),('HEAD_far',9,12)]}
    return result


def run(a):
    out=fresh_output(a.output);start=time.perf_counter();_,valid,_=fixed_projection();valid=valid.numpy()
    receipt=read(a.probes/'receipt.json');pointcuts=read(a.probes/'selection.json');assert sha(a.probes/'selection.json')==receipt['selection_sha256']
    inputs={};cached={};selection={};chosen={}
    devrec,devgt=truth(a.cache,'dev')
    for target in ('ray','local'):
        for name in ('B','R1','XYZ'):
            path=a.probes/f'{target}-{name}-logits.npz';assert sha(path)==receipt['arrays'][path.name];inputs[str(path)]=sha(path);z=np.load(path)
            for role in ('train','dev','eval'):
                for method,q in aggregate(z[role],valid,pointcuts[target][name]).items():
                    key=f'{target}-{name}-{method}';cached[key,role]=(q,q.reshape(-1,2,6).max(-1))
            ranks=[]
            for i,method in enumerate(METHODS):
                key=f'{target}-{name}-{method}';s=cached[key,'dev'][1]
                heads=[select_threshold(s[:,h],devgt['near'][:,h],min_count=8) for h in range(2)]
                selection[key]=[h['threshold'] for h in heads];recalls=[h['recall'] for h in heads]
                ranks.append(((min(recalls),sum(recalls),-sum(h['FPR'] for h in heads),-i),key))
            chosen[target+'-'+name]=max(ranks)[1]
    write(out/'selection.json',dict(thresholds=selection,DEV_selected=chosen));selection_hash=sha(out/'selection.json')
    result=dict(methods={},baselines={},DEV_selected=chosen)
    for key,thresholds in selection.items():
        result['methods'][key]={};payload={}
        for role in ('train','dev','eval'):
            rec,gt=truth(a.cache,role);q,s=cached[key,role];payload[role+'_query']=q;payload[role+'_frame']=s
            result['methods'][key][role]=evaluate(s,q,gt,rec['records'],thresholds)
            if '-XYZ-' in key:assert all(p['frame_delta']==0 and all(v==0 for v in p['query_delta']) for p in result['methods'][key][role]['pairs'])
        np.savez_compressed(out/f'{key}.npz',**payload)
    for name,root,prefix in [('B',a.baseline,'B'),('R1',a.r1,'R1')]:
        thresholds=read(root/'selection.json');thresholds=thresholds['B']['thresholds'] if name=='B' else thresholds['thresholds']
        result['baselines'][name]={}
        for role in ('train','dev','eval'):
            rec,gt=truth(a.cache,role);path=root/f'{prefix}-{role}.npz';inputs[str(path)]=sha(path)
            result['baselines'][name][role]=evaluate(np.load(path)['near'],None,gt,rec['records'],thresholds)
    assert sha(out/'selection.json')==selection_hash
    result['seconds']=time.perf_counter()-start;write(out/'result.json',result)
    write(out/'receipt.json',dict(status='PASS',training_steps=0,inference_frames=0,backend='CPU',reason='TASK_NOT_GPU_SUITABLE cached scalar scoring',
        inputs=inputs,code_sha256=sha(Path(__file__)),result_sha256=sha(out/'result.json'),selection_sha256=selection_hash))
    print({k:result['methods'][v]['eval']['heads'] for k,v in chosen.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('cache','probes','baseline','r1','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
