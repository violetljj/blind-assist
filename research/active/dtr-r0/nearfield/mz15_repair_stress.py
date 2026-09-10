"""Repair diagnostic label alignment only; replay frozen models without fitting."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from mz5_ensemble_readout import read, write, sha, load_npz
from mz15_shared_support import LocalSupportReadout
from mz15_evaluator import align_stress_labels


def main(root, run, output):
    output.mkdir(parents=True, exist_ok=False); started=time.perf_counter()
    inputs={}
    receipt=read(run/'receipt.json')
    for name,digest in receipt['outputs'].items():
        assert sha(run/name)==digest
        inputs[str(run/name)]=digest
    original=read(run/'start.json')['code_sha256']['mz15_train.py']
    assert sha(run/'mz15_train.original.py')==original
    work=root/'artifacts.local/work';cache=work/'mz8-attribution-20260910/cache-v5'
    d=load_npz(cache/'observations.npz')
    labels=load_npz(work/'mz9-source-supervision-20260910/labels-v1/evaluator.npz')
    normal=work/'mz9-source-supervision-20260910/run-v1/normalization.npz'
    n=load_npz(normal);maps=np.load(cache/'dense.npy',mmap_mode='r')
    for p in [cache/'observations.npz',cache/'dense.npy',normal,work/'mz9-source-supervision-20260910/labels-v1/evaluator.npz']:
        inputs[str(p)]=sha(p)
    source,query,moved=align_stress_labels(labels['source_counts'][3500:3700]>0,labels['query_counts'][3500:3700]>0,
        d['ranges'][3500:3700],d['valid'][3500:3700],d['stress_ranges'],d['stress_valid'])
    deleted=load_npz(work/'mz6-short-sequence-20260910/evaluation-v1/evaluator.npz')['deleted']
    np.testing.assert_array_equal(moved,deleted)
    a=load_npz(run/'predictions.npz');result=read(run/'result.json')
    torch.set_num_threads(1);torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    changes={};verified=0
    for arm in ['SHARED','QUERY']:
        model=LocalSupportReadout(arm=='SHARED').cuda().eval()
        model.load_state_dict(torch.load(run/(arm+'.pt'),weights_only=True))
        cutoff=np.load(run/(arm+'-cutoff.npy'))
        for wrong in [False,True]:
            name='stress_wrong' if wrong else 'stress';prefix=arm+'/'+name+'/'
            srcwin,qwin,best=[],[],[]
            with torch.inference_mode():
                for begin in range(0,200,16):
                    end=min(200,begin+16)
                    dense=(np.array(maps[3500+begin:3500+end])-n['mean'])/n['std']
                    out=model.inspect(torch.from_numpy(dense).cuda(),torch.from_numpy(d['stress_ranges'][begin:end].astype(np.float32)).cuda(),torch.from_numpy(d['stress_valid'][begin:end]).cuda(),wrong)
                    np.testing.assert_array_equal(out['logits'].cpu().numpy(),a[prefix+'raw'][begin:end])
                    np.testing.assert_array_equal(out['support'].cpu().numpy(),a[prefix+'support'][begin:end])
                    verified+=(end-begin)*4
                    score=out['candidate_logits'].masked_fill(~out['eligible'],-1e6)
                    winner=score.flatten(1,3).argmax(1)
                    s=torch.from_numpy(source[begin:end]).cuda()[...,None].expand_as(score)
                    q=torch.from_numpy(query[begin:end]).cuda()
                    srcwin.append(s.flatten(1,3).gather(1,winner[:,None]).squeeze(1).cpu().numpy())
                    qwin.append(q.flatten(1,3).gather(1,winner[:,None]).squeeze(1).cpu().numpy())
                    best.append(score.masked_fill(~q,-1e6).flatten(1,3).amax(1).cpu().numpy())
            for key,vals in [('winning_source',srcwin),('winning_query',qwin),('best_true',best)]:
                value=np.concatenate(vals)
                changes[prefix+key]=int(np.count_nonzero(value!=a[prefix+key]))
                a[prefix+key]=value
            accepted,truth=a[prefix+'accepted'],a[prefix+'truth']
            result['metrics'][arm][name]['added_positive_wrong_source']=(accepted & ~a[prefix+'winning_source']).sum(0).tolist()
            result['metrics'][arm][name]['added_tp_correct_evidence']=(accepted & truth & (a[prefix+'best_true']>=cutoff)).sum(0).tolist()
    np.savez_compressed(output/'predictions.npz',**a)
    write(output/'result.json',result)
    write(output/'receipt.json',dict(status='PASS',training_steps=0,moved_zones=int(moved.sum()),verified_frozen_raw_bits=verified,
        changes=changes,inputs=inputs,original_train_source_sha256=original,
        corrected_train_source_sha256=sha(Path(__file__).with_name('mz15_train.py')),
        code_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('mz15_evaluator.py')]},
        backend='CUDA',seconds=time.perf_counter()-started,
        invariant='All predictions, cutoffs, TP/FP and admission decisions unchanged; evaluator-source statistics only',
        outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print('PASS',verified,'raw bits unchanged; shifted',int(moved.sum()),'zones',changes)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','run','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();main(a.root,a.run,a.output)
