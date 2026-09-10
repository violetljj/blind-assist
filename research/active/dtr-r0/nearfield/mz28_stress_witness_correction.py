"""Correct only MZ28 stress attribution; preserve original run and predictions."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from mz5_ensemble_readout import read,write,sha,load_npz
from mz16_detail_readout import make_model
from mz15_evaluator import align_stress_labels


def main(root,run,output):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();receipt=read(run/'receipt.json');assert receipt['status']=='PASS'
    for name,digest in receipt['outputs'].items():assert sha(run/name)==digest,name
    start=read(run/'start.json')
    for path,digest in start['inputs'].items():assert sha(path)==digest,path
    work=root/'artifacts.local/work';oldpath=work/'mz8-attribution-20260910/cache-v5';labelpath=work/'mz9-source-supervision-20260910/labels-v1'
    cache=work/'mz16-visual-detail-20260910/cache-v2';rank=work/'mz20-rank-objective-20260910/run-v1';prior=work/'mz15-shared-support-20260910/run-v1'
    old=load_npz(oldpath/'observations.npz');labels=load_npz(labelpath/'evaluator.npz');pred=load_npz(run/'predictions.npz');ii=np.arange(3500,3700)
    source,query,moved=align_stress_labels(labels['source_counts'][ii]>0,labels['query_counts'][ii]>0,old['ranges'][ii],old['valid'][ii],old['stress_ranges'],old['stress_valid'])
    del labels
    ids=np.load(cache/'ids.npy');lookup=np.full(14200,-1,int);lookup[ids]=np.arange(len(ids));maps=np.load(cache/'dense_HIGH_DETAIL.npy',mmap_mode='r')
    norm=load_npz(work/'mz9-source-supervision-20260910/run-v1/normalization.npz');votes=pred['stress/votes']
    torch.set_num_threads(1);assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    model=make_model(torch.load(prior/'QUERY-initial.pt',map_location='cpu',weights_only=True));model.load_state_dict(torch.load(rank/'BODY_RANK.pt',map_location='cpu',weights_only=True));model.cuda().eval().requires_grad_(False)
    inputs=dict(start['inputs']);inputs[str(run/'receipt.json')]=sha(run/'receipt.json');inputs[str(run/'predictions.npz')]=sha(run/'predictions.npz')
    write(output/'start.json',dict(status='STARTED',training_steps=0,frames=200,scope='Stress attribution-only correction; no packet retrieval, task decisions or operating thresholds changed',inputs=inputs,code_sha256=sha(Path(__file__)),alignment_sha256=sha(Path(__file__).with_name('mz15_evaluator.py'))))
    arrays={};pieces=[]
    try:
        with torch.inference_mode():
            for begin in range(0,200,16):
                end=min(begin+16,200);batch=ii[begin:end];x=torch.from_numpy((np.array(maps[lookup[batch]])-norm['mean'])/norm['std']).cuda()
                r=torch.from_numpy(old['stress_ranges'][begin:end].astype(np.float32)).cuda();v=torch.from_numpy(old['stress_valid'][begin:end]).cuda()
                o=model.inspect(x,r,v);e=o['eligible'].cpu().numpy();scores=o['candidate_logits'].cpu().numpy()
                np.testing.assert_allclose(o['logits'].cpu().numpy(),pred['stress/original_raw'][begin:end],atol=2e-5,rtol=1e-5)
                positive=e&query[begin:end];present=positive.any((1,2,3));positions=np.where(positive,scores,-np.inf).reshape(len(batch),-1,4).argmax(1)
                allvotes=np.broadcast_to(votes[begin:end,:,None,:,None],e.shape).reshape(len(batch),-1,4)
                selected=np.take_along_axis(allvotes,positions[:,None,:],axis=1)[:,0].astype(np.int16);selected[~present]=-1
                pieces.append(dict(teacher_witness_index=positions,teacher_witness_valid=present,teacher_witness_votes=selected,
                    actual_contributors=positive.sum((1,2,3)),rejected_contributors=(positive&(votes[begin:end,:,None,:,None]<3)).sum((1,2,3))))
        arrays={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]};arrays['global_ids']=ii;arrays['moved_echo_zones']=moved
        summary=dict(valid=arrays['teacher_witness_valid'].sum(0).tolist(),retained=((arrays['teacher_witness_votes']>=3)&arrays['teacher_witness_valid']).sum(0).tolist(),
            actual_contributors=arrays['actual_contributors'].sum(0).tolist(),rejected_contributors=arrays['rejected_contributors'].sum(0).tolist())
        changes={k:int((arrays[k]!=pred['stress/'+k]).sum()) for k in pieces[0]}
        np.savez_compressed(output/'corrected.npz',**arrays)
        write(output/'result.json',dict(status='PASS',training_steps=0,frames=200,moved_echo_zones=int(moved.sum()),corrected_stress_witnesses=summary,changed_elements=changes,
            supersedes=['run-v1/result.json witnesses.stress']+['run-v1/predictions.npz stress/'+k for k in pieces[0]],
            unchanged=['all candidate/raw/support/votes/neighbor arrays','all task metrics and gates','normal-cohort witness diagnostics'],original_receipt_sha256=sha(run/'receipt.json')))
        write(output/'receipt.json',dict(status='PASS',training_steps=0,backend='CUDA frozen MZ20 scorer for200stress frames; aligned evaluator labels and saved votes',seconds=time.perf_counter()-started,inputs=inputs,
            code_sha256=sha(Path(__file__)),alignment_sha256=sha(Path(__file__).with_name('mz15_evaluator.py')),outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
        print('PASS',summary,changes,flush=True)
    finally:model.cpu();del maps;torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.run,a.output)
