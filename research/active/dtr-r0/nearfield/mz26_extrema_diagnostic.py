"""Fixed MZ26 endpoint TRAIN-versus-DEV extrema; no fit or snapshot selection."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from mz5_ensemble_readout import read, sha, write, load_npz
from mz16_detail_readout import make_model
from mz25_fixed_sampler import StableAngularAvailability

ORDER=['BODY_NEAR','BODY_FAR','HEAD_NEAR','HEAD_FAR']


def summarize(values, valid, positive=False):
    x=values[valid].astype(float)
    if not len(x):return dict(bags=0,nonnegative=0,nonnegative_rate=None,quantiles=None,mean_softplus_loss=None)
    levels=[0,.1,.25,.5,.75,.9,.99,1]
    return dict(bags=len(x),nonnegative=int((x>=0).sum()),nonnegative_rate=float((x>=0).mean()),
        quantiles=dict(zip(['min','p10','p25','median','p75','p90','p99','max'],np.quantile(x,levels).tolist())),
        mean_softplus_loss=float(np.logaddexp(0.,-x if positive else x).mean()))


def main(root,output):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    work=root/'artifacts.local/work';cache=work/'mz16-visual-detail-20260910/cache-v2'
    rank=work/'mz20-rank-objective-20260910/run-v1';prior=work/'mz15-shared-support-20260910/run-v1'
    oldpath=work/'mz8-attribution-20260910/cache-v5';newpath=work/'mz15-shared-support-20260910/cache-v1'
    labels=work/'mz9-source-supervision-20260910/labels-v1';normpath=work/'mz9-source-supervision-20260910/run-v1/normalization.npz'
    inputs={}
    def bind(path,expected=None):
        digest=sha(path);assert expected is None or digest==expected,str(path);inputs[str(path)]=digest
    for folder,names in [(cache,['dense_HIGH_DETAIL.npy','ids.npy']),(oldpath,['observations.npz']),
                          (newpath,['observations.npz','evaluator.npz','selected.json'])]:
        receipt=read(folder/'receipt.json');assert receipt['status']=='PASS'
        for name in names:bind(folder/name,receipt['files'][name])
    for name in ['BODY_RANK.pt','batches.npy']:bind(rank/name,read(rank/'receipt.json')['outputs'][name])
    bind(prior/'QUERY-initial.pt',read(prior/'receipt.json')['outputs']['QUERY-initial.pt'])
    bind(labels/'evaluator.npz',read(labels/'receipt.json')['labels_sha256'])
    bind(normpath)
    run=work/'mz26-deterministic-convergence-20260910/run-v1';runs={tag:run for tag in ['step1200','step4800']}
    for tag,run in runs.items():
        receipt=read(run/'receipt.json');assert receipt['status']=='PASS'
        for name in ['availability-'+tag+'.pt','initial.pt','batches.npy','start.json']:bind(run/name,receipt['outputs'][name])
        np.testing.assert_array_equal(np.load(run/'batches.npy'),np.tile(np.load(rank/'batches.npy'),(4,1)))
    tailnote='Two predeclared budget endpoints from one completed deterministic trajectory; no selection or updates.'
    old,new=load_npz(oldpath/'observations.npz'),load_npz(newpath/'observations.npz')
    ol,nl=load_npz(labels/'evaluator.npz'),load_npz(newpath/'evaluator.npz')
    known=np.concatenate([ol['cell_known_counts']>0,nl['known']])
    query=np.concatenate([ol['query_counts']>0,nl['query_presence']]);del ol,nl
    ranges=np.concatenate([old['ranges'],new['ranges']]);valid=np.concatenate([old['valid'],new['valid']])
    batches=np.load(rank/'batches.npy');train=np.unique(batches);assert len(train)==7562
    cohortids=dict(TRAIN_UNIQUE=train,oldDEV=np.flatnonzero(old['role']=='DEV_ONLY'),
        relationDEV=np.arange(11200,13200),distanceDEV=np.arange(13200,14200),clean=np.arange(3500,3700),pole_clean=np.arange(3600,3625))
    assert np.isin(cohortids['clean'],train).all()
    allowed=np.r_[np.flatnonzero(old['role']=='TRAIN_ONLY'),np.arange(3500,11200)]
    assert np.isin(train,allowed).all() and not np.isin(train,cohortids['oldDEV']).any()
    allids=np.unique(np.concatenate(list(cohortids.values())))
    cacheids=np.load(cache/'ids.npy');np.testing.assert_array_equal(allids,cacheids)
    lookup=np.full(14200,-1,int);lookup[cacheids]=np.arange(len(cacheids))
    rows=read(newpath/'selected.json');source=np.empty(14200,dtype='<U20')
    source[:3500]='old5000';source[3500:3700]='MZ6_sequence'
    source[3700:]=[r['dataset'] for r in rows]
    slices=dict(cohortids)
    for label in sorted(set(source[train])):slices['TRAIN_SOURCE/'+label]=train[source[train]==label]
    for name in ['TRAIN_UNIQUE','oldDEV','relationDEV','distanceDEV']:
        assert len(np.unique(cohortids[name]))==len(cohortids[name])
    write(output/'start.json',dict(status='STARTED',training_steps=0,input_unique_frames=len(allids),cohort_counts={n:len(v) for n,v in slices.items()},
        inputs=inputs,code_sha256=sha(Path(__file__)),source_label='Consumed Development',endpoint_scope=tailnote))
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    frozen=make_model(torch.load(prior/'QUERY-initial.pt',map_location='cpu',weights_only=True))
    frozen.load_state_dict(torch.load(rank/'BODY_RANK.pt',map_location='cpu',weights_only=True));frozen.cuda().eval();frozen.requires_grad_(False)
    heads={}
    for tag,run in runs.items():
        head=StableAngularAvailability(frozen.grid.cpu());head.load_state_dict(torch.load(run/('availability-'+tag+'.pt'),map_location='cpu',weights_only=True))
        heads[tag]=head.cuda().eval();heads[tag].requires_grad_(False)
    norm=load_npz(normpath);maps=np.load(cache/'dense_HIGH_DETAIL.npy',mmap_mode='r')
    arrays=dict(global_ids=allids,source=source[allids])
    for tag in heads:
        for name in ['negative_max','positive_witness']:arrays[tag+'/'+name]=np.full((len(allids),4),np.nan,dtype=np.float32)
    arrays['negative_valid']=np.zeros((len(allids),4),bool);arrays['positive_valid']=np.zeros((len(allids),4),bool)
    arrays['negative_candidates']=np.zeros((len(allids),4),np.int32);arrays['positive_candidates']=np.zeros((len(allids),4),np.int32)
    arrays['positive_teacher_index']=np.full((len(allids),4),-1,np.int32)
    try:
        for begin in range(0,len(allids),16):
            ii=allids[begin:begin+16];end=begin+len(ii)
            x=torch.from_numpy((np.array(maps[lookup[ii]])-norm['mean'])/norm['std']).cuda()
            r=torch.from_numpy(ranges[ii].astype(np.float32)).cuda();v=torch.from_numpy(valid[ii]).cuda()
            with torch.inference_mode():
                out=frozen.inspect(x,r,v)
                availability={tag:head(x,r,v).cpu().numpy() for tag,head in heads.items()}
            eligible=out['eligible'].cpu().numpy();teacher=out['candidate_logits'].cpu().numpy()
            negative=eligible&~known[ii,:,None,:,None]
            positive=eligible&query[ii]
            assert not (query[ii]&~known[ii,:,None,:,None]).any()
            nvalid=negative.any((1,2,3));pvalid=positive.any((1,2,3))
            selected=np.where(positive,teacher,-np.inf).reshape(len(ii),-1,4).argmax(1)
            arrays['negative_valid'][begin:end]=nvalid;arrays['positive_valid'][begin:end]=pvalid
            arrays['negative_candidates'][begin:end]=negative.sum((1,2,3));arrays['positive_candidates'][begin:end]=positive.sum((1,2,3))
            arrays['positive_teacher_index'][begin:end]=np.where(pvalid,selected,-1)
            for tag,a in availability.items():
                expanded=np.broadcast_to(a[:,:,None,:,None],teacher.shape)
                maximum=np.where(negative,expanded,-np.inf).max((1,2,3))
                witness=np.take_along_axis(expanded.reshape(len(ii),-1,4),selected[:,None,:],axis=1).squeeze(1)
                arrays[tag+'/negative_max'][begin:end]=np.where(nvalid,maximum,np.nan)
                arrays[tag+'/positive_witness'][begin:end]=np.where(pvalid,witness,np.nan)
            if begin%1024==0 or end==len(allids):
                write(output/'progress.json',dict(frames=end,total=len(allids),seconds=time.perf_counter()-started));print('INFERENCE',end,'/',len(allids),flush=True)
        torch.cuda.synchronize()
    finally:
        frozen.cpu()
        for head in heads.values():head.cpu()
        del maps;torch.cuda.empty_cache()
    positions=np.full(14200,-1,int);positions[allids]=np.arange(len(allids));results={}
    for name,ids in slices.items():
        ix=positions[ids];assert (ix>=0).all();arrays['ids/'+name]=ids
        results[name]=dict(frames=len(ids),models={})
        for tag in heads:
            n,p=arrays[tag+'/negative_max'][ix],arrays[tag+'/positive_witness'][ix]
            nv,pv=arrays['negative_valid'][ix],arrays['positive_valid'][ix]
            results[name]['models'][tag]=dict(negative=summarize(n,nv),positive=summarize(p,pv,True),
                per_query={q:dict(negative=summarize(n[:,j],nv[:,j]),positive=summarize(p[:,j],pv[:,j],True)) for j,q in enumerate(ORDER)})
    result=dict(results=results,inputs=inputs,code_sha256=sha(Path(__file__)),event_order=ORDER,training_steps=0,
        inference_unique_frames=len(allids),inference_head_frames={tag:len(allids) for tag in heads},
        definition='Negative bag: any geometrically eligible candidate with known0, summarized by maximum availability; positive witness: frozenMZ20argmax inside actualquery&eligible.',
        interpretation='Negative availability bags are NOT task false alarms; no MZ20task cutoff or task truth selection enters these masks. Known0 never means no obstacle/CLEAR.',
        source_scope='Exact7562uniqueTRAIN IDs; disjointoldDEV1000 and consumedplacementDEV3000. clean200 and pole25 are already-trained overlapping slices.',
        endpoint_scope=tailnote,causal_limit='Residual TRAIN tails can indicate unresolved objective weighting, representation or optimization; this diagnostic cannot isolate those causes. No threshold was selected.')
    write(output/'result.json',result);np.savez_compressed(output/'extrema.npz',**arrays)
    write(output/'receipt.json',dict(status='PASS',training_steps=0,backend='CUDA frozen MZ20 and MZ26 two fixed endpoint heads; independent NumPy extrema reduction',
        seconds=time.perf_counter()-started,inputs=inputs,code_sha256=sha(Path(__file__)),outputs={n:sha(output/n) for n in ['result.json','extrema.npz','start.json','progress.json']}))
    for name in ['TRAIN_UNIQUE','oldDEV','relationDEV','distanceDEV','pole_clean']:
        for tag,row in results[name]['models'].items():print(name,tag,'negative',row['negative'],'positive',row['positive'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();main(args.root,args.output)
