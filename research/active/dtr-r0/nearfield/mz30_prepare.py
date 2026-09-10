"""Reconstruct frozen sensor branches and run the predeclared opportunity gate."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from mz5_ensemble_readout import CompactEnsemble,read,write,sha,load_npz
from body_query_context_decoder import CountDecoder
from body_query_range import range_from_counts
from body_query_fresh_size_eval import FROZEN
from mz24_audit import geometry


def main(root,output):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();work=root/'artifacts.local/work';inputs={}
    def bind(path,expected=None):
        h=sha(path);assert expected is None or h==expected,str(path);inputs[str(path)]=h
    oldpath=work/'mz8-attribution-20260910/cache-v5';newpath=work/'mz15-shared-support-20260910/cache-v1'
    rank=work/'mz20-rank-objective-20260910/run-v1';run28=work/'mz28-packet-availability-20260910/run-v1'
    compact=work/'mz5-fixed-ensemble-20260910/compact-v1';context=work/'body-query-context-decoder-20260909/run-v1';basepath=work/'body-query-10000-b-20260909/run-v1'
    for folder,names in [(oldpath,['observations.npz','evaluator.npz']),(newpath,['observations.npz','evaluator.npz','dense.npy','selected.json'])]:
        receipt=read(folder/'receipt.json');assert receipt['status']=='PASS';bind(folder/'receipt.json')
        for name in names:bind(folder/name,receipt['files'][name])
    for folder,names in [(rank,['predictions.npz','BODY_RANK.pt','batches.npy']),(run28,['predictions.npz'])]:
        receipt=read(folder/'receipt.json');assert receipt['status']=='PASS';bind(folder/'receipt.json')
        for name in names:bind(folder/name,receipt['outputs'][name])
    bind(compact/'export-receipt.json');bind(compact/'compact.pt',read(compact/'export-receipt.json')['compact_sha256'])
    for name,digest in FROZEN.items():bind((basepath if name=='NEW-step2000.pt' else context)/name,digest)
    p11path=work/'mz11-selective-addition-20260910/run-v1';p13path=work/'mz13-training-coverage-20260910/features-v1'
    bind(p11path/'predictions.npz',read(p11path/'receipt.json')['outputs']['predictions.npz']);bind(p13path/'features.npz',read(p13path/'receipt.json')['files']['features.npz'])
    old,new=load_npz(oldpath/'observations.npz'),load_npz(newpath/'observations.npz');ref=load_npz(rank/'predictions.npz');mz28=load_npz(run28/'predictions.npz')
    truth=np.concatenate([load_npz(oldpath/'evaluator.npz')['truth'],load_npz(newpath/'evaluator.npz')['truth']]);ranges=np.concatenate([old['ranges'],new['ranges']]);valid=np.concatenate([old['valid'],new['valid']])
    batches=np.load(rank/'batches.npy');train=np.unique(batches);assert len(train)==7562
    cohorts=dict(DEV=np.flatnonzero(old['role']=='DEV_ONLY'),clean=np.arange(3500,3700),stress=np.arange(3500,3700),relation10000=np.arange(11200,13200),distance5000=np.arange(13200,14200))
    ids=np.unique(np.r_[train,*cohorts.values()]);assert len(ids)==11562;lookup=np.full(14200,-1,int);lookup[ids]=np.arange(len(ids))
    expected=np.full((14200,4),np.nan);p11=load_npz(p11path/'predictions.npz');p13=load_npz(p13path/'features.npz')
    expected[np.flatnonzero(old['role']=='TRAIN_ONLY')]=p11['TRAIN/baseline'];expected[3700:8700]=p13['baseline'][p13['dataset']=='relation'];expected[8700:11200]=p13['baseline'][p13['dataset']=='distance']
    for name,ii in cohorts.items():
        if name!='stress':expected[ii]=ref['BODY_RANK/'+name+'/baseline']
        np.testing.assert_array_equal(truth[ii],ref['BODY_RANK/'+name+'/truth'])
    assert np.isfinite(expected[ids]).all()
    write(output/'start.json',dict(status='STARTED',training_steps=0,inputs=inputs,normal_frames=len(ids),train_frames=len(train),
        protocol_sha256=sha(Path(__file__).with_name('MZ30_BRANCH_RESPONSIBILITY_PROTOCOL_20260910.md')),code_sha256=sha(Path(__file__)),
        gate=dict(minimum_placement_fp_disagreements=41,minimum_train_each_branch_correct=20),scope='Frozen cached-feature branches; no backbone or fit'))
    torch.set_num_threads(1);assert torch.cuda.is_available();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    model=CompactEnsemble.from_checkpoint(compact/'compact.pt').cuda().eval()
    state=torch.load(basepath/'NEW-step2000.pt',map_location='cpu',weights_only=True)
    projection=state['query_projection'].cuda();mask=state['query_valid'][None,:,:,None].cuda();del state
    decstate=torch.load(context/'JOINT.pt',map_location='cpu',weights_only=True);decoder=CountDecoder('JOINT',decstate['xyz']).cuda();decoder.load_state_dict(decstate);decoder.eval().requires_grad_(False)
    norm=load_npz(context/'normalization.npz');mean=torch.from_numpy(norm['mean']).cuda();std=torch.from_numpy(norm['std']).cuda()
    rays=torch.load(rank/'BODY_RANK.pt',map_location='cpu',weights_only=True)['rays'];maps=np.load(newpath/'dense.npy',mmap_mode='r');arrays={};maxerr=0.
    def extract(ii,stress=False):
        nonlocal maxerr
        pieces=[]
        with torch.inference_mode():
            for begin in range(0,len(ii),32):
                batch=ii[begin:begin+32];vis=torch.empty((len(batch),772),device='cuda')
                isold=batch<3700
                if isold.any():vis[torch.from_numpy(isold).cuda()]=torch.from_numpy(old['visual'][batch[isold]]).cuda()
                if (~isold).any():
                    dense=torch.from_numpy(np.array(maps[batch[~isold]-3700])).cuda();sampled=(dense.flatten(2)@projection.T).transpose(1,2).reshape(len(dense),12,27,64)
                    pooled=(sampled*mask).sum(2)/mask.sum(2).clamp_min(1);normalized=(pooled-mean)/std
                    visual=torch.cat([normalized.flatten(1),range_from_counts(decoder(normalized)).sigmoid().flatten(1)],1)
                    vis[torch.from_numpy(~isold).cuda()]=visual
                rr,vv=(old['stress_ranges'][batch-3500],old['stress_valid'][batch-3500]) if stress else (ranges[batch],valid[batch])
                tof=torch.from_numpy(np.concatenate([rr.reshape(len(batch),-1)/4,vv.reshape(len(batch),-1).astype(np.float32)],1).astype(np.float32)).cuda()
                rh=model.rgb[:2](vis);th=model.tof[:2](tof);rgb=model.rgb[2](rh);toflog=model.tof[2](th);average=(rgb+toflog)*.5
                reference=ref['BODY_RANK/stress/baseline'][begin:begin+len(batch)] if stress else expected[batch]
                error=float(np.abs(average.cpu().numpy()-reference).max());maxerr=max(maxerr,error)
                np.testing.assert_allclose(average.cpu().numpy(),reference,atol=1e-4,rtol=1e-5);np.testing.assert_array_equal(average.cpu().numpy()>=0,reference>=0)
                supported=geometry(rr,vv,rays).any((1,2,3));disagree=(rgb>=0)!=(toflog>=0)
                eligibility=(reference>=0)&~supported&disagree.cpu().numpy()
                target=(rgb.cpu().numpy()>=0)==truth[batch]
                pieces.append(dict(features=torch.cat([rh,th,rgb,toflog],1).cpu().numpy(),rgb=rgb.cpu().numpy(),tof=toflog.cpu().numpy(),baseline=reference,
                    support=supported,eligible=eligibility,target=target,truth=truth[batch]))
        return {k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    try:
        normal=extract(ids);stress=extract(cohorts['stress'],True)
        for k,v in normal.items():arrays['normal/'+k]=v
        for k,v in stress.items():arrays['stress/'+k]=v
        arrays.update(global_ids=ids,train_ids=train,batches=batches)
        for name,ii in cohorts.items():arrays['ids/'+name]=ii
        summaries={}
        for name,ii in cohorts.items():
            data=stress if name=='stress' else {k:v[lookup[ii]] for k,v in normal.items()}
            np.testing.assert_array_equal(data['support'],mz28[name+'/original_support'])
            summaries[name]={}
            for label,title in [(False,'baseline_false'),(True,'baseline_true')]:
                take=(data['baseline']>=0)&(data['truth']==label);disagree=(data['rgb']>=0)!=(data['tof']>=0)
                summaries[name][title]=dict(total=take.sum(0).tolist(),unsupported=(take&~data['support']).sum(0).tolist(),eligible=(take&data['eligible']).sum(0).tolist(),
                    unsupported_both_positive=(take&~data['support']&~disagree).sum(0).tolist(),eligible_rgb_correct=(take&data['eligible']&data['target']).sum(0).tolist(),eligible_tof_correct=(take&data['eligible']&~data['target']).sum(0).tolist())
        ti=lookup[train];mask=normal['eligible'][ti];target=normal['target'][ti];traincounts=[int((mask&target).sum()),int((mask&~target).sum())]
        opportunity=sum(sum(summaries[n]['baseline_false']['eligible']) for n in ['relation10000','distance5000'])
        gate=dict(placement_opportunity=opportunity,required=41,train_rgb_tof_correct=traincounts,passed=opportunity>=41 and min(traincounts)>=20)
        np.savez_compressed(output/'branches.npz',**arrays);write(output/'gate.json',dict(status='PASS' if gate['passed'] else 'NOT_ADMITTED',gate=gate,summaries=summaries,baseline_parity_max_abs=maxerr,training_steps=0))
        write(output/'prepare-receipt.json',dict(status='PASS',training_steps=0,backend='CUDA frozen branch/context readouts on saved features; CPU geometry',seconds=time.perf_counter()-started,inputs=inputs,code_sha256=sha(Path(__file__)),outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
        print('PREPARE PASS',gate,'parity',maxerr,flush=True)
    finally:model.cpu();decoder.cpu();del maps;torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.root,a.output)
