"""One frozen-evidence residual fit with task-only labels and oldDEV calibration."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from mz5_ensemble_readout import read,write,sha,load_npz
from mz16_detail_readout import make_model
from mz22_evidence_arbiter import EvidenceArbiter,observable_features
from mz8_train import metrics,thresholds_at_budget


def main(root,output):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    work=root/'artifacts.local/work';old=work/'mz8-attribution-20260910/cache-v5'
    new=work/'mz15-shared-support-20260910/cache-v1';cache=work/'mz16-visual-detail-20260910/cache-v2'
    rank=work/'mz20-rank-objective-20260910/run-v1';prior=work/'mz15-shared-support-20260910/run-v1'
    inputs={}
    def bind(p,expected=None):
        digest=sha(p);assert expected is None or digest==expected,str(p);inputs[str(p)]=digest
    for folder,names in [(old,['observations.npz','evaluator.npz']),(new,['observations.npz','evaluator.npz','selected.json'])]:
        for name in names:bind(folder/name,read(folder/'receipt.json')['files'][name])
    for name,h in read(cache/'receipt.json')['files'].items():bind(cache/name,h)
    for name in ['BODY_RANK.pt','predictions.npz','batches.npy']:bind(rank/name,read(rank/'receipt.json')['outputs'][name])
    bind(prior/'QUERY-initial.pt',read(prior/'receipt.json')['outputs']['QUERY-initial.pt'])
    norm=work/'mz9-source-supervision-20260910/run-v1/normalization.npz';bind(norm)
    p11folder=work/'mz11-selective-addition-20260910/run-v1'
    p13folder=work/'mz13-training-coverage-20260910/features-v1'
    bind(p11folder/'predictions.npz',read(p11folder/'receipt.json')['outputs']['predictions.npz'])
    bind(p13folder/'features.npz',read(p13folder/'receipt.json')['files']['features.npz'])
    d,nd=load_npz(old/'observations.npz'),load_npz(new/'observations.npz')
    truth=np.concatenate([load_npz(old/'evaluator.npz')['truth'],load_npz(new/'evaluator.npz')['truth']])
    ranges=np.concatenate([d['ranges'],nd['ranges']]);valid=np.concatenate([d['valid'],nd['valid']])
    reference=load_npz(rank/'predictions.npz');p11=load_npz(p11folder/'predictions.npz');p13=load_npz(p13folder/'features.npz')
    base=np.full((14200,4),np.nan,np.float64)
    base[np.flatnonzero(d['role']=='TRAIN_ONLY')]=p11['TRAIN/baseline']
    base[3700:8700]=p13['baseline'][p13['dataset']=='relation'];base[8700:11200]=p13['baseline'][p13['dataset']=='distance']
    cohorts=dict(DEV=np.flatnonzero(d['role']=='DEV_ONLY'),clean=np.arange(3500,3700),stress=np.arange(3500,3700),
        relation10000=np.arange(11200,13200),distance5000=np.arange(13200,14200))
    for name,ids in cohorts.items():
        if name!='stress':base[ids]=reference['BODY_RANK/'+name+'/baseline']
        np.testing.assert_array_equal(truth[ids],reference['BODY_RANK/'+name+'/truth'])
    batches=np.load(rank/'batches.npy');ids=np.load(cache/'ids.npy')
    trainids=np.unique(batches);np.testing.assert_array_equal(ids,np.unique(np.r_[batches.flatten(),*cohorts.values()]))
    assert np.isfinite(base[ids]).all()
    lookup=np.full(14200,-1,int);lookup[ids]=np.arange(len(ids));np.save(output/'batches.npy',batches)
    n=load_npz(norm);maps=np.load(cache/'dense_HIGH_DETAIL.npy',mmap_mode='r')
    torch.set_num_threads(1);torch.manual_seed(122);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    model=make_model(torch.load(prior/'QUERY-initial.pt',weights_only=True)).cuda().eval()
    model.load_state_dict(torch.load(rank/'BODY_RANK.pt',weights_only=True));model.requires_grad_(False)
    write(output/'start.json',dict(status='STARTED',inputs=inputs,steps=1200,train_draws=int(batches.size),unique_train=len(trainids),
        code_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('mz22_evidence_arbiter.py')]},
        protocol_sha256=sha(Path(__file__).with_name('MZ22_EVIDENCE_ARBITRATION_PROTOCOL_20260910.md'))))
    def extract(ii,stress=False,wrong=False,baseline_override=None):
        parts=[];sups=[];raws=[]
        with torch.inference_mode():
            for begin in range(0,len(ii),16):
                batch=ii[begin:begin+16];x=torch.from_numpy((np.array(maps[lookup[batch]])-n['mean'])/n['std']).cuda()
                rr=d['stress_ranges'][batch-3500] if stress else ranges[batch]
                vv=d['stress_valid'][batch-3500] if stress else valid[batch]
                r=torch.from_numpy(rr.astype(np.float32)).cuda();v=torch.from_numpy(vv).cuda()
                out=model.inspect(x,r,v,wrong)
                currentbase=base[batch] if baseline_override is None else baseline_override[begin:begin+16]
                b=torch.from_numpy(currentbase.astype(np.float32)).cuda()
                parts.append(observable_features(out,b,r,v,model.rays).cpu().numpy());sups.append(out['support'].cpu().numpy());raws.append(out['logits'].cpu().numpy())
        return dict(features=np.concatenate(parts),support=np.concatenate(sups),raw=np.concatenate(raws))
    allnormal=extract(ids);arrays={};evals={}
    for name,ii in cohorts.items():
        for wrong in ([False] if name=='DEV' else [False,True]):
            key=name+('_wrong' if wrong else '')
            ref='BODY_RANK/'+key+'/'
            a=extract(ii,name=='stress',wrong,reference[ref+'baseline']) if name=='stress' or wrong else {k:v[lookup[ii]] for k,v in allnormal.items()}
            np.testing.assert_allclose(a['raw'],reference[ref+'raw'],atol=2e-5,rtol=1e-5)
            np.testing.assert_array_equal(a['support'],reference[ref+'support'])
            a.update(baseline=reference[ref+'baseline'],truth=truth[ii],mz20=reference[ref+'candidate'])
            evals[key]=a
    model.cpu();del maps
    x=allnormal['features'];normal=x[lookup[trainids]]
    mean=normal.mean(0,keepdims=True);std=normal.std(0,keepdims=True).clip(.1)
    np.savez(output/'normalization.npz',mean=mean,std=std)
    features=torch.from_numpy((x-mean)/std).cuda();bb=torch.from_numpy(base[ids].astype(np.float32)).cuda()
    ss=torch.from_numpy(allnormal['support']).cuda();yy=torch.from_numpy(truth[ids]).cuda()
    arbiter=EvidenceArbiter(x.shape[-1]).cuda();torch.save(arbiter.cpu().state_dict(),output/'initial.pt');arbiter.cuda()
    with torch.no_grad():assert torch.equal(arbiter(features,bb,ss),bb)
    opt=torch.optim.Adam(arbiter.parameters(),lr=.001);history=[];fitstart=time.perf_counter()
    for step,batch in enumerate(batches):
        ix=lookup[batch];z=arbiter(features[ix],bb[ix],ss[ix]);mask=ss[ix]
        loss=(F.binary_cross_entropy_with_logits(z,yy[ix].float(),reduction='none')*mask).sum()/mask.sum().clamp_min(1)
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
        if step%200==0 or step==1199:
            row=dict(step=step+1,loss=float(loss.detach()));history.append(row);print(row,flush=True)
            write(output/'progress.json',row)
    torch.cuda.synchronize();fitseconds=time.perf_counter()-fitstart
    arbiter.eval();torch.save(arbiter.cpu().state_dict(),output/'arbiter.pt');arbiter.cuda()
    for a in evals.values():
        with torch.inference_mode():
            a['score']=arbiter(torch.from_numpy((a['features']-mean)/std).cuda(),torch.from_numpy(a['baseline'].astype(np.float32)).cuda(),torch.from_numpy(a['support']).cuda()).cpu().numpy()
    dev=evals['DEV'];budget=((dev['baseline']>=0)&dev['support']&~dev['truth']).sum(0)
    cut=thresholds_at_budget(dev['score'],dev['support'],dev['truth'],budget);np.save(output/'cutoff.npy',cut)
    result=dict(metrics={},thin={},fit=dict(steps=1200,parameters=sum(p.numel() for p in arbiter.parameters()),seconds=fitseconds,history=history),cutoff=cut.tolist(),groups={})
    for name,a in evals.items():
        z=np.where(a['support'],a['score'].astype(np.float64)-cut,a['baseline']);a['candidate']=z
        p,b,t=z>=0,a['baseline']>=0,a['truth']
        changed=(p!=b);m=dict(baseline=metrics(a['baseline'],t),candidate=metrics(z,t),mz20=metrics(a['mz20'],t),
            gained_tp=(changed&p&t).sum(0).tolist(),lost_tp=(changed&~p&t).sum(0).tolist(),
            added_fp=(changed&p&~t).sum(0).tolist(),removed_fp=(changed&~p&~t).sum(0).tolist(),
            unsupported_changed=int((changed&~a['support']).sum()))
        result['metrics'][name]=m;assert m['unsupported_changed']==0
        for k,v in a.items():arrays[name+'/'+k]=v
        if name.startswith(('clean','stress')):result['thin'][name]=metrics(z[100:125],t[100:125])
        print(name,m,flush=True)
    rows=read(new/'selected.json')
    for name in ['relation10000','distance5000']:
        rr=[r for r in rows if r['dataset']==name];a=evals[name];result['groups'][name]={}
        for field in ['family','group','site']:
            units={}
            for i,r in enumerate(rr):units.setdefault(r[field],[]).append(i)
            result['groups'][name][field]={u:dict(frames=len(ii),baseline=metrics(a['baseline'][ii],a['truth'][ii]),candidate=metrics(a['candidate'][ii],a['truth'][ii])) for u,ii in units.items()}
    m=result['metrics'];names=list(cohorts)
    gates=dict(no_fp_increase=all(all(m[n]['candidate']['fp'][q]<=m[n]['baseline']['fp'][q] for q in range(4)) for n in names),
        near_retained=all(all(m[n]['candidate']['tp'][q]>=m[n]['baseline']['tp'][q] for q in [0,2]) for n in names),
        placement_far_gain=all(sum(m[n]['candidate']['tp'][1::2])>sum(m[n]['baseline']['tp'][1::2]) for n in ['relation10000','distance5000']),
        pole_clean=sum(result['thin']['clean']['tp'][1::2])>=48,pole_stress=sum(result['thin']['stress']['tp'][1::2])>=48)
    gates['useful_effect']=all(gates.values());result['gates']=gates
    assert all(m['DEV']['candidate']['fp'][q]<=m['DEV']['baseline']['fp'][q] for q in range(4))
    np.savez_compressed(output/'predictions.npz',**arrays);write(output/'result.json',result)
    write(output/'receipt.json',dict(status='PASS',backend='CUDA',device=torch.cuda.get_device_name(),seconds=time.perf_counter()-started,
        outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}));print('PASS',gates,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)
