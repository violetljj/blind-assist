"""One learned availability veto with frozen MZ20 scores and cutoffs."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from mz5_ensemble_readout import read,write,sha,load_npz
from mz16_detail_readout import make_model
from mz15_train import balanced_local
from mz23_availability import AngularAvailability,restrict_candidates
from mz8_train import metrics


def main(root,output):
    work=root/'artifacts.local/work';oracle=work/'mz23-availability-20260910/oracle-v1'
    oracle_receipt=read(oracle/'receipt.json')
    assert oracle_receipt['status']=='PASS'
    assert sha(oracle/'result.json')==oracle_receipt['outputs']['result.json']
    oracle_result=read(oracle/'result.json')
    assert oracle_result['placement_far_before']==75
    assert oracle_result['placement_far_retained']>=68
    assert oracle_result['placement_added_fp_before']==8
    assert oracle_result['placement_added_fp_after']==0
    assert min(oracle_result['pole_retained'].values())>=48
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();inputs={}
    def bind(p,h=None):
        digest=sha(p);assert h is None or digest==h,str(p);inputs[str(p)]=digest
    old=work/'mz8-attribution-20260910/cache-v5';new=work/'mz15-shared-support-20260910/cache-v1'
    cache=work/'mz16-visual-detail-20260910/cache-v2';rank=work/'mz20-rank-objective-20260910/run-v1'
    prior=work/'mz15-shared-support-20260910/run-v1';labels=work/'mz9-source-supervision-20260910/labels-v1'
    for folder,names in [(old,['observations.npz','evaluator.npz']),(new,['observations.npz','evaluator.npz','selected.json'])]:
        for name in names:bind(folder/name,read(folder/'receipt.json')['files'][name])
    for name,h in read(cache/'receipt.json')['files'].items():bind(cache/name,h)
    for name in ['BODY_RANK.pt','BODY_RANK-cutoff.npy','predictions.npz','batches.npy']:bind(rank/name,read(rank/'receipt.json')['outputs'][name])
    bind(labels/'evaluator.npz',read(labels/'receipt.json')['labels_sha256'])
    bind(prior/'QUERY-initial.pt',read(prior/'receipt.json')['outputs']['QUERY-initial.pt'])
    norm=work/'mz9-source-supervision-20260910/run-v1/normalization.npz';bind(norm)
    for name in ['receipt.json','result.json']:bind(oracle/name)
    d,nd=load_npz(old/'observations.npz'),load_npz(new/'observations.npz')
    ol,nl=load_npz(labels/'evaluator.npz'),load_npz(new/'evaluator.npz')
    known=np.concatenate([ol['cell_known_counts']>0,nl['known']]);del ol,nl
    ranges=np.concatenate([d['ranges'],nd['ranges']]);valid=np.concatenate([d['valid'],nd['valid']])
    ref=load_npz(rank/'predictions.npz');cut=np.load(rank/'BODY_RANK-cutoff.npy');np.save(output/'cutoff.npy',cut)
    batches=np.load(rank/'batches.npy');np.save(output/'batches.npy',batches)
    ids=np.load(cache/'ids.npy');lookup=np.full(14200,-1,int);lookup[ids]=np.arange(len(ids))
    cohorts=dict(DEV=np.flatnonzero(d['role']=='DEV_ONLY'),clean=np.arange(3500,3700),stress=np.arange(3500,3700),relation10000=np.arange(11200,13200),distance5000=np.arange(13200,14200))
    np.testing.assert_array_equal(ids,np.unique(np.r_[batches.flatten(),*cohorts.values()]))
    n=load_npz(norm);maps=np.load(cache/'dense_HIGH_DETAIL.npy',mmap_mode='r')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;assert torch.cuda.is_available()
    frozen=make_model(torch.load(prior/'QUERY-initial.pt',weights_only=True)).cuda().eval()
    frozen.load_state_dict(torch.load(rank/'BODY_RANK.pt',weights_only=True));frozen.requires_grad_(False)
    torch.manual_seed(123);head=AngularAvailability(frozen.grid).cuda();torch.save(head.cpu().state_dict(),output/'initial.pt');head.cuda()
    write(output/'start.json',dict(status='STARTED',inputs=inputs,steps=1200,seed=123,unique_train=len(np.unique(batches)),draws=int(batches.size),
        code_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('mz23_availability.py')]},
        protocol_sha256=sha(Path(__file__).with_name('MZ23_AVAILABILITY_PROTOCOL_20260910.md'))))
    def get(ii,stress=False):
        assert (lookup[ii]>=0).all()
        x=torch.from_numpy((np.array(maps[lookup[ii]])-n['mean'])/n['std']).cuda()
        rr=d['stress_ranges'][ii-3500] if stress else ranges[ii];vv=d['stress_valid'][ii-3500] if stress else valid[ii]
        return x,torch.from_numpy(rr.astype(np.float32)).cuda(),torch.from_numpy(vv).cuda()
    head.train();opt=torch.optim.Adam(head.parameters(),lr=.001);history=[];fitstart=time.perf_counter()
    for step,ii in enumerate(batches):
        x,r,v=get(ii);a=head(x,r,v);target=torch.from_numpy(known[ii]).cuda()
        loss=balanced_local(a[...,None],target[...,None],torch.ones_like(target[...,None]))
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
        if step%200==0 or step==1199:
            row=dict(step=step+1,loss=float(loss.detach()));history.append(row);print(row,flush=True);write(output/'progress.json',row)
    torch.cuda.synchronize();fitseconds=time.perf_counter()-fitstart;head.eval();torch.save(head.cpu().state_dict(),output/'availability.pt');head.cuda()
    result=dict(metrics={},availability={},thin={},groups={},fit=dict(steps=1200,seconds=fitseconds,parameters=sum(p.numel() for p in head.parameters()),history=history));arrays={}
    for name,ii in cohorts.items():
        for wrong in ([False] if name=='DEV' else [False,True]):
            key=name+('_wrong' if wrong else '');parts=[]
            with torch.inference_mode():
                for begin in range(0,len(ii),16):
                    batch=ii[begin:begin+16];x,r,v=get(batch,name=='stress');o=frozen.inspect(x,r,v,wrong);a=head(x,r,v,wrong);z=restrict_candidates(o,a)
                    parts.append(dict(raw=z['logits'].cpu().numpy(),support=z['support'].cpu().numpy(),original_raw=o['logits'].cpu().numpy(),
                        original_support=o['support'].cpu().numpy(),availability=a.cpu().numpy()))
            a={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};prefix='BODY_RANK/'+key+'/'
            np.testing.assert_allclose(a['original_raw'],ref[prefix+'raw'],atol=2e-5,rtol=1e-5);np.testing.assert_array_equal(a['original_support'],ref[prefix+'support'])
            assert (a['support']<=a['original_support']).all();assert (a['raw'][a['support']]<=a['original_raw'][a['support']]+2e-5).all()
            b,t,previous=(ref[prefix+k] for k in ['baseline','truth','candidate'])
            margin=np.where(a['support'],a['raw'].astype(np.float64)-cut,-1e6)
            accept=(b<0)&a['support']&(margin>=0);candidate=np.where(accept,margin,b)
            assert not ((candidate>=0)&(previous<0)).any()
            a.update(baseline=b,truth=t,mz20=previous,margin=margin,accepted=accept,candidate=candidate,known=known[ii])
            result['metrics'][key]=dict(baseline=metrics(b,t),mz20=metrics(previous,t),candidate=metrics(candidate,t),added_tp=(accept&t).sum(0).tolist(),added_fp=(accept&~t).sum(0).tolist(),
                mz20_tp_lost=((previous>=0)&(candidate<0)&t).sum(0).tolist(),mz20_fp_removed=((previous>=0)&(candidate<0)&~t).sum(0).tolist())
            if not wrong:
                p=a['availability']>=0;k=known[ii];tp=int((p&k).sum());fp=int((p&~k).sum());fn=int((~p&k).sum())
                result['availability'][key]=dict(tp=tp,fp=fp,fn=fn,positive=int(k.sum()),negative=int((~k).sum()),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1))
            if name in ['clean','stress']:result['thin'][key]=metrics(candidate[100:125],t[100:125])
            for k,v in a.items():arrays[key+'/'+k]=v
            print(key,result['metrics'][key],flush=True)
    rows=read(new/'selected.json')
    for name in ['relation10000','distance5000']:
        rr=[r for r in rows if r['dataset']==name];result['groups'][name]={}
        for field in ['family','group','site']:
            units={}
            for i,r in enumerate(rr):units.setdefault(r[field],[]).append(i)
            result['groups'][name][field]={u:dict(frames=len(ix),baseline=metrics(arrays[name+'/baseline'][ix],arrays[name+'/truth'][ix]),candidate=metrics(arrays[name+'/candidate'][ix],arrays[name+'/truth'][ix])) for u,ix in units.items()}
    m=result['metrics'];retained=sum(sum(m[n]['added_tp'][1::2]) for n in ['relation10000','distance5000'])
    gates=dict(no_added_fp=all(sum(m[n]['added_fp'])==0 for n in cohorts),far_retention=retained>=68,
        placement_far_gain=all(sum(m[n]['added_tp'][1::2])>0 for n in ['relation10000','distance5000']),
        pole_clean=sum(result['thin']['clean']['tp'][1::2])>=48,pole_stress=sum(result['thin']['stress']['tp'][1::2])>=48,
        baseline_positives_retained=all(not ((arrays[n+'/baseline']>=0)&(arrays[n+'/candidate']<0)).any() for n in cohorts))
    gates['useful_effect']=all(gates.values());result.update(gates=gates,retained_far_additions=retained)
    np.savez_compressed(output/'predictions.npz',**arrays);write(output/'result.json',result)
    write(output/'receipt.json',dict(status='PASS',backend='CUDA',device=torch.cuda.get_device_name(),seconds=time.perf_counter()-started,outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}));print('PASS',gates,retained,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.root,a.output)
