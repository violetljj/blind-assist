"""Three bounded equal-new-exposure fits and frozen DEV threshold comparison."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from mz5_ensemble_readout import CompactEnsemble,read,write,sha,load_npz
from mz8_train import metrics,thresholds_at_budget
from mz9_source_readout import SourceReadout


def main(root,cache,supervision,output):
    output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    cr=read(cache/'receipt.json');sr=read(supervision/'receipt.json')
    assert cr['status']==sr['status']=='PASS' and sha(cache/'receipt.json')==sr['cache_receipt_sha256']
    assert sha(supervision/'evaluator.npz')==sr['labels_sha256']
    for name,digest in cr['files'].items():assert sha(cache/name)==digest
    torch.set_num_threads(1);torch.manual_seed(109);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.is_available()
    d=load_npz(cache/'observations.npz');e=load_npz(cache/'evaluator.npz');lab=load_npz(supervision/'evaluator.npz')
    train=np.flatnonzero(d['role']=='TRAIN_ONLY');dev=np.flatnonzero(d['role']=='DEV_ONLY');seq=np.arange(3500,3700)
    old_run=root/'artifacts.local/work/mz8-attribution-20260910/run-v1'
    assert sha(old_run/'normalization.npz')==read(old_run/'receipt.json')['outputs']['normalization.npz']
    n=load_npz(old_run/'normalization.npz');maps=np.load(cache/'dense.npy',mmap_mode='r')
    np.savez(output/'normalization.npz',**n)
    source=SourceReadout().cuda();source_init=copy.deepcopy(source.state_dict())
    tokens=[]
    with torch.inference_mode():
        for begin in range(0,3700,32):
            x=torch.from_numpy((np.array(maps[begin:begin+32])-n['mean'])/n['std']).cuda()
            tokens.append(source.sample_visual(x).cpu())
    tokens=torch.cat(tokens).cuda()
    ranges=torch.from_numpy(d['ranges']).cuda();valid=torch.from_numpy(d['valid']).cuda()
    visual=torch.from_numpy(d['visual']).cuda();tof=torch.cat([ranges.flatten(1)/4,valid.float().flatten(1)],1)
    truth=torch.from_numpy(e['truth'].astype(np.float32)).cuda()
    local=torch.from_numpy(lab['query_counts']>0).cuda()
    known=torch.from_numpy(lab['cell_known_counts']>0).cuda()[:,:,None,:,None]&valid[:,:,:,None,None]
    checkpoint=root/'artifacts.local/work/mz5-fixed-ensemble-20260910/compact-v1/compact.pt'
    assert sha(checkpoint)=='ccbfd6817bfec226f0db74e7ff4b22ce4f153255a9102d5c8684de91197486e1'
    fixed=CompactEnsemble.from_checkpoint(checkpoint).cuda()
    with torch.inference_mode():baseline=fixed(visual,tof).cpu().numpy()
    previous=load_npz(root/'artifacts.local/work/mz6-short-sequence-20260910/evaluation-v1/predictions.npz')
    np.testing.assert_array_equal(baseline[seq]>=0,previous['clean/CURRENT']>=0)
    with torch.no_grad():
        _,_,eligible=source.eligibility(ranges[seq],valid[seq])
        supported=(eligible&local[seq]).flatten(1,3).any(1).cpu().numpy()
    write(output/'supervision-diagnostic.json',dict(
        sequence_positive_exact_source_support=(supported&e['truth'][seq]).sum(0).tolist(),
        thin_positive_exact_source_support=(supported[100:125]&e['truth'][3600:3625]).sum(0).tolist(),
        empty_pole_query_contributors=lab['query_counts'][3625:3650].sum((0,1,2,3)).tolist(),
        source_labels='actual selected-bin pixel membership; cells may have multiple contributors'))
    rng=np.random.default_rng(109);batches=np.stack([np.r_[rng.choice(train,8),rng.choice(seq,8)] for _ in range(1200)])
    np.save(output/'batches.npy',batches)
    write(output/'start-receipt.json',dict(status='STARTED',steps_per_arm=1200,seed=109,
        source_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('mz9_source_readout.py'),Path(__file__).with_name('MZ9_SOURCE_SUPERVISION_PROTOCOL_20260910.md')]},
        batches_sha256=sha(output/'batches.npy')))
    preds={'FROZEN':baseline[seq]};details={};raws={};fits={}
    budget=((baseline[dev]>=0)&~e['truth'][dev]).sum(0)
    for arm in ['MZ5_ADAPT','SOURCE_RGB','SOURCE_NO_RGB']:
        arm_start=time.perf_counter()
        if arm=='MZ5_ADAPT':model=CompactEnsemble.from_checkpoint(checkpoint).cuda().requires_grad_(True)
        else:
            model=SourceReadout(no_rgb=arm=='SOURCE_NO_RGB').cuda();model.load_state_dict(source_init)
        model.train();opt=torch.optim.Adam(model.parameters(),lr=.001);history=[]
        for step,batch in enumerate(batches):
            if arm=='MZ5_ADAPT':
                r,t=model.branches(visual[batch],tof[batch])
                loss=.5*(F.binary_cross_entropy_with_logits(r,truth[batch])+F.binary_cross_entropy_with_logits(t,truth[batch]))
            else:
                out=model.forward_tokens(tokens[batch],ranges[batch],valid[batch],return_details=True)
                query=F.binary_cross_entropy_with_logits(out['logits'],truth[batch])
                local_loss=F.binary_cross_entropy_with_logits(out['candidate_logits'],local[batch].float(),reduction='none')
                pos=known[batch]&local[batch];neg=known[batch]&~local[batch]
                aux=.5*((local_loss*pos).sum()/pos.sum().clamp_min(1)+(local_loss*neg).sum()/neg.sum().clamp_min(1))
                loss=query+.25*aux
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();opt.step()
            if step%300==0 or step==1199:
                row=dict(step=step+1,loss=float(loss.detach()));history.append(row);print(arm,row,flush=True)
        model.eval();torch.save(model.cpu().state_dict(),output/f'{arm}.pt');model.cuda()
        def infer(ids,wrong=False,stress=False):
            outputs=[];supports=[]
            with torch.inference_mode():
                for begin in range(0,len(ids),32):
                    batch=ids[begin:begin+32];r=ranges[batch];v=valid[batch]
                    if stress:
                        r=torch.from_numpy(d['stress_ranges'][batch-3500].astype(np.float32)).cuda()
                        v=torch.from_numpy(d['stress_valid'][batch-3500]).cuda()
                    if arm=='MZ5_ADAPT':
                        packed=torch.cat([r.flatten(1)/4,v.float().flatten(1)],1)
                        z=model(visual[batch],packed);support=torch.ones_like(z,dtype=torch.bool)
                    else:
                        out=model.forward_tokens(tokens[batch],r,v,wrong_zone=wrong,return_details=True)
                        z,support=out['logits'],out['support']
                    outputs.append(z.cpu().numpy());supports.append(support.cpu().numpy())
            return np.concatenate(outputs),np.concatenate(supports)
        z,s=infer(dev);threshold=thresholds_at_budget(z,s,e['truth'][dev],budget)
        fitted=metrics(np.where(s,z-threshold,-1e6),e['truth'][dev])
        details[arm]=dict(threshold=threshold.tolist(),dev=fitted,fp_budget=budget.tolist())
        for condition in ['clean','stress']:
            for wrong in ([False,True] if arm=='SOURCE_RGB' else [False]):
                name=arm+'/'+condition+('/wrong' if wrong else '')
                z,s=infer(seq,wrong,condition=='stress');preds[name]=np.where(s,z-threshold,-1e6)
                raws[name+'/logits']=z;raws[name+'/support']=s
        fits[arm]=dict(steps=1200,seconds=time.perf_counter()-arm_start,history=history,
            parameters=sum(p.numel() for p in model.parameters()),checkpoint_sha256=sha(output/f'{arm}.pt'))
        write(output/'fit-progress.json',fits)
    results={}
    for arm,z in preds.items():
        results[arm]={}
        for clip in ['ALL',*dict.fromkeys(d['clip'].tolist())]:
            mask=np.ones(200,dtype=bool) if clip=='ALL' else d['clip']==clip
            results[arm][clip]=metrics(z[mask],e['truth'][seq][mask])
    base=results['FROZEN']['ALL'];base_dev=metrics(baseline[dev],e['truth'][dev]);gates={}
    for arm in fits:
        c=results[arm+'/clean']['ALL'];dv=details[arm]['dev'];thin=results[arm+'/clean']['thin_pole_background_target']
        gates[arm]=dict(thin_recovery=sum(thin['tp'][1::2])>0,far_recovery=sum(c['tp'][1::2])>sum(base['tp'][1::2]),
            no_fp_increase=all(x<=y for x,y in zip(c['fp'],base['fp'])),no_near_loss=all(c['tp'][q]>=base['tp'][q] for q in [0,2]),
            dev_near_preserved=all(dv['tp'][q]>=base_dev['tp'][q] for q in [0,2]),dev_far_preserved=sum(dv['tp'][1::2])>=sum(base_dev['tp'][1::2]))
    qualifying=[a for a,g in gates.items() if all(g.values())]
    selected=max(qualifying,key=lambda a:(results[a+'/clean']['ALL']['exact'],sum(results[a+'/clean']['ALL']['tp']),a=='MZ5_ADAPT',a=='SOURCE_NO_RGB')) if qualifying else None
    write(output/'result.json',dict(metrics=results,dev=details,baseline_dev=base_dev,gates=gates,confirmation_candidate=selected))
    np.savez_compressed(output/'predictions.npz',**preds,truth=e['truth'][seq],clip=d['clip'])
    np.savez_compressed(output/'raw-support.npz',**raws)
    torch.cuda.synchronize()
    write(output/'receipt.json',dict(status='PASS',backend='CUDA',device=torch.cuda.get_device_name(),seconds=time.perf_counter()-start,
        cache_receipt_sha256=sha(cache/'receipt.json'),labels_receipt_sha256=sha(supervision/'receipt.json'),
        outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print('SUMMARY',{a:v['ALL'] for a,v in results.items()},'DEV',details,'GATES',gates,'SELECTED',selected,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','cache','supervision','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();main(a.root,a.cache,a.supervision,a.output)
