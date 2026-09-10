"""One bounded angular-attribution fit, DEV thresholds, and causal control."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from mz5_ensemble_readout import CompactEnsemble,read,write,sha,load_npz
from mz8_attribution import AngularReturnReadout


def metrics(z,truth):
    p=z>=0
    return dict(exact=int((p==truth).all(1).sum()),tp=(p&truth).sum(0).tolist(),
        fp=(p&~truth).sum(0).tolist(),positives=truth.sum(0).tolist())


def thresholds_at_budget(z,support,truth,budget):
    thresholds=[]
    for q in range(4):
        values=z[support[:,q],q]
        candidates=np.r_[np.unique(values),np.nextafter(values.max(),np.inf)] if len(values) else [0.]
        scores=[]
        for threshold in candidates:
            p=(z[:,q]>=threshold)&support[:,q]
            tp=int((p&truth[:,q]).sum());fp=int((p&~truth[:,q]).sum())
            if fp<=budget[q]:scores.append((tp,-fp,float(threshold)))
        thresholds.append(max(scores)[2])
    return np.array(thresholds)


def main(root,cache,output):
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    receipt=read(cache/'receipt.json');assert receipt['status']=='PASS'
    for name,digest in receipt['files'].items():assert sha(cache/name)==digest
    assert torch.cuda.is_available();torch.set_num_threads(1);torch.manual_seed(108);np.random.seed(108)
    torch.backends.cuda.matmul.allow_tf32=False
    maps=np.load(cache/'dense.npy',mmap_mode='r');data=load_npz(cache/'observations.npz');ev=load_npz(cache/'evaluator.npz')
    train=np.flatnonzero(data['role']=='TRAIN_ONLY');dev=np.flatnonzero(data['role']=='DEV_ONLY');seq=np.arange(3500,3700)
    assert len(train)==2500 and len(dev)==1000
    mean=np.asarray(maps[train]).mean((0,2,3),keepdims=True);std=np.asarray(maps[train]).std((0,2,3),keepdims=True).clip(.01)
    np.savez(output/'normalization.npz',mean=mean,std=std)
    model=AngularReturnReadout().cuda()
    write(output/'start-receipt.json',dict(status='STARTED',seed=108,steps=1200,backend='CUDA',
        source_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('mz8_attribution.py'),Path(__file__).with_name('MZ8_ATTRIBUTION_PROTOCOL_20260910.md')]}))
    grid=load_npz(cache/'grid.npz')['grid'];np.testing.assert_allclose(model.grid.cpu().numpy(),grid,atol=1e-7)
    # Cache normalized tokens once; model-visible tensors never contain sampled native depth.
    tokens=[]
    with torch.inference_mode():
        for begin in range(0,3700,32):
            x=torch.from_numpy((np.array(maps[begin:begin+32])-mean)/std).cuda()
            tokens.append(model.sample_visual(x).cpu())
    tokens=torch.cat(tokens).cuda();ranges=torch.from_numpy(data['ranges']).cuda();valid=torch.from_numpy(data['valid']).cuda()
    labels=torch.from_numpy(ev['truth'].astype(np.float32)).cuda()
    sampled=torch.from_numpy(ev['sampled_radial']).cuda()
    compatible=(sampled[:,:,None,:]-ranges[:,:,:,None]).abs()<=.10
    known=torch.isfinite(sampled[:,:,None,:])&valid[:,:,:,None]
    # Descriptive local response on the targeted training clip, not an oracle feature input.
    thin_delta=tokens[3600:3625]-tokens[3625:3650]
    with torch.no_grad():
        _,_,eligible=model.eligibility(ranges[seq],valid[seq])
        possible=eligible.flatten(1,3).any(1).cpu().numpy()
        compatible_support=(eligible&compatible[seq][...,None]).flatten(1,3).any(1).cpu().numpy()
    write(output/'representation-diagnostic.json',dict(
        scope='Consumed thin-pole25 paired frames; standardized frozen local feature differences, not semantic separability',
        pair_rms=thin_delta.square().mean((1,2,3)).sqrt().cpu().tolist(),
        pair_max_token_l2=thin_delta.norm(dim=-1).flatten(1).max(1).values.cpu().tolist(),
        backbone_spatial_size=[18,32],frozen_channels=64,
        sequence_positive_geometry_support=(possible&ev['truth'][seq]).sum(0).tolist(),
        sequence_positive_native_compatible_support=(compatible_support&ev['truth'][seq]).sum(0).tolist(),
        thin_positive_geometry_support=(possible[100:125]&ev['truth'][seq][100:125]).sum(0).tolist(),
        thin_positive_native_compatible_support=(compatible_support[100:125]&ev['truth'][seq][100:125]).sum(0).tolist()))
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    generator=np.random.default_rng(108);history=[]
    for step in range(1200):
        batch=np.r_[generator.choice(train,8),generator.choice(seq,8)]
        out=model.forward_tokens(tokens[batch],ranges[batch],valid[batch],return_details=True)
        query_loss=F.binary_cross_entropy_with_logits(out['logits'],labels[batch])
        local_loss=F.binary_cross_entropy_with_logits(out['candidate_logits'],compatible[batch].float(),reduction='none')
        pos=known[batch]&compatible[batch];neg=known[batch]&~compatible[batch]
        aux=.5*((local_loss*pos).sum()/pos.sum().clamp_min(1)+(local_loss*neg).sum()/neg.sum().clamp_min(1))
        loss=query_loss+.25*aux
        assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();opt.step()
        if step%100==0 or step==1199:
            row=dict(step=step+1,loss=float(loss),query=float(query_loss),aux=float(aux));history.append(row);print(row,flush=True)
    torch.save(model.cpu().state_dict(),output/'model.pt');model.cuda().eval()
    write(output/'fit.json',dict(steps=1200,seed=108,history=history,checkpoint_sha256=sha(output/'model.pt'),
        parameter_count=sum(p.numel() for p in model.parameters()),training_roles=['TRAIN_ONLY','MZ6_TRAIN']))
    def infer(ids,wrong=False,stress=False):
        outputs=[];supports=[]
        with torch.inference_mode():
            for begin in range(0,len(ids),32):
                batch=ids[begin:begin+32]
                r=ranges[batch];v=valid[batch]
                if stress:
                    r=torch.from_numpy(data['stress_ranges'][batch-3500].astype(np.float32)).cuda()
                    v=torch.from_numpy(data['stress_valid'][batch-3500]).cuda()
                out=model.forward_tokens(tokens[batch],r,v,wrong_zone=wrong,return_details=True)
                outputs.append(out['logits'].cpu().numpy());supports.append(out['support'].cpu().numpy())
        return np.concatenate(outputs),np.concatenate(supports)
    checkpoint=root/'artifacts.local/work/mz5-fixed-ensemble-20260910/compact-v1/compact.pt'
    assert sha(checkpoint)=='ccbfd6817bfec226f0db74e7ff4b22ce4f153255a9102d5c8684de91197486e1'
    fixed=CompactEnsemble.from_checkpoint(checkpoint).cuda()
    with torch.inference_mode():
        tof=torch.cat([ranges.flatten(1)/4,valid.float().flatten(1)],1)
        baseline=fixed(torch.from_numpy(data['visual']).cuda(),tof).cpu().numpy()
    previous=load_npz(root/'artifacts.local/work/mz6-short-sequence-20260910/evaluation-v1/predictions.npz')
    np.testing.assert_array_equal(baseline[seq]>=0,previous['clean/CURRENT']>=0)
    baseline_max_error=float(np.abs(baseline[seq]-previous['clean/CURRENT']).max())
    dz,ds=infer(dev);budget=((baseline[dev]>=0)&~ev['truth'][dev]).sum(0)
    thresholds=thresholds_at_budget(dz,ds,ev['truth'][dev],budget)
    write(output/'thresholds.json',dict(thresholds=thresholds.tolist(),fitting_role='DEV_ONLY',
        fp_budget=budget.tolist(),baseline=metrics(baseline[dev],ev['truth'][dev]),
        fitted=metrics(np.where(ds,dz-thresholds,-np.inf),ev['truth'][dev])))
    preds={'baseline':baseline[seq]};raws={};supports={};results={}
    for condition in ['clean','stress']:
        for wrong in [False,True]:
            name=condition+('/wrong_zone' if wrong else '/correct')
            z,s=infer(seq,wrong,condition=='stress');preds[name]=np.where(s,z-thresholds,-1e6);raws[name]=z;supports[name]=s
    for name,z in preds.items():
        results[name]={}
        for clip in ['ALL',*dict.fromkeys(data['clip'].tolist())]:
            mask=np.ones(200,dtype=bool) if clip=='ALL' else data['clip']==clip
            results[name][clip]=metrics(z[mask],ev['truth'][seq][mask])
    b=results['baseline']['ALL'];c=results['clean/correct']['ALL'];w=results['clean/wrong_zone']['ALL']
    thin='thin_pole_background_target'
    thin_tp=sum(results['clean/correct'][thin]['tp'][1::2]);thin_wrong=sum(results['clean/wrong_zone'][thin]['tp'][1::2])
    gate=dict(thin_recovered=thin_tp>0,far_tp_improved=sum(c['tp'][1::2])>sum(b['tp'][1::2]),
        fp_budget_preserved=all(x<=y for x,y in zip(c['fp'],b['fp'])),
        correspondence_use=thin_wrong<thin_tp)
    write(output/'result.json',dict(metrics=results,gate=gate,retain_for_confirmation=all(gate.values())))
    np.savez_compressed(output/'predictions.npz',**preds,truth=ev['truth'][seq],clip=data['clip'])
    np.savez_compressed(output/'raw-support.npz',**{k+'/logits':v for k,v in raws.items()},**{k+'/support':v for k,v in supports.items()})
    torch.cuda.synchronize()
    write(output/'receipt.json',dict(status='PASS',backend='CUDA',device=torch.cuda.get_device_name(),seconds=time.perf_counter()-started,
        baseline_flag_parity=800,baseline_max_error=baseline_max_error,
        cache_receipt_sha256=sha(cache/'receipt.json'),source_sha256={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('mz8_attribution.py'),Path(__file__).with_name('MZ8_ATTRIBUTION_PROTOCOL_20260910.md')]},
        outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}))
    print('RESULT',results,'GATE',gate,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['root','cache','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();main(a.root,a.cache,a.output)
