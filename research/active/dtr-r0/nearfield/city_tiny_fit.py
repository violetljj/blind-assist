"""Two fixed tiny-TRAIN fits and loss-gradient diagnostics; no held-out access."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
from city_data import CitySupervisedDataset, pixel_support_bce
from city_pilot_metrics import evaluate
from decoupled_model import DecoupledModel
from representation_model import pretrained_manifest
from city_score_separation import curve

INITIAL_SHA='0c14179486102993508dd2385a3b6f9e17d51a6d53e657c0240325e5f78d5c7b'
SITE='TRAIN-north-x-76'
FAMILIES=('crossbar','cabinet','oblique_rod','hanging_sign')
MODULES=('backbone','deep_projection','detail','support','near')
LOG_STEPS=(1,200,500,1000,2000)
GRAD_STEPS=(1,200,1000,2000)


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def state_sha(items):
    h=hashlib.sha256()
    for name,t in items:
        h.update(name.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def select_indices(records):
    groups=[]
    for family in FAMILIES:
        eligible=sorted({r['group_id'] for r in records if r['source_site_id']==SITE and r['family']==family})
        if len(eligible)<2:raise ValueError('Insufficient first-site family groups')
        groups.extend(eligible[:2])
    indices=sorted(r['sample_index'] for r in records if r['group_id'] in groups)
    selected=sorted([r for r in records if r['sample_index'] in indices],key=lambda r:r['sample_index'])
    if len(indices)!=32 or len(set(indices))!=32 or Counter(r['group_id'] for r in selected)!=Counter({g:4 for g in groups}):
        raise ValueError('Require eight complete unique quartets')
    for g in groups:
        if {r['geometry_relation'] for r in selected if r['group_id']==g}!={'CLEAR','BODY_ONLY','HEAD_ONLY','BOTH'}:
            raise ValueError('Missing quartet relation')
    return indices,selected


def gradient_report(loss, model):
    named=list(model.named_parameters())
    if any(name.split('.')[0] not in MODULES for name,_ in named):raise ValueError('Unclassified parameter group')
    active=[(name,p) for name,p in named if p.requires_grad]
    gradients=torch.autograd.grad(loss,[p for _,p in active],retain_graph=True,allow_unused=True)
    mapping={name:g for (name,_),g in zip(active,gradients)}
    result={}
    for module in MODULES:
        params=[(name,p) for name,p in named if name.split('.')[0]==module]
        gs=[mapping[name] for name,p in params if p.requires_grad and mapping[name] is not None]
        if gs and not bool(torch.stack([torch.isfinite(g).all() for g in gs]).all()):raise RuntimeError('Nonfinite diagnostic gradient')
        result[module]=dict(parameters=sum(p.numel() for _,p in params),parameter_tensors=len(params),
            frozen_tensors=sum(not p.requires_grad for _,p in params),
            none_tensors=sum(p.requires_grad and mapping[name] is None for name,p in params),
            zero_tensors=sum(int(torch.count_nonzero(g))==0 for g in gs),
            zero_elements=sum(int((g==0).sum()) for g in gs),
            gradient_elements=sum(g.numel() for g in gs),
            l2_norm=float(torch.stack([g.double().square().sum() for g in gs]).sum().sqrt()) if gs else None)
    return result


def run(a):
    started=time.perf_counter();out=a.output.resolve()
    artifacts=(Path(__file__).resolve().parents[4]/'artifacts.local').resolve()
    if out.exists() or not out.is_relative_to(artifacts) or out==artifacts:raise ValueError('Fresh canonical artifact output required')
    if sha(a.initial)!=INITIAL_SHA:raise ValueError('Original G13 initialization mismatch')
    manifest=read(a.cache/'manifest.json')
    if manifest.get('role')!='TRAIN_ONLY' or manifest['admission']['status']!='PASS':raise ValueError('Admitted new TRAIN384 required')
    dataset=CitySupervisedDataset(a.cache,'train')
    if len(dataset)!=384:raise ValueError('Only new TRAIN384 supported')
    groups_path=a.cache/'evaluator/groups.json';records=read(groups_path)
    indices,selected=select_indices(records)
    if dataset.ids!=list(range(384)):raise ValueError('Unexpected cache sample order')
    batch=[dataset[i] for i in indices]
    y=torch.stack([s['near'] for s in batch]);m=torch.stack([s['support'] for s in batch])
    if not torch.all((y==0)|(y==1)) or not torch.equal(y.sum(0),torch.tensor([16.,16.])):raise ValueError('Actual native near must be16+/16- per head')
    if not torch.equal((m==1).any(dim=(2,3)),y.bool()):raise ValueError('Tiny near/support positive mismatch')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    out.mkdir(parents=True)
    steps={};results={}
    try:
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        initial=torch.load(a.initial,map_location='cpu',weights_only=True)
        pretrained=pretrained_manifest(a.pretrained)
        protocol=Path(__file__).with_name('CITY_TINY_FIT_20260908.md')
        paths=[a.initial,a.cache/'manifest.json',a.cache/'supervision/train.json',groups_path,protocol]
        labels=read(a.cache/'supervision/train.json')
        paths.extend(a.cache/p['path'] for p in (manifest['partitions']['train']['rgb'],labels['near'],labels['support']))
        source_names=['city_tiny_fit.py','city_data.py','city_pilot_metrics.py','city_score_separation.py','decoupled_model.py','representation_model.py','whisker_model.py']
        inputs={str(p.resolve()):sha(p) for p in paths}
        write(out/'inputs.json',dict(input_sha256=inputs,source_sha256={n:sha(Path(__file__).with_name(n)) for n in source_names},
            initial_state_sha256=state_sha(initial.items()),pretrained=pretrained,selected_indices=indices,selected_samples=selected,
            protocol=dict(arms=['B','F'],steps_per_arm=2000,seed=17,batch='All32 every step',optimizer='AdamW',lr=1e-5,weight_decay=1e-4,
                loss='mean near BCE + .25 global class-balanced known-pixel support BCE',BN='All running buffers frozen',
                diagnostic_gradients='Separate near and weighted support autograd.grad before actual backward; no accumulation into .grad',
                diagnostic_per_head_support='Reported separately only; not used to replace global support loss',
                evaluation='Tiny TRAIN only after each fit, fixed .5, no threshold selection'),
            native_positive_per_head=y.sum(0).tolist(),native_negative_per_head=(32-y.sum(0)).tolist()))
        np.save(out/'batch_indices.npy',np.array(indices),allow_pickle=False)
        rgb=torch.stack([s['rgb'] for s in batch]).cuda();y=y.cuda();m=m.cuda()
        for arm in ('B','F'):
            random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
            model=DecoupledModel(a.pretrained).cuda();model.load_state_dict(initial,strict=True)
            if state_sha(model.state_dict().items())!=state_sha(initial.items()):raise ValueError('Model initialization differs')
            for p in model.backbone.parameters():p.requires_grad_(arm=='F')
            backbone=state_sha(model.backbone.state_dict().items());buffers=state_sha(model.named_buffers())
            optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=1e-4)
            model.train()
            for layer in model.modules():
                if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()
            history=[];diagnostics=[];steps[arm]=0;arm_started=time.perf_counter()
            for step in range(1,2001):
                n,s=model(rgb);near_loss=F.binary_cross_entropy_with_logits(n,y);support_loss=pixel_support_bce(s,m);loss=near_loss+.25*support_loss
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                if step in GRAD_STEPS:
                    diagnostics.append(dict(step=step,phase='PRE_UPDATE',near=gradient_report(near_loss,model),weighted_support=gradient_report(.25*support_loss,model)))
                    write(out/f'{arm}-gradients.json',diagnostics)
                if step in LOG_STEPS:
                    row=dict(step=step,phase='PRE_UPDATE',near_BCE=float(near_loss),near_BCE_per_head=[float(F.binary_cross_entropy_with_logits(n[:,h],y[:,h])) for h in range(2)],
                        support_BCE=float(support_loss),support_BCE_per_head=[float(pixel_support_bce(s[:,h:h+1],m[:,h:h+1])) for h in range(2)],total=float(loss))
                    history.append(row);write(out/f'{arm}-losses.json',history);print(json.dumps(dict(arm=arm,**row)),flush=True)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                if not grads or not bool(torch.stack([torch.isfinite(g).all() for g in grads]).all()):raise RuntimeError('Nonfinite actual gradient')
                optimizer.step();steps[arm]=step
            if state_sha(model.named_buffers())!=buffers:raise ValueError('BN/buffers changed')
            backbone_unchanged=state_sha(model.backbone.state_dict().items())==backbone
            if arm=='B' and not backbone_unchanged:raise ValueError('B backbone changed')
            checkpoint=out/f'{arm}-step2000.pt';torch.save(model.state_dict(),checkpoint)
            torch.save(dict(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),completed_steps=2000),out/f'{arm}-optimizer-state.pt')
            write(out/f'{arm}-fit-complete.json',dict(status='PASS',steps=2000,checkpoint_sha256=sha(checkpoint),seconds=time.perf_counter()-arm_started,BN_unchanged=True,backbone_unchanged=backbone_unchanged))
            model.eval()
            with torch.inference_mode():
                n,s=model(rgb);pn=n.sigmoid().cpu().numpy();ps=s.sigmoid().cpu().numpy()
                final_loss=dict(near=float(F.binary_cross_entropy_with_logits(n,y)),support=float(pixel_support_bce(s,m)))
            np.savez_compressed(out/f'{arm}-tiny-predictions.npz',near=pn,support=ps,sample_indices=np.array(indices))
            results[arm]=dict(metrics=evaluate(pn,ps,y.cpu().numpy(),m.cpu().numpy(),[r['group_id'] for r in selected]),final_losses=final_loss,
                checkpoint_sha256=sha(checkpoint),BN_unchanged=True,backbone_unchanged=backbone_unchanged,
                curves={head:curve(pn[:,h].astype(float).tolist(),y[:,h].cpu().numpy().astype(int).tolist()) for h,head in enumerate(('BODY','HEAD'))},
                phase='POST_2000_UPDATES')
            metrics=results[arm]['metrics']
            results[arm]['strong_tiny_fit']=metrics['group_joint']['all_correct_groups']==8 and all(
                metrics['heads'][h]['support']['positive_frame_mean_iou'] is not None and
                metrics['heads'][h]['support']['positive_frame_mean_iou']>=.5 for h in ('BODY','HEAD'))
            write(out/f'{arm}-result.json',results[arm]);del model,optimizer,n,s,loss,near_loss,support_loss,grads;torch.cuda.empty_cache()
        if any(sha(p)!=h for p,h in inputs.items()) or pretrained_manifest(a.pretrained)!=pretrained:raise ValueError('Input changed during fits')
        write(out/'result.json',dict(status='PASS',results=results,scope='Balanced32 tiny TRAIN memorization diagnostic; no DEV/plaza/Willow evaluation or automatic promotion'))
        write(out/'receipt.json',dict(status='PASS',new_fits=2,optimizer_steps=steps,backend='CUDA',device=torch.cuda.get_device_name(),torch_version=torch.__version__,seconds=time.perf_counter()-started,result_sha256=sha(out/'result.json')))
    except BaseException as exc:
        write(out/'failure.json',dict(status='FAIL',completed_steps=steps,error=repr(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('cache','initial','pretrained','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
