"""One full-parameter2000-step fit; frozen-B schedule, final-only DEV selection."""
import argparse
from collections import Counter
from pathlib import Path
import os
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from city_data import CityRGBDataset, CitySupervisedDataset, pixel_support_bce, _array
from city_dev_baseline import sha, read, write, tensor_digest, operating_metrics
from city_dev_selection import select_threshold
from city_finetune_pilot import predict
from city_pilot_metrics import evaluate
from decoupled_model import DecoupledModel


def evaluator(cache, split):
    dataset = CityRGBDataset(cache, split)
    folder = 'supervision' if split == 'train' else 'evaluator'
    record = read(cache/folder/f'{split}.json')
    if record['sample_indices'] != dataset.ids:
        raise ValueError('Label ordering mismatch')
    return dataset, np.array(_array(cache.resolve(),record['near']),copy=True), np.array(_array(cache.resolve(),record['support']),copy=True), record.get('group_ids')



@torch.inference_mode()
def fixed_losses(model, dataset):
    """Exact logits BCE aggregated over the entire fixed TRAIN partition."""
    model.eval()
    near_sum=torch.zeros(2,device='cuda',dtype=torch.float64)
    pos_sum=near_sum.clone();neg_sum=near_sum.clone()
    pc=near_sum.clone();nc=near_sum.clone()
    for start in range(0,len(dataset),32):
        rows=[dataset[i] for i in range(start,min(start+32,len(dataset)))]
        x=torch.stack([r['rgb'] for r in rows]).cuda()
        y=torch.stack([r['near'] for r in rows]).cuda()
        target=torch.stack([r['support'] for r in rows]).cuda()
        n,m=model(x)
        near_sum+=F.binary_cross_entropy_with_logits(n,y,reduction='none').sum(0).double()
        loss=F.binary_cross_entropy_with_logits(m,target.clamp_min(0).to(m.dtype),reduction='none')
        positive=target==1;negative=target==0
        pos_sum+=(loss*positive).sum((0,2,3)).double()
        neg_sum+=(loss*negative).sum((0,2,3)).double()
        pc+=positive.sum((0,2,3));nc+=negative.sum((0,2,3))
    head=(pos_sum/pc.clamp_min(1)+neg_sum/nc.clamp_min(1))/((pc>0).double()+(nc>0).double()).clamp_min(1)
    support=(pos_sum.sum()/pc.sum().clamp_min(1)+neg_sum.sum()/nc.sum().clamp_min(1))/((pc.sum()>0).double()+(nc.sum()>0).double()).clamp_min(1)
    near=near_sum.sum()/(2*len(dataset))
    return dict(near_BCE=float(near),near_BCE_per_head=(near_sum/len(dataset)).tolist(),support_BCE=float(support),support_BCE_per_head=head.tolist(),total=float(near+.25*support),frames=len(dataset),scope='Fixed partition, exact logits, global support class balance; per-head support descriptive only')


def run(a):
    started = time.perf_counter()
    short_result=read(a.short_run/'result.json')
    short_receipt=read(a.short_run/'receipt.json')
    short_protocol=read(a.short_run/'protocol.json')
    short_checkpoint=a.short_run/'B-relational-step200.pt'
    if sha(a.short_run/'result.json')!=short_receipt['result_sha256'] or sha(short_checkpoint)!=short_result['fit']['checkpoint_sha256']:
        raise ValueError('Short-run baseline identity mismatch')
    frozen_result=read(a.frozen_run/'result.json')
    frozen_receipt=read(a.frozen_run/'receipt.json')
    if sha(a.frozen_run/'result.json')!=frozen_receipt['result_sha256']:
        raise ValueError('Frozen2000 baseline identity mismatch')
    previous = read(a.previous/'protocol.json')
    receipt = read(a.previous/'receipt.json')
    previous_result = read(a.previous/'result.json')
    if sha(a.previous/'result.json') != receipt['result_sha256'] or sha(a.previous/'selection.json') != receipt['selection_sha256']:
        raise ValueError('Previous result/selection identity mismatch')
    if not torch.cuda.is_available() or torch.__version__ != receipt['torch_version']:
        raise RuntimeError('Original worker CUDA/Torch environment required')
    original = read(a.pilot/'protocol.json')
    if sha(a.initial) != original['checkpoint_sha256'] or sha(a.old_cache/'manifest.json') != original['cache_manifest_sha256']:
        raise ValueError('Original initialization/TRAIN required')
    if sha(a.dev_cache/'manifest.json') != previous_result['selection']['dev_manifest_sha256']:
        raise ValueError('DEV cache changed')
    for cache in (a.old_cache,a.new_cache,a.dev_cache):
        if sha(cache/'manifest.json') not in short_protocol['input_sha256'].values():
            raise ValueError('Dataset identity differs from step200 run')
    for name in ('city_data.py','decoupled_model.py','representation_model.py','city_pilot_metrics.py'):
        if sha(Path(__file__).with_name(name)) != original['source_sha256'][name]:
            raise ValueError('Common source drift: '+name)
    for name in ('city_dev_selection.py','city_dev_baseline.py'):
        if sha(Path(__file__).with_name(name)) != previous['source_sha256'][name]:
            raise ValueError('Previous selection/helper source drift: '+name)
    manifest = read(a.new_cache/'manifest.json')
    admission = manifest['admission']
    if manifest.get('role') != 'TRAIN_ONLY' or admission['status'] != 'PASS':
        raise ValueError('Admitted TRAIN-only cache required')
    if admission['minimum_dev_camera_distance_m'] < 75 or admission['minimum_inter_site_camera_distance_m'] < 30 or not admission['dev_parent_groups_disjoint']:
        raise ValueError('Region or parent identity isolation failed')
    old = CitySupervisedDataset(a.old_cache, 'train')
    new = CitySupervisedDataset(a.new_cache, 'train')
    if len(old) != 750 or len(new) != 384:
        raise ValueError('Expected old750 plus new384')
    # TRAIN supervision only, for source admission before fitting.
    new_record = read(a.new_cache/'supervision/train.json')
    if len(Counter(new_record['group_ids'])) != 96 or set(Counter(new_record['group_ids']).values()) != {4}:
        raise ValueError('Require 96 complete TRAIN quartets')
    train = ConcatDataset([old,new])
    out = a.output.resolve()
    artifacts = (Path(__file__).resolve().parents[4]/'artifacts.local').resolve()
    if out.exists() or not out.is_relative_to(artifacts) or out == artifacts:
        raise ValueError('Fresh canonical artifact output required')
    out.mkdir(parents=True)
    immutable = {str(p):sha(p) for p in (a.initial,a.old_cache/'manifest.json',a.new_cache/'manifest.json',a.dev_cache/'manifest.json',a.previous/'result.json',a.previous/'selection.json',a.short_run/'result.json',short_checkpoint,a.frozen_run/'result.json',a.frozen_run/'training_indices.npy')}
    schedule = np.random.default_rng(17).integers(0,len(train),size=(2000,32))
    if not np.array_equal(schedule[:200],np.load(a.short_run/'training_indices.npy',allow_pickle=False)):
        raise ValueError('Short-run schedule prefix mismatch')
    if not np.array_equal(schedule,np.load(a.frozen_run/'training_indices.npy',allow_pickle=False)):
        raise ValueError('Frozen2000 full schedule mismatch')
    np.save(out/'training_indices.npy',schedule,allow_pickle=False)
    all_near = np.concatenate([old.near,new.near])
    write(out/'protocol.json',dict(seed=17,steps=2000,batch=32,train_frames=len(train),new_frames=len(new),
        backbone='TRAINABLE_ALL_PARAMETERS',BN='FROZEN_RUNNING_BUFFERS',head_lr=1e-5,weight_decay=1e-4,support_weight=.25,
        sampling='Same uniform-replacement procedure; new indices for1134 entries',
        sampled_old=int((schedule<750).sum()),sampled_new=int((schedule>=750).sum()),
        sampled_positive_by_head=all_near[schedule].sum(axis=(0,1)).tolist(),
        schedule_sha256=sha(out/'training_indices.npy'),input_sha256=immutable,
        protocol_sha256=sha(Path(__file__).with_name('CITY_FULL_FIT_20260908.md')),
        source_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('city_dev_baseline.py'),Path(__file__).with_name('city_dev_selection.py'),Path(__file__).with_name('city_score_separation.py'))},
        scope='One full-parameter Development fit; final2000 DEV selection; consumed plaza diagnostic; no fresh TEST or promotion'))
    steps = 0
    try:
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.manual_seed(17); torch.cuda.manual_seed_all(17)
        torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
        model = DecoupledModel(a.pretrained).cuda()
        model.load_state_dict(torch.load(a.initial,map_location='cpu',weights_only=True),strict=True)
        for p in model.parameters():p.requires_grad_(True)
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=1e-4)
        backbone_hash = tensor_digest(model.backbone); buffers_hash = tensor_digest(model,buffers_only=True)
        model.train()
        for layer in model.modules():
            if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()
        history=[]; snapshots={}; fit_started=time.perf_counter()
        for step,indices in enumerate(schedule,1):
            samples=[train[int(i)] for i in indices]
            rgb=torch.stack([s['rgb'] for s in samples]).cuda()
            near=torch.stack([s['near'] for s in samples]).cuda()
            support=torch.stack([s['support'] for s in samples]).cuda()
            n,s=model(rgb)
            near_loss=F.binary_cross_entropy_with_logits(n,near)
            support_loss=pixel_support_bce(s,support)
            loss=near_loss+.25*support_loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            if not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all():
                raise RuntimeError('Nonfinite gradient')
            optimizer.step();steps=step
            if step in (200,500,1000,2000):
                snapshot=out/f'F-step{step}.pt';torch.save(model.state_dict(),snapshot)
                snapshots[step]=snapshot
            if step%100==0:
                history.append(dict(step=step,loss=float(loss.detach()),near_BCE=float(near_loss.detach()),support_BCE=float(support_loss.detach()),phase='PRE_UPDATE'))
                write(out/'progress.json',history[-1])
        torch.cuda.synchronize()
        if tensor_digest(model,buffers_only=True)!=buffers_hash:
            raise ValueError('Frozen BN changed')
        checkpoint=snapshots[2000]
        fit=dict(steps=steps,seconds=time.perf_counter()-fit_started,history=history,checkpoint_sha256=sha(checkpoint),backbone_unchanged=tensor_digest(model.backbone)==backbone_hash,BN_unchanged=True,schedule_parity_frozen2000=True)
        write(out/'fit-complete.json',fit)
        torch.save(dict(optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),completed_steps=steps),out/'optimizer-resume-state.pt')
        del optimizer,rgb,near,support,n,s,loss,samples
        # DEV is opened only after the sole fixed2000-step fit has completed. No intermediate DEV.
        dev,dy,ds,dg=evaluator(a.dev_cache,'dev')
        dn,dm=predict(model,dev)
        np.savez_compressed(out/'dev-predictions.npz',near=dn,support=dm,sample_indices=np.array(dev.ids))
        heads={h:select_threshold(dn[:,j],dy[:,j],min_count=48) for j,h in enumerate(('BODY','HEAD'))}
        thresholds=[heads[h]['threshold'] for h in ('BODY','HEAD')]
        selection=dict(heads=heads,thresholds=thresholds,checkpoint_sha256=sha(checkpoint),dev_manifest_sha256=sha(a.dev_cache/'manifest.json'),rule='Unchanged per-head empirical FPR<=10%, max recall, lower FPR, higher threshold')
        write(out/'selection.json',selection); selection_hash=sha(out/'selection.json')
        result=dict(status='PASS',fit=fit,selection=selection,
            DEV_fixed_05=evaluate(dn,dm,dy,ds,dg),DEV_selected=operating_metrics(dn,dm,dy,ds,thresholds,dg),
            frozen2000_DEV=frozen_result['DEV_selected'],frozen2000_plaza=frozen_result['consumed_plaza_diagnostic'],TRAIN={})
        for name,cache in (('old750',a.old_cache),('new384',a.new_cache)):
            data,y,m,g=evaluator(cache,'train');pn,pm=predict(model,data)
            np.savez_compressed(out/f'{name}-predictions.npz',near=pn,support=pm,sample_indices=np.array(data.ids))
            result['TRAIN'][name]=evaluate(pn,pm,y,m,g)
        plaza,py,ps,pg=evaluator(a.old_cache,'test')
        pg=[f'original_city_triplet_{i//3}' for i in plaza.ids]
        pn,pm=predict(model,plaza)
        np.savez_compressed(out/'plaza-predictions.npz',near=pn,support=pm,sample_indices=np.array(plaza.ids))
        result['consumed_plaza_diagnostic']=dict(fixed_05=evaluate(pn,pm,py,ps,pg),DEV_thresholds=operating_metrics(pn,pm,py,ps,thresholds,pg))
        if sha(out/'selection.json')!=selection_hash:raise ValueError('Selection changed after diagnostics')
        for path,digest in immutable.items():
            if sha(path)!=digest:raise ValueError('Input changed')
        # Descriptive TRAIN-only trajectories; no checkpoint selection from these scores.
        from city_score_separation import curve
        trajectory={}
        for step,path in snapshots.items():
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
            row={}
            for name,cache in (('old750',a.old_cache),('new384',a.new_cache)):
                data,y,m,g=evaluator(cache,'train')
                if step==2000:
                    with np.load(out/f'{name}-predictions.npz',allow_pickle=False) as z:
                        pn,pm=z['near'].copy(),z['support'].copy()
                else:
                    pn,pm=predict(model,data)
                    np.savez_compressed(out/f'step{step}-{name}-predictions.npz',near=pn,support=pm,sample_indices=np.array(data.ids))
                row[name]=dict(losses=fixed_losses(model,old if name=='old750' else new),fixed05=evaluate(pn,pm,y,m,g),curves={h:curve(pn[:,j].astype(float).tolist(),y[:,j].astype(int).tolist()) for j,h in enumerate(('BODY','HEAD'))})
            trajectory[str(step)]=row
        result['TRAIN_trajectory']=trajectory
        result['selection_scope']='Final2000 only; intermediate snapshots TRAIN diagnostic only'
        write(out/'result.json',result)
        write(out/'receipt.json',dict(status='PASS',new_fit_count=1,optimizer_steps=steps,backend='CUDA',device=torch.cuda.get_device_name(),
            torch_version=torch.__version__,seconds=time.perf_counter()-started,result_sha256=sha(out/'result.json'),selection_sha256=selection_hash,
            scope='Full-parameter fit comparison; final2000 DEV selection and consumed plaza diagnostic; no fresh TEST or Willow regression, no promotion'))
    except BaseException as exc:
        write(out/'failure.json',dict(status='FAIL',completed_steps=steps,error=str(exc)));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot','previous','initial','pretrained','old-cache','new-cache','dev-cache','short-run','frozen-run','output'):
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
