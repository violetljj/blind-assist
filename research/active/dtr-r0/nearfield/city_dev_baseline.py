"""Fixed A/B/C adaptation, DEV-only selection, then consumed-plaza diagnostics."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
from city_data import CityRGBDataset, CitySupervisedDataset, pixel_support_bce, _array
from city_dev_selection import select_checkpoint, select_threshold, apply_thresholds
from city_finetune_pilot import predict
from city_pilot_metrics import evaluate
from decoupled_model import DecoupledModel


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def tensor_digest(module, buffers_only=False):
    digest = hashlib.sha256()
    items = module.named_buffers() if buffers_only else module.state_dict().items()
    for name, tensor in items:
        digest.update(name.encode()); digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def operating_metrics(prob, maps, truth, support, thresholds, groups=None):
    # Convert only near decisions; support retains its original fixed .5 rule.
    decisions = apply_thresholds(prob, thresholds)
    result = evaluate(decisions.astype(np.float64), maps, truth, support, groups)
    result['thresholds']['near'] = list(thresholds)
    result['thresholds']['comparison'] = 'inclusive >= in float64 on original probabilities'
    return result


def run(a):
    started = time.perf_counter()
    pilot, fit = read(a.pilot/'protocol.json'), read(a.pilot/'fit-complete.json')
    receipt = read(a.pilot/'receipt.json')
    if not torch.cuda.is_available() or torch.__version__ != receipt['torch_version']:
        raise RuntimeError('Use original pilot CUDA/Torch environment for matched A/B/C')
    if sha(a.initial) != pilot['checkpoint_sha256'] or sha(a.full_ft) != fit['checkpoint_sha256']:
        raise ValueError('Matched original/full-FT weights required')
    if sha(a.train_cache/'manifest.json') != pilot['cache_manifest_sha256']:
        raise ValueError('Original TRAIN cache required')
    if sha(a.schedule) != fit['schedule_sha256']:
        raise ValueError('Original 200x32 schedule required')
    for name in ('city_data.py','decoupled_model.py','representation_model.py','city_pilot_metrics.py'):
        if sha(Path(__file__).with_name(name)) != pilot['source_sha256'][name]:
            raise ValueError('Frozen common source changed: '+name)
    out = a.output.resolve()
    artifacts = (Path(__file__).resolve().parents[4]/'artifacts.local').resolve()
    if not out.is_relative_to(artifacts) or out == artifacts or out.exists():
        raise ValueError('Fresh canonical artifact output required')
    dm = read(a.dev_cache/'manifest.json')
    if dm.get('role') != 'DEV_ONLY' or dm['admission']['status'] != 'PASS' or dm['admission']['minimum_train_camera_distance_m'] < 100:
        raise ValueError('Admitted distant-region DEV required before fitting')
    dev = CityRGBDataset(a.dev_cache, 'dev')
    labels = read(a.dev_cache/'evaluator/dev.json')
    if labels['sample_indices'] != dev.ids:
        raise ValueError('DEV label ordering mismatch')
    dev_groups=labels['group_ids']
    if len(dev_groups) != 128 or len(dev_groups) != len(dev) or len(set(dev_groups)) != 32 or set(Counter(dev_groups).values()) != {4}:
        raise ValueError('Expected 32 complete DEV quartets')
    dy = np.array(_array(a.dev_cache.resolve(),labels['near']),copy=True)
    ds = np.array(_array(a.dev_cache.resolve(),labels['support']),copy=True)
    # Coverage check only, with constant synthetic scores; no model predictions.
    for h in range(2):
        select_threshold(np.zeros(len(dev)), dy[:,h], min_count=48)
    train = CitySupervisedDataset(a.train_cache,'train')
    schedule = np.load(a.schedule,allow_pickle=False)
    if schedule.shape != (200,32) or not np.array_equal(schedule,np.random.default_rng(17).integers(0,len(train),size=(200,32))):
        raise ValueError('Schedule semantic mismatch')
    out.mkdir(parents=True)
    initial_hashes = {str(p):sha(p) for p in (a.initial,a.full_ft,a.schedule,a.dev_cache/'manifest.json',a.train_cache/'manifest.json')}
    protocol = dict(seed=17,steps_per_arm=200,batch_size=32,head_lr=1e-5,weight_decay=1e-4,
        C_unfreeze_step=101,C_backbone_lr=1e-6,BN='All running statistics frozen throughout',
        A='Reused original full-FT step200; zero additional fits',B='RepViT frozen; projection/detail/support/near trainable',
        C='B for100 steps, then backbone LR1e-6 for100; heads retain LR1e-5 and optimizer state',
        selection='Final step200 only; per-head DEV FPR<=.10; choose one whole model',
        fresh_TEST='NOT_CAPTURED_OR_ACCESSED; old plaza diagnostic only after DEV selection',
        input_sha256=initial_hashes,dev_site=dm['source_site_id'],
        protocol_sha256=sha(Path(__file__).with_name('CITY_DEV_BASELINE_20260908.md')),
        source_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('city_dev_selection.py'))})
    write(out/'protocol.json',protocol)
    steps = {}
    try:
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
        weights = {'A':a.full_ft}; fits = {}; prefix = None
        for arm in ('B','C'):
            torch.manual_seed(17); torch.cuda.manual_seed_all(17)
            model = DecoupledModel(a.pretrained).cuda()
            model.load_state_dict(torch.load(a.initial,map_location='cpu',weights_only=True),strict=True)
            backbone_ids = {id(p) for p in model.backbone.parameters()}
            head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
            for p in model.backbone.parameters():p.requires_grad_(False)
            frozen_backbone = tensor_digest(model.backbone)
            frozen_buffers = tensor_digest(model,buffers_only=True)
            optimizer = torch.optim.AdamW(head_params,lr=1e-5,weight_decay=1e-4)
            model.train()
            for layer in model.modules():
                if isinstance(layer,torch.nn.modules.batchnorm._BatchNorm):layer.eval()
            arm_start = time.perf_counter(); history=[];steps[arm]=0
            for step,indices in enumerate(schedule,1):
                if arm == 'C' and step == 101:
                    for p in model.backbone.parameters():p.requires_grad_(True)
                    optimizer.add_param_group(dict(params=list(model.backbone.parameters()),lr=1e-6,weight_decay=1e-4))
                samples = [train[int(i)] for i in indices]
                rgb = torch.stack([s['rgb'] for s in samples]).cuda()
                target = torch.stack([s['near'] for s in samples]).cuda()
                support = torch.stack([s['support'] for s in samples]).cuda()
                near, mask = model(rgb)
                loss = F.binary_cross_entropy_with_logits(near,target)+.25*pixel_support_bce(mask,support)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                grads = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                if not torch.stack(grads).all():raise RuntimeError('Nonfinite gradients')
                optimizer.step();steps[arm]=step
                if step == 100:
                    if tensor_digest(model.backbone) != frozen_backbone:raise ValueError('Frozen backbone changed')
                    digest = tensor_digest(model)
                    if arm == 'B':prefix=digest
                    elif digest != prefix:raise ValueError('B/C first100 trajectories differ')
                if step%25==0:
                    history.append(dict(step=step,loss=float(loss.detach())))
                    write(out/'progress.json',dict(arm=arm,step=step,loss=float(loss.detach())))
            torch.cuda.synchronize()
            if tensor_digest(model,buffers_only=True) != frozen_buffers:raise ValueError('BN buffers changed')
            if arm == 'B' and tensor_digest(model.backbone) != frozen_backbone:raise ValueError('B backbone changed')
            path=out/f'{arm}-step200.pt';torch.save(model.state_dict(),path);weights[arm]=path
            fits[arm]=dict(steps=200,seconds=time.perf_counter()-arm_start,checkpoint_sha256=sha(path),history=history,
                first100_model_sha256=prefix,BN_unchanged=True,backbone_unchanged=tensor_digest(model.backbone)==frozen_backbone)
            write(out/f'{arm}-fit.json',fits[arm])
            del model,optimizer,rgb,target,support,near,mask,loss,samples
        write(out/'fits-complete.json',fits)
        # All model fitting has stopped before predictions are exposed to DEV selection.
        dev_predictions={};dev_maps={}; fixed={}
        model=DecoupledModel(a.pretrained).cuda().eval()
        for arm,path in weights.items():
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
            n,s=predict(model,dev);dev_predictions[arm]=n;dev_maps[arm]=s
            np.savez_compressed(out/f'{arm}-dev-predictions.npz',near=n,support=s,sample_indices=np.array(dev.ids))
            fixed[arm]=evaluate(n,s,dy,ds,dev_groups)
        selection=select_checkpoint(dev_predictions,dy,min_count=48)
        selection['dev_manifest_sha256']=sha(a.dev_cache/'manifest.json')
        selection['checkpoint_sha256']={arm:sha(path) for arm,path in weights.items()}
        write(out/'selection.json',selection)
        selection_hash=sha(out/'selection.json')
        dev_selected={}
        for arm in weights:
            thresholds=[selection['arms'][arm]['heads'][h]['threshold'] for h in ('BODY','HEAD')]
            dev_selected[arm]=operating_metrics(dev_predictions[arm],dev_maps[arm],dy,ds,thresholds,dev_groups)
        # Only now open the consumed plaza labels; never revise selection from them.
        plaza=CityRGBDataset(a.train_cache,'test')
        pl=read(a.train_cache/'evaluator/test.json')
        if pl['sample_indices'] != plaza.ids:raise ValueError('Plaza identity mismatch')
        py=np.array(_array(a.train_cache.resolve(),pl['near']),copy=True)
        ps=np.array(_array(a.train_cache.resolve(),pl['support']),copy=True)
        plaza_groups=[f'original_city_triplet_{i//3}' for i in plaza.ids]
        diagnostic={}
        for arm,path in weights.items():
            if arm == 'A':
                with np.load(a.pilot/'city_finetuned-city-predictions.npz',allow_pickle=False) as archive:
                    n,s=archive['near'].copy(),archive['support'].copy()
            else:
                model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
                n,s=predict(model,plaza)
                np.savez_compressed(out/f'{arm}-plaza-predictions.npz',near=n,support=s,sample_indices=np.array(plaza.ids))
            thresholds=[selection['arms'][arm]['heads'][h]['threshold'] for h in ('BODY','HEAD')]
            diagnostic[arm]=dict(fixed_05=evaluate(n,s,py,ps,plaza_groups),DEV_thresholds=operating_metrics(n,s,py,ps,thresholds,plaza_groups))
        if sha(out/'selection.json') != selection_hash:raise ValueError('Selection changed after plaza access')
        for path,digest in initial_hashes.items():
            if sha(path) != digest:raise ValueError('Input changed')
        write(out/'result.json',dict(status='PASS',fits=fits,DEV_fixed_05=fixed,DEV_selected=dev_selected,
            selection=selection,consumed_plaza_diagnostic=diagnostic,
            scope='DEV-selected candidate only; no fresh TEST result, no deployment or natural-world claim'))
        write(out/'receipt.json',dict(status='PASS',new_fit_count=2,optimizer_steps=steps,
            backend='CUDA',device=torch.cuda.get_device_name(),torch_version=torch.__version__,
            seconds=time.perf_counter()-started,result_sha256=sha(out/'result.json'),selection_sha256=selection_hash))
    except BaseException as exc:
        write(out/'failure.json',dict(status='FAIL',completed_steps=steps,error=str(exc)))
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('pilot','initial','full-ft','pretrained','train-cache','schedule','dev-cache','output'):
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
