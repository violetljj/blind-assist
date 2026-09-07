"""G10 frozen 2x2 data-diversity and predicted-region-head comparison."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from PIL import Image
import torch
from grounding_predict import read, write, sha, checkpoint_inputs, torch_observation
from factorial_train import configure_trainable, balanced_indices, VARIANTS, EXPECTED
from whisker_model import IMAGE_SIZE, SUPPORT_SIZE, TARGET_ORDER, masked_bce, support_bce
from diversity_model import DiversityModel

SEEDS=(17,29,43)
ARMS=('existing_plain','existing_region','expanded_plain','expanded_region')
CONFIG=dict(schema='nf-g10-diversity-training-v1', seeds=list(SEEDS), arms=list(ARMS),
    steps=600, batch_size=32, validation_every_steps=50, optimizer='AdamW', learning_rate=1e-4,
    weight_decay=1e-4, support_weight=.25, selection='common new VAL near BCE minimum; first tie',
    loss='near BCE + .25 unchanged class-balanced support BCE', region='predicted sigmoid support times spatial feature; AvgPool6x8; original corresponding near linear row; no mask-sum normalization; gate gradients enabled',
    parameter_count='identical plain/region G8 weights', warm_start='original G8 ordinary per seed, never G9B fit',
    train_images=dict(existing=160,expanded=800), train_variant_sampling='8 of each variant per batch32, uniform with replacement within variant',
    trainable_modules=['spatial','near','support'], steps_total=7200, presentations_per_fit=19200,
    closing='-1 unscored; temporal/approach weights unchanged, shared spatial features change',
    target_order=list(TARGET_ORDER), image_size=list(IMAGE_SIZE), support_size=list(SUPPORT_SIZE),
    validation_groups=dict(narrow=12,diverse=12), test_groups=dict(narrow=16,diverse=16),
    coverage_claim='existing versus expanded changes unique images and appearance/geometry coverage at equal steps and variant priors')

def load_inputs(capture, expected_groups):
    capture=capture.resolve(); model_root=capture/'model'
    if read(capture/'verification.json').get('status')!='PASS': raise ValueError('Capture verification must PASS')
    dataset=read(model_root/'dataset.json'); calibration=dataset['calibration']
    if (calibration['width'],calibration['height'],float(calibration['horizontal_fov_degrees']))!=(640,360,100.): raise ValueError('Frozen calibration mismatch')
    frames={}; paths={}
    for frame in dataset['frames']:
        if set(frame)!={'sample_index','rgb_path','time_s','clip_id','frame_in_clip'}: raise ValueError('Privileged frame metadata prohibited')
        index=int(frame['sample_index']); rel=Path(frame['rgb_path']); path=(model_root/rel).resolve()
        if index in frames or not np.isfinite(float(frame['time_s'])): raise ValueError('Invalid frame index/time')
        if rel.is_absolute() or not path.is_relative_to(model_root) or path.suffix.lower()!='.png': raise ValueError('RGB path escape')
        frames[index]=frame; paths[index]=path
    samples=dataset['samples']; groups={}; clips=set(); used=set(); ids=set()
    for sample in samples:
        if set(sample)!={'sample_id','clip_id','group_id','split','frame_indices'}: raise ValueError('Unexpected sample metadata')
        sid=sample['sample_id']; split=sample['split']; group=sample['group_id']
        if sid in ids or sample['clip_id'] in clips or split not in ('train','val','test'): raise ValueError('Invalid sample identity/split')
        if group in groups and groups[group]!=split: raise ValueError('Group crosses splits')
        groups[group]=split; ids.add(sid); clips.add(sample['clip_id'])
        indices=sample['frame_indices']
        if len(indices)!=1 or int(indices[0]) in used: raise ValueError('Exactly one unique current frame required')
        index=int(indices[0]); used.add(index)
        if frames[index]['clip_id']!=sample['clip_id']: raise ValueError('Sample/frame clip mismatch')
    if used!=set(frames): raise ValueError('Unused frame inventory')
    if {s:sum(v==s for v in groups.values()) for s in ('train','val','test')}!=expected_groups: raise ValueError('Frozen split group counts mismatch')
    if any(sum(s['group_id']==g for s in samples)!=4 for g in groups): raise ValueError('Require four samples per group')
    labels=read(capture/'training/labels.json')['targets']; variants=read(capture/'training/partition.json')['variants']
    train_ids={s['sample_id'] for s in samples if s['split']=='train'}
    known_ids={s['sample_id'] for s in samples if s['split']!='test'}
    if set(labels)!=known_ids: raise ValueError('Labels must contain exactly train+val; test labels prohibited')
    if set(variants)!=train_ids: raise ValueError('Variant metadata must contain exactly TRAIN IDs')
    for value in labels.values():
        if np.asarray(value).shape!=(4,) or not np.isin(value[:2],[0,1]).all() or value[2:]!=[-1,-1]: raise ValueError('Require binary near and UNKNOWN closing')
    for sample in samples:
        if sample['split']=='train':
            variant=variants[sample['sample_id']]
            if variant not in EXPECTED or labels[sample['sample_id']][:2]!=EXPECTED[variant]: raise ValueError('Train variant/near prior mismatch')
    for group,split in groups.items():
        if split=='train' and {variants[s['sample_id']] for s in samples if s['group_id']==group}!=set(VARIANTS): raise ValueError('Train group missing factorial variant')
    with np.load(capture/'training/support.npz',allow_pickle=False) as archive:
        if set(archive.files)!=known_ids: raise ValueError('Support must contain exactly train+val')
        support={key:archive[key].copy() for key in archive.files}
    if any(v.dtype!=np.uint8 or v.shape!=(2,*SUPPORT_SIZE) or not np.isin(v,[0,1]).all() for v in support.values()): raise ValueError('Invalid binary support shape')
    inputs=[capture/p for p in ('model/dataset.json','verification.json','training/labels.json','training/partition.json','training/support.npz')]
    return dataset,paths,labels,support,variants,{p.relative_to(capture).as_posix():sha(p) for p in inputs}


def cache_inputs(loaded, selected):
    dataset,paths,labels,support,variants,input_hashes=loaded
    arrays=[]; hashes={}; timing=dict(decode_seconds=0.,resize_seconds=0.,hash_seconds=0.,cuda_cache_seconds=0.)
    for sample in selected:
        index=int(sample['frame_indices'][0]); path=paths[index]
        tick=time.perf_counter(); hashes[str(index)]=sha(path); timing['hash_seconds']+=time.perf_counter()-tick
        tick=time.perf_counter()
        with Image.open(path) as image:
            if image.size!=(640,360): raise ValueError('RGB dimensions mismatch')
            raw=image.convert('RGB'); raw.load()
        timing['decode_seconds']+=time.perf_counter()-tick; tick=time.perf_counter()
        arrays.append(np.asarray(raw.resize((IMAGE_SIZE[1],IMAGE_SIZE[0]),Image.Resampling.BOX),np.uint8).copy())
        timing['resize_seconds']+=time.perf_counter()-tick
    tick=time.perf_counter()
    rgb=torch.from_numpy(np.stack(arrays)).cuda().permute(0,3,1,2).float().div_(255.)
    target=torch.tensor([labels.get(s['sample_id'],[-1]*4)[:2] for s in selected],dtype=torch.float32,device='cuda')
    masks=torch.from_numpy(np.stack([support.get(s['sample_id'],np.zeros((2,*SUPPORT_SIZE),np.uint8)) for s in selected])).cuda().float()
    torch.cuda.synchronize(); timing['cuda_cache_seconds']=time.perf_counter()-tick
    return rgb,target,masks,dict(input_sha256=input_hashes,rgb_sha256=hashes,timing=timing,decoded_images=len(selected))


def make_pools(old_samples, old_variants, new_samples, new_variants, expanded):
    pools={variant:[] for variant in VARIANTS}
    for i,s in enumerate(old_samples):
        if s['split']=='train': pools[old_variants[s['sample_id']]].append(i)
    if expanded:
        for i,s in enumerate(new_samples):
            if s['split']=='train': pools[new_variants[s['sample_id']]].append(len(old_samples)+i)
    return pools


@torch.inference_mode()
def predict(model,images,samples):
    model.eval(); scores=[]; maps=[]
    test=[i for i,s in enumerate(samples) if s['split']=='test']
    for start in range(0,len(images),64):
        near,support=model(images[start:start+64])
        scores.append(torch.cat((near.sigmoid(),near.new_full((len(near),2),-1.)),1).cpu().numpy())
        maps.append(support.sigmoid().cpu().numpy())
    scores,maps=np.concatenate(scores),np.concatenate(maps)[test]
    if not np.isfinite(scores).all() or not np.isfinite(maps).all(): raise RuntimeError('Nonfinite prediction')
    return scores,maps


def prediction_container(samples):
    return dict(schema='nf-g8-whisker-predictions-v1',target_order=list(TARGET_ORDER),unknown_score=-1,
        sample_ids=[s['sample_id'] for s in samples],arms={}), {'test_sample_ids':np.asarray([s['sample_id'] for s in samples if s['split']=='test'])}


def run(capture,existing_capture,learned,output,regression_capture=None):
    if output.exists(): raise FileExistsError(f'Refuse overwrite: {output}')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU fallback')
    new=load_inputs(capture,{'train':160,'val':24,'test':32})
    old=load_inputs(existing_capture,{'train':40,'val':12,'test':12})
    if regression_capture and regression_capture.resolve()!=existing_capture.resolve(): raise ValueError('Regression must be the supplied old G9B capture')
    old_ids={s['group_id'] for s in old[0]['samples']}; new_ids={s['group_id'] for s in new[0]['samples']}
    if old_ids & new_ids: raise ValueError('Old/new group identities overlap')
    checkpoints,source=checkpoint_inputs(learned)
    output.mkdir(parents=True,exist_ok=False); write(output/'config.json',CONFIG)
    started=time.perf_counter(); records=[]; cache_receipts={}
    try:
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True; torch.use_deterministic_algorithms(True)
        new_samples=new[0]['samples']
        old_samples=[s for s in old[0]['samples'] if regression_capture or s['split']=='train']
        new_rgb,new_target,new_masks,cache_receipts['new']=cache_inputs(new,new_samples)
        old_rgb,old_target,old_masks,cache_receipts['existing']=cache_inputs(old,old_samples)
        all_rgb=torch.cat((old_rgb,new_rgb)); all_target=torch.cat((old_target,new_target)); all_masks=torch.cat((old_masks,new_masks))
        val=torch.tensor([i for i,s in enumerate(new_samples) if s['split']=='val'],device='cuda')
        predictions,map_arrays=prediction_container(new_samples)
        if regression_capture:
            regression_predictions,regression_maps=prediction_container(old_samples); (output/'regression').mkdir()
        for arm in ARMS:
            predictions['arms'][arm]={'seeds':{}}; arm_scores=[]; arm_maps=[]
            if regression_capture:
                regression_predictions['arms'][arm]={'seeds':{}}; reg_scores=[]; reg_maps=[]
            pools=make_pools(old_samples,old[4],new_samples,new[4],arm.startswith('expanded'))
            expected=800 if arm.startswith('expanded') else 160
            if sum(map(len,pools.values()))!=expected or len({len(v) for v in pools.values()})!=1: raise ValueError('Frozen sampling inventory mismatch')
            for seed in SEEDS:
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); rng=np.random.default_rng(seed)
                model=DiversityModel(region=arm.endswith('region')).cuda()
                initial=torch.load(checkpoints[('ordinary_video',seed)],map_location='cuda',weights_only=True)
                model.load_state_dict(initial,strict=True)
                optimizer=torch.optim.AdamW(configure_trainable(model),lr=1e-4,weight_decay=1e-4)
                best=float('inf'); best_step=None; history=[]; seen=set(); fit_started=time.perf_counter()
                checkpoint=output/f'{arm}_seed{seed}.pt'
                for step in range(1,601):
                    model.train(); indices=balanced_indices(pools,rng); seen.update(map(int,indices)); selected=torch.as_tensor(indices,device='cuda')
                    near,support=model(all_rgb[selected])
                    loss=masked_bce(near,all_target[selected])+.25*support_bce(support,all_masks[selected],all_target[selected])
                    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                    if step%50==0:
                        model.eval()
                        with torch.inference_mode():
                            near,_=model(new_rgb[val]); value=float(masked_bce(near,new_target[val]).item())
                        if not np.isfinite(value): raise RuntimeError('Nonfinite validation')
                        history.append(dict(step=step,validation_near_bce=value))
                        if value<best:
                            best=value; best_step=step; torch.save(model.state_dict(),checkpoint)
                        progress=dict(stage='training',arm=arm,seed=seed,step=step,total_steps=600,fit_index=len(records)+1,total_fits=12,validation_near_bce=value,best_step=best_step,fit_elapsed_s=time.perf_counter()-fit_started)
                        write(output/'progress.json',progress); print(json.dumps(progress),flush=True)
                torch.cuda.synchronize(); elapsed=time.perf_counter()-fit_started
                model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
                frozen_unchanged=all(torch.equal(value,initial[name]) for name,value in model.state_dict().items() if name.split('.')[0] not in {'spatial','near','support'})
                if not frozen_unchanged: raise RuntimeError('Frozen temporal/approach weights changed')
                torch.cuda.synchronize(); tick=time.perf_counter(); values,maps=predict(model,new_rgb,new_samples); torch.cuda.synchronize()
                record=dict(arm=arm,seed=seed,optimizer_steps=600,presentations=19200,eligible_unique_images=expected,observed_unique_images=len(seen),best_step=best_step,best_validation_near_bce=best,validation_history=history,fit_seconds=elapsed,inference_seconds=time.perf_counter()-tick,
                    warm_start_sha256=source['checkpoint_sha256'][f'ordinary_video_seed{seed}.pt'],checkpoint_sha256=sha(checkpoint),parameters=sum(p.numel() for p in model.parameters()),frozen_parameters_unchanged=True,actual_backend_observation=asdict(torch_observation(model=model)))
                predictions['arms'][arm]['seeds'][str(seed)]={'normal':values.tolist()}; arm_scores.append(values); arm_maps.append(maps)
                map_arrays[f'{arm}__seed{seed}__normal']=maps.astype(np.float16)
                if regression_capture:
                    torch.cuda.synchronize(); tick=time.perf_counter(); rv,rm=predict(model,old_rgb,old_samples); torch.cuda.synchronize()
                    record['regression_inference_seconds']=time.perf_counter()-tick
                    regression_predictions['arms'][arm]['seeds'][str(seed)]={'normal':rv.tolist()}
                    regression_maps[f'{arm}__seed{seed}__normal']=rm.astype(np.float16); reg_scores.append(rv); reg_maps.append(rm)
                records.append(record); write(output/'fit_records.json',records)
            predictions['arms'][arm]['ensemble']={'normal':np.mean(arm_scores,axis=0).tolist()}
            map_arrays[f'{arm}__ensemble__normal']=np.mean(arm_maps,axis=0).astype(np.float16)
            write(output/'predictions.json',predictions)
            if regression_capture:
                regression_predictions['arms'][arm]['ensemble']={'normal':np.mean(reg_scores,axis=0).tolist()}
                regression_maps[f'{arm}__ensemble__normal']=np.mean(reg_maps,axis=0).astype(np.float16)
                write(output/'regression/predictions.json',regression_predictions)
        np.savez_compressed(output/'support_predictions.npz',**map_arrays)
        if regression_capture:
            np.savez_compressed(output/'regression/support_predictions.npz',**regression_maps)
            write(output/'regression/receipt.json',dict(status='complete',sample_count=len(old_samples),support_sample_count=sum(s['split']=='test' for s in old_samples),scope='disclosed old G9B regression; no old val/test fitting or evaluator/test target reads'))
        write(output/'receipt.json',dict(schema='nf-g10-diversity-training-receipt-v1',status='complete',actual_backend='cuda',actual_device=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,
            records=records,cache_receipts=cache_receipts,learned_source=source,optimizer_steps_total=7200,total_wall_seconds=time.perf_counter()-started,source_sha256={name:sha(Path(__file__).with_name(name)) for name in ('diversity_train.py','diversity_model.py','whisker_model.py','factorial_train.py','grounding_predict.py')},
            boundary='Only training and common new VAL targets influence fits/selection; old VAL labels loaded for input validation only and never indexed for fit/selection; test targets never read; regression unscored during training',regression_requested=bool(regression_capture),peak_cuda_memory_bytes=torch.cuda.max_memory_allocated()))
        write(output/'progress.json',dict(stage='complete'))
    except BaseException as error:
        write(output/'receipt.json',dict(status='failed',error=repr(error),records=records,cache_receipts=cache_receipts,total_wall_seconds=time.perf_counter()-started,actual_backend='cuda',actual_device=torch.cuda.get_device_name())); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',required=True,type=Path); parser.add_argument('--existing-capture',required=True,type=Path)
    parser.add_argument('--learned-source',required=True,type=Path); parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--regression-capture',type=Path)
    args=parser.parse_args(); run(args.capture,args.existing_capture,args.learned_source,args.output,args.regression_capture)
