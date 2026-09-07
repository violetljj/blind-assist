"""NF-G9-B bounded ordinary-model data intervention; no evaluator/test targets."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from PIL import Image
import torch
from grounding_predict import read, write, sha, checkpoint_inputs, torch_observation, load_model_inputs
from whisker_model import WhiskerModel, IMAGE_SIZE, SUPPORT_SIZE, TARGET_ORDER, masked_bce, support_bce

SEEDS=(17,29,43)
ARMS=('single_data','factorial_data')
VARIANTS=('both','bar_only','box_only','neither')
EXPECTED={'both':[1,1], 'bar_only':[0,1], 'box_only':[1,0], 'neither':[0,0]}
CONFIG=dict(schema='nf-g9-b-factorial-training-v1', seeds=list(SEEDS), steps=300, batch_size=32,
    optimizer='AdamW', learning_rate=1e-4, weight_decay=1e-4, validation_every_steps=25,
    selection='minimum common all-variant validation near BCE; first tie', support_weight=.25,
    loss='near BCE + .25 existing known-class-balanced support BCE', pair_loss=False,
    arms=list(ARMS), frozen_baseline='frozen_ordinary', input='current RGB only',
    trainable_modules=['spatial','near','support'], sampling='equal variant counts per batch with replacement',
    planned_unique_train_images={'single_data':80,'factorial_data':160}, presentations_per_fit=9600,
    comparison_scope='equal optimizer steps/presentations and 50% per-head prior; unique-image/variant coverage differs; no pair-loss mechanism claim',
    closing_scope='unscored sentinel -1; temporal/approach weights unchanged but shared spatial features change',
    image_size=list(IMAGE_SIZE), support_size=list(SUPPORT_SIZE), target_order=list(TARGET_ORDER))


def load_inputs(capture):
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
    if {s:sum(v==s for v in groups.values()) for s in ('train','val','test')}!={'train':40,'val':12,'test':12}: raise ValueError('Require frozen 40/12/12 groups')
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


def variant_pools(samples,variants,arm):
    allowed=('bar_only','box_only') if arm=='single_data' else VARIANTS
    return {v:[i for i,s in enumerate(samples) if s['split']=='train' and variants[s['sample_id']]==v] for v in allowed}


def balanced_indices(pools,rng,batch_size=32):
    if batch_size%len(pools) or any(not values for values in pools.values()): raise ValueError('Invalid balanced pools')
    chosen=np.concatenate([rng.choice(values,batch_size//len(pools),replace=True) for values in pools.values()])
    rng.shuffle(chosen)
    return chosen


def spatial_only(model,rgb):
    features=model.spatial(rgb)
    return model.near(features),model.support(features)


def configure_trainable(model):
    for name,p in model.named_parameters(): p.requires_grad_(name.split('.')[0] in {'spatial','near','support'})
    return [p for p in model.parameters() if p.requires_grad]


@torch.inference_mode()
def predict(model,images,test_indices):
    model.eval(); scores=[]; maps=[]
    for start in range(0,len(images),64):
        near,support=spatial_only(model,images[start:start+64])
        value=torch.cat((near.sigmoid(),near.new_full((len(near),2),-1.)),1)
        scores.append(value.cpu().numpy()); maps.append(support.sigmoid().cpu().numpy())
    scores,maps=np.concatenate(scores),np.concatenate(maps)[test_indices]
    if not np.isfinite(scores).all() or not np.isfinite(maps).all(): raise RuntimeError('Nonfinite model output')
    return scores,maps


def run(capture,learned,output,regression_capture=None):
    if output.exists(): raise FileExistsError(f'Refuse overwrite: {output}')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    dataset,paths,labels,support,variants,input_hashes=load_inputs(capture)
    checkpoints,source=checkpoint_inputs(learned)
    regression_input=load_model_inputs(regression_capture) if regression_capture else None
    output.mkdir(parents=True,exist_ok=False); write(output/'config.json',CONFIG)
    started=time.perf_counter(); timing=dict(decode_seconds=0.,resize_seconds=0.,hash_seconds=0.,cuda_cache_seconds=0.)
    try:
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True; torch.use_deterministic_algorithms(True)
        samples=dataset['samples']; hashes={}; cache={}
        for index,path in paths.items():
            tick=time.perf_counter(); hashes[str(index)]=sha(path); timing['hash_seconds']+=time.perf_counter()-tick
            tick=time.perf_counter()
            with Image.open(path) as image:
                if image.size!=(640,360): raise ValueError('Source image resolution mismatch')
                raw=image.convert('RGB'); raw.load()
            timing['decode_seconds']+=time.perf_counter()-tick; tick=time.perf_counter()
            cache[index]=np.asarray(raw.resize((IMAGE_SIZE[1],IMAGE_SIZE[0]),Image.Resampling.BOX),np.uint8).copy()
            timing['resize_seconds']+=time.perf_counter()-tick
        tick=time.perf_counter()
        images=torch.from_numpy(np.stack([cache[int(s['frame_indices'][0])] for s in samples])).cuda().permute(0,3,1,2).float().div_(255.)
        targets=torch.tensor([labels.get(s['sample_id'],[-1]*4)[:2] for s in samples],device='cuda',dtype=torch.float32)
        masks=torch.from_numpy(np.stack([support.get(s['sample_id'],np.zeros((2,*SUPPORT_SIZE),np.uint8)) for s in samples])).cuda().float()
        torch.cuda.synchronize(); timing['cuda_cache_seconds']=time.perf_counter()-tick
        val=[i for i,s in enumerate(samples) if s['split']=='val']; val_index=torch.tensor(val,device='cuda')
        test=[i for i,s in enumerate(samples) if s['split']=='test']
        predictions=dict(schema='nf-g8-whisker-predictions-v1',target_order=list(TARGET_ORDER),sample_ids=[s['sample_id'] for s in samples],arms={},unknown_score=-1)
        map_arrays={'test_sample_ids':np.asarray([samples[i]['sample_id'] for i in test])}; records=[]
        regression_receipt=None
        if regression_input:
            reg_dataset,reg_paths,reg_input_hashes=regression_input
            reg_arrays=[]; reg_hashes={}; tick=time.perf_counter()
            for sample in reg_dataset['samples']:
                index=int(sample['frame_indices'][-1]); path=reg_paths[index]; reg_hashes[str(index)]=sha(path)
                with Image.open(path) as image:
                    if image.size!=(640,360): raise ValueError('Regression image resolution mismatch')
                    reg_arrays.append(np.asarray(image.convert('RGB').resize((IMAGE_SIZE[1],IMAGE_SIZE[0]),Image.Resampling.BOX),np.uint8).copy())
            reg_images=torch.from_numpy(np.stack(reg_arrays)).cuda().permute(0,3,1,2).float().div_(255.)
            torch.cuda.synchronize()
            regression_receipt=dict(input_sha256=reg_input_hashes,current_rgb_sha256=reg_hashes,cache_seconds=time.perf_counter()-tick,sample_count=len(reg_arrays),scope='disclosed unchanged G9-A regression; last/current RGB only; no evaluator or labels read')
            reg_predictions=dict(schema='nf-g8-whisker-predictions-v1',target_order=list(TARGET_ORDER),sample_ids=[s['sample_id'] for s in reg_dataset['samples']],arms={},unknown_score=-1)
            reg_maps={'test_sample_ids':np.asarray(reg_predictions['sample_ids'])}
            (output/'regression').mkdir()
        for arm in ('frozen_ordinary',*ARMS):
            predictions['arms'][arm]={'seeds':{}}; arm_scores=[]; arm_maps=[]
            if regression_input:
                reg_predictions['arms'][arm]={'seeds':{}}; reg_arm_scores=[]; reg_arm_maps=[]
            for seed in SEEDS:
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); rng=np.random.default_rng(seed)
                model=WhiskerModel('ordinary_video').cuda()
                initial=torch.load(checkpoints[('ordinary_video',seed)],map_location='cuda',weights_only=True)
                model.load_state_dict(initial); parameters=configure_trainable(model)
                record=dict(arm=arm,seed=seed,warm_start_sha256=source['checkpoint_sha256'][f'ordinary_video_seed{seed}.pt'],optimizer_steps=0)
                if arm!='frozen_ordinary':
                    pools=variant_pools(samples,variants,arm); optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=1e-4)
                    best=float('inf'); history=[]; best_step=None; fit_start=time.perf_counter(); seen=set()
                    for step in range(1,301):
                        model.train(); indices=balanced_indices(pools,rng); seen.update(map(int,indices)); selected=torch.as_tensor(indices,device='cuda')
                        near,spatial=spatial_only(model,images[selected]); loss=masked_bce(near,targets[selected])+.25*support_bce(spatial,masks[selected],targets[selected])
                        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                        if step%25==0:
                            model.eval()
                            with torch.inference_mode():
                                near,_=spatial_only(model,images[val_index]); value=float(masked_bce(near,targets[val_index]).item())
                            if not np.isfinite(value): raise RuntimeError('Nonfinite validation loss')
                            history.append(dict(step=step,validation_near_bce=value))
                            if value<best:
                                best=value; best_step=step; torch.save(model.state_dict(),output/f'{arm}_seed{seed}.pt')
                            progress=dict(stage='training',arm=arm,seed=seed,step=step,total_steps=300,best_step=best_step,validation_near_bce=value,fit_elapsed_s=time.perf_counter()-fit_start)
                            write(output/'progress.json',progress); print(__import__('json').dumps(progress),flush=True)
                    torch.cuda.synchronize(); elapsed=time.perf_counter()-fit_start
                    model.load_state_dict(torch.load(output/f'{arm}_seed{seed}.pt',map_location='cuda',weights_only=True))
                    record.update(optimizer_steps=300,presentations=9600,eligible_unique_images=sum(map(len,pools.values())),observed_unique_images=len(seen),best_step=best_step,best_validation_near_bce=best,validation_history=history,fit_seconds=elapsed,checkpoint_sha256=sha(output/f'{arm}_seed{seed}.pt'))
                if any(not torch.equal(value,initial[name]) for name,value in model.state_dict().items() if name.split('.')[0] not in {'spatial','near','support'}): raise RuntimeError('Frozen temporal/approach weights changed')
                torch.cuda.synchronize(); tick=time.perf_counter(); scores,maps=predict(model,images,test); torch.cuda.synchronize()
                record.update(inference_seconds=time.perf_counter()-tick,actual_backend_observation=asdict(torch_observation(model=model)),frozen_parameters_unchanged=True)
                predictions['arms'][arm]['seeds'][str(seed)]={'normal':scores.tolist()}; arm_scores.append(scores); arm_maps.append(maps)
                map_arrays[f'{arm}__seed{seed}__normal']=maps.astype(np.float16); records.append(record)
                if regression_input:
                    torch.cuda.synchronize(); tick=time.perf_counter()
                    reg_scores,reg_support=predict(model,reg_images,list(range(len(reg_images))))
                    torch.cuda.synchronize(); record['regression_inference_seconds']=time.perf_counter()-tick
                    reg_predictions['arms'][arm]['seeds'][str(seed)]={'normal':reg_scores.tolist()}
                    reg_maps[f'{arm}__seed{seed}__normal']=reg_support.astype(np.float16)
                    reg_arm_scores.append(reg_scores); reg_arm_maps.append(reg_support)
            predictions['arms'][arm]['ensemble']={'normal':np.mean(arm_scores,axis=0).tolist()}
            map_arrays[f'{arm}__ensemble__normal']=np.mean(arm_maps,axis=0).astype(np.float16)
            write(output/'predictions.json',predictions)
            if regression_input:
                reg_predictions['arms'][arm]['ensemble']={'normal':np.mean(reg_arm_scores,axis=0).tolist()}
                reg_maps[f'{arm}__ensemble__normal']=np.mean(reg_arm_maps,axis=0).astype(np.float16)
                write(output/'regression/predictions.json',reg_predictions)
        np.savez_compressed(output/'support_predictions.npz',**map_arrays)
        if regression_input:
            np.savez_compressed(output/'regression/support_predictions.npz',**reg_maps)
            write(output/'regression/receipt.json',dict(status='complete',**regression_receipt))
        write(output/'receipt.json',dict(status='complete',schema='nf-g9-b-factorial-training-receipt-v1',actual_backend='cuda',actual_device=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,input_sha256=input_hashes,rgb_sha256=hashes,learned_source=source,records=records,timing=timing,total_wall_seconds=time.perf_counter()-started,optimizer_steps_total=1800,sample_count=len(samples),frame_count=len(paths),source_sha256={name:sha(Path(__file__).with_name(name)) for name in ('factorial_train.py','whisker_model.py')},boundary='Only train+val targets, TRAIN variant map; no evaluator or test target reads; closing sentinel -1 unscored',cache_bytes=images.numel()*images.element_size(),regression=regression_receipt))
        write(output/'progress.json',dict(stage='complete'))
    except BaseException as error:
        write(output/'receipt.json',dict(status='failed',error=repr(error),timing=timing,total_wall_seconds=time.perf_counter()-started,actual_backend='cuda',actual_device=torch.cuda.get_device_name())); raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True); parser.add_argument('--learned-source',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--regression-capture',type=Path)
    args=parser.parse_args(); run(args.capture,args.learned_source,args.output,args.regression_capture)
