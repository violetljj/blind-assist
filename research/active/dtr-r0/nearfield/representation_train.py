"""G12 fixed RepViT representation comparison; six new fits and a TRAIN-only probe."""
from __future__ import annotations
import argparse
import ast
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from PIL import Image
import torch
from diversity_train import load_inputs,cache_inputs,make_pools,balanced_indices,configure_trainable
from grounding_predict import read,write,sha,checkpoint_inputs,torch_observation
from whisker_model import IMAGE_SIZE,SUPPORT_SIZE,TARGET_ORDER,masked_bce,support_bce
from detached_model import DetachedModel
from representation_model import RepresentationModel,pretrained_manifest

SEEDS=(17,29,43)
ARMS=('original_gate','repvit','repvit_detail')
CONFIG=dict(schema='nf-g12-representation-training-v1',seeds=list(SEEDS),arms=['original_gate','repvit','repvit_detail'],new_fits=6,baseline_fits=0,steps_per_fit=600,total_optimizer_steps=3600,batch_size=32,validation_interval=50,optimizer='AdamW',learning_rate=1e-4,weight_decay=1e-4,support_weight=.25,selection='common G10 VAL near BCE first minimum',pretraining='official RepViT m0_9 distillation strict checkpoint',head='fresh paired initialization; predicted-support gated weighted mean without normalization',detail='last stride4 feature projected16 then stride2conv32 added to deep projection',batchnorm='train/update during fine-tuning, eval during validation and inference; identical policy B/C',closing='-1 unscored',image_size=list(IMAGE_SIZE),normalization='ImageNet mean/std, full frame BOX resize, no crop',target_order=list(TARGET_ORDER))


def load_fresh(capture):
    capture=capture.resolve(); model_root=capture/'model'
    if read(capture/'verification.json').get('status')!='PASS': raise ValueError('Fresh verification must PASS')
    data=read(model_root/'dataset.json'); c=data['calibration']
    if (c['width'],c['height'],float(c['horizontal_fov_degrees']))!=(640,360,100.): raise ValueError('Fresh calibration mismatch')
    frames={}; paths={}
    for frame in data['frames']:
        if set(frame)!={'sample_index','rgb_path','time_s','clip_id','frame_in_clip'}: raise ValueError('Fresh frame metadata prohibited')
        index=int(frame['sample_index']); rel=Path(frame['rgb_path']); path=(model_root/rel).resolve()
        if index in frames or not np.isfinite(float(frame['time_s'])): raise ValueError('Invalid fresh index/time')
        if rel.is_absolute() or not path.is_relative_to(model_root) or path.suffix.lower()!='.png': raise ValueError('Fresh RGB path escape')
        frames[index]=frame; paths[index]=path
    samples=data['samples']; used=[]
    if len(samples)!=128 or len(frames)!=128 or len({s['sample_id'] for s in samples})!=128 or len({s['group_id'] for s in samples})!=32: raise ValueError('Frozen fresh128 samples/32groups required')
    for s in samples:
        if set(s)!={'sample_id','clip_id','group_id','split','frame_indices'} or s['split']!='test' or len(s['frame_indices'])!=1: raise ValueError('Fresh sanitized all-test current frame required')
        index=int(s['frame_indices'][0]); used.append(index)
        if frames[index]['clip_id']!=s['clip_id']: raise ValueError('Fresh clip mismatch')
    if len(set(used))!=128 or set(used)!=set(frames): raise ValueError('Fresh shared/unused frames')
    hashes={p.relative_to(capture).as_posix():sha(p) for p in (model_root/'dataset.json',capture/'verification.json')}
    return data,paths,hashes


def baseline_inputs(learned,new_inputs,old_inputs):
    receipt=read(learned/'receipt.json')
    if receipt.get('status')!='complete' or receipt.get('actual_backend')!='cuda': raise ValueError('Completed G10 CUDA receipt required')
    for name,inputs in (('new',new_inputs),('existing',old_inputs)):
        if receipt['cache_receipts'][name]['input_sha256']!=inputs: raise ValueError('G10 source inputs changed')
    records={(r['arm'],r['seed']):r for r in receipt['records']}; paths={}
    for seed in SEEDS:
        p=learned/f'expanded_region_seed{seed}.pt'
        if sha(p)!=records[('expanded_region',seed)]['checkpoint_sha256']: raise ValueError('G10 baseline checkpoint changed')
        paths[seed]=p
    for name in ('diversity_model.py','whisker_model.py','factorial_train.py'):
        if sha(Path(__file__).with_name(name))!=receipt['source_sha256'][name]: raise ValueError('Baseline model source changed')
    current=Path(__file__).with_name('diversity_train.py')
    source_format=dict(byte_identical=sha(current)==receipt['source_sha256']['diversity_train.py'])
    if not source_format['byte_identical']:
        executed=learned.parent.parent/'executed-diversity-train.py'
        if sha(executed)!=receipt['source_sha256']['diversity_train.py']: raise ValueError('Executed G10 trainer source hash mismatch')
        if ast.dump(ast.parse(executed.read_text(encoding='utf-8-sig')))!=ast.dump(ast.parse(current.read_text(encoding='utf-8-sig'))): raise ValueError('G10 trainer AST changed')
        source_format.update(ast_identical=True,executed_source_sha256=sha(executed),current_source_sha256=sha(current),reason='source formatting only; full AST verified identical')
    return paths,receipt,source_format


def cache_fresh(data,paths):
    arrays=[]; hashes={}; timing=dict(hash_seconds=0.,decode_seconds=0.,resize_seconds=0.,cuda_transfer_seconds=0.)
    for sample in data['samples']:
        index=int(sample['frame_indices'][0]); path=paths[index]; tick=time.perf_counter()
        hashes[str(index)]=sha(path); timing['hash_seconds']+=time.perf_counter()-tick; tick=time.perf_counter()
        with Image.open(path) as image:
            if image.size!=(640,360): raise ValueError('Fresh RGB dimensions mismatch')
            raw=image.convert('RGB'); raw.load()
        timing['decode_seconds']+=time.perf_counter()-tick; tick=time.perf_counter()
        arrays.append(np.asarray(raw.resize((IMAGE_SIZE[1],IMAGE_SIZE[0]),Image.Resampling.BOX),np.uint8).copy())
        timing['resize_seconds']+=time.perf_counter()-tick
    tick=time.perf_counter(); rgb=torch.from_numpy(np.stack(arrays)).cuda().permute(0,3,1,2).float().div_(255.)
    torch.cuda.synchronize(); timing['cuda_transfer_seconds']=time.perf_counter()-tick
    return rgb,dict(rgb_sha256=hashes,timing=timing)


@torch.inference_mode()
def predict(model,rgb,map_indices):
    batch=64 if isinstance(model,DetachedModel) else 32
    model.eval(); scores=[]; maps=[]
    for start in range(0,len(rgb),batch):
        near,support=model(rgb[start:start+batch])
        scores.append(torch.cat((near.sigmoid(),near.new_full((len(near),2),-1.)),1).cpu().numpy())
        maps.append(support.sigmoid().cpu().numpy())
    scores,maps=np.concatenate(scores),np.concatenate(maps)[map_indices]
    if not np.isfinite(scores).all() or not np.isfinite(maps).all(): raise RuntimeError('Nonfinite predictions')
    return scores,maps


def container(samples,map_indices,mixed=False):
    return (dict(schema='nf-g8-whisker-predictions-v1',target_order=list(TARGET_ORDER),unknown_score=-1,
        sample_ids=[s['sample_id'] for s in samples],arms={}),
        {'sample_ids' if mixed else 'test_sample_ids':np.asarray([samples[i]['sample_id'] for i in map_indices])})


def parameter_sha(model,common_only=False):
    digest=hashlib.sha256()
    for name,value in model.named_parameters():
        if common_only and name.startswith('detail.'): continue
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def validation_loss(model,rgb,target,indices):
    total=0.
    for start in range(0,len(indices),32):
        selected=indices[start:start+32]
        total+=float(masked_bce(model(rgb[selected])[0],target[selected]).item())*len(selected)
    return total/len(indices)


def run_probe(args,pretrained):
    output=args.output; output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter(); records=[]
    try:
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(True)
        new=load_inputs(args.training_capture,{'train':160,'val':24,'test':32})
        old=load_inputs(args.existing_capture,{'train':40,'val':12,'test':12})
        ns=[s for s in new[0]['samples'] if s['split']=='train']; osamples=[s for s in old[0]['samples'] if s['split']=='train']
        nr,ny,nm,nreceipt=cache_inputs(new,ns); or_,oy,om,oreceipt=cache_inputs(old,osamples)
        rgb=torch.cat((or_,nr)); target=torch.cat((oy,ny)); masks=torch.cat((om,nm))
        pools=make_pools(osamples,old[4],ns,new[4],True)
        for arm in ('repvit','repvit_detail'):
            torch.manual_seed(17); torch.cuda.manual_seed_all(17); rng=np.random.default_rng(17)
            model=RepresentationModel(args.pretrained_root,detail=arm=='repvit_detail').cuda().train()
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
            torch.cuda.reset_peak_memory_stats(); times=[]; initial_hash=parameter_sha(model)
            for step in range(15):
                torch.cuda.synchronize(); tick=time.perf_counter()
                selected=torch.as_tensor(balanced_indices(pools,rng),device='cuda')
                near,support=model(rgb[selected]); loss=masked_bce(near,target[selected])+.25*support_bce(support,masks[selected],target[selected])
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                torch.cuda.synchronize()
                if step>=5: times.append(time.perf_counter()-tick)
            record=dict(arm=arm,warmup_steps=5,timed_steps=10,batch_size=32,step_seconds=times,p50_step_seconds=float(np.median(times)),p95_step_seconds=float(np.percentile(times,95)),estimated_1800_optimizer_seconds=float(np.median(times))*1800,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),parameters=sum(p.numel() for p in model.parameters()),initial_parameter_sha256=initial_hash,actual_backend_observation=asdict(torch_observation(model=model)))
            records.append(record); print(json.dumps(record),flush=True)
            del model,optimizer,near,support,loss; torch.cuda.empty_cache()
        if pretrained_manifest(args.pretrained_root)!=pretrained: raise RuntimeError('Pretrained files changed during probe')
        write(output/'receipt.json',dict(status='complete',schema='nf-g12-train-only-throughput-probe-v1',records=records,actual_backend='cuda',actual_device=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,pretrained=pretrained,
            cache_receipts=dict(existing=oreceipt,new=nreceipt),total_wall_seconds=time.perf_counter()-started,scope='TRAIN RGB/targets only; no VAL or TEST inference/outcomes; all probe models and optimizers discarded; no checkpoint saved; full runs recreate pretrained weights and seeds',estimated_six_fits_optimizer_seconds=sum(r['estimated_1800_optimizer_seconds'] for r in records),estimate_excludes='validation, checkpoint IO, prediction, startup and changing GPU contention'))
    except BaseException as error:
        write(output/'receipt.json',dict(status='failed',error=repr(error),records=records,total_wall_seconds=time.perf_counter()-started,actual_backend='cuda',actual_device=torch.cuda.get_device_name())); raise


def run(args):
    output=args.output
    if output.exists(): raise FileExistsError(f'Refuse overwrite: {output}')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    pretrained=pretrained_manifest(args.pretrained_root)
    if args.probe: return run_probe(args,pretrained)
    if not args.fresh_capture: raise ValueError('--fresh-capture required outside probe')
    new=load_inputs(args.training_capture,{'train':160,'val':24,'test':32})
    old=load_inputs(args.existing_capture,{'train':40,'val':12,'test':12})
    fresh,fresh_paths,fresh_hashes=load_fresh(args.fresh_capture)
    prior_groups={s['group_id'] for s in new[0]['samples']+old[0]['samples']}
    if prior_groups & {s['group_id'] for s in fresh['samples']}: raise ValueError('Fresh groups overlap previous sources')
    baselines,baseline_receipt,baseline_source_format=baseline_inputs(args.baseline_learned,new[5],old[5])
    warm,source=checkpoint_inputs(args.g8_learned)
    saved=read(args.baseline_learned/'predictions.json')
    if saved['sample_ids']!=[s['sample_id'] for s in new[0]['samples']]: raise ValueError('G10 prediction order changed')
    output.mkdir(parents=True,exist_ok=False); (output/'reference').mkdir(); write(output/'config.json',CONFIG)
    started=time.perf_counter(); records=[]; caches={}
    try:
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True; torch.use_deterministic_algorithms(True)
        new_samples=new[0]['samples']; old_samples=old[0]['samples']
        new_rgb,new_y,new_m,caches['training']=cache_inputs(new,new_samples)
        old_rgb,old_y,old_m,caches['existing']=cache_inputs(old,old_samples)
        for name,prior in (('training','new'),('existing','existing')):
            if caches[name]['rgb_sha256']!=baseline_receipt['cache_receipts'][prior]['rgb_sha256']: raise ValueError('Cached RGB differs from executed G10')
        fresh_rgb,caches['fresh']=cache_fresh(fresh,fresh_paths); caches['fresh']['input_sha256']=fresh_hashes
        rgb=torch.cat((old_rgb,new_rgb)); y=torch.cat((old_y,new_y)); masks=torch.cat((old_m,new_m))
        pools=make_pools(old_samples,old[4],new_samples,new[4],True)
        if sum(map(len,pools.values()))!=800 or len(old_samples)!=256: raise ValueError('Exact G10 pool/index layout required')
        expected_pools={v:[i for i,s in enumerate(old_samples) if s['split']=='train' and old[4][s['sample_id']]==v]
            +[len(old_samples)+i for i,s in enumerate(new_samples) if s['split']=='train' and new[4][s['sample_id']]==v]
            for v in ('both','bar_only','box_only','neither')}
        if list(pools.items())!=list(expected_pools.items()): raise ValueError('Canonical G10 variant/pool order changed')
        val=torch.tensor([i for i,s in enumerate(new_samples) if s['split']=='val'],device='cuda')
        ref_indices=[i for i,s in enumerate(new_samples) if s['split']!='train']
        ref_test_positions=[j for j,i in enumerate(ref_indices) if new_samples[i]['split']=='test']
        predictions,maps=container(fresh['samples'],list(range(len(fresh_rgb))))
        reference,ref_maps=container(new_samples,ref_indices,True)
        common_initializations={}; paired_batches={}
        with np.load(args.baseline_learned/'support_predictions.npz',allow_pickle=False) as saved_maps:
            expected_test=[s['sample_id'] for s in new_samples if s['split']=='test']
            if saved_maps['test_sample_ids'].tolist()!=expected_test: raise ValueError('G10 test support order changed')
            for arm in ARMS:
                predictions['arms'][arm]={'seeds':{}}; reference['arms'][arm]={'seeds':{}}
                arm_scores=[]; arm_maps=[]; reference_scores=[]; reference_maps=[]
                for seed in SEEDS:
                    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                    model=(DetachedModel(False) if arm=='original_gate' else RepresentationModel(args.pretrained_root,detail=arm=='repvit_detail')).cuda()
                    record=dict(arm=arm,seed=seed,optimizer_steps=0)
                    if arm=='original_gate':
                        model.load_state_dict(torch.load(baselines[seed],map_location='cuda',weights_only=True))
                        record['checkpoint_sha256']=sha(baselines[seed])
                    else:
                        initial_parameter_sha=parameter_sha(model)
                        common_sha=parameter_sha(model,common_only=True)
                        if arm=='repvit': common_initializations[seed]=common_sha
                        elif common_initializations[seed]!=common_sha: raise RuntimeError('B/C shared initialization mismatch')
                        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
                        rng=np.random.default_rng(seed); batch_hash=hashlib.sha256(); best=float('inf'); best_step=None; history=[]; fit_start=time.perf_counter()
                        checkpoint=output/f'{arm}_seed{seed}.pt'
                        for step in range(1,601):
                            model.train(); selected_np=balanced_indices(pools,rng); batch_hash.update(np.asarray(selected_np,dtype='<i8').tobytes())
                            selected=torch.as_tensor(selected_np,device='cuda'); near,support=model(rgb[selected])
                            loss=masked_bce(near,y[selected])+.25*support_bce(support,masks[selected],y[selected])
                            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                            if step%50==0:
                                model.eval()
                                with torch.inference_mode(): value=validation_loss(model,new_rgb,new_y,val)
                                if not np.isfinite(value): raise RuntimeError('Nonfinite validation loss')
                                history.append(dict(step=step,validation_near_bce=value))
                                if value<best: best=value; best_step=step; torch.save(model.state_dict(),checkpoint)
                                progress=dict(stage='training',arm=arm,seed=seed,step=step,best_step=best_step,validation_near_bce=value,fit_elapsed_s=time.perf_counter()-fit_start)
                                write(output/'progress.json',progress); print(json.dumps(progress),flush=True)
                        torch.cuda.synchronize(); elapsed=time.perf_counter()-fit_start
                        model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
                        replay_rng=np.random.default_rng(seed); replay_hash=hashlib.sha256()
                        for _ in range(600): replay_hash.update(np.asarray(balanced_indices(pools,replay_rng),dtype='<i8').tobytes())
                        if replay_hash.hexdigest()!=batch_hash.hexdigest(): raise RuntimeError('G10 batch sequence replay mismatch')
                        if arm=='repvit': paired_batches[seed]=batch_hash.hexdigest()
                        elif paired_batches[seed]!=batch_hash.hexdigest(): raise RuntimeError('B/C minibatch sequence mismatch')
                        record.update(optimizer_steps=600,presentations=19200,best_step=best_step,best_validation_near_bce=best,validation_history=history,fit_seconds=elapsed,
                            checkpoint_sha256=sha(checkpoint),pretrained_checkpoint_sha256=pretrained['checkpoint_sha256'],initial_parameter_sha256=initial_parameter_sha,common_initial_parameter_sha256=common_sha,batch_sequence_sha256=batch_hash.hexdigest(),
                            batch_sequence_evidence='same unchanged G10 make_pools/balanced_indices, old full256 offset and RNGseed; reconstructed sequence hash matches; G10 did not persist an independent executed-batch hash')
                    torch.cuda.synchronize(); tick=time.perf_counter()
                    fv,fm=predict(model,fresh_rgb,list(range(len(fresh_rgb)))); rv,rm=predict(model,new_rgb,ref_indices)
                    torch.cuda.synchronize(); record['inference_seconds']=time.perf_counter()-tick
                    if arm=='original_gate':
                        expected=np.asarray(saved['arms']['expanded_region']['seeds'][str(seed)]['normal'],np.float32)
                        if not np.array_equal(rv,expected): raise RuntimeError('Original G10 scores not bit-exact reproduced')
                        if not np.array_equal(rm[ref_test_positions].astype(np.float16),saved_maps[f'expanded_region__seed{seed}__normal']): raise RuntimeError('Original G10 test support not bit-exact reproduced')
                        record['g10_reference_scores_bit_exact']=True; record['g10_test_support_float16_bit_exact']=True
                    record['parameters']=sum(p.numel() for p in model.parameters()); record['actual_backend_observation']=asdict(torch_observation(model=model)); records.append(record); write(output/'fit_records.json',records)
                    predictions['arms'][arm]['seeds'][str(seed)]={'normal':fv.tolist()}; reference['arms'][arm]['seeds'][str(seed)]={'normal':rv.tolist()}
                    maps[f'{arm}__seed{seed}__normal']=fm.astype(np.float16); ref_maps[f'{arm}__seed{seed}__normal']=rm.astype(np.float16)
                    arm_scores.append(fv); arm_maps.append(fm); reference_scores.append(rv); reference_maps.append(rm)
                predictions['arms'][arm]['ensemble']={'normal':np.mean(arm_scores,axis=0).tolist()}; reference['arms'][arm]['ensemble']={'normal':np.mean(reference_scores,axis=0).tolist()}
                maps[f'{arm}__ensemble__normal']=np.mean(arm_maps,axis=0).astype(np.float16); ref_maps[f'{arm}__ensemble__normal']=np.mean(reference_maps,axis=0).astype(np.float16)
                write(output/'predictions.json',predictions); write(output/'reference/predictions.json',reference)
        np.savez_compressed(output/'support_predictions.npz',**maps); np.savez_compressed(output/'reference/support_predictions.npz',**ref_maps)
        write(output/'receipt.json',dict(schema='nf-g12-representation-training-receipt-v1',status='complete',records=records,caches=caches,g8_source=source,pretrained=pretrained,
            baseline_receipt_sha256=sha(args.baseline_learned/'receipt.json'),baseline_predictions_sha256=sha(args.baseline_learned/'predictions.json'),
            baseline_executed_source_sha256=baseline_receipt['source_sha256'],
            baseline_source_format=baseline_source_format,
            sampling_verification='factorial_train.py including balanced_indices source hash pinned; full G10 trainer AST pinned to retained exact executed source; make_pools output independently matches canonical G10 ordered variant/index pools',
            actual_backend='cuda',actual_device=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,optimizer_steps_total=3600,new_fits=6,baseline_fits=0,total_wall_seconds=time.perf_counter()-started,
            source_sha256={name:sha(Path(__file__).with_name(name)) for name in ('representation_model.py','representation_train.py','diversity_model.py','diversity_train.py','whisker_model.py','factorial_train.py')},
            reference_map_scope='sample_ids contains G10 VAL96+TEST128; evaluator may calibrate masks on VAL only; oldTEST is consumed regression',
            boundary='No evaluator or test labels read; only fixed training pools and G10 VAL checkpoint selection; fresh outcomes not used in fitting'))
        write(output/'progress.json',dict(stage='complete'))
    except BaseException as error:
        write(output/'receipt.json',dict(status='failed',error=repr(error),records=records,caches=caches,actual_backend='cuda',actual_device=torch.cuda.get_device_name(),total_wall_seconds=time.perf_counter()-started)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training-capture','existing-capture','g8-learned','baseline-learned','output','pretrained-root'): p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--fresh-capture',type=Path);p.add_argument('--probe',action='store_true')
    run(p.parse_args())
