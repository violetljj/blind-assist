"""G13 one fixed decoupled-detail intervention; G12 B/C predictions reused."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from representation_train import (load_inputs,cache_inputs,load_fresh,cache_fresh,make_pools,
    balanced_indices,parameter_sha,validation_loss,predict,pretrained_manifest,read,write,sha,torch_observation)
from whisker_model import masked_bce,support_bce
from decoupled_model import DecoupledModel

SEEDS=(17,29,43)
CONFIG=dict(schema='nf-g13-decoupled-training-v1',arms=['repvit','repvit_detail','decoupled'],
    new_fits=3,baseline_fits=0,baseline_inferences=0,steps=600,total_steps=1800,batch_size=32,
    seeds=list(SEEDS),optimizer='AdamW',learning_rate=1e-4,weight_decay=1e-4,support_weight=.25,
    validation_interval=50,selection='same G10 VAL near BCE first minimum',
    intervention='support=Support(D+E); gated contents=D*sigmoid(support); attached gate, no detach',
    initialization='exact G12 C full parameter initialization per seed, same official pretrained',
    sampling='exact saved G12 C batch sequence hash, old256+new864 index layout, TRAIN800 only',
    scope='consumed G12 TEST comparison plus G10 reference; no fresh-confirmation claim',
    batchnorm='same end-to-end train/update; eval for validation/inference',closing='-1 unscored')


def reuse_predictions(learned,relative):
    path=learned/relative/'predictions.json'; support_path=learned/relative/'support_predictions.npz'
    predictions=read(path)
    predictions['arms']={arm:copy.deepcopy(predictions['arms'][arm]) for arm in ('repvit','repvit_detail')}
    with np.load(support_path,allow_pickle=False) as archive:
        maps={key:archive[key].copy() for key in archive.files if key in ('sample_ids','test_sample_ids') or key.startswith(('repvit__','repvit_detail__'))}
    return predictions,maps,{str(path.relative_to(learned)).replace('\\','/'):sha(path),str(support_path.relative_to(learned)).replace('\\','/'):sha(support_path)}


def run(args):
    output=args.output
    if output.exists(): raise FileExistsError(f'Refuse overwrite: {output}')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    baseline=read(args.baseline_learned/'receipt.json')
    if baseline.get('status')!='complete' or baseline.get('schema')!='nf-g12-representation-training-receipt-v1': raise ValueError('Completed G12 source receipt required')
    for name in ('representation_model.py','diversity_train.py','whisker_model.py','factorial_train.py'):
        if sha(Path(__file__).with_name(name))!=baseline['source_sha256'][name]: raise ValueError('G12 reused model/data helper source changed')
    pretrained=pretrained_manifest(args.pretrained_root)
    if pretrained!=baseline['pretrained']: raise ValueError('G12 pretraining provenance changed')
    new=load_inputs(args.training_capture,{'train':160,'val':24,'test':32})
    old=load_inputs(args.existing_capture,{'train':40,'val':12,'test':12})
    consumed,consumed_paths,consumed_hashes=load_fresh(args.fresh_capture)
    for name,inputs in (('training',new[5]),('existing',old[5]),('fresh',consumed_hashes)):
        if inputs!=baseline['caches'][name]['input_sha256']: raise ValueError('G12 capture inputs changed')
    parent_records={(r['arm'],r['seed']):r for r in baseline['records']}
    checkpoint_hashes={}
    for arm in ('repvit','repvit_detail'):
        for seed in SEEDS:
            path=args.baseline_learned/f'{arm}_seed{seed}.pt'; digest=sha(path)
            if digest!=parent_records[(arm,seed)]['checkpoint_sha256']: raise ValueError('G12 baseline checkpoint changed')
            checkpoint_hashes[path.name]=digest
    predictions,maps,prediction_hashes=reuse_predictions(args.baseline_learned,Path('.'))
    reference,ref_maps,reference_hashes=reuse_predictions(args.baseline_learned,Path('reference'))
    new_samples=new[0]['samples']; old_samples=old[0]['samples']; ref_indices=[i for i,s in enumerate(new_samples) if s['split']!='train']
    if predictions['sample_ids']!=[s['sample_id'] for s in consumed['samples']] or reference['sample_ids']!=[s['sample_id'] for s in new_samples]: raise ValueError('Saved prediction identity/order mismatch')
    if maps['test_sample_ids'].tolist()!=predictions['sample_ids'] or ref_maps['sample_ids'].tolist()!=[new_samples[i]['sample_id'] for i in ref_indices]: raise ValueError('Saved support identity/order mismatch')
    output.mkdir(parents=True,exist_ok=False); (output/'reference').mkdir(); write(output/'config.json',CONFIG)
    started=time.perf_counter(); records=[]; caches={}
    try:
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True; torch.use_deterministic_algorithms(True)
        nr,ny,nm,caches['training']=cache_inputs(new,new_samples); or_,oy,om,caches['existing']=cache_inputs(old,old_samples)
        cr,caches['consumed']=cache_fresh(consumed,consumed_paths); caches['consumed']['input_sha256']=consumed_hashes
        for current,previous in (('training','training'),('existing','existing'),('consumed','fresh')):
            if caches[current]['rgb_sha256']!=baseline['caches'][previous]['rgb_sha256']: raise ValueError('G12 decoded RGB hash changed')
        rgb=torch.cat((or_,nr)); target=torch.cat((oy,ny)); support_target=torch.cat((om,nm))
        pools=make_pools(old_samples,old[4],new_samples,new[4],True)
        if len(old_samples)!=256 or sum(map(len,pools.values()))!=800: raise ValueError('G12 index layout changed')
        val=torch.tensor([i for i,s in enumerate(new_samples) if s['split']=='val'],device='cuda')
        predictions['arms']['decoupled']={'seeds':{}}; reference['arms']['decoupled']={'seeds':{}}
        scores=[]; supports=[]; ref_scores=[]; ref_supports=[]
        for seed in SEEDS:
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            model=DecoupledModel(args.pretrained_root).cuda()
            initial_sha=parameter_sha(model); common_sha=parameter_sha(model,common_only=True)
            parent=parent_records[('repvit_detail',seed)]
            if initial_sha!=parent['initial_parameter_sha256'] or common_sha!=parent['common_initial_parameter_sha256']: raise ValueError('Exact G12 C initialization mismatch')
            rng=np.random.default_rng(seed); digest=hashlib.sha256()
            # Verify full intended sequence before spending any fitting steps.
            planned=[]
            for _ in range(600):
                indices=balanced_indices(pools,rng); planned.append(indices); digest.update(np.asarray(indices,dtype='<i8').tobytes())
            if digest.hexdigest()!=parent['batch_sequence_sha256']: raise ValueError('Exact G12 C batch sequence mismatch')
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
            history=[]; best=float('inf'); best_step=None; fit_start=time.perf_counter(); checkpoint=output/f'decoupled_seed{seed}.pt'
            for step,indices in enumerate(planned,1):
                model.train(); selected=torch.as_tensor(indices,device='cuda'); near,support=model(rgb[selected])
                loss=masked_bce(near,target[selected])+.25*support_bce(support,support_target[selected],target[selected])
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
                if step%50==0:
                    model.eval(); value=validation_loss(model,nr,ny,val)
                    if not np.isfinite(value): raise RuntimeError('Nonfinite validation loss')
                    history.append(dict(step=step,validation_near_bce=value))
                    if value<best: best=value; best_step=step; torch.save(model.state_dict(),checkpoint)
                    progress=dict(stage='training',arm='decoupled',seed=seed,step=step,best_step=best_step,validation_near_bce=value,fit_elapsed_s=time.perf_counter()-fit_start)
                    write(output/'progress.json',progress); print(json.dumps(progress),flush=True)
            torch.cuda.synchronize(); fit_seconds=time.perf_counter()-fit_start
            model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True)); tick=time.perf_counter()
            value,spatial=predict(model,cr,list(range(len(cr)))); rv,rm=predict(model,nr,ref_indices); torch.cuda.synchronize()
            records.append(dict(arm='decoupled',seed=seed,optimizer_steps=600,presentations=19200,best_step=best_step,best_validation_near_bce=best,validation_history=history,
                fit_seconds=fit_seconds,inference_seconds=time.perf_counter()-tick,initial_parameter_sha256=initial_sha,common_initial_parameter_sha256=common_sha,batch_sequence_sha256=digest.hexdigest(),exact_g12_C_initialization_and_batches=True,
                checkpoint_sha256=sha(checkpoint),parameters=sum(p.numel() for p in model.parameters()),actual_backend_observation=asdict(torch_observation(model=model))))
            write(output/'fit_records.json',records)
            predictions['arms']['decoupled']['seeds'][str(seed)]={'normal':value.tolist()}; reference['arms']['decoupled']['seeds'][str(seed)]={'normal':rv.tolist()}
            maps[f'decoupled__seed{seed}__normal']=spatial.astype(np.float16); ref_maps[f'decoupled__seed{seed}__normal']=rm.astype(np.float16)
            scores.append(value); supports.append(spatial); ref_scores.append(rv); ref_supports.append(rm)
        predictions['arms']['decoupled']['ensemble']={'normal':np.mean(scores,axis=0).tolist()}; reference['arms']['decoupled']['ensemble']={'normal':np.mean(ref_scores,axis=0).tolist()}
        maps['decoupled__ensemble__normal']=np.mean(supports,axis=0).astype(np.float16); ref_maps['decoupled__ensemble__normal']=np.mean(ref_supports,axis=0).astype(np.float16)
        for relative,expected in {**prediction_hashes,**reference_hashes}.items():
            if sha(args.baseline_learned/relative)!=expected: raise RuntimeError('Reused G12 outputs changed during fitting')
        write(output/'predictions.json',predictions); write(output/'reference/predictions.json',reference)
        np.savez_compressed(output/'support_predictions.npz',**maps); np.savez_compressed(output/'reference/support_predictions.npz',**ref_maps)
        write(output/'receipt.json',dict(status='complete',schema='nf-g13-decoupled-training-receipt-v1',records=records,caches=caches,pretrained=pretrained,
            reused_prediction_sha256={**prediction_hashes,**reference_hashes},reused_checkpoint_sha256=checkpoint_hashes,baseline_receipt_sha256=sha(args.baseline_learned/'receipt.json'),
            new_fits=3,baseline_fits=0,baseline_inferences=0,optimizer_steps_total=1800,actual_backend='cuda',actual_device=torch.cuda.get_device_name(),torch_version=torch.__version__,cuda_version=torch.version.cuda,total_wall_seconds=time.perf_counter()-started,
            source_sha256={name:sha(Path(__file__).with_name(name)) for name in ('decoupled_model.py','decoupled_train.py','representation_model.py','representation_train.py','diversity_train.py','factorial_train.py')},
            boundary='Only TRAIN800 and G10 VAL96 influence fits/selection; no evaluator or test labels read; G12 test consumed, B/C saved outputs reused without inference; reference map sample_ids=VAL96+TEST128'))
        write(output/'progress.json',dict(stage='complete'))
    except BaseException as error:
        write(output/'receipt.json',dict(status='failed',error=repr(error),records=records,caches=caches,actual_backend='cuda',actual_device=torch.cuda.get_device_name(),total_wall_seconds=time.perf_counter()-started)); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training-capture','existing-capture','baseline-learned','fresh-capture','pretrained-root','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--g8-learned',type=Path,help='Compatibility metadata only; G13 never loads G8 weights')
    run(p.parse_args())
