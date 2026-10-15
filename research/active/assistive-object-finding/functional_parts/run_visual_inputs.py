#!/usr/bin/env python3
"""Observation-only visual inputs; passive DINO and pre-classifier dense features."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
sys.dont_write_bytecode = True
from run_baseline import ACTIONS, HEADS, append, crop_box, load_official, observe, rows, save, sha256


from run_multicandidate import plan_views

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--compare-predictions', type=Path)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=900)
    args = parser.parse_args()
    started = time.perf_counter()
    cpu_started = time.process_time()
    old, output = args.old_root.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'writer.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    completed, actual_calls, stage = 0, 0, 'startup'
    stop = threading.Event()
    def status():
        return dict(pid=os.getpid(), completed=completed, actual_new_calls=actual_calls,
                    stage=stage, elapsed_s=time.perf_counter()-started, cpu_s=time.process_time()-cpu_started)
    def watchdog():
        if not stop.wait(args.max_seconds):
            save(output/'terminal.json', dict(status(), status='TIME_BUDGET_EXHAUSTED'))
            lock.unlink(missing_ok=True)
            os._exit(124)
    hooks = []
    try:
        predictions = output/'predictions.jsonl'
        if predictions.exists():
            raise ValueError('Preserve existing predictions; use explicit residual manifest/new output for repair')
        config = dict(script_sha256=sha256(__file__), inherited_runner_sha256=sha256(Path(__file__).with_name('run_baseline.py')),
                      manifest_sha256=sha256(args.manifest), candidate_geometry_sha256=sha256(Path(__file__).with_name('run_multicandidate.py')),
                      protocol_sha256=sha256(args.protocol), max_seconds=args.max_seconds, primary_head='ree',
                      seed=0, logical_calls_per_method=4,
                      hook='passive aff_proj input and aff_classifier input/output; no mutation', downloads=0,
                      compare_predictions_sha256=sha256(args.compare_predictions) if args.compare_predictions else None)
        save(output/'config.json', config)
        threading.Thread(target=watchdog, daemon=True).start()
        observations = observe(args.manifest, 0)
        import numpy as np
        import cv2
        import torch
        import torchvision
        from torchvision import transforms
        from PIL import Image, ImageOps
        torch.set_num_threads(4); cv2.setNumThreads(2)
        torch.manual_seed(0); np.random.seed(0)
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        free, capacity = torch.cuda.mem_get_info()
        if free < 3_000_000_000: raise RuntimeError(f'Less than 3 GB free VRAM: {free}')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats()
        os.environ['TORCH_HOME'] = str(old/'torch-cache')
        os.environ['HF_HOME'] = str(old/'hf-cache')
        os.environ['HF_HUB_OFFLINE'] = '1'
        stage = 'load_model'
        model = load_official(old/'vendor'/'SelectiveCL', old/'models'/'AGD20K-Seen.pth', output)
        save(output/'runtime.json', dict(torch=torch.__version__, torchvision=torchvision.__version__,
             python=sys.version, backend='CUDA', device=torch.cuda.get_device_name(), free_total_bytes=[free,capacity]))
        captured = []; dense_dino = []; dense_head = []
        comparisons = {r['sample_id']: r for r in rows(args.compare_predictions)} if args.compare_predictions else {}
        comparison_counts = dict(images=0, maps=0, logits=0, max_abs_error=0.0)
        def capture(_module, _inputs, result):
            captured.append(result.detach())
            dense_head.append(_inputs[0].detach())
        hooks.append(model.aff_classifier.register_forward_hook(capture))
        def capture_dino(_module, inputs):
            dense_dino.append(inputs[0].detach())
        hooks.append(model.aff_proj.register_forward_pre_hook(capture_dino))
        transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                    transforms.Normalize(mean=(.485,.456,.406),std=(.229,.224,.225))])
        map_root = output/'maps'; map_root.mkdir(exist_ok=True)
        def predict(image, action):
            nonlocal actual_calls
            captured.clear(); dense_dino.clear(); dense_head.clear()
            tick=time.perf_counter()
            tensor=transform(image).unsqueeze(0).cuda()
            label=torch.tensor([ACTIONS.index(action)],dtype=torch.long,device='cuda')
            torch.cuda.synchronize(); preprocess=time.perf_counter()-tick
            tick=time.perf_counter()
            with torch.inference_mode(): result=model.test_forward(tensor,label)
            torch.cuda.synchronize(); forward=time.perf_counter()-tick
            actual_calls += 1
            if len(captured)!=1 or tuple(captured[0].shape)!=(1,36,14,14):
                raise ValueError('Unexpected classifier hook shape/count')
            if len(dense_dino)!=1 or tuple(dense_dino[0].shape)!=(1,196,384):
                raise ValueError('Unexpected DINO hook shape/count')
            if len(dense_head)!=1 or tuple(dense_head[0].shape)!=(1,384,14,14):
                raise ValueError('Unexpected head hook shape/count')
            dino=dense_dino[0][0].reshape(14,14,384).permute(2,0,1).float().cpu().numpy().copy()
            head_dense=dense_head[0][0].float().cpu().numpy().copy()
            if not np.isfinite(dino).all() or not np.isfinite(head_dense).all():
                raise ValueError('Invalid dense features')
            logits=captured[0][0].float().cpu().numpy()
            maps={name:value[0].detach().float().cpu().numpy() for name,value in zip(HEADS,result)}
            if any(v.shape!=(14,14) or not np.isfinite(v).all() for v in maps.values()) or not np.isfinite(logits).all():
                raise ValueError('Invalid model maps')
            if not np.array_equal(logits[ACTIONS.index(action)],maps['raw']):
                raise ValueError('Passive all-action hook disagrees with returned raw map')
            return maps,logits,dino,head_dense,dict(preprocess_s=preprocess,forward_s=forward,actual_view_s=preprocess+forward)
        stage='inference'
        for row in observations:
            image=Image.open(row['image_path']).convert('RGB')
            width,height=image.size
            image_hash=sha256(row['image_path'])
            if row.get('image_sha256') and row['image_sha256'] != image_hash:
                raise ValueError('Observation hash mismatch')
            # The base map alone determines proposals; no evaluator fields are read.
            base_maps,base_logits,base_dino,base_head,base_elapsed=predict(image,row['action'])
            candidates,fixed=plan_views(base_maps['ree'],width,height)
            views={}
            specifications=[dict(name='base',bbox_xyxy=[0,0,width,height])]+candidates+fixed+[
                dict(name='flip',bbox_xyxy=[0,0,width,height],flip=True),
                dict(name='center',bbox_xyxy=crop_box(width,height)),
                dict(name='flip_center',bbox_xyxy=crop_box(width,height),flip=True)]
            dedup={}; image_calls=0
            for spec in specifications:
                name,box,flip=spec['name'],spec['bbox_xyxy'],spec.get('flip',False)
                key=(tuple(box),flip)
                if key in dedup:
                    views[name]=dict(views[dedup[key]],cache_source=f'new_run:{dedup[key]}')
                    continue
                view_image=image.crop(tuple(box))
                if flip: view_image=ImageOps.mirror(view_image)
                if name=='base':
                    maps,logits,dino,head_dense,elapsed=base_maps,base_logits,base_dino,base_head,base_elapsed
                else:
                    maps,logits,dino,head_dense,elapsed=predict(view_image,row['action'])
                image_calls+=1
                identifier=hashlib.sha256((row['sample_id']+name).encode()).hexdigest()[:24]
                paths={}
                raw_paths={}
                for head,value in maps.items():
                    path=map_root/f'{identifier}_{head}.npy'
                    if flip:
                        raw_path=map_root/f'{identifier}_{head}_unrestored.npy';np.save(raw_path,value);raw_paths[head]=str(raw_path)
                        value=value[:,::-1].copy()
                    np.save(path,value);paths[head]=str(path)
                logits_path=map_root/f'{identifier}_all_action_logits.npy'
                raw_logits_path=None
                if flip:
                    raw_logits_path=map_root/f'{identifier}_all_action_logits_unrestored.npy'
                    np.save(raw_logits_path,logits);logits=logits[:,:,::-1].copy()
                np.save(logits_path,logits)
                dense_paths={}
                for dense_name,dense_value in [('dino',dino),('head',head_dense)]:
                    dense_path=map_root/f'{identifier}_{dense_name}_dense.npy'
                    if flip:
                        unrestored=map_root/f'{identifier}_{dense_name}_dense_unrestored.npy'
                        np.save(unrestored,dense_value)
                        dense_paths[f'unrestored_{dense_name}_dense_path']=str(unrestored)
                        dense_value=dense_value[:,:,::-1].copy()
                    np.save(dense_path,dense_value)
                    dense_paths[f'{dense_name}_dense_path']=str(dense_path)
                views[name]=dict(**dense_paths,heads=paths,all_action_logits_path=str(logits_path),bbox_xyxy=box,
                    inverse_flip=flip,map_inverse_restored=True,elapsed=elapsed,cache_source=None,
                    unrestored_heads=raw_paths,unrestored_all_action_logits_path=str(raw_logits_path) if raw_logits_path else None)
                dedup[key]=name
            record=dict(row,image_sha256=image_hash,width=width,height=height,primary_head='ree',
              candidates=candidates,fixed=fixed,candidate_shortage=3-len(candidates),views=views,actual_new_calls=image_calls,logical_method_calls=4,
              protocol_sha256=config['protocol_sha256'])
            if comparisons:
                reference=comparisons[row['sample_id']]
                if candidates!=reference['candidates'] or fixed!=reference['fixed']:
                    raise ValueError('Recomputed candidate geometry differs from sealed old predictions')
                for name,view in views.items():
                    prior=reference['views'][name]
                    if view['bbox_xyxy']!=prior['bbox_xyxy']:
                        raise ValueError('Recomputed view differs from sealed old predictions')
                    pairs=[(view['heads'][h],prior['heads'][h],'maps') for h in HEADS]
                    if prior.get('all_action_logits_path'):
                        pairs.append((view['all_action_logits_path'],prior['all_action_logits_path'],'logits'))
                    for current_path,prior_path,kind in pairs:
                        current_value=np.load(current_path);prior_value=np.load(prior_path)
                        error=float(np.max(np.abs(current_value-prior_value)))
                        comparison_counts['max_abs_error']=max(comparison_counts['max_abs_error'],error)
                        comparison_counts[kind]+=1
                        if not np.array_equal(current_value,prior_value):
                            save(output/'comparison_failure.json',dict(sample_id=row['sample_id'],view=name,kind=kind,max_abs_error=error,current=current_path,prior=prior_path))
                            raise ValueError('Recomputed map/logits not bitwise equal; inspect before proceeding')
                comparison_counts['images']+=1
            append(predictions,record);completed+=1
            if completed==1 or completed%40==0 or completed==len(observations): print(json.dumps(status()),flush=True)
        stage='complete'
        result=dict(status(),status='SUCCESS',observations=len(observations),
          comparisons=comparison_counts,predictions_sha256=sha256(predictions),raw_map_files=len(list(map_root.glob('*.npy'))),
          peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save(output/'terminal.json',result); print(json.dumps(result),flush=True)
        return 0
    except BaseException as exc:
        stage='failed'
        result=dict(status(),status='FAILED',error=str(exc),traceback=traceback.format_exc())
        save(output/'terminal.json',result);print(json.dumps(result),flush=True)
        return 1
    finally:
        for hook in hooks: hook.remove()
        stop.set();lock.unlink(missing_ok=True)

if __name__=='__main__':
    raise SystemExit(main())
