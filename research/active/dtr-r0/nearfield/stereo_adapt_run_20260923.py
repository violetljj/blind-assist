"""Fixed one-pass stereo adapters and sealed public-only inference."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
from PIL import Image
import torch

from foundation_geometry_infer import sha,write,disparity_to_depth
from stereo_adapt_model import load_adapter,state_hashes,SELECTED,selected_name
from stereo_adapt_loss import targets,loss_weights,sequence_loss
from vpp_geometry_core import Projector,ray_hints,frame_seed

SEED=20260923
REPO=Path(__file__).resolve().parents[4]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def admitted_examples(manifest,stage):
    rows=manifest['frames']
    if stage=='train':
        assert manifest['authority']=='SUPERVISED_TRAIN_EXAMPLES_ONLY'
        assert manifest['target_role']=='training_ground_truth_NOT_SENSOR_INPUT'
        assert len(rows)==192 and len({r['layout_id'] for r in rows})==48
        assert all(r['split']=='train' and 'target_depth' in r for r in rows)
    else:
        assert manifest['authority']=='PUBLIC_RGB_TOF_CALIBRATION_ONLY_NO_TARGETS'
        assert len(rows)==96 and len({r['layout_id'] for r in rows})==24
        assert all(r['split']=='eval' for r in rows)
        forbidden=('target','native','truth','camera','bounds','objects')
        assert all(not any(any(word in k for word in forbidden) for k in r) for r in rows)
    assert len({(r['panel'],r['id']) for r in rows})==len(rows)
    for r in rows:
        assert all(r[k] and not any(x in r[k] for x in ('/','\\','..')) for k in ('panel','id'))
        for k in ('left','right','ranges','valid'):assert sha(r[k])==r[k+'_sha256']
    return rows


def historical(rgb_path,tof_path,rig):
    a,b=read(rgb_path),read(tof_path)
    assert a['authority']=='RGB_AND_CALIBRATION_ONLY' and b['authority']=='TOF_AND_CALIBRATION_ONLY'
    assert a['rig']==b['rig']==rig
    assert len(a['frames'])==len(b['frames'])==576
    lookup={(r['panel'],r['id']):r for r in b['frames']};assert len(lookup)==576
    result=[]
    for r in a['frames']:
        assert set(r)=={'panel','id','left','right','left_sha256','right_sha256'}
        t=lookup[(r['panel'],r['id'])]
        assert set(t)=={'panel','id','ranges','valid','ranges_sha256','valid_sha256'}
        q=dict(r,**{k:v for k,v in t.items() if k not in ('panel','id')})
        for k in ('left','right','ranges','valid'):assert sha(q[k])==q[k+'_sha256']
        result.append(q)
    return result


def project(row,rig,projector):
    images=[np.array(Image.open(row[k]).convert('RGB')) for k in ('left','right')]
    assert all(a.shape==(360,640,3) for a in images)
    hints,seeds=ray_hints(np.load(row['ranges'],allow_pickle=False),np.load(row['valid'],allow_pickle=False),rig)
    seed=frame_seed(row['panel'],row['id'])
    projected=projector.apply(*images,hints,seed)
    metadata=dict(seed_panel=row['panel'],id=row['id'],seed=seed,total_frame_hints=len(seeds),seeds=seeds,
        changed_left_pixels=int(np.any(projected[0]!=images[0],axis=2).sum()),
        changed_right_pixels=int(np.any(projected[1]!=images[1],axis=2).sum()),
        left_pixel_sha256=hashlib.sha256(projected[0].tobytes()).hexdigest(),
        right_pixel_sha256=hashlib.sha256(projected[1].tobytes()).hexdigest())
    return projected,metadata


def run(args):
    journal=read(os.environ['BLINDASSIST_ASSET_RUN_JOURNAL'])
    assert journal['state']=='running' and journal['reuse_preflight']['status']=='PASS'
    seal_path=REPO/'artifacts.local/evidence/ba-stereo-adapt-20260923/plan/launch-seal.json'
    seal=read(seal_path)
    for path,digest in seal['code'].items():assert sha(path)==digest,path
    for path in (args.config,args.inputs):assert sha(path)==seal['inputs'][str(path)],path
    out=args.output;assert out.resolve().is_relative_to((REPO/'artifacts.local').resolve())
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    manifest=read(args.inputs);rig=manifest['rig']
    assert rig==dict(width=640,height=360,hfov_deg=70.,baseline_m=.1,tof_hfov_deg=45.)
    rows=admitted_examples(manifest,args.stage)
    if args.stage=='train':
        assert args.arm in ('ordinary','balanced') and args.updates is None
        assert args.historical_rgb is None and args.historical_tof is None
        for row in rows:assert sha(row['target_depth'])==row['target_depth_sha256']
    elif args.arm=='baseline':
        assert args.updates is None and args.historical_rgb is None and args.historical_tof is None
    else:
        assert args.updates is not None and args.historical_rgb is not None and args.historical_tof is not None
        receipt=read(args.updates.parent/'receipt.json')
        assert receipt['status']=='PASS' and receipt['arm']==args.arm and receipt['stage']=='train'
        assert sha(args.updates)==receipt['hashes'][args.updates.name]
        rows=rows+historical(args.historical_rgb,args.historical_tof,rig)
    plan=read(args.config)
    cache=REPO/'artifacts.local/work/stereo-adapt-20260923/cache'
    for key,name in [('HF_HOME','hf'),('TORCH_HOME','torch'),('TMP','tmp'),('TEMP','tmp')]:
        p=cache/name;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
    os.environ['XFORMERS_DISABLED']='1'
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available()
    sys.path.insert(0,str(REPO));from tools.research_backend import torch_observation
    started=time.perf_counter();adapter=None;hashes={}
    receipt=dict(status='RUNNING',stage=args.stage,arm=args.arm,frames=0,seed=SEED,
        launch_seal_sha256=sha(seal_path),
        config_sha256=sha(args.config),input_manifest_sha256=sha(args.inputs),runner_sha256=sha(__file__),
        model_code_sha256=sha(Path(__file__).with_name('stereo_adapt_model.py')),
        loss_code_sha256=sha(Path(__file__).with_name('stereo_adapt_loss.py')))
    if args.historical_rgb:
        receipt.update(historical_rgb_sha256=sha(args.historical_rgb),historical_tof_sha256=sha(args.historical_tof))
    try:
        adapter=load_adapter(args.config,checkpoint_recurrence=False)
        if args.updates:adapter.load_updates(args.updates);receipt['updates_sha256']=sha(args.updates)
        projector=Projector(plan['vpp_source'],plan['vpp_sha256'])
        torch.cuda.reset_peak_memory_stats()
        if args.stage=='train':
            before=state_hashes(adapter.model);write(out/'before-hashes.json',before)
            order=list(range(192));random.Random(SEED).shuffle(order)
            write(out/'sample-order.json',[dict(step=i,id=rows[k]['id'],layout_id=rows[k]['layout_id']) for i,k in enumerate(order)])
            optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-5,weight_decay=1e-4)
            scaler=torch.amp.GradScaler('cuda',init_scale=128.)
            with (out/'steps.jsonl').open('x',encoding='utf-8') as log:
                for step,index in enumerate(order):
                    row=rows[index];begin=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                    images,hint=project(row,rig,projector)
                    target,valid,regions=targets(np.load(row['target_depth'],allow_pickle=False))
                    weights=loss_weights(valid,regions,args.arm)
                    t=torch.as_tensor(target,device='cuda');w=torch.as_tensor(weights,device='cuda')
                    sequence=adapter.predict(*images,iters=8,training=True,return_sequence=True)
                    assert len(sequence)==8 and all(p.requires_grad and torch.isfinite(p).all() for p in sequence)
                    loss=sequence_loss(sequence,t,w);assert torch.isfinite(loss)
                    scaler.scale(loss).backward();scaler.unscale_(optimizer)
                    maxima={k:[] for k in SELECTED};missing=[]
                    for name,param in adapter.model.named_parameters():
                        if not selected_name(name):assert param.grad is None
                        elif param.grad is None:missing.append(name)
                        else:
                            assert torch.isfinite(param.grad).all(),name
                            for prefix in SELECTED:
                                if name.startswith(prefix+'.'):maxima[prefix].append(param.grad.detach().abs().max())
                    gradmax={k:float(torch.stack(v).max()) if v else 0. for k,v in maxima.items()}
                    assert all(v>0 for v in gradmax.values())
                    norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.);assert torch.isfinite(norm)
                    scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
                    entry=dict(step=step,id=row['id'],layout_id=row['layout_id'],family=row['family'],appearance=row['appearance'],
                        loss=float(loss.detach()),valid_pixels=int(valid.sum()),stratum_counts={str(k):int((regions==k).sum()) for k in range(4)},
                        stratum_weight_mass={str(k):float(weights[regions==k].sum()) for k in range(4)},
                        gradient_norm=float(norm),gradient_max_by_module=gradmax,missing_gradient_parameters=missing,
                        scaler_scale=float(scaler.get_scale()),seconds=time.perf_counter()-begin,
                        memory_allocated_bytes=torch.cuda.memory_allocated(),memory_reserved_bytes=torch.cuda.memory_reserved(),
                        total_frame_hints=hint['total_frame_hints'])
                    log.write(json.dumps(entry,allow_nan=False)+'\n');log.flush()
                    receipt['frames']=step+1
                    write(out/'progress.json',entry)
                    if step%16==0:print('ADAPT_TRAIN',args.arm,step+1,round(entry['loss'],5),flush=True)
                    receipt['observed_backend']=asdict(torch_observation(model=adapter.model,output=sequence[-1]))
                    del sequence,loss,t,w
            after=state_hashes(adapter.model);write(out/'after-hashes.json',after)
            assert before['frozen']==after['frozen'] and before['buffers']==after['buffers']
            changed=[k for k in before['selected'] if before['selected'][k]!=after['selected'][k]]
            assert all(any(k.startswith(p+'.') for k in changed) for p in SELECTED)
            adapter.save_updates(out/'final-delta.pt')
            receipt.update(steps=192,iterations=8,selected_modules=adapter.selected_counts,
                changed_selected_parameters=changed,frozen_parameters_unchanged=True,all_registered_buffers_unchanged=True,
                final_checkpoint_only=True,optimizer=dict(name='AdamW',lr=1e-5,weight_decay=1e-4,gradient_norm_cap=1.,batch=1))
            del optimizer,scaler
        else:
            hints=[];times=[]
            for index,row in enumerate(rows):
                begin=time.perf_counter();images,hint=project(row,rig,projector)
                prediction=adapter.predict(*images,iters=32,training=False)
                assert not prediction.requires_grad and torch.isfinite(prediction).all()
                torch.cuda.synchronize()
                panel='new_eval' if row.get('split')=='eval' else row['panel']
                hint['panel']=panel;hints.append(hint)
                disparity=prediction.float().cpu().numpy()
                raw,depth=disparity_to_depth(disparity,rig)
                for folder,array in [('raw_disparity',disparity),('raw_depth',raw),('depth',depth)]:
                    p=out/panel/folder/(row['id']+'.npy');p.parent.mkdir(parents=True,exist_ok=True);np.save(p,array,allow_pickle=False)
                times.append(dict(panel=panel,id=row['id'],seconds=time.perf_counter()-begin))
                receipt['frames']=index+1;receipt['observed_backend']=asdict(torch_observation(model=adapter.model,output=prediction))
                write(out/'progress.json',dict(frames=index+1,total=len(rows),**times[-1]))
                if index%24==0:print('ADAPT_INFER',args.arm,index+1,'/',len(rows),flush=True)
                del prediction
            write(out/'hints.json',dict(input_manifest_sha256=sha(args.inputs),frames=hints))
            write(out/'manifest.json',dict(rig=rig,arm=args.arm,iterations=32,frames=times,
                new_eval_seed_panel='adapt',new_eval_output_panel='new_eval',historical_seed_panels='mz101/mz102',
                target_access='NONE_PUBLIC_INPUTS_ONLY',configuration=plan))
            receipt.update(iterations=32,target_access='NONE_PUBLIC_INPUTS_ONLY',pair_mean_s=float(np.mean([r['seconds'] for r in times])))
        hashes={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}
        receipt.update(status='PASS',hashes=hashes,elapsed_s=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as exc:
        receipt.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),elapsed_s=time.perf_counter()-started,
            hashes={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='receipt.json'})
        write(out/'receipt.json',receipt);raise
    finally:
        adapter=None;gc.collect();torch.cuda.empty_cache()
    receipt['owned_model_released']=True;write(out/'receipt.json',receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('train','infer'),required=True)
    p.add_argument('--arm',choices=('ordinary','balanced','baseline'),required=True)
    for name in ('output','config','inputs'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('historical-rgb','historical-tof','updates'):p.add_argument('--'+name,type=Path)
    run(p.parse_args())
