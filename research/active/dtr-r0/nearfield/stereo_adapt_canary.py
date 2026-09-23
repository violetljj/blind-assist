"""At most two synthetic optimizer steps; discard all adapted weights afterward."""
import argparse
from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from stereo_adapt_model import load_adapter,state_hashes,SELECTED
from foundation_geometry_infer import write,sha
from vpp_geometry_core import ray_hints,Projector


def run(config,output,checkpoint_recurrence):
    if output.exists() and any(output.iterdir()):raise ValueError('Preserve existing canary')
    output.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[4]
    assert output.resolve().is_relative_to((repo/'artifacts.local').resolve())
    sys.path.insert(0,str(repo));from tools.research_backend import torch_observation
    for key,name in [('HF_HOME','hf'),('TORCH_HOME','torch'),('TMP','tmp'),('TEMP','tmp')]:
        path=output/'cache'/name;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    torch.set_num_threads(4);torch.manual_seed(73);torch.cuda.manual_seed_all(73)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    started=time.perf_counter();receipt=dict(status='RUNNING',authority='SYNTHETIC_ENGINEERING_ONLY',
        checkpoint_recurrence=checkpoint_recurrence,config_sha256=sha(config),steps=[],weights_saved=False)
    adapter=None
    try:
        adapter=load_adapter(config,checkpoint_recurrence=checkpoint_recurrence)
        plan=adapter.plan;rig=dict(width=640,height=360,hfov_deg=70.,baseline_m=.1,tof_hfov_deg=45.)
        rng=np.random.default_rng(73);left=rng.integers(0,256,(360,640,3),dtype=np.uint8)
        disparity=20.;right=np.zeros_like(left);right[:,:620]=left[:,20:]
        # Exact synthetic axial plane; radial ranges reflect centre-ray obliquity.
        z=320/math.tan(math.radians(35))*.1/disparity
        ranges=[]
        for row in range(8):
            for col in range(8):
                az=math.radians(-22.5+(col+.5)*45/8);el=math.radians(22.5-(row+.5)*45/8)
                ranges.append(z*math.sqrt(1+math.tan(az)**2+math.tan(el)**2))
        hints,seeds=ray_hints(np.array(ranges),np.ones(64,bool),rig)
        projector=Projector(plan['vpp_source'],plan['vpp_sha256'])
        left,right=projector.apply(left,right,hints,73)
        receipt.update(selected_modules=adapter.selected_counts,total_parameters=adapter.checkpoint_info['parameters'],
            synthetic_disparity=disparity,synthetic_axial_depth=z,public_hints=len(seeds),
            all_modules_eval=True,batch_norm_count=sum(isinstance(m,torch.nn.modules.batchnorm._BatchNorm) for m in adapter.model.modules()),
            selected_batch_norm=[name for name,m in adapter.model.named_modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)
                and any(name.startswith(p+'.') for p in SELECTED)])
        before=state_hashes(adapter.model)
        write(output/'before-hashes.json',before)
        optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-5,weight_decay=1e-4)
        scaler=torch.amp.GradScaler('cuda',init_scale=128.)
        torch.cuda.reset_peak_memory_stats()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();begin=time.perf_counter()
            sequence=adapter.predict(left,right,iters=8,training=True,return_sequence=True)
            prediction=sequence[-1]
            assert prediction.shape==(360,640) and prediction.requires_grad
            target=torch.full_like(prediction,disparity)
            weights=[.9**(7-i) for i in range(8)]
            loss=sum(w*torch.nn.functional.l1_loss(p[:,20:],target[:,20:]) for w,p in zip(weights,sequence))/sum(weights)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            grads={prefix:[] for prefix in SELECTED};missing=[]
            for name,param in adapter.model.named_parameters():
                if not param.requires_grad:assert param.grad is None,name
                elif param.grad is None:missing.append(name)
                else:
                    assert torch.isfinite(param.grad).all(),name
                    for prefix in SELECTED:
                        if name.startswith(prefix+'.'):grads[prefix].append(float(param.grad.abs().max()))
            assert all(v and max(v)>0 for v in grads.values())
            gradnorm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.)
            scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
            receipt['steps'].append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(gradnorm),
                gradient_max_by_module={k:max(v) for k,v in grads.items()},missing_gradient_parameters=missing,
                seconds=time.perf_counter()-begin,backend=asdict(torch_observation(model=adapter.model,output=prediction)),
                allocated_peak_bytes=torch.cuda.max_memory_allocated(),reserved_peak_bytes=torch.cuda.max_memory_reserved()))
            write(output/'progress.json',receipt)
            del prediction,target,loss,sequence
        after=state_hashes(adapter.model);write(output/'after-hashes.json',after)
        assert before['frozen']==after['frozen'],'Frozen parameters changed'
        assert before['buffers']==after['buffers'],'Registered buffers changed'
        changed=[name for name in before['selected'] if before['selected'][name]!=after['selected'][name]]
        assert all(any(name.startswith(prefix+'.') for name in changed) for prefix in SELECTED)
        # Full32-iteration eval verifies same API without gradients after updates.
        begin=time.perf_counter();pred=adapter.predict(left,right,iters=32,training=False)
        torch.cuda.synchronize();assert torch.isfinite(pred).all() and not pred.requires_grad
        receipt.update(status='PASS',changed_selected_parameters=changed,changed_selected_parameter_count=len(changed),
            frozen_parameters_unchanged=True,all_registered_buffers_unchanged=True,
            eval32_seconds=time.perf_counter()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_s=time.perf_counter()-started,
            model_code_sha256=sha(Path(__file__).with_name('stereo_adapt_model.py')),canary_code_sha256=sha(Path(__file__)))
        del pred,optimizer,scaler
    except BaseException as exc:
        receipt.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),elapsed_s=time.perf_counter()-started)
        write(output/'receipt.json',receipt);raise
    finally:
        adapter=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    receipt['owned_weights_discarded']=True
    write(output/'receipt.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint-recurrence',action='store_true');args=p.parse_args()
    run(args.config,args.output,args.checkpoint_recurrence)
