"""All-bin public-query weak-mass features on consumed ideal Development.

Observations, ambient/bias and public membership only. Split positive/negative
before history averaging; no target boxes/categories or fitted amplitude gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009/data'
GEOMETRY = ROOT/'artifacts.local/work/cnh-graded-corridor-dev-20261010/features'
OUTPUT = ROOT/'artifacts.local/work/cnh-graded-weak-mass-dev-20261010/features'
BIAS = ROOT/'artifacts.local/work/cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy'
FRAMES = np.arange(3,16)
REGIONS = ('inner','ring')
METRICS = ('current_positive_density','current_negative_density',
    'current_positive_participation_fraction','current_top8_bin_mass_share',
    'past_positive_density_mean','past_positive_density_max','past_positive_density_std',
    'past_negative_density_mean','effective_positive_mass_frames',
    'current_over_past_max','current_over_past_mean','past_bounded_positive_density_mean',
    'past_positive_cancellation_density','available_history_frames')
NAMES = [f'{r}_{m}' for r in REGIONS for m in METRICS]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<23),b''): h.update(block)
    return h.hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def normalize(hist,ambient,bias):
    denominator=np.sqrt(np.maximum(16*ambient.astype(np.float32)[...,None]+np.maximum(bias,0),1e-9))
    return ((hist.astype(np.float32)-bias)/denominator).astype(np.float16)


def extract(z,geometry,backend='cuda'):
    """Return observation x frame x query x 28, using only frames <= now.

    Density denominator is sum membership, not supported-bin count. Each
    historical observation is normalized separately before temporal averaging.
    Participation denominator counts bins with w>0. Invalid ratios stay NaN.
    """
    if backend=='cuda':
        import torch
        a=lambda x:torch.as_tensor(x,device='cuda')
        sm=lambda x,d:x.sum(dim=d)
        stack=lambda xs,d:torch.stack(xs,dim=d)
        maximum=lambda x,v:x.clamp_min(v)
        where=torch.where
        top=lambda x:x.topk(8,dim=-1).values.sum(-1)
        sqrt=torch.sqrt
        mx=lambda x,d:x.amax(dim=d)
        finite=torch.isfinite
        host=lambda x:x.detach().cpu().numpy()
    else:
        a=np.asarray; sm=lambda x,d:x.sum(axis=d)
        stack=lambda xs,d:np.stack(xs,axis=d)
        maximum=np.maximum; where=np.where
        top=lambda x:np.partition(x,-8,axis=-1)[...,-8:].sum(-1)
        sqrt=np.sqrt; mx=lambda x,d:x.max(axis=d)
        finite=np.isfinite; host=lambda x:x
    z=a(z.astype(np.float32).reshape(len(z),16,1024))
    logs= (torch.sign(z)*torch.log1p(z.abs()) if backend=='cuda' else np.sign(z)*np.log1p(np.abs(z)))
    out,valids=[],[]
    for j,f in enumerate(FRAMES):
        length=int(geometry['length'][j]); begin=int(f)-length+1
        # Explicit axis selection preserves Q,R,T,B order.
        w=a(np.take(geometry['membership'][j], [0,2],axis=1)[:,:,8-length:])
        mass=sm(w,-1)
        supported=mass>0
        count=sm(supported,-1)
        safe_mass=maximum(mass,1e-20)
        signed=logs[:,None,None,begin:int(f)+1]*w[None]
        positive=maximum(signed,0)
        negative=maximum(-signed,0)
        pd=sm(positive,-1)/safe_mass[None]
        nd=sm(negative,-1)/safe_mass[None]
        def avg(x): return sm(x*supported[None],-1)/maximum(count[None],1)
        mean=avg(pd); maxpos=mx(pd,-1)
        std=sqrt(maximum(avg(pd*pd)-mean*mean,0))
        current=positive[...,-1,:]
        total=sm(current,-1); squares=sm(current*current,-1)
        nbins=sm(w[...,-1,:]>0,-1)
        participation=total*total/maximum(squares,1e-20)/maximum(nbins[None],1)
        peakshare=top(current)/maximum(total,1e-20)
        frame_square=sm(pd*pd,-1)
        effective=sm(pd,-1)**2/maximum(frame_square,1e-20)
        # Normalize each observation before signed history mean: cancellation
        # difference is then a nonnegative Jensen gap, not a mass-change proxy.
        signed_density_mean=sm(signed/safe_mass[None,...,None],-2)/maximum(count[None,...,None],1)
        cancellation=mean-sm(maximum(signed_density_mean,0),-1)
        poslogs=maximum(logs[:,None,None,begin:int(f)+1],0)
        bounded=sm((poslogs/(1+poslogs))*w[None],-1)/safe_mass[None]
        vals=stack((pd[...,-1],nd[...,-1],participation,peakshare,mean,maxpos,std,
            avg(nd),effective,pd[...,-1]/maximum(maxpos,1e-20),
            pd[...,-1]/maximum(mean,1e-20),avg(bounded),cancellation,
            pd[...,-1]*0+length),-1)
        v=finite(vals)&(count[None,...,None]>0)
        if backend=='cuda': v=v.expand_as(vals).clone()
        else: v=np.broadcast_to(v,vals.shape).copy()
        v[...,:4]&=supported[None,...,-1,None]
        v[...,2]&=(squares>0)&(nbins[None]>0)
        v[...,3]&=total>0
        v[...,8]&=frame_square>0
        v[...,9]&=(maxpos>0)&supported[None,...,-1]
        v[...,10]&=(mean>0)&supported[None,...,-1]
        vals=where(v,vals,float('nan'))
        out.append(host(vals).reshape(len(z),2,28).astype(np.float32))
        valids.append(host(v).reshape(len(z),2,28))
    if backend=='cuda': torch.cuda.synchronize()
    return np.stack(out,1),np.stack(valids,1)


def run(output):
    start=time.monotonic(); gpu_seconds=0.; phase='declare'
    output=output.resolve()
    if not output.is_relative_to((ROOT/'artifacts.local').resolve()): raise ValueError('Canonical artifact routing required')
    if output.exists(): raise FileExistsError('Preserve existing output')
    output.mkdir(parents=True)
    save(output/'PLAN.json',dict(task='CNH_GRADED_WEAK_MASS_FEATURES_DEV_20261010',
        goal='Preserve low-amplitude all-bin positive mass before temporal cancellation',
        scope='cal/validation 384 scenes x4 replicas x13 frames x2 public queries x28 features',
        CPU_command_wall_seconds_cap=240,GPU_phase_wall_seconds_cap=120,
        adjustable='Implementation repair only; fixed feature definitions, no amplitude cutoff or outcome-driven selection',
        deliverables='features/valid/names/identity; source/input hashes; GPU receipt; CPU parity and future-prefix checks',
        truth_boundary='Read hist, physics ambient, bias and geometry public sensor/query/identity/membership; no boxes/category/scene_rows',
        history='past<=8 including available f0..2, membership at current query; per-observation density then positive/negative temporal summaries',
        missing='zero denominator => NaN/valid false; no evidence is never free',source_sha256=sha(__file__)))
    checks={}; inputs={}
    try:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required for tensor reduce')
        bias=np.load(BIAS).astype(np.float32)
        for split in ('cal','validation'):
            phase=f'{split} load public inputs'
            directory=SOURCE/split
            with np.load(directory/'geometry.npz',allow_pickle=False) as f:
                sensor,query,ids,uids=(f[k] for k in ('sensor','public_query','scene_ids','scene_uids'))
            with np.load(directory/'physics.npz',allow_pickle=False) as f:
                ambient=f['ambient']
                np.testing.assert_array_equal(f['sensor'],sensor)
                np.testing.assert_array_equal(f['public_query'],query)
            gpath=GEOMETRY/f'{split}_public_geometry.npz'
            with np.load(gpath,allow_pickle=False) as f:
                geom={k:f[k] for k in ('membership','length')}
                np.testing.assert_array_equal(f['sensor'],sensor)
                np.testing.assert_array_equal(f['public_query'],query)
            assert geom['membership'].shape==(13,2,3,8,1024)
            np.testing.assert_array_equal(geom['length'],np.minimum(FRAMES+1,8))
            hist=np.load(directory/'hist.npy',mmap_mode='r',allow_pickle=False)
            assert hist.shape==(384,4,16,8,8,16)
            out=np.empty((384,4,13,2,28),np.float32); valid=np.empty(out.shape,bool)
            phase=f'{split} CPU/CUDA parity and causal prefix'
            sample=normalize(np.array(hist[[0,383],0],copy=True),ambient,bias)
            ref,rv=extract(sample,geom,'cpu')
            mark=time.monotonic(); actual,av=extract(sample,geom,'cuda'); gpu_seconds+=time.monotonic()-mark
            np.testing.assert_array_equal(rv,av)
            np.testing.assert_allclose(actual[rv],ref[rv],atol=3e-4,rtol=3e-5)
            changed=sample.copy(); changed[:,8:]=np.float16(-25)
            alt,altv=extract(changed,geom,'cpu')
            np.testing.assert_array_equal(ref[:,:5],alt[:,:5])
            np.testing.assert_array_equal(rv[:,:5],altv[:,:5])
            # Separate-before-mean cancellation must be >=0 up to FP32 sum error.
            for offset in (12,26): assert np.nanmin(ref[...,offset])>=-1e-5
            checks[split]=dict(CPU_CUDA_validity_exact=True,CPU_CUDA_max_absolute_difference=float(np.max(np.abs(actual[rv]-ref[rv]))),
                sample_scene_indices=[0,383],replica=0,future_mutation_from_frame=8,through_frame=7,prefix_exact=True)
            phase=f'{split} all-bin GPU reduction'
            for begin in range(0,384,16):
                if time.monotonic()-start>=240 or gpu_seconds>=120: raise TimeoutError('Declared feature budget reached')
                normalized=normalize(np.array(hist[begin:begin+16],copy=True),ambient,bias).reshape(-1,16,8,8,16)
                mark=time.monotonic(); vals,v=extract(normalized,geom,'cuda'); gpu_seconds+=time.monotonic()-mark
                out[begin:begin+16]=vals.reshape(16,4,13,2,28)
                valid[begin:begin+16]=v.reshape(16,4,13,2,28)
            np.savez_compressed(output/f'{split}_features.npz',features=out,valid=valid,names=np.array(NAMES),
                scene_ids=ids,scene_uids=uids,replica=np.arange(4),frames=FRAMES,
                queries=np.array(['HEAD','BODY']),cache_row_index=np.arange(19968).reshape(384,4,13))
            inputs[split]={str(p.relative_to(ROOT)):sha(p) for p in
                (directory/'hist.npy',directory/'physics.npz',directory/'geometry.npz',gpath)}
            del hist
        phase='receipts'
        save(output/'schema.json',dict(names=NAMES,metrics=METRICS,regions=REGIONS,
            shape_axes=['scene384','replica4','frame13','query2','feature28'],dtype='float32',valid_dtype='bool',
            input_sha256=inputs,bias_sha256=sha(BIAS),source_sha256=sha(__file__),physical_output=str(output),
            density='sum weighted signedlog positive/negative / sum membership per historical observation; positive/negative split before history',
            participation='(sum weighted positive)^2/sum(weighted positive squared)/count(membership>0)',
            top8='8 largest weighted positive bin masses across all1024bins; no local-peak selection',
            bounded='positive log amplitude/(1+positive log amplitude), before membership density and time mean',
            cancellation='mean positive per-observation normalized bin density minus positive part of signed history normalized-bin mean; no target attribution',
            history='means/std over available supported past<=8 observations; available_history_frames is observed history length',
            missing='NaN/valid false for no membership or ratio denominator zero; never labels free'))
        save(output/'checks.json',checks)
        save(output/'result.json',dict(status='COMPLETE',backend='cuda',device=torch.cuda.get_device_name(),
            torch_version=torch.__version__,GPU_stage_seconds=gpu_seconds,command_wall_seconds=time.monotonic()-start,
            source_sha256=sha(__file__),feature_shape=[384,4,13,2,28],splits=['cal','validation'],
            inference=0,training=0,new_sampling=0,limitation='Consumed simulated Development; public angular-node membership; no target identification/free evidence/device proof'))
    except BaseException as e:
        save(output/f'failure_{time.time_ns()}.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),
            command_wall_seconds=time.monotonic()-start,GPU_stage_seconds=gpu_seconds))
        raise
    finally:
        if 'torch' in locals(): torch.cuda.empty_cache()
        save(output/'execution_receipt.json',dict(command=[sys.executable,*sys.argv],command_wall_seconds=time.monotonic()-start,
            GPU_stage_seconds=gpu_seconds,source_sha256=sha(__file__),plan_sha256=sha(output/'PLAN.json'),
            persistent_processes=0,outputs_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='execution_receipt.json'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    run(parser.parse_args().output)
