"""Localized all-bin weak-mass profiles; post-hoc consumed Development pilot.

Public mean supported-node forward depth assigns each native cell to one of
12 fixed .225m bins [.3,3]. Positive/negative mass is split before history;
this approximation is not a node splat, physical occupancy or target identity.
"""
import argparse
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import cnh_graded_weak_mass_features_dev as M

ROOT=M.ROOT
PARENT=ROOT/'artifacts.local/work/cnh-graded-weak-mass-dev-20261010'
OUTPUT=PARENT/'localized/features'
FRAMES=M.FRAMES
REGIONS=M.REGIONS
METRICS=('current_positive_density','current_negative_density',
         'past_positive_density_mean','past_negative_density_mean')
EDGES=np.linspace(.3,3,13,dtype=np.float32)
NAMES=[f'{r}_{m}_depth{d:02d}' for r in REGIONS for m in METRICS for d in range(12)]


def extract(normalized,geometry,backend='cuda'):
    """Only past/current observed histograms and cached public geometry."""
    if backend=='cuda':
        import torch
        a=lambda v:torch.as_tensor(v,device='cuda')
        sm=lambda v,d:v.sum(dim=d)
        maximum=lambda v,k:v.clamp_min(k)
        stack=lambda vs,d:torch.stack(vs,dim=d)
        where=torch.where
        finite=torch.isfinite
        host=lambda v:v.detach().cpu().numpy()
    else:
        a=np.asarray; sm=lambda v,d:v.sum(axis=d)
        maximum=np.maximum; stack=lambda vs,d:np.stack(vs,axis=d)
        where=np.where; finite=np.isfinite; host=lambda v:v
    z=a(normalized.astype(np.float32).reshape(len(normalized),16,1024))
    logs=torch.sign(z)*torch.log1p(z.abs()) if backend=='cuda' else np.sign(z)*np.log1p(np.abs(z))
    output=[]; validity=[]
    for j,frame in enumerate(FRAMES):
        length=int(geometry['length'][j]); begin=int(frame)-length+1
        weights=np.take(geometry['membership'][j],[0,2],axis=1)[:,:,8-length:]
        depths=np.take(geometry['depth'][j],[0,2],axis=1)[:,:,8-length:]
        # Supported-cell depth is within the public gate. Upper edge z=3 goes
        # to the last bin, like a closed histogram endpoint.
        if np.any(weights>0):
            assert np.min(depths[weights>0])>=.3-1e-6 and np.max(depths[weights>0])<=3+1e-6
        index=np.clip(np.floor((depths-.3)/.225).astype(np.int64),0,11)
        membership_by_depth=(index[...,None,:]==np.arange(12)[:,None])*weights[...,None,:]
        pw=a(membership_by_depth.astype(np.float32))  # Q,R,T,D,C
        mass=sm(pw,-1); supported=mass>0
        safe=maximum(mass,1e-20)
        past=logs[:,None,None,begin:int(frame)+1,:]
        positive=maximum(past,0); negative=maximum(-past,0)
        if backend=='cuda':
            ps=torch.einsum('bqrtc,qrtdc->bqrtd',positive.expand(-1,2,2,-1,-1),pw)
            ns=torch.einsum('bqrtc,qrtdc->bqrtd',negative.expand(-1,2,2,-1,-1),pw)
        else:
            # Independent reductions by depth masks, no Torch/device kernel.
            ps=np.sum(positive[...,None,:]*pw[None],axis=-1)
            ns=np.sum(negative[...,None,:]*pw[None],axis=-1)
        pd=ps/safe[None]; nd=ns/safe[None]
        count=sm(supported,-2)
        meanpositive=sm(pd*supported[None],-2)/maximum(count[None],1)
        meannegative=sm(nd*supported[None],-2)/maximum(count[None],1)
        values=stack((pd[...,-1,:],nd[...,-1,:],meanpositive,meannegative),-2)
        currentv=supported[..., -1,:][None] & finite(values[...,0,:])
        pastv=(count[None]>0)&finite(values[...,2,:])
        valid=stack((currentv,currentv,pastv,pastv),-2)
        if backend=='cuda': valid=valid.expand_as(values)
        else: valid=np.broadcast_to(valid,values.shape)
        values=where(valid,values,float('nan'))
        output.append(host(values).reshape(len(normalized),2,96).astype(np.float32))
        validity.append(host(valid).reshape(len(normalized),2,96))
    if backend=='cuda': torch.cuda.synchronize()
    return np.stack(output,1),np.stack(validity,1)


def run(output):
    start=time.monotonic(); gpu_seconds=0.; phase='declare'
    output=output.resolve()
    if not output.is_relative_to((ROOT/'artifacts.local').resolve()): raise ValueError('Canonical artifact routing required')
    if output.exists(): raise FileExistsError('Preserve old output')
    output.mkdir(parents=True)
    M.save(output/'PLAN.json',dict(task='CNH_GRADED_WEAK_PROFILE_FEATURES_DEV_20261010',
        amendment_of=str(PARENT/'TASK.json'),parent_task_sha256=M.sha(PARENT/'TASK.json'),
        failed_original_bank='28 global weak-mass features failed stable weak-horizontal rescue; 955 HEAD rescue2/loss0 same as control; 956/957 losses. Preserve original features and run.',
        failed_run_metrics_sha256=M.sha(PARENT/'metrics.json'),
        goal='Materially different localized depth profile preserves spatial weak evidence before global pooling',
        scope='cal/validation384x4x13x2x96; no new hold, model fit, sampling or outcome-driven feature choice',
        CPU_command_wall_seconds_remaining_cap=230,GPU_phase_wall_seconds_remaining_cap=113,
        previous_feature_CPU_command_seconds=9.875,previous_feature_GPU_phase_seconds=6.864,
        profile='12 fixed z bins [.3,3], width .225m; assign cached average supported-node depth per native cell, weight membership inner/ring',
        history='Current positive/negative density and mean per-observation positive/negative density over supported history in each depthbin; split sign before mean, include actual f0..2',
        missing='Depthbin membershipmass=0 => current NaN/false; no supported past observations => history NaN/false; zero positive density with membership remains measured zero, never free label',
        truth_boundary='Feature path only hist, ambient/bias and public geometry sensor/query/membership/depth/length/scene identity',
        lane='Post-hoc EXPLORE on consumed simulated Development, prompted by failed global pooling; not fresh confirmation',
        adjustable='Implementation repairs; preserve fixed depth bins and 96-feature design',
        acceptance='Focused CPU/CUDA value/valid parity and future-mutation prefix once; source/input hashes and GPU receipt',
        source_sha256=M.sha(__file__),normalization_source_sha256=M.sha(M.__file__)))
    checks={}; inputs={}
    try:
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        bias=np.load(M.BIAS).astype(np.float32)
        for split in ('cal','validation'):
            directory=M.SOURCE/split; phase=f'{split} load public inputs'
            with np.load(directory/'geometry.npz',allow_pickle=False) as f:
                sensor,query,ids,uids=(f[k] for k in ('sensor','public_query','scene_ids','scene_uids'))
            with np.load(directory/'physics.npz',allow_pickle=False) as f:
                ambient=f['ambient']; np.testing.assert_array_equal(f['sensor'],sensor)
                np.testing.assert_array_equal(f['public_query'],query)
            gpath=M.GEOMETRY/f'{split}_public_geometry.npz'
            with np.load(gpath,allow_pickle=False) as f:
                geometry={k:f[k] for k in ('membership','depth','length')}
                np.testing.assert_array_equal(f['sensor'],sensor); np.testing.assert_array_equal(f['public_query'],query)
            assert geometry['membership'].shape==(13,2,3,8,1024)
            np.testing.assert_array_equal(geometry['length'],np.minimum(FRAMES+1,8))
            hist=np.load(directory/'hist.npy',mmap_mode='r',allow_pickle=False)
            assert hist.shape==(384,4,16,8,8,16)
            phase=f'{split} focused parity and prefix'
            sample=M.normalize(np.array(hist[[0,383],0],copy=True),ambient,bias)
            ref,rv=extract(sample,geometry,'cpu')
            mark=time.monotonic(); actual,av=extract(sample,geometry,'cuda'); gpu_seconds+=time.monotonic()-mark
            np.testing.assert_array_equal(rv,av)
            np.testing.assert_allclose(actual[rv],ref[rv],rtol=3e-5,atol=3e-4)
            altered=sample.copy(); altered[:,8:]=np.float16(-25)
            other,ov=extract(altered,geometry,'cpu')
            np.testing.assert_array_equal(ref[:,:5],other[:,:5]); np.testing.assert_array_equal(rv[:,:5],ov[:,:5])
            checks[split]=dict(sample_scene_indices=[0,383],replica=0,CPU_CUDA_validity_exact=True,
                CPU_CUDA_max_absolute_difference=float(np.max(np.abs(actual[rv]-ref[rv]))),
                future_mutation_from_frame=8,through_frame=7,prefix_exact=True)
            out=np.empty((384,4,13,2,96),np.float32); valid=np.empty(out.shape,bool)
            phase=f'{split} all profile features'
            for begin in range(0,384,16):
                if time.monotonic()-start>=230 or gpu_seconds>=113: raise TimeoutError('Remaining feature budget reached')
                z=M.normalize(np.array(hist[begin:begin+16],copy=True),ambient,bias).reshape(-1,16,8,8,16)
                mark=time.monotonic(); values,v=extract(z,geometry,'cuda'); gpu_seconds+=time.monotonic()-mark
                out[begin:begin+16]=values.reshape(16,4,13,2,96); valid[begin:begin+16]=v.reshape(16,4,13,2,96)
            np.savez_compressed(output/f'{split}_features.npz',features=out,valid=valid,names=np.array(NAMES),
                scene_ids=ids,scene_uids=uids,replica=np.arange(4),frames=FRAMES,queries=np.array(['HEAD','BODY']),
                cache_row_index=np.arange(19968).reshape(384,4,13),depth_edges=EDGES)
            inputs[split]={str(p.relative_to(ROOT)):M.sha(p) for p in
                (directory/'hist.npy',directory/'geometry.npz',directory/'physics.npz',gpath)}
            del hist
        M.save(output/'schema.json',dict(names=NAMES,metrics=METRICS,regions=REGIONS,
            shape_axes=['scene384','replica4','frame13','query2','feature96'],dtype='float32',valid_dtype='bool',
            depth_edges_m=EDGES.tolist(),depth_width_m=.225,
            profile='Membership weighted each native-cell contribution assigned by cached average supported-node z to one forward depth bin; approximate mean-node assignment, not full node splat',
            density='Per-observation sum positive/negative signedlog native amplitudes weighted by membership / per-depthbin membershipmass',
            history='Mean over past<=8 observations with positive public membershipmass in that depth bin; sign split before temporal averaging',
            missing='No public mass => NaN/false. Nonnegative zero density with public mass is measured zero, not free/clear evidence.',
            truth_boundary='No target boxes/categories/scene_rows/target-attribution report in feature extraction',
            input_sha256=inputs,bias_sha256=M.sha(M.BIAS),source_sha256=M.sha(__file__),normalization_source_sha256=M.sha(M.__file__)))
        M.save(output/'checks.json',checks)
        M.save(output/'result.json',dict(status='COMPLETE',backend='cuda',device=torch.cuda.get_device_name(),
            torch_version=torch.__version__,GPU_stage_seconds=gpu_seconds,command_wall_seconds=time.monotonic()-start,
            feature_shape=[384,4,13,2,96],source_sha256=M.sha(__file__),inference=0,training=0,new_sampling=0,
            limitation='Post-hoc consumed simulated Development; public mean-node depth approximation; no physical occupancy/target identity/device proof'))
    except BaseException as e:
        M.save(output/f'failure_{time.time_ns()}.json',dict(phase=phase,error=repr(e),traceback=traceback.format_exc(),
            command_wall_seconds=time.monotonic()-start,GPU_stage_seconds=gpu_seconds))
        raise
    finally:
        if 'torch' in locals(): torch.cuda.empty_cache()
        M.save(output/'execution_receipt.json',dict(command=[sys.executable,*sys.argv],command_wall_seconds=time.monotonic()-start,
            GPU_stage_seconds=gpu_seconds,source_sha256=M.sha(__file__),plan_sha256=M.sha(output/'PLAN.json'),
            persistent_processes=0,outputs_sha256={p.name:M.sha(p) for p in output.iterdir() if p.is_file() and p.name!='execution_receipt.json'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    run(parser.parse_args().output)
