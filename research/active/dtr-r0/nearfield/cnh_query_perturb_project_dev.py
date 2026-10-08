"""Exact FP64 finite-volume projection for small direction-perturbation batches.

Preserves BatchedProjector's FP64 geometry/mass, per-frame FP32 evidence and
coverage, sequential FP32 accumulation, and final FP16 raw voxel cache. No
signed-log preprocessing, model, labels, or scene geometry enter this helper.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from cnh_cvr_projection import EDGE, WIDTH, SUB, SHAPE

ROOT = Path(__file__).resolve().parents[4]
BENCH = ROOT/'artifacts.local/work/cnh-query-perturb-train-dev-20261008/bench'


def perturb_query(query, delta):
    """Left-multiply query by Ry(-delta), including its coordinate translation."""
    q = np.asarray(query, float)
    if q.shape[-2:] != (4,4): raise ValueError('Homogeneous query required')
    d = np.broadcast_to(np.asarray(delta,float),q.shape[:-2])
    rotation = np.broadcast_to(np.eye(4), q.shape).copy()
    a = np.radians(-d); c,s=np.cos(a),np.sin(a)
    rotation[...,0,0]=c;rotation[...,0,2]=s
    rotation[...,2,0]=-s;rotation[...,2,2]=c
    out = rotation@q
    out[d == 0] = q[d == 0]
    return out


def perturb_matrices(matrices, delta):
    """One signed direction delta per independent row, constant over history."""
    m = np.asarray(matrices,float)
    if m.ndim != 4 or m.shape[-2:] != (4,4): raise ValueError('B,L,4,4 transforms required')
    d = np.broadcast_to(np.asarray(delta,float),(len(m),))
    return perturb_query(m,d[:,None])


class QueryPerturbProjector:
    def __init__(self, projector=None):
        if projector is None:
            from cnh_cvr_v2_materialize import BatchedProjector
            projector = BatchedProjector()
        self.projector = projector

    @torch.no_grad()
    def project(self, z, matrices):
        """B<=4, equal L<=8 per row; return GPU raw voxels B,3,*SHAPE FP16."""
        if len(z.shape) != 5 or tuple(z.shape[2:]) != (8,8,16):
            raise ValueError('B,L,8,8,16 CNH required')
        b,l = z.shape[:2]
        if not 1 <= b <= 4 or not 1 <= l <= 8 or tuple(matrices.shape) != (b,l,4,4):
            raise ValueError('One to four independent equal-length histories, L<=8 required')
        projector = self.projector
        t = torch.as_tensor(matrices,dtype=torch.float64,device=projector.device).reshape(b*l,4,4)
        points = torch.bmm(projector.points[None]-t[:,None,:3,3],t[:,:3,:3])
        radius = torch.linalg.vector_norm(points,dim=2)
        xy = points[:,:,:2]/points[:,:,2:3].clamp_min(1e-30)
        ij = torch.floor((xy+EDGE)/(2*EDGE)*8).long()
        bins = torch.floor(radius/WIDTH).long()
        valid = (points[:,:,2]>0)&(ij>=0).all(2)&(ij<8).all(2)&(bins>=0)&(bins<16)
        index = (ij[:,:,1].clamp(0,7)*8+ij[:,:,0].clamp(0,7))*16+bins.clamp(0,15)
        weights = valid*projector.voxel_volume/(SUB**3)/projector.volumes[index]
        values = torch.as_tensor(z,dtype=torch.float64,device=projector.device).reshape(b*l,1024)
        mass = torch.gather(values,1,index)*weights
        evidence = mass.reshape(b,l,-1,SUB**3).sum(3).reshape(b,l,*SHAPE).float()
        coverage = valid.reshape(b,l,-1,SUB**3).double().mean(3).reshape(b,l,*SHAPE).float()
        total = torch.zeros_like(evidence[:,0]);count=torch.zeros_like(total)
        for frame in range(l):
            total += evidence[:,frame];count += coverage[:,frame]
        return torch.stack((total,count,evidence[:,-1]),dim=1).half()

    __call__ = project


def nr_fixture(frame):
    """Read two original NR training rows, without labels or scene generation."""
    from cnh_cvr_pilot import motion_metadata,relative_transforms
    base = ROOT/'artifacts.local/work/cnh-near-range-20261001'
    unit = 93000; configs=(0,1);histories=[];matrices=[];cache=[]
    with np.load(base/f'features/train/unit{unit}.npz') as raw:
        scenes=raw['scene'];frames=raw['frame'];z=raw['z1']
        for config in configs:
            ids=np.flatnonzero(scenes==config)
            np.testing.assert_array_equal(frames[ids],np.arange(16))
            histories.append(z[ids[max(0,frame-7):frame+1]])
            sensor,travel,noisy=motion_metadata(unit,config)
            matrices.append(relative_transforms(sensor,travel,noisy,frame))
    with np.load(base/'data/train/metadata_c0.npz') as metadata:
        voxels=np.load(base/'data/train/features_c0.npy',mmap_mode='r')
        for config in configs:
            row=np.flatnonzero((metadata['unit']==unit)&(metadata['config']==config)&(metadata['frame']==frame))
            if len(row)!=1:raise ValueError('Exactly one existing NR cache row required')
            cache.append(np.asarray(voxels[int(row[0])]).copy())
    return np.asarray(histories),np.asarray(matrices),np.asarray(cache)


def benchmark():
    started=time.monotonic();cap=120.;receipt=BENCH/'benchmark.json'
    if receipt.exists():raise FileExistsError('Preserve prior representative benchmark')
    def check():
        if time.monotonic()-started>=cap:raise TimeoutError('120-second representative benchmark cap')
    status='FAIL';error=None;records=[];projector=None;helper=None
    try:
        check();torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        from cnh_cvr_v2_materialize import BatchedProjector
        projector=BatchedProjector();helper=QueryPerturbProjector(projector)
        torch.cuda.reset_peak_memory_stats()
        for frame in (3,10):
            check();z,m,cached=nr_fixture(frame)
            actual=helper.project(z,m).cpu().numpy()
            reference=np.stack([projector.sequence(row,transform).half().cpu().numpy() for row,transform in zip(z,m)])
            np.testing.assert_array_equal(actual,reference)
            np.testing.assert_array_equal(actual,cached)
            shifted=perturb_matrices(m,np.array([10.,-10.]))
            np.testing.assert_array_equal(helper.project(z,shifted).cpu().numpy(),
                np.stack([projector.sequence(row,transform).half().cpu().numpy() for row,transform in zip(z,shifted)]))
            # Only three repeated two-row batches; these are throughput samples.
            torch.cuda.synchronize();tick=time.monotonic()
            for _ in range(3):
                check();result=helper.project(z,m)
            torch.cuda.synchronize();elapsed=time.monotonic()-tick
            records.append(dict(unit=93000,configs=[0,1],frame=frame,history_length=len(z[0]),
                batch_rows=2,repeated_batches=3,total_rows=6,seconds=elapsed,row_seconds=elapsed/6,
                reference_fp16_bitwise_equal=True,original_cache_bitwise_equal=True,
                perturb_plus_minus10_reference_fp16_bitwise_equal=True))
            del result
        lengths=np.minimum(np.arange(3,16)+1,8)
        early,late=(r['row_seconds'] for r in records)
        avg=float(np.mean(early+(lengths-4)/4*(late-early)))
        report=dict(status='PASS',records=records,projection=dict(mean_history_length=float(lengths.mean()),
            assumption='Linear interpolation between measured L4 and L8, same two-row batches; no inference, full-cohort I/O or cache writes included.',
            estimated_row_seconds=avg,training_rows=27456,training_seconds=avg*27456,
            evaluation_rows=144*40*13*3*2,evaluation_seconds=avg*(144*40*13*3*2)),
            gpu=torch.cuda.get_device_name(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            numeric_contract='FP64 geometry/mass; frame FP32 then sequential FP32 total/count; FP16 raw voxel output; no signedlog',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limits='Four distinct original NR rows only, single GPU; representative projection benchmark, not training or task efficacy. Batch4 throughput remains unmeasured.')
        check();status='PASS'
    except BaseException as exc:
        error=repr(exc);report=dict(status='FAIL',records=records,error=error)
        raise
    finally:
        helper=None;projector=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        report.update(elapsed_seconds=time.monotonic()-started,budget_seconds=cap,training_started=False)
        BENCH.mkdir(parents=True,exist_ok=True)
        with receipt.open('x',encoding='utf-8') as stream:
            json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
        print(json.dumps(report,ensure_ascii=False),flush=True)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark',action='store_true');args=parser.parse_args()
    if args.benchmark:benchmark()
