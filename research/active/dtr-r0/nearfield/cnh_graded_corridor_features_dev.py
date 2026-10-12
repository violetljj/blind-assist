"""Public angular-node corridor evidence on consumed ideal Development.

Native peaks are observations, never object attribution, visibility coverage or
free-space evidence. No category, frame-category or authored target box is read
by the feature path. Histograms are copied once per observation, normalized once
and shared by all current/history public gates. No voxel reconstruction/model.
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
SOURCE = ROOT / 'artifacts.local/work/cnh-counterfactual-dev-20261009'
OUTPUT = ROOT / 'artifacts.local/work/cnh-graded-corridor-dev-20261010/features'
BIAS = ROOT / 'artifacts.local/work/cnh-track-a-v5-20260928/data/readouts-gpu/primary-mount-10-snr6/bias.npy'
FRAMES = np.arange(3, 16)
REGIONS = ('inner', 'expanded', 'ring')
MODES = ('current', 'past8')
METRICS = ('top1_positive_peak_log', 'top3_positive_peak_log_sum',
           'signed_log_total', 'positive_log_total', 'positive_participation',
           'positive_bin_count', 'positive_radial_peak_count', 'membership_mass',
           'top1_forward_depth_m', 'top1_radial_distance_m',
           'top1_extent_membership', 'top1_native_zscore_max',
           'top1_forward_depth_low_m', 'top1_forward_depth_high_m')
COMPETITION = ('expanded_current_anchor_bin', 'inner_current_samebin_peak_log',
               'ring_current_samebin_peak_log', 'inner_current_samebin_peak_share')
NAMES = [f'{region}_{mode}_{metric}' for region in REGIONS for mode in MODES for metric in METRICS] + list(COMPETITION)
WIDTH = np.float32(8 * .0375348)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 23), b''):
            h.update(block)
    return h.hexdigest()


def public_geometry(sensor, query):
    """Only public transforms and fixed H/B corridors; no scene parameters."""
    offsets = np.array([1/6, .5, 5/6], np.float32)
    edge = np.float32(np.tan(np.pi/8))
    slopes = -edge + (np.arange(8, dtype=np.float32)[:, None]+offsets)*(2*edge/8)
    yy = np.broadcast_to(slopes[:, None, :, None], (8, 8, 3, 3))
    xx = np.broadcast_to(slopes[None, :, None, :], yy.shape)
    rays = np.stack((xx, yy, np.ones_like(xx)), -1).reshape(64, 9, 3)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    radii = (np.arange(16, dtype=np.float32)+.5)*WIDTH
    points = (rays[:, None] * radii[None, :, None, None]).reshape(1024, 9, 3)
    weights, depth, low, high, lengths = [], [], [], [], []
    node_membership = []
    for frame in FRAMES:
        begin = max(0, int(frame)-7)
        transform = (query[frame] @ np.linalg.inv(sensor[frame]))[None] @ sensor[begin:frame+1]
        xyz = np.einsum('tij,bnj->tbni', transform[:, :3, :3].astype(np.float32), points)
        xyz += transform[:, None, None, :3, 3].astype(np.float32)
        x, y, z = np.moveaxis(xyz, -1, 0)
        masks = []
        for ylo, yhi in ((-.20, .42), (.42, .90)):
            vertical = (y >= ylo) & (y <= yhi) & (z >= .30) & (z <= 3)
            inner = vertical & (np.abs(x) <= .30)
            expanded = vertical & (np.abs(x) <= .40)
            masks.append(np.stack((inner, expanded, expanded & ~inner)))
        nodes = np.stack(masks).astype(bool)  # Q,R,T,B,node
        mass = nodes.mean(-1, dtype=np.float32)
        d = np.divide((nodes*z[None, None]).sum(-1), nodes.sum(-1),
                      out=np.zeros_like(mass), where=nodes.sum(-1)>0)
        lo = np.where(nodes, z[None, None], np.inf).min(-1)
        hi = np.where(nodes, z[None, None], -np.inf).max(-1)
        length = int(frame)-begin+1
        weights.append(np.pad(mass, ((0,0),(0,0),(8-length,0),(0,0))))
        depth.append(np.pad(d, ((0,0),(0,0),(8-length,0),(0,0))))
        low.append(np.pad(lo, ((0,0),(0,0),(8-length,0),(0,0)), constant_values=np.inf))
        high.append(np.pad(hi, ((0,0),(0,0),(8-length,0),(0,0)), constant_values=-np.inf))
        node_membership.append(np.pad(nodes, ((0,0),(0,0),(8-length,0),(0,0),(0,0))))
        lengths.append(length)
    return dict(membership=np.stack(weights), depth=np.stack(depth),
                depth_low=np.stack(low), depth_high=np.stack(high),
                node_membership=np.stack(node_membership), length=np.array(lengths),
                radius=np.tile(radii,64))


class Ops:
    def __init__(self, backend):
        self.backend = backend
        self.torch = None
        if backend == 'cuda':
            import torch
            self.torch = torch
    def array(self, value):
        if self.torch:
            return self.torch.as_tensor(value, device='cuda')
        return np.asarray(value)
    def maximum(self, x, value):
        return x.clamp_min(value) if self.torch else np.maximum(x,value)
    def sum(self, x, axis):
        return x.sum(dim=axis) if self.torch else x.sum(axis=axis)
    def max(self, x, axis):
        return x.amax(dim=axis) if self.torch else x.max(axis=axis)
    def min(self, x, axis):
        return x.amin(dim=axis) if self.torch else x.min(axis=axis)
    def stack(self, x, axis):
        return self.torch.stack(x,dim=axis) if self.torch else np.stack(x,axis=axis)
    def where(self, cond, x, y):
        return self.torch.where(cond,x,y) if self.torch else np.where(cond,x,y)
    def top3(self, x):
        return x.topk(3,dim=-1).values.sum(-1) if self.torch else np.partition(x,-3,axis=-1)[...,-3:].sum(-1)
    def argmax(self,x):
        return x.argmax(dim=-1) if self.torch else x.argmax(axis=-1)
    def take(self, x, index):
        return x.gather(-1,index[...,None])[...,0] if self.torch else np.take_along_axis(x,index[...,None],axis=-1)[...,0]
    def numpy(self,x):
        return x.detach().cpu().numpy() if self.torch else x
    def sync(self):
        if self.torch:
            self.torch.cuda.synchronize()


def extract(normalized_z, geometry, backend='cpu'):
    """Inputs only normalized observations and public mask/coordinates.

    Past8 averages native-bin signed-log contributions after each historical
    bin is gated in the *current* public query. It is not a tracked/registered
    object echo. No future frames are referenced.
    """
    op = Ops(backend)
    z = op.array(normalized_z.astype(np.float32).reshape(len(normalized_z),16,1024))
    logs = op.torch.sign(z)*op.torch.log1p(z.abs()) if op.torch else np.sign(z)*np.log1p(np.abs(z))
    radius = op.array(geometry['radius'])
    results, valid_results = [], []
    for fj, frame in enumerate(FRAMES):
        length = int(geometry['length'][fj]); begin = int(frame)-length+1
        w = op.array(geometry['membership'][fj,:,:,8-length:])
        ds = op.array(geometry['depth'][fj,:,:,8-length:])
        dl = op.array(geometry['depth_low'][fj,:,:,8-length:])
        dh = op.array(geometry['depth_high'][fj,:,:,8-length:])
        raw = z[:,None,None,begin:int(frame)+1]
        signed = logs[:,None,None,begin:int(frame)+1]*w[None]
        pos_observation = op.maximum(signed,0)
        current = signed[...,-1,:]
        past = op.sum(signed,-2)/length
        maps = op.stack((current,past),-2)  # B,Q,R,mode,bin
        positive = op.maximum(maps,0)
        grid = positive.reshape(*positive.shape[:-1],64,16)
        # Leftmost maximum convention, endpoints allowed, strictly positive.
        if op.torch:
            pad = op.torch.full((*grid.shape[:-1],1),-float('inf'),device='cuda')
            left = op.torch.cat((pad,grid[...,:-1]),-1)
            right = op.torch.cat((grid[...,1:],pad),-1)
        else:
            pad = np.full((*grid.shape[:-1],1),-np.inf,np.float32)
            left = np.concatenate((pad,grid[...,:-1]),-1)
            right = np.concatenate((grid[...,1:],pad),-1)
        peakflag = ((grid>0)&(grid>left)&(grid>=right)).reshape(positive.shape)
        peaks = op.where(peakflag,positive,0)
        index = op.argmax(peaks)
        top1 = op.max(peaks,-1)
        haspeak = top1>0
        positive_sum = op.sum(positive,-1)
        square_sum = op.sum(positive*positive,-1)
        member_mass = op.stack((op.sum(w[...,-1,:],-1),op.sum(w,(-1,-2))/length),-1)
        # Selected native-bin geometry; conditional node depth with positive
        # contribution weights for past8, not object tracking.
        denom = op.sum(pos_observation,-2)
        past_depth = op.sum(pos_observation*ds[None],-2)/op.maximum(denom,1e-20)
        depths = op.stack((op.array(geometry['depth'][fj,:,:,-1])[None]+current*0,past_depth),-2)
        coordinate_peak = op.take(depths,index)
        radii = radius[None,None,None,None]+positive*0
        positive_gate = pos_observation>0
        low = op.stack((op.array(geometry['depth_low'][fj,:,:,-1])[None]+current*0,
                        op.min(op.where(positive_gate,dl[None],float('inf')),-2)),-2)
        high = op.stack((op.array(geometry['depth_high'][fj,:,:,-1])[None]+current*0,
                         op.max(op.where(positive_gate,dh[None],-float('inf')),-2)),-2)
        membership_peak = op.stack((w[...,-1,:][None]+current*0,op.sum(w,-2)[None]/length+past*0),-2)
        max_z = op.stack((op.where(w[None,...,-1,:]>0,raw[...,-1,:],0),
                         op.max(op.where(w[None]>0,raw,0),-2)),-2)
        metrics = op.stack((top1,op.top3(peaks),op.sum(maps,-1),positive_sum,
                           positive_sum*positive_sum/op.maximum(square_sum,1e-20),
                           op.sum(positive>0,-1),op.sum(peakflag,-1),
                           member_mass[None]+top1*0,coordinate_peak,op.take(radii,index),
                           op.take(membership_peak,index),op.take(max_z,index),
                           op.take(low,index),op.take(high,index)),-1)
        v = (member_mass[None,...,None]>0) & (metrics==metrics)
        if op.torch:
            v = v.expand_as(metrics).clone()
        else:
            v = np.broadcast_to(v,metrics.shape).copy()
        v[...,8:] = v[...,8:] & haspeak[...,None]
        metrics = op.where(v,metrics,float('nan'))
        # At expanded current's strongest observed peak radial bin, compare
        # inner/ring local peak amplitudes within +/- one radial bin.
        anchor = index[:,:,1,0]%16
        native_bin = op.array(np.tile(np.arange(16),64))
        nearby = abs(native_bin[None,None]-anchor[...,None])<=1
        inner = op.max(op.where(nearby,peaks[:,:,0,0],0),-1)
        ring = op.max(op.where(nearby,peaks[:,:,2,0],0),-1)
        comp = op.stack((anchor,inner,ring,inner/op.maximum(inner+ring,1e-20)),-1)
        comp_valid = haspeak[:,:,1,0,None] & (comp==comp)
        if op.torch:
            comp_valid = comp_valid.expand_as(comp).clone()
        else:
            comp_valid = np.broadcast_to(comp_valid,comp.shape).copy()
        comp_valid[...,1] &= member_mass[None,:,0,0]>0
        comp_valid[...,2] &= member_mass[None,:,2,0]>0
        comp_valid[...,3] &= (inner+ring>0) & comp_valid[...,1] & comp_valid[...,2]
        comp = op.where(comp_valid,comp,float('nan'))
        flat = metrics.reshape(len(normalized_z),2,-1)
        vf = v.reshape(len(normalized_z),2,-1)
        results.append(op.numpy(op.torch.cat((flat,comp),-1) if op.torch else np.concatenate((flat,comp),-1)).astype(np.float32))
        valid_results.append(op.numpy(op.torch.cat((vf,comp_valid),-1) if op.torch else np.concatenate((vf,comp_valid),-1)))
    op.sync()
    return np.stack(results,1), np.stack(valid_results,1)


def normalize(hist, ambient, bias):
    den = np.sqrt(np.maximum(16*ambient.astype(np.float32)[...,None]+np.maximum(bias,0),1e-9))
    return ((hist.astype(np.float32)-bias)/den).astype(np.float16)


def declare(output):
    save(output/'PLAN.json',dict(task='CNH_GRADED_CORRIDOR_FEATURES_DEV_20261010',lane='EXPLORE consumed simulated Development',
        scope='All existing ideal cal/validation 384 scenes x4 replicas x f3..15 x HEAD/BODY; no sampling/model/training/voxel rebuild',
        CPU_command_wall_seconds_cap=600,GPU_stage_wall_seconds_cap=180,
        goal='Public extended corridor and sparse radial-peak support for separately evaluated grading candidates',
        geometry='Nine fixed angular nodes per native bin, hard query AABB membership fraction; x inner+/-.30 expanded+/-.40, identical query y, z[.30,3]; ring=expanded-inner per node',
        time='Current or past<=8 frames ending now, each historical node transformed by query[f]@inv(sensor[f])@sensor[t]; no object registration',
        peaks='Strictly positive signedlog local radial maxima, leftmost equal plateau, endpoints; top3 from 64 zones x16 bins. No fitted amplitude cutoff.',
        competition='Expanded current largest positive local-peak radial bin anchors bin+-1 comparison of inner/ring peak maxima, inner/(inner+ring); only observations',
        backend='Representative 32-observation CPU/CUDA end-to-end normalized/summarized wall benchmark; choose faster equivalent path; CUDA failure is retained',
        truth_boundary='Feature path loads only hist, ambient, bias, sensor/public_query, scene/cache row identity; no category/author box. Evaluation downstream only.',
        missing='Per-feature valid bool; NaN if no geometry support or no valid positive peak for coordinates/competition. Zero evidence never labels negative/free.',
        acceptance='CPU/CUDA representative equivalence, exact source FP16 histories parity at deterministic rows, representative independent geometry/peak checks and past-prefix invariance',
        provenance='Physical canonical artifact junction, input/source/geometry hashes, explicit feature names/dtypes/units and cache row mapping',
        source_sha256=sha(__file__)))


def run(output):
    began = time.monotonic()
    output = output.resolve()
    if not output.is_relative_to((ROOT/'artifacts.local').resolve()):
        raise ValueError('Use canonical artifacts.local junction')
    if (output/'PLAN.json').exists():
        raise FileExistsError('Preserve prior output and PLAN')
    output.mkdir(parents=True,exist_ok=True)
    declare(output)
    phase, gpu_seconds = 'load public inputs', 0.
    try:
        bias = np.load(BIAS).astype(np.float32)
        inputs, benchmark = {}, None
        backend = 'cpu'
        for split in ('cal','validation'):
            directory = SOURCE/'data'/split
            with np.load(directory/'geometry.npz',allow_pickle=False) as a:
                sensor, query, scene_ids, scene_uids = (a[k] for k in ('sensor','public_query','scene_ids','scene_uids'))
            with np.load(directory/'physics.npz',allow_pickle=False) as a:
                ambient = a['ambient']
                np.testing.assert_array_equal(a['sensor'],sensor)
                np.testing.assert_array_equal(a['public_query'],query)
            hist = np.load(directory/'hist.npy',mmap_mode='r',allow_pickle=False)
            assert hist.shape == (384,4,16,8,8,16)
            geometry = public_geometry(sensor,query)
            if benchmark is None:
                phase = 'representative backend benchmark'
                sample = np.array(hist[:8],copy=True).reshape(32,16,8,8,16)
                cpu_start = time.monotonic()
                z = normalize(sample,ambient,bias)
                ref,ref_valid = extract(z,geometry,'cpu')
                cpu_seconds = time.monotonic()-cpu_start
                benchmark = dict(observations=32,CPU_seconds=cpu_seconds)
                gpu_start = time.monotonic()
                try:
                    import torch
                    if not torch.cuda.is_available():
                        raise RuntimeError('CUDA unavailable')
                    torch.cuda.synchronize()
                    # Warm runtime only; scientific extraction is one timed pass.
                    torch.zeros(1,device='cuda').sum().item()
                    timed = time.monotonic()
                    actual,actual_valid = extract(normalize(sample,ambient,bias),geometry,'cuda')
                    seconds = time.monotonic()-timed
                    np.testing.assert_array_equal(actual_valid,ref_valid)
                    delta = np.abs(actual[ref_valid]-ref[ref_valid])
                    np.testing.assert_allclose(actual[ref_valid],ref[ref_valid],atol=3e-4,rtol=3e-5)
                    benchmark.update(CUDA_seconds=seconds,device=torch.cuda.get_device_name(),torch_version=torch.__version__,
                                     max_abs_difference=float(delta.max()),validity_exact=True)
                    backend = 'cuda' if seconds<cpu_seconds else 'cpu'
                except Exception as error:
                    benchmark['CUDA_failure'] = repr(error)
                    save(output/'benchmark_cuda_failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
                gpu_seconds += time.monotonic()-gpu_start
                benchmark.update(selected=backend,placement='GPU_FASTER_MEASURED' if backend=='cuda' else
                                 'CPU_FASTER_MEASURED' if 'CUDA_seconds' in benchmark else 'GPU_BACKEND_UNAVAILABLE',
                                 GPU_stage_seconds=gpu_seconds)
                save(output/'benchmark.json',benchmark)
                if gpu_seconds>=180:
                    raise TimeoutError('GPU stage cap reached')
            phase = f'{split} all observations'
            out = np.empty((384,4,13,2,len(NAMES)),np.float32)
            validity = np.empty(out.shape,bool)
            summary = dict(missing_support_by_region_mode={},missing_positive_peak_by_region_mode={})
            for begin in range(0,384,16):
                if time.monotonic()-began>=600:
                    raise TimeoutError('CPU/I/O command wall cap reached')
                h = np.array(hist[begin:begin+16],copy=True)
                z = normalize(h,ambient,bias).reshape(-1,16,8,8,16)
                stage = time.monotonic()
                values,valid = extract(z,geometry,backend)
                if backend=='cuda':
                    gpu_seconds += time.monotonic()-stage
                    if gpu_seconds>=180:
                        raise TimeoutError('GPU stage cap reached')
                out[begin:begin+16] = values.reshape(16,4,13,2,-1)
                validity[begin:begin+16] = valid.reshape(16,4,13,2,-1)
                if begin in (0,368):
                    save(output/'progress.json',dict(split=split,completed_scenes=begin+16,total_scenes=384,
                         elapsed_seconds=time.monotonic()-began,GPU_stage_seconds=gpu_seconds))
            phase = f'{split} identity/parity/prefix checks'
            with np.load(directory/'rows.npz',allow_pickle=False) as a:
                row_scene,row_replica,row_frame = (a[k] for k in ('scene_id','replica','frame'))
            expected_scene = np.repeat(scene_ids,4*13)
            expected_replica = np.tile(np.repeat(np.arange(4),13),384)
            expected_frame = np.tile(FRAMES,384*4)
            for actual,expected in ((row_scene,expected_scene),(row_replica,expected_replica),(row_frame,expected_frame)):
                np.testing.assert_array_equal(actual,expected)
            parity, prefix = [], []
            histories = np.load(directory/'histories.npy',mmap_mode='r',allow_pickle=False)
            for i,k,frame in ((0,0,3),(127,1,7),(383,3,15)):
                z = normalize(np.array(hist[i,k],copy=True),ambient,bias)
                length=min(frame+1,8); row=(i*4+k)*13+frame-3
                np.testing.assert_array_equal(histories[row,8-length:],z[frame-length+1:frame+1])
                parity.append(dict(scene=int(scene_ids[i]),replica=k,frame=frame,cache_row=row,normalized_values=length*1024))
                # Deterministic future mutation: prefix below deadline unchanged.
                base, basev = extract(z[None],geometry,'cpu')
                changed = z.copy(); changed[frame+1:]=np.float16(-10)
                future, futurev = extract(changed[None],geometry,'cpu')
                np.testing.assert_array_equal(basev[:,:frame-2],futurev[:,:frame-2])
                np.testing.assert_array_equal(base[:,:frame-2],future[:,:frame-2])
                prefix.append(dict(scene=int(scene_ids[i]),replica=k,through_frame=frame,exact=True))
            np.savez_compressed(output/f'{split}_features.npz',features=out,valid=validity,names=np.array(NAMES),
                                scene_ids=scene_ids,scene_uids=scene_uids,replica=np.arange(4),frames=FRAMES,
                                queries=np.array(['HEAD','BODY']),cache_row_index=np.arange(19968).reshape(384,4,13))
            np.savez_compressed(output/f'{split}_public_geometry.npz',**geometry,sensor=sensor,public_query=query)
            for r,region in enumerate(REGIONS):
                for m,mode in enumerate(MODES):
                    offset=(r*2+m)*len(METRICS)
                    summary['missing_support_by_region_mode'][f'{region}_{mode}']=int((~validity[...,offset]).sum())
                    summary['missing_positive_peak_by_region_mode'][f'{region}_{mode}']=int((~validity[...,offset+8]).sum())
            summary.update(shape=list(out.shape),dtype='float32',valid_dtype='bool',histories_parity=parity,
                           future_mutation_prefix_checks=prefix,scene_count=384,replica_count=4,query_slot_count=39936)
            save(output/f'{split}_summary.json',summary)
            inputs[split] = {str(p.relative_to(SOURCE)):sha(p) for p in
                             (directory/'hist.npy',directory/'geometry.npz',directory/'physics.npz',directory/'rows.npz',directory/'histories.npy')}
            del hist,histories
        descriptions = dict(positive_participation='positive_log_total squared / sum positive gated bin log squared; 0 if no positive energy',
            membership_mass='sum mean angular-node membership; past8 averages across available historical frames; node hypothesis mass, not physical coverage',
            top1_extent_membership='membership at selected native-bin peak; past8 unweighted mean membership over available history, not independent sensor support',
            top1_native_zscore_max='maximum original FP16-normalized z across supported history observations at selected native bin (current uses one)',
            top1_forward_depth_m='public forward z of within-gate nodes; past8 positive gated-log weighted across history, not tracked object distance',
            top1_forward_depth_low_m='smallest supported node forward depth among positive contributions at selected bin',
            top1_forward_depth_high_m='largest supported node forward depth among positive contributions at selected bin',
            positive_radial_peak_count='local maxima along 16 radial bins per zone, after gate and current/history aggregation, strictly positive',
            signed_log_total='sum signed-log gated map, current or mean past8; includes negative observation fluctuation',
            positive_log_total='sum positive part of gated aggregated map',
            inner_current_samebin_peak_share='inner maximum/(inner maximum+ring maximum), same expanded observed anchor bin+-1; no target attribution')
        save(output/'schema.json',dict(names=NAMES,shape_axes=['scene384','replica4','frame13','query2','feature88'],
            metrics=METRICS,regions=REGIONS,modes=MODES,descriptions=descriptions,units='log scores except *_m meters, *_count bins/peaks, membership fractions or node mass',
            valid='False+NaN is missing geometry or positive peak; never a negative label',
            input_sha256=inputs,bias_sha256=sha(BIAS),physical_output=str(output),source_sha256=sha(__file__)))
        save(output/'result.json',dict(status='COMPLETE',backend=backend,seconds=time.monotonic()-began,GPU_stage_seconds=gpu_seconds,
             feature_count=len(NAMES),identity='Scene and cache row arrays exact for cal/validation; geometry common to all scenes/replicas',
             limitation='Angular-node and observed-peak support only. No physical coverage, free or target attribution. Consumed simulated Development.',
             inference=0,training=0,new_sampling=0))
    except BaseException as error:
        save(output/f'failure_{time.time_ns()}.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),seconds=time.monotonic()-began,GPU_stage_seconds=gpu_seconds))
        raise
    finally:
        save(output/'execution_receipt.json',dict(seconds=time.monotonic()-began,GPU_stage_seconds=gpu_seconds,
            command=[sys.executable,*sys.argv],source_sha256=sha(__file__),plan_sha256=sha(output/'PLAN.json'),
            outputs_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name!='execution_receipt.json'}))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    run(args.output)
