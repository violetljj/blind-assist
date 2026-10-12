"""Frozen M3 pose-prior score ensemble; observations and public mount only.

No scene, target, visibility, travel or true sensor anchor enters a prediction.
Per-unit sealed chunks permit interruption without dropping hard examples.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.spatial.transform import Rotation

import cnh_cvr_pilot as CP
import cnh_displacement_ceiling as D
import cnh_margin_confirm as MC
from cnh_cvr_projection import Projector, SHAPE, SUB, EDGE, WIDTH, query_masks
from cnh_temporal_readout_model import prepare_voxels

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-m3-pose-ensemble-20261004'
FRAMES = np.arange(3, 16)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def public_query(unit, frame):
    """Declared mount/head-scan command, no scene or true current pose."""
    mode = int(unit) % 3
    yaw = 20*np.sin(np.linspace(-np.pi/2, np.pi/2, 16))[frame] if mode == 1 else 15. if mode == 0 else 0.
    result = np.eye(4)
    result[:3, :3] = CP.rotation(yaw, 'y')@CP.rotation(-10., 'x')
    return result


def corrected_sequence(noisy, scale, bias):
    poses = np.asarray(noisy, np.float64)
    if scale == 0. and not np.any(bias):
        return poses.copy()
    corrected = poses.copy()
    cancel = Rotation.from_rotvec(-np.deg2rad(bias)*.2).as_matrix()
    for t in range(1, len(poses)):
        increment = np.linalg.inv(poses[t-1])@poses[t]
        increment[:3, 3] /= 1.+scale
        increment[:3, :3] = increment[:3, :3]@cancel
        corrected[t] = corrected[t-1]@increment
    return corrected


def transforms(unit, noisy, grid, frame):
    begin = max(0, int(frame)-7)
    sequences = [corrected_sequence(noisy[:frame+1], h['scale'], h['bias_deg_per_s']) for h in grid]
    return np.stack([public_query(unit, frame)@np.linalg.inv(c[frame])@c[begin:frame+1] for c in sequences])


@torch.no_grad()
def project_shared(projector, z, matrices):
    """[variants,time,H3] and [hyp,time,4,4] -> [variant,hyp,3,vox].

    FP64 geometry is computed once and reused across independent variants.
    FP32 exposure sums follow the original sequential accumulation exactly.
    """
    H, L = matrices.shape[:2]
    t = torch.as_tensor(matrices.reshape(H*L, 4, 4), dtype=torch.float64, device='cuda')
    p = torch.bmm(projector.points[None]-t[:, None, :3, 3], t[:, :3, :3])
    radius = torch.linalg.vector_norm(p, dim=2)
    xy = p[:, :, :2]/p[:, :, 2:3].clamp_min(1e-30)
    ij = torch.floor((xy+EDGE)/(2*EDGE)*8).long()
    bins = torch.floor(radius/WIDTH).long()
    valid = (p[:, :, 2] > 0)&(ij >= 0).all(2)&(ij < 8).all(2)&(bins >= 0)&(bins < 16)
    index = ((ij[:, :, 1].clamp(0, 7)*8+ij[:, :, 0].clamp(0, 7))*16+bins.clamp(0, 15)).reshape(H, L, -1)
    weight = (valid*projector.voxel_volume/(SUB**3)/projector.volumes[index.reshape(H*L, -1)]).reshape(H, L, -1)
    values = torch.as_tensor(z, dtype=torch.float64, device='cuda').reshape(len(z), L, 1024)
    total = torch.zeros((len(z), H, *SHAPE), dtype=torch.float32, device='cuda')
    count = torch.zeros((H, *SHAPE), dtype=torch.float32, device='cuda')
    coverage = valid.reshape(H, L, -1, SUB**3).double().mean(-1).reshape(H, L, *SHAPE).float()
    for past in range(L):
        gathered = torch.gather(values[:, past, None].expand(-1, H, -1), 2, index[None, :, past].expand(len(z), -1, -1))
        evidence = (gathered*weight[None, :, past]).reshape(len(z), H, -1, SUB**3).sum(-1).reshape(len(z), H, *SHAPE).float()
        total += evidence
        count += coverage[:, past]
    return torch.stack((total, count[None].expand(len(z), -1, *SHAPE), evidence), 2).half()


class Predictor:
    def __init__(self, plan, hypothesis_batch=4):
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError('This frozen geometry/inference stage requires CUDA')
        self.plan = plan
        self.grid = [dict(scale=0., bias_deg_per_s=[0., 0., 0.], weight=0.)]+plan['grid']
        self.weights = np.array([g['weight'] for g in plan['grid']], np.float64)
        if len(self.weights) != 24 or not np.isclose(self.weights.sum(), 1.):
            raise ValueError('Expected frozen 24-node normalized prior')
        self.batch = hypothesis_batch
        self.projector = Projector()
        self.masks = torch.as_tensor(query_masks(), device='cuda')
        self.model_hashes = {str(p): sha(p) for p in MC.model_paths('M3')}
        self.nets = []
        for path in MC.model_paths('M3'):
            net = CP.CVR().cuda().eval()
            net.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
            self.nets.append(net)
        self.timing = dict(projection_s=0., network_s=0., transforms_s=0., sequences=0, physical_variants=0)

    @torch.inference_mode()
    def predict(self, unit, z, noisy):
        """z [variant,16,8,8,16]; no metadata containing scene truth."""
        scores = np.empty((len(z), 25, 13, 2), np.float32)
        for fi, frame in enumerate(FRAMES):
            tick = time.monotonic()
            matrices = transforms(unit, noisy, self.grid, int(frame))
            self.timing['transforms_s'] += time.monotonic()-tick
            window = z[:, max(0, int(frame)-7):frame+1]
            for begin in range(0, 25, self.batch):
                end = min(25, begin+self.batch)
                torch.cuda.synchronize(); tick = time.monotonic()
                voxels = project_shared(self.projector, window, matrices[begin:end])
                torch.cuda.synchronize(); self.timing['projection_s'] += time.monotonic()-tick
                tick = time.monotonic()
                voxels = voxels.reshape(-1, 3, *SHAPE)
                batches = []
                for j in range(0, len(voxels), 64):
                    x = prepare_voxels(voxels[j:j+64], self.masks)
                    batches.append(torch.stack([net(x) for net in self.nets]).mean(0).cpu().numpy())
                scores[:, begin:end, fi] = np.concatenate(batches).reshape(len(z), end-begin, 2)
                torch.cuda.synchronize(); self.timing['network_s'] += time.monotonic()-tick
        self.timing['sequences'] += 1
        self.timing['physical_variants'] += len(z)
        return dict(Z=scores[:, 0], P24=np.einsum('h,vhfq->vfq', self.weights, scores[:, 1:]))

    def close(self):
        for path, digest in self.model_hashes.items():
            if sha(path) != digest:
                raise ValueError('Frozen model changed: '+path)
        self.nets.clear()
        self.projector = self.masks = None
        gc.collect(); torch.cuda.empty_cache()


def units_for(plan, split):
    return plan['original_units'] if split == 'original' else list(range(95000, 95048)) if split == 'calibration' else list(range(96000, 96096))


def paths_for(split, unit):
    if split == 'original':
        return [D.OUT/'observations'/f'unit{unit}.npz', D.BIAS]
    return [MC.OUT/'features'/('calib' if split == 'calibration' else 'evaluation')/f'unit{unit}.npz']


def check_frozen_inputs(paths):
    """Check root's before-inference identity, not newly admitted bytes."""
    identity = read(OUT/'input_identity.json')['input_sha256']
    canonical = {str((ROOT/Path(k)).resolve()).lower(): v for k, v in identity.items()}
    result = {}
    for path in paths:
        key = str(path.resolve()).lower()
        digest = sha(path)
        if key not in canonical or canonical[key] != digest:
            raise ValueError('Input absent from frozen identity or changed: '+str(path))
        result[str(path)] = digest
    return result


def unit_inputs(split, unit):
    """Yield only z, estimated ego-motion and serialization identity."""
    if split == 'original':
        with np.load(paths_for(split, unit)[0], allow_pickle=False) as data:
            hist = torch.as_tensor(data['hist'], dtype=torch.float32, device='cuda')
            ambient = torch.as_tensor(data['ambient'], dtype=torch.float32, device='cuda')
            noisy = data['noisy'].copy()
        bias = torch.as_tensor(np.load(D.BIAS), dtype=torch.float32, device='cuda')
        z = ((hist-bias)/(16*ambient[..., None]+bias.clamp_min(0)).clamp_min(1e-9).sqrt()).half().cpu().numpy()
        del hist, ambient, bias
        for replica in range(4):
            rows = dict(unit=np.full(7*13, unit), variant=np.repeat(np.arange(7), 13), replica=np.full(7*13, replica), frame=np.tile(FRAMES, 7))
            yield z[:, replica], noisy[replica], rows
    else:
        with np.load(paths_for(split, unit)[0], allow_pickle=False) as data:
            z, config, frames = data['z1'].copy(), data['scene'].copy(), data['frame'].copy()
        for c in sorted(set(config.tolist())):
            selection = np.flatnonzero(config == c)
            if not np.array_equal(frames[selection], np.arange(16)):
                raise ValueError('Natural sequence must contain sixteen ordered exposures')
            # Reproduce inherited estimated ego-motion only; discard true metadata.
            _, _, noisy = CP.motion_metadata(unit, int(c))
            rows = dict(unit=np.full(13, unit), config=np.full(13, c), frame=FRAMES.copy())
            yield z[selection][None], noisy, rows


def run(split, limit_units=None, benchmark=False, hypothesis_batch=4, limit_sequences=None):
    if limit_sequences is not None and not benchmark:
        raise ValueError('Partial sequence sampling is only allowed for the engineering benchmark')
    plan = read(OUT/'PLAN.json'); digest = sha(OUT/'PLAN.json')
    source_hash = sha(__file__)
    units = units_for(plan, split)
    if limit_units is not None:
        units = units[:limit_units]
    start = time.monotonic(); runner = None
    owned_status = 'FAILED'
    chunk_folder = OUT/('benchmark_chunks' if benchmark else 'chunks')/split
    chunk_folder.mkdir(parents=True, exist_ok=True)
    try:
        runner = Predictor(plan, hypothesis_batch)
        check_frozen_inputs(MC.model_paths('M3'))
        for unit in units:
            path = chunk_folder/f'unit{unit}.npz'; receipt_path = path.with_suffix('.json')
            input_hashes = check_frozen_inputs(paths_for(split, unit))
            identity = dict(plan_sha256=digest, source_sha256=source_hash, input_sha256=input_hashes, models_sha256=runner.model_hashes)
            if benchmark:
                identity['engineering_sample_sequence_limit'] = limit_sequences
            if receipt_path.exists():
                receipt = read(receipt_path)
                if any(receipt.get(k) != v for k, v in identity.items()) or receipt['score_sha256'] != sha(path):
                    raise ValueError('Sealed chunk identity changed: '+str(path))
                continue
            raws = dict(Z=[], P24=[]); identities = []
            tick = time.monotonic()
            for z, noisy, rows in unit_inputs(split, unit):
                scores = runner.predict(unit, z, noisy)
                for arm in raws:
                    raws[arm].append(scores[arm].reshape(-1, 2))
                identities.append(rows)
                if limit_sequences is not None and len(identities) >= limit_sequences:
                    break
            payload = {arm: np.concatenate(values) for arm, values in raws.items()}
            payload.update({k: np.concatenate([r[k] for r in identities]) for k in identities[0]})
            if not all(np.isfinite(payload[a]).all() for a in ('Z', 'P24')):
                raise ValueError('Nonfinite score; do not seal')
            np.savez_compressed(path, **payload)
            save(receipt_path, dict(status='COMPLETE', unit=unit, score_sha256=sha(path), elapsed_s=time.monotonic()-tick,
                engineering_only=benchmark, sequence_count=len(identities), row_count=len(payload['Z']), **identity))
            print(split, unit, 'seconds', round(time.monotonic()-tick, 2), 'cumulative', round(time.monotonic()-start, 2), flush=True)
        if not benchmark and limit_units is None:
            collect(plan, split)
        elapsed = time.monotonic()-start
        status = dict(status='BENCHMARK_COMPLETE' if benchmark else 'COMPLETE', split=split, units=units, elapsed_s=elapsed,
                      timing=runner.timing, hypothesis_batch=hypothesis_batch, plan_sha256=digest,
                      models_sha256=runner.model_hashes, source_sha256=source_hash,
                      runtime=dict(torch=torch.__version__, cuda=torch.version.cuda, device=torch.cuda.get_device_name(), no_tf32=True),
                      forecast_full_split_s=elapsed*len(units_for(plan, split))*(4 if split=='original' else 40)/max(1, runner.timing['sequences']))
        save(OUT/f'{"benchmark" if benchmark else "runtime"}_{split}.json', status)
        owned_status = status['status']
        return status
    finally:
        timing = None if runner is None else runner.timing
        if runner is not None:
            runner.close()
        save(OUT/f'release_{split}.json', dict(status=owned_status, plan_sha256=digest, elapsed_s=time.monotonic()-start, timing=timing,
            cuda_allocated_bytes=torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
            cuda_reserved_bytes=torch.cuda.memory_reserved() if torch.cuda.is_available() else 0))


def collect(plan, split):
    units = units_for(plan, split); digest = sha(OUT/'PLAN.json')
    arrays = []; hashes = {}
    for unit in units:
        path = OUT/'chunks'/split/f'unit{unit}.npz'
        receipt = read(path.with_suffix('.json'))
        if receipt['status'] != 'COMPLETE' or receipt['plan_sha256'] != digest or receipt['score_sha256'] != sha(path):
            raise ValueError('Unsealed or mismatched unit: '+str(path))
        with np.load(path, allow_pickle=False) as data:
            arrays.append({k: data[k].copy() for k in data.files})
        hashes[str(path)] = receipt['score_sha256']
    rows = {k: np.concatenate([a[k] for a in arrays]) for k in arrays[0] if k not in ('Z', 'P24')}
    row_path = OUT/'inputs'/split/'rows.npz'; row_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(row_path, **rows)
    for arm in ('Z', 'P24'):
        path = OUT/'predictions'/f'{arm}_{split}.npz'; path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, raw=np.concatenate([a[arm] for a in arrays]))
        save(path.with_suffix('.json'), dict(status='COMPLETE', score_sha256=sha(path), plan_sha256=digest,
            rows_sha256=sha(row_path), chunks_sha256=hashes, units=units))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', required=True, choices=('original', 'calibration', 'evaluation'))
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--limit-units', type=int)
    parser.add_argument('--hypothesis-batch', type=int, default=4)
    parser.add_argument('--limit-sequences', type=int)
    parser.add_argument('--collect', action='store_true')
    args = parser.parse_args()
    if args.collect:
        collect(read(OUT/'PLAN.json'), args.split)
    else:
        print(json.dumps(run(args.split, args.limit_units, args.benchmark, args.hypothesis_batch, args.limit_sequences), indent=2), flush=True)
