"""Matched local/pooled RGB-ToF readouts on frozen, consumed Development inputs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn

from body_query_model import fixed_projection
from mz1_tiny_fusion import sha, write, state_sha, schedule, STEPS, BATCH_SIZE, ROLES
from mz3_error_attribution import read, load, metrics, compare

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'artifacts.local/work'
sys.path.insert(0, str(ROOT / 'tools'))
from research_backend import BackendCandidate, Workload, select_backend, torch_observation

ARMS = ('POOLED_FUSION', 'LOCAL_FUSION', 'LOCAL_RGB_ONLY')


def geometry():
    grid, valid, xyz = fixed_projection()
    physical = xyz.double() * torch.tensor([3.18, .28, 1.85], dtype=torch.float64)
    relative = physical - torch.tensor([0., 0., 1.7], dtype=torch.float64)
    az = torch.rad2deg(torch.atan2(relative[..., 1], relative[..., 0]))
    el = torch.rad2deg(torch.atan2(relative[..., 2], relative[..., 0]))
    covered = (az.abs() <= 22.5) & (el.abs() <= 22.5) & valid
    col = ((az + 22.5) / (45/8)).floor().long().clamp(0, 7)
    row = ((22.5 - el) / (45/8)).floor().long().clamp(0, 7)
    return dict(valid=valid, xyz=xyz, zone=row*8+col, covered=covered,
                radial=(relative.norm(dim=-1) / 4).float(), grid=grid)


class SpatialFusion(nn.Module):
    def __init__(self, arm='LOCAL_FUSION'):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown arm')
        self.arm = arm
        for name, value in geometry().items():
            self.register_buffer(name, value)
        self.point = nn.Sequential(nn.Linear(74, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.readout = nn.Sequential(nn.Linear(1412, 88), nn.ReLU(), nn.Linear(88, 4))

    def forward(self, points, visual, tof):
        if points.shape[1:] != (12, 27, 64) or visual.shape != (len(points), 772) or tof.shape != (len(points), 256):
            raise ValueError('Expected point[B,12,27,64], visual[B,772], tof[B,256]')
        mask = self.valid[None, :, :, None]
        if self.arm == 'POOLED_FUSION':
            points = ((points * mask).sum(2, keepdim=True) / mask.sum(2, keepdim=True).clamp_min(1)).expand_as(points)
        if self.arm == 'LOCAL_RGB_ONLY':
            tof = torch.zeros_like(tof)
        ranges = tof[:, :128].reshape(-1, 64, 2)[:, self.zone]
        valid = tof[:, 128:].reshape(-1, 64, 2)[:, self.zone] * self.covered[None, :, :, None]
        ranges = ranges * valid
        residual = (ranges - self.radial[None, :, :, None]) * valid
        xyz = self.xyz[None].expand(len(points), -1, -1, -1)
        covered = self.covered[None, :, :, None].expand(len(points), -1, -1, -1)
        local = self.point(torch.cat((points, xyz, ranges, valid, residual, covered), -1))
        pooled = (local * mask).sum(2) / mask.sum(2).clamp_min(1)
        return self.readout(torch.cat((visual, tof, pooled.flatten(1)), 1))


def point_permutation(valid):
    rng = np.random.default_rng(83)
    permutation = np.broadcast_to(np.arange(27), (12, 27)).copy()
    for q in range(12):
        index = np.flatnonzero(valid[q])
        permutation[q, index] = rng.permutation(index)
    return permutation


def choose_device(initial, points, visual, tof, target, output):
    candidates = []
    references = []
    # Probe exactly one forward+BCE+backward on genuine TRAIN_ONLY examples.
    # Separate probe models never update or supply the trained checkpoint state.
    for device in ('cpu', 'cuda'):
        model = SpatialFusion().to(device)
        model.load_state_dict(initial)
        p, v, t, y = [torch.as_tensor(a, device=device) for a in (points, visual, tof, target)]
        def probe(model=model, p=p, v=v, t=t, y=y):
            model.zero_grad(set_to_none=True)
            out = model(p, v, t)
            nn.functional.binary_cross_entropy_with_logits(out, y).backward()
            return out.detach()
        references.append(probe().cpu().numpy())
        candidates.append(BackendCandidate(device, device, probe,
            lambda result: torch_observation(output=result),
            torch.cuda.synchronize if device == 'cuda' else lambda: None))
    np.testing.assert_allclose(references[0], references[1], rtol=2e-5, atol=2e-5)
    record = select_backend(Workload.BATCH_TENSOR, cpu=candidates[0], gpu=candidates[1],
        warmups=2, repeats=5, record_path=output / 'backend.json')
    return record['selected_device_type']


def run(cache, output):
    cache, output = Path(cache).resolve(), Path(output).resolve()
    allowed = (WORK / 'mz5-spatial-fusion-20260910').resolve()
    if not output.is_relative_to(allowed) or output == allowed or output.exists():
        raise ValueError('Use a fresh task output child')
    source = WORK / 'mz1-tiny-fusion-20260910'
    protocol = Path(__file__).with_name('MZ5_SPATIAL_FUSION_PROTOCOL_20260910.md')
    cr, fr = read(cache/'receipt.json'), read(source/'cache-v1/features-receipt.json')
    assert cr['status'] == fr['status'] == 'PASS'
    assert cr['points_sha256'] == sha(cache/'points.npy')
    assert cr['geometry_sha256'] == sha(cache/'geometry.npz')
    assert cr['original_features_sha256'] == fr['feature_sha256'] == sha(source/'cache-v1/features.npz')
    data = load(source/'cache-v1/features.npz')
    p = np.load(cache/'points.npy', mmap_mode='r')
    assert p.shape == (5000, 12, 27, 64) and np.isfinite(p).all()
    for key, value in load(cache/'geometry.npz').items():
        if key in ('valid', 'grid', 'xyz'):
            np.testing.assert_array_equal(value, geometry()[key])
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(53)
    initial = copy.deepcopy(SpatialFusion().state_dict())
    init_sha = state_sha(initial)
    assert sum(x.numel() for x in SpatialFusion().parameters()) == 131580
    indices = schedule(data['role'])
    np.testing.assert_array_equal(indices, np.load(source/'run-v1/schedule.npy'))
    train = np.flatnonzero(data['role'] == 'TRAIN_ONLY')
    assert len(train) == 2500 and np.all(data['role'][indices] == 'TRAIN_ONLY')
    output.mkdir(parents=True)
    shutil.copyfile(__file__, output/'source.py')
    shutil.copyfile(protocol, output/'protocol.md')
    write(output/'start-receipt.json', dict(status='STARTED', utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), protocol_sha256=sha(protocol), cache_receipt_sha256=sha(cache/'receipt.json'),
        source_features_sha256=fr['feature_sha256'], initial_state_sha256=init_sha))
    started = time.perf_counter()
    try:
        torch.save(initial, output/'initial.pt')
        np.save(output/'schedule.npy', indices, allow_pickle=False)
        dev = choose_device(initial, np.array(p[train[:128]]), data['visual'][train[:128]],
            data['tof'][train[:128]], data['truth'][train[:128]].astype(np.float32), output)
        points = torch.tensor(np.array(p), device=dev)
        visual = torch.as_tensor(data['visual'], device=dev)
        tof = torch.as_tensor(data['tof'], device=dev)
        target = torch.tensor(data['truth'][train].astype(np.float32), device=dev)
        mapping = np.full(5000, -1, dtype=np.int64)
        mapping[train] = np.arange(len(train))
        batch_indices = torch.as_tensor(indices, device=dev)
        target_indices = torch.as_tensor(mapping[indices], device=dev)
        permutation = point_permutation(geometry()['valid'].numpy())
        np.save(output/'permutation.npy', permutation, allow_pickle=False)
        perm = torch.as_tensor(permutation, device=dev)
        query = torch.arange(12, device=dev)[:, None]
        predictions = {k: data[k].copy() for k in ('role', 'truth', 'original_alerts', 'joint')}
        results, receipts = {}, {}
        for arm in ARMS:
            model = SpatialFusion(arm).to(dev)
            model.load_state_dict(initial)
            assert state_sha({k: v.cpu() for k, v in model.state_dict().items()}) == init_sha
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
            losses = []
            fit_start = time.perf_counter()
            for step in range(STEPS):
                b = batch_indices[step]
                optimizer.zero_grad(set_to_none=True)
                logits = model(points[b], visual[b], tof[b])
                loss = nn.functional.binary_cross_entropy_with_logits(logits, target[target_indices[step]])
                if not torch.isfinite(loss):
                    raise ValueError(f'Nonfinite loss in {arm} step {step}')
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            if dev == 'cuda':
                torch.cuda.synchronize()
            fit_seconds = time.perf_counter()-fit_start
            model.eval()
            logits = []
            with torch.inference_mode():
                for begin in range(0, 5000, 128):
                    b = slice(begin, begin+128)
                    logits.append(model(points[b], visual[b], tof[b]).cpu().numpy())
            logits = np.concatenate(logits)
            assert np.isfinite(logits).all()
            flags = logits >= 0
            predictions[arm+'_logits'], predictions[arm+'_flags'] = logits, flags
            results[arm] = {role: metrics(flags[data['role']==role], data['truth'][data['role']==role]) for role in ROLES}
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(state, output/f'{arm}.pt')
            receipts[arm] = dict(steps=STEPS, initial_state_sha256=init_sha,
                checkpoint_sha256=sha(output/f'{arm}.pt'), final_state_sha256=state_sha(state),
                train_losses=losses, fit_seconds=fit_seconds)
            if arm == 'LOCAL_FUSION':
                ev_indices = torch.as_tensor(np.flatnonzero(data['role']=='EVAL_ONLY'), device=dev)
                out = []
                with torch.inference_mode():
                    for begin in range(0, 1500, 128):
                        b = ev_indices[begin:begin+128]
                        shuffled = points[b][:, query, perm]
                        out.append(model(shuffled, visual[b], tof[b]).cpu().numpy())
                predictions['LOCAL_PERMUTED_logits'] = np.concatenate(out)
                predictions['LOCAL_PERMUTED_flags'] = predictions['LOCAL_PERMUTED_logits'] >= 0
            print(f'{arm} fit complete: {fit_seconds:.3f}s', flush=True)
            del model, optimizer
        np.testing.assert_array_equal(predictions['original_alerts'], data['original_alerts'])
        np.savez_compressed(output/'predictions.npz', **predictions)
        write(output/'metrics.json', results)
        write(output/'receipt.json', dict(status='PASS', cache_receipt_sha256=sha(cache/'receipt.json'),
            predictions_sha256=sha(output/'predictions.npz'), metrics_sha256=sha(output/'metrics.json'),
            source_features_sha256=fr['feature_sha256'], source_index_sha256=fr['source_index_sha256'],
            protocol_sha256=sha(protocol), source_sha256=sha(__file__), arms=receipts,
            initial_state_sha256=init_sha, original_alert_parity=5000,
            parameters=131580, initial_seed=53, schedule_seed=59, permutation_seed=83,
            steps_per_arm=300, batch_size=128, learning_rate=.001, weight_decay=.0001,
            backend=dev, device=torch.cuda.get_device_name() if dev=='cuda' else 'CPU',
            seconds=time.perf_counter()-started, scope='Consumed controlled Development; zeros are not CLEAR'))
        print('FIT PASS', time.perf_counter()-started, flush=True)
    except Exception as error:
        write(output/'failure.json', dict(status='ENGINEERING_FAILED', type=type(error).__name__, message=str(error)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.cache, args.output)
