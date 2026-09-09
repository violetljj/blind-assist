"""Fixed MZ1 ablations on frozen cached observations; no feature or label fitting.

Training uses TRAIN_ONLY, with final-only DEV/EVAL scoring. Spatial zeros never
assert safe clearance; original B alerts are retained independently and unchanged.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from mz0_clean import metric

ARMS = ('RGB_ONLY', 'TOF_ONLY', 'FUSION')
ROLES = ('TRAIN_ONLY', 'DEV_ONLY', 'EVAL_ONLY')
STEPS, BATCH_SIZE = 300, 128


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def state_sha(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


class TinyFusion(nn.Module):
    def __init__(self, arm='FUSION'):
        super().__init__()
        if arm not in ARMS:
            raise ValueError('Unknown arm')
        self.arm = arm
        self.layers = nn.Sequential(nn.Linear(1028, 128), nn.ReLU(), nn.Linear(128, 4))

    def forward(self, visual, tof):
        if visual.ndim != 2 or visual.shape[1] != 772 or tof.shape != (visual.shape[0], 256):
            raise ValueError('Expected visual[B,772] and tof[B,256]')
        if self.arm == 'RGB_ONLY':
            tof = torch.zeros_like(tof)
        elif self.arm == 'TOF_ONLY':
            visual = torch.zeros_like(visual)
        return self.layers(torch.cat((visual, tof), dim=1))


def schedule(role):
    """Frozen sampling with replacement; indices always refer to TRAIN_ONLY rows."""
    train = np.flatnonzero(role == 'TRAIN_ONLY')
    if not len(train):
        raise ValueError('No TRAIN_ONLY rows')
    return np.random.default_rng(59).choice(train, size=(STEPS, BATCH_SIZE), replace=True)


def run(features, output):
    features, output = Path(features).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Fresh output directory required')
    receipt_path = features.with_name('features-receipt.json')
    source_receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
    feature_sha = sha(features)
    if source_receipt['status'] != 'PASS' or source_receipt['feature_sha256'] != feature_sha:
        raise ValueError('Feature source receipt mismatch')
    with np.load(features, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ('visual', 'tof', 'truth', 'role', 'original_alerts', 'joint')}
    n = len(data['role'])
    for name, shape in (('visual', (n, 772)), ('tof', (n, 256)), ('truth', (n, 4)),
                        ('original_alerts', (n, 2)), ('joint', (n, 4))):
        if data[name].shape != shape:
            raise ValueError(f'Invalid {name} shape')
    if data['role'].ndim != 1 or set(data['role'].tolist()) != set(ROLES):
        raise ValueError('All three explicit roles required')
    for name in ('truth', 'original_alerts', 'joint'):
        if data[name].dtype != np.bool_:
            raise ValueError(f'{name} must be Boolean')
    if not np.isfinite(data['visual']).all() or not np.isfinite(data['tof']).all():
        raise ValueError('Cached observations must be finite; invalid ToF must already be zero-filled')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA training required')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(53)
    initial = copy.deepcopy(TinyFusion().state_dict())
    initial_sha = state_sha(initial)
    indices = schedule(data['role'])
    assert np.all(data['role'][indices] == 'TRAIN_ONLY')
    output.mkdir(parents=True)
    np.save(output/'schedule.npy', indices, allow_pickle=False)
    torch.save(initial, output/'initial.pt')
    started = time.perf_counter()
    visual = torch.as_tensor(data['visual'], dtype=torch.float32, device='cuda')
    tof = torch.as_tensor(data['tof'], dtype=torch.float32, device='cuda')
    # Only training labels enter CUDA/loss; DEV/EVAL truth stays evaluator-side.
    train_rows = np.flatnonzero(data['role'] == 'TRAIN_ONLY')
    train_target = torch.as_tensor(data['truth'][train_rows], dtype=torch.float32, device='cuda')
    global_to_train = np.full(n, -1, dtype=np.int64)
    global_to_train[train_rows] = np.arange(len(train_rows))
    targets = torch.as_tensor(global_to_train[indices], device='cuda')
    batches = torch.as_tensor(indices, device='cuda')
    original_alerts = data['original_alerts'].copy()
    predictions = dict(role=data['role'], truth=data['truth'], original_alerts=original_alerts.copy(), joint=data['joint'])
    results = {}
    arm_receipts = {}
    for arm in ARMS:
        model = TinyFusion(arm)
        model.load_state_dict(initial)
        assert state_sha(model.state_dict()) == initial_sha
        model = model.cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
        losses = []
        for step in range(STEPS):
            batch = batches[step]
            optimizer.zero_grad(set_to_none=True)
            logits = model(visual[batch], tof[batch])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, train_target[targets[step]])
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite loss at{arm} step{step}')
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        all_logits = []
        with torch.inference_mode():
            for begin in range(0, n, 256):
                all_logits.append(model(visual[begin:begin+256], tof[begin:begin+256]).cpu().numpy())
        all_logits = np.concatenate(all_logits)
        if not np.isfinite(all_logits).all():
            raise ValueError('Nonfinite final logits')
        flags = (torch.from_numpy(all_logits).sigmoid() >= .5).numpy()
        predictions[arm+'_logits'] = all_logits
        predictions[arm+'_flags'] = flags
        results[arm] = {role: metric(flags[data['role']==role], data['truth'][data['role']==role]) for role in ROLES}
        checkpoint = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        torch.save(checkpoint, output/f'{arm}.pt')
        arm_receipts[arm] = dict(initial_state_sha256=initial_sha, checkpoint_sha256=sha(output/f'{arm}.pt'),
                                 final_state_sha256=state_sha(checkpoint), steps=STEPS,
                                 train_loss_by_step=losses)
        del model, optimizer
    assert np.array_equal(predictions['original_alerts'], data['original_alerts'])
    assert np.array_equal(original_alerts, data['original_alerts'])
    np.savez_compressed(output/'predictions.npz', **predictions)
    with np.load(output/'predictions.npz', allow_pickle=False) as stored:
        assert np.array_equal(stored['original_alerts'], data['original_alerts'])
    baseline = {role: metric(data['joint'][data['role']==role], data['truth'][data['role']==role]) for role in ROLES}
    write(output/'metrics.json', dict(arms=results, frozen_joint=baseline))
    torch.cuda.synchronize()
    write(output/'receipt.json', dict(status='PASS', frames=n, role_counts={role:int(np.sum(data['role']==role)) for role in ROLES},
        feature_sha256=feature_sha, features_receipt_sha256=sha(receipt_path), source_sha256=sha(__file__),
        metric_source_sha256=sha(Path(__file__).with_name('mz0_clean.py')), initial_state_sha256=initial_sha,
        initial_file_sha256=sha(output/'initial.pt'), schedule_sha256=sha(output/'schedule.npy'),
        predictions_sha256=sha(output/'predictions.npz'), metrics_sha256=sha(output/'metrics.json'), arms=arm_receipts,
        parameters=dict(architecture=[1028,128,4], activation='ReLU', initial_seed=53, schedule_seed=59,
                        sampling='TRAIN_ONLY with replacement', steps=300, batch_size=128,
                        optimizer='AdamW', learning_rate=.001, weight_decay=.0001,
                        loss='BCEWithLogits mean', threshold=.5, early_selection=False),
        original_alert_parity=n, backend='CUDA', device=torch.cuda.get_device_name(), torch_version=torch.__version__,
        seconds=time.perf_counter()-started, scope='Consumed controlled Development; spatial zeros and UNKNOWN are not CLEAR; no original alert changes'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.features, args.output)
