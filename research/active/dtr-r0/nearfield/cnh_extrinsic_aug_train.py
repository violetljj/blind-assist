"""Frozen V-recipe M3 extrinsic-augmentation fitting; final epoch, no teacher.

Only training opens supervision. Seed 0 precedes the calibration-only pilot gate;
seeds 1/2 require a sealed PASS. All outputs belong to the new task tree.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import cnh_temporal_readout_train as V
import cnh_temporal_readout_model as M

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-extrinsic-aug-20261006'
EPOCHS, BATCH, LR, WD = 8, 64, 3e-4, 1e-4
sha = V.sha


def write(path, value, exclusive=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def contract(root, later=False):
    root = Path(root)
    plan = json.loads((root/'PLAN.json').read_text(encoding='utf8'))
    deadline = float(plan['deadline_unix'])
    cfg = plan['training']
    if (cfg['epochs'], cfg['batch_size'], cfg['learning_rate'], cfg['weight_decay']) != (EPOCHS, BATCH, LR, WD):
        raise ValueError('PLAN differs from frozen V recipe')
    if time.time() >= deadline:
        raise TimeoutError('Authorized compute deadline reached')
    if later:
        gate = json.loads((root/plan['calibration']['pilot_gate_path']).read_text(encoding='utf8'))
        if gate.get('judgment') != 'PASS':
            raise ValueError('Remaining seeds and evaluation require pilot PASS')
    return plan, deadline


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError('Authorized compute deadline reached; preserve completed evidence')


def sources():
    import cnh_cvr_pilot as architecture
    import cnh_cvr_projection as projection
    return {str(Path(p).resolve()): sha(p) for p in
            (__file__, V.__file__, M.__file__, architecture.__file__, projection.__file__)}


class Inputs(V.Inputs):
    def __init__(self, root):
        self.root = Path(root)
        super().__init__(root, 'train', 'V', training=True)
        if self.maps['voxels'].dtype != np.float16 or self.maps['voxels'].shape[1:] != (3, 24, 17, 33):
            raise ValueError('Frozen raw fp16 voxel shape required')

    def hashes(self):
        # Manifest belongs to this new experiment; referenced old inputs are read-only.
        path = self.root/'training_input_hashes.json'
        entries = json.loads(path.read_text(encoding='utf8')) if path.exists() else {}
        result = {}
        for name, p in self.paths.items():
            stat = p.stat(); signature = {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            entry = entries.get(name)
            if entry is None:
                entry = dict(signature, path=str(p.resolve()), sha256=sha(p)); entries[name] = entry
            elif entry['path'] != str(p.resolve()) or any(entry[k] != v for k, v in signature.items()):
                raise ValueError(f'Sealed training input changed: {p}')
            result[name] = entry['sha256']
        if not path.exists():
            write(path, entries)
        return result


def train(root=OUT, seed=0, device='cuda'):
    if seed not in (0, 1, 2):
        raise ValueError('Only predeclared seeds 0/1/2')
    root = Path(root); plan, deadline = contract(root, later=seed != 0)
    ready = json.loads((root/'inputs/train/receipt.json').read_text(encoding='utf8'))
    if ready.get('status') != 'COMPLETE':
        raise ValueError('Augmented training data must be completely sealed')
    folder = root/'models/M3_aug'/f'seed{seed}'; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'model.pt').exists() or (folder/'training_receipt.json').exists():
        raise FileExistsError('Completed checkpoint is immutable')
    V.configure(device)
    began = time.monotonic(); data = None; net = opt = scheduler = None
    plan_hash = sha(root/'PLAN.json'); source_hash = sources()
    try:
        data = Inputs(root); input_hash = data.hashes()
        if data.n != int(ready['rows']) or data.n != int(plan['training']['expected_rows']):
            raise ValueError('Data receipt row count mismatch')
        first_ids = np.arange(min(BATCH, data.n))
        before = data.batch(first_ids, device)
        cache = data.enable_gpu_cache(device)
        after = data.batch(first_ids, device)
        parity = {k: bool(torch.equal(before[k], after[k])) for k in before}
        if not all(parity.values()):
            raise ValueError('GPU cache changed input bytes')
        del before, after
        net, baseline = V.network('V', seed, device)
        if sum(p.numel() for p in net.parameters()) != 46097:
            raise ValueError('Original CVR architecture changed')
        baseline_hash = sha(baseline)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        rng = np.random.default_rng(seed); history = []; fitting_start = time.monotonic()
        for epoch in range(EPOCHS):
            start = time.monotonic(); order = rng.permutation(data.n)
            net.train(); total = 0.; trained = updates = 0
            epoch_lr = float(opt.param_groups[0]['lr'])
            for offset in range(0, data.n, BATCH):
                check_deadline(deadline)
                ids = np.sort(order[offset:offset+BATCH]); batch = data.batch(ids, device)
                opt.zero_grad(set_to_none=True)
                # Exact V recipe: fp16 input bytes, float32 preprocessing/forward/loss.
                loss = V.loss_for(net(M.prepare_voxels(batch['voxels'])), batch)
                if loss is None:
                    continue
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward(); opt.step()
                total += float(loss.detach())*len(ids); trained += len(ids); updates += 1
            scheduler.step()
            history.append(dict(epoch=epoch+1, loss=total/max(trained, 1),
                                seconds=time.monotonic()-start, rows=trained, updates=updates,
                                learning_rate=epoch_lr, next_learning_rate=float(opt.param_groups[0]['lr']),
                                order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
            write(folder/'progress.json', dict(history=history, seconds=time.monotonic()-fitting_start), exclusive=False)
            print(f'seed{seed} epoch{epoch+1}: loss={history[-1]["loss"]:.6f} seconds={history[-1]["seconds"]:.2f}', flush=True)
        if data.hashes() != input_hash or sha(root/'PLAN.json') != plan_hash or sources() != source_hash:
            raise ValueError('Frozen source, plan or input changed during fitting')
        if sha(baseline) != baseline_hash:
            raise ValueError('Original matching M3 checkpoint changed')
        checkpoint = folder/'model.pt'
        torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, checkpoint)
        receipt = dict(status='COMPLETE', seed=seed, rows=data.n, epochs=EPOCHS, batch_size=BATCH,
                       learning_rate=LR, weight_decay=WD, scheduler='CosineAnnealingLR(T_max=8), epoch-end step',
                       precision='float32; fp16 raw voxels; TF32 disabled', final_epoch_selection=True,
                       parameters=46097, seconds=time.monotonic()-began, fitting_seconds=time.monotonic()-fitting_start,
                       history=history, input_sha256=input_hash, data_receipt_sha256=sha(root/'inputs/train/receipt.json'),
                       plan_sha256=plan_hash, source_sha256=source_hash, baseline_sha256=baseline_hash,
                       model_sha256=sha(checkpoint), gpu_input_cache=cache, cache_input_bitwise_equal=parity,
                       initialization='Matching original M3 seed', teacher=False,
                       loss='sum(BCEWithLogits(hardlabels)*(mask*weights))/batch_rows',
                       input_validation='Full SHA once; sealed size/mtime on reuse; final owner full SHA audit')
        write(folder/'training_receipt.json', receipt)
        return receipt
    finally:
        if data is not None:
            data.close()
        net = opt = scheduler = None
        if 'batch' in locals():
            del batch
        if 'loss' in locals():
            del loss
        if 'before' in locals():
            del before
        if 'after' in locals():
            del after
        gc.collect()
        if device.startswith('cuda'):
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, default=OUT)
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0); parser.add_argument('--device', default='cuda')
    args = parser.parse_args(); print(json.dumps(train(args.root, args.seed, args.device), indent=2), flush=True)
