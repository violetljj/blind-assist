"""Fixed V/T training and label-blind inference for the temporal pilot.

Memory maps are sliced lazily; labels/masks/weights are opened only in train and
smoke. Epochs use one common seed permutation for both arms. No evaluation score
is read during fitting; final epoch only, three seeds required by the parent.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cnh_temporal_readout_model as M

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-temporal-readout-20261004'
EPOCHS, BATCH, LR, WD = 8, 64, 3e-4, 1e-4


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temp.replace(path)


def fixed_ambient():
    # This is the same frozen public sensor calibration that generated cached
    # natural z1. Its ambient is a scene-independent constant, not truth input.
    import cnh_proposal_attribution_scenes as S
    params, _ = S.nominal_parameters()
    return float(params.ambient_counts*params.output_gain)


def check_budget(root):
    try:
        import cnh_temporal_readout as common
    except ModuleNotFoundError:
        common = None
    if common is not None and Path(root).resolve() == Path(common.OUT).resolve():
        common.check_budget()
        return
    path = Path(root)/'budget.json'
    if path.exists():
        data = json.loads(path.read_text(encoding='utf8'))
        deadline = data.get('deadline_unix', data.get('compute_deadline_unix'))
        if deadline is not None and time.time() >= float(deadline):
            raise TimeoutError('Pilot computation budget reached; preserve completed evidence')


def recipe(root):
    import cnh_temporal_readout as common
    p = common.load_plan() if Path(root).resolve() == Path(common.OUT).resolve() else json.loads((Path(root)/'PLAN.json').read_text(encoding='utf8'))
    return dict(epochs=int(p['epochs']), batch=int(p['batch']), lr=float(p['lr']), wd=float(p['wd']))


class Inputs:
    """Lazy, arm-specific observations; no evaluator metadata or labels."""
    def __init__(self, root, split, arm, training=False):
        self.folder = Path(root)/'inputs'/split
        names = ['voxels'] if arm == 'V' else ['histories', 'transforms', 'length', 'ambient']
        if training:
            names += ['labels', 'mask', 'weights']
        self.paths = {k: self.folder/f'{k}.npy' for k in names}
        self.maps = {k: np.load(p, mmap_mode='r', allow_pickle=False) for k, p in self.paths.items()}
        self.n = len(self.maps[names[0]])
        if not self.n or any(len(v) != self.n for v in self.maps.values()):
            raise ValueError('Input/label row counts differ or empty')
        if training and any(self.maps[k].shape != (self.n, 2) for k in ('labels', 'mask', 'weights')):
            raise ValueError('Training supervision must be [N,2]')
        self.arm = arm
        self.gpu_maps = {}
        self.cache_receipt = dict(used=False, bytes=0, reason='not_requested')

    def enable_gpu_cache(self, device):
        """Cache identical input bytes on GPU to release resident mmap pages.

        This addresses observed host RAM pressure; it changes neither input
        construction nor precision/loss. Reserve 2GiB for models/optimizer and
        never dedicate more than 70% of the device to this process's cache.
        """
        if self.gpu_maps:
            return self.cache_receipt
        size = sum(v.nbytes for v in self.maps.values())
        self.cache_receipt = dict(used=False, bytes=size, reason='non_cuda')
        if not device.startswith('cuda'):
            return self.cache_receipt
        free, total = torch.cuda.mem_get_info(device)
        self.cache_receipt.update(free_before=free, total=total, reserve_bytes=2*1024**3)
        if size > free-2*1024**3 or size > .7*total:
            self.cache_receipt['reason'] = 'memory_budget'
            return self.cache_receipt
        start = time.monotonic(); copied = {}
        try:
            for k, v in self.maps.items():
                # Read-only numpy view is never modified; .to performs a
                # synchronous device copy without a full CPU array duplicate.
                host = torch.from_numpy(v)
                copied[k] = host.to(device, non_blocking=False, copy=True)
                del host
        except torch.cuda.OutOfMemoryError:
            copied.clear(); gc.collect(); torch.cuda.empty_cache()
            self.cache_receipt.update(reason='allocation_failed', seconds=time.monotonic()-start)
            return self.cache_receipt
        self.gpu_maps = copied
        self.maps.clear()
        gc.collect()
        self.cache_receipt.update(used=True, reason='cached_exact_bytes', seconds=time.monotonic()-start)
        return self.cache_receipt

    def batch(self, indices, device):
        if self.gpu_maps:
            ids = torch.as_tensor(indices, device=device, dtype=torch.long)
            return {k: v[ids] for k, v in self.gpu_maps.items()}
        return {k: torch.from_numpy(np.array(v[indices], copy=True)).to(device, non_blocking=False)
                for k, v in self.maps.items()}

    def hashes(self):
        # Each immutable input is fully hashed once. Subsequent seeds verify its
        # size/mtime against that recorded hash, and root performs a final full
        # payload hash audit. This avoids rescanning GB-sized voxels 12 times.
        cache_path = self.folder/'input_hash_manifest.json'
        cache = json.loads(cache_path.read_text(encoding='utf8')) if cache_path.exists() else {}
        result = {}; changed = False
        for p in self.paths.values():
            stat = p.stat(); key = str(p)
            signature = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            entry = cache.get(key)
            if entry is None:
                entry = dict(**signature, sha256=sha(p)); cache[key] = entry; changed = True
            elif any(entry[k] != v for k, v in signature.items()):
                raise ValueError(f'Previously sealed input stat changed: {p}')
            result[key] = entry['sha256']
        if changed: write(cache_path, cache)
        return result

    def close(self):
        self.maps.clear()
        self.gpu_maps.clear()


def network(arm, seed, device, initialize=True):
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    net = M.CVR() if arm == 'V' else M.TemporalReadout()
    baseline = None
    if arm == 'V' and initialize:
        baseline = ROOT/f'artifacts.local/work/cnh-margin-labels-20261002/models/M3/model_seed{seed}.pt'
        net.load_state_dict(torch.load(baseline, map_location='cpu', weights_only=True))
    return net.to(device), baseline


def forward(net, arm, batch, ambient):
    if arm == 'V':
        return net(M.prepare_voxels(batch['voxels']))
    return net(batch['histories'], batch['transforms'], batch['length'].long(), batch.get('ambient', ambient))


def loss_for(logits, batch):
    labels, mask, weights = (batch[k].float() for k in ('labels', 'mask', 'weights'))
    effective = mask*weights
    if not bool(torch.isfinite(labels).all() and torch.isfinite(effective).all()) or bool((effective < 0).any()):
        raise ValueError('Nonfinite or negative fixed training supervision')
    if not bool(effective.sum() > 0):
        return None
    # Each domain's fixed masked far-weights sum to N/2, so all query weights
    # together sum to N. Per-row normalization preserves the global mixture.
    return (F.binary_cross_entropy_with_logits(logits.float(), labels, reduction='none')*effective).sum()/len(labels)


def autocast(device, precision):
    if precision == 'bf16':
        if not device.startswith('cuda') or not torch.cuda.is_bf16_supported():
            raise ValueError('Requested BF16 requires CUDA BF16 support')
    return torch.autocast(device_type='cuda' if device.startswith('cuda') else 'cpu',
                          dtype=torch.bfloat16, enabled=precision == 'bf16')


def configure(device):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')


def smoke(root=OUT, device='cuda', precision='float32'):
    configure(device); check_budget(root); began = time.monotonic()
    cfg = recipe(root)
    ambient = fixed_ambient(); counts = M.parameter_counts()
    if counts != {'V': 46097, 'T': 51794} or counts['T'] > 5*counts['V']:
        raise ValueError('Frozen model size constraint failed')
    results = {}
    for arm in ('V', 'T'):
        data = Inputs(root, 'smoke', arm, training=True)
        # One independent smoke world, all variants/replicas/frames; a complete
        # epoch. This dataset is never part of formal training or evaluation.
        batch = data.batch(np.arange(min(cfg['batch'], data.n)), device)
        net, _ = network(arm, 0, device)
        net.eval()
        with torch.no_grad():
            full = forward(net, arm, batch, ambient)
            if full.shape != (len(batch['labels']), 2) or not bool(torch.isfinite(full).all()):
                raise ValueError('Smoke logits shape/finiteness failed')
            with autocast(device, precision):
                mixed = forward(net, arm, batch, ambient).float()
            delta = float((full.float()-mixed).abs().max())
            if delta > .05:
                raise ValueError('Requested precision logits parity exceeds .05')
            padding_delta = 0.
            if arm == 'T':
                changed = dict(batch)
                changed['histories'] = batch['histories'].clone()
                padding = torch.arange(8, device=device)[None] < 8-batch['length'].long()[:, None]
                changed['histories'][padding] = 123
                modified = forward(net, arm, changed, ambient)
                padding_delta = float((full-modified).abs().max())
                if padding_delta > 1e-6:
                    raise ValueError('Padding influenced T prediction')
        net.train(); opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
        loss_total, updates, seen = 0., 0, 0
        before = data.hashes()
        for offset in range(0, data.n, cfg['batch']):
            check_budget(root); ids = np.arange(offset, min(offset+cfg['batch'], data.n))
            batch = data.batch(ids, device); opt.zero_grad(set_to_none=True)
            with autocast(device, precision):
                loss = loss_for(forward(net, arm, batch, ambient), batch)
            if loss is None: continue
            if not bool(torch.isfinite(loss)): raise ValueError('Nonfinite smoke loss')
            loss.backward()
            grads = [p.grad for p in net.parameters() if p.grad is not None]
            if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                raise ValueError('Nonfinite/missing smoke gradients')
            opt.step(); updates += 1; seen += len(ids); loss_total += float(loss.detach())*len(ids)
        if not updates or data.hashes() != before:
            raise ValueError('Empty smoke updates or changed smoke inputs')
        results[arm] = dict(parameters=counts[arm], logits_first_batch=full.detach().float().cpu().tolist(),
                            epochs=1, rows=data.n, processed_evaluable_rows=seen, updates=updates,
                            loss=loss_total/max(seen, 1), precision_max_abs=delta, padding_max_abs=padding_delta,
                            finite_gradients=True, input_sha256=before)
        data.close(); del net, opt, batch; gc.collect()
        if device.startswith('cuda'): torch.cuda.empty_cache()
    receipt = dict(status='PASS', seconds=time.monotonic()-began, device=device, precision=precision,
                   ambient=fixed_ambient(), checks=results, input_boundary='observations/transforms/query constants only',
                   source_sha256={str(Path(p).resolve()): sha(p) for p in (__file__, M.__file__)})
    write(Path(root)/'smoke_receipt.json', receipt)
    return receipt


def train(root=OUT, arm='V', seed=0, device='cuda', precision='float32'):
    if arm not in ('V', 'T') or seed not in (0, 1, 2):
        raise ValueError('Exactly V/T and seeds0/1/2 are permitted')
    root = Path(root); configure(device); check_budget(root)
    smoke_receipt = json.loads((root/'smoke_receipt.json').read_text(encoding='utf8'))
    if smoke_receipt['status'] != 'PASS' or smoke_receipt['precision'] != precision:
        raise ValueError('Matching successful smoke is required')
    plan_digest = sha(root/'PLAN.json')
    cfg = recipe(root)
    folder = root/'models'/arm/f'seed{seed}'; folder.mkdir(parents=True, exist_ok=True)
    if (folder/'model.pt').exists() or (folder/'training_receipt.json').exists():
        raise FileExistsError('Completed model cannot be overwritten')
    data = Inputs(root, 'train', arm, training=True)
    before = data.hashes(); cache = data.enable_gpu_cache(device)
    net, baseline = network(arm, seed, device)
    source_hashes = {str(Path(p).resolve()): sha(p) for p in (__file__, M.__file__)}
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['epochs'])
    rng = np.random.default_rng(seed); ambient = fixed_ambient(); history = []
    start = time.monotonic()
    try:
        for epoch in range(cfg['epochs']):
            t0 = time.monotonic(); order = rng.permutation(data.n); net.train(); total = 0.; trained = 0
            for offset in range(0, data.n, cfg['batch']):
                check_budget(root)
                indices = np.sort(order[offset:offset+cfg['batch']]); batch = data.batch(indices, device)
                opt.zero_grad(set_to_none=True)
                with autocast(device, precision):
                    loss = loss_for(forward(net, arm, batch, ambient), batch)
                if loss is None: continue
                if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite training loss')
                loss.backward(); opt.step(); total += float(loss.detach())*len(indices); trained += len(indices)
            scheduler.step()
            history.append(dict(epoch=epoch+1, loss=total/max(trained, 1), seconds=time.monotonic()-t0,
                                order_sha256=hashlib.sha256(order.tobytes()).hexdigest()))
            write(folder/'progress.json', dict(epoch=epoch+1, history=history, seconds=time.monotonic()-start))
            print(f'{arm} seed{seed} epoch{epoch+1} loss={history[-1]["loss"]:.6f} seconds={history[-1]["seconds"]:.2f}', flush=True)
        if data.hashes() != before: raise ValueError('Training input bytes changed')
        if sha(root/'PLAN.json') != plan_digest or any(sha(p) != h for p, h in source_hashes.items()):
            raise ValueError('PLAN or implementation changed during training')
        checkpoint = folder/'model.pt'
        torch.save({k: v.detach().cpu() for k, v in net.state_dict().items()}, checkpoint)
        receipt = dict(status='COMPLETE', arm=arm, seed=seed, rows=data.n, epochs=cfg['epochs'], batch_size=cfg['batch'],
                       learning_rate=cfg['lr'], weight_decay=cfg['wd'], precision=precision, final_epoch_selection=True,
                       parameters=sum(p.numel() for p in net.parameters()), seconds=time.monotonic()-start,
                       history=history, model_sha256=sha(checkpoint), plan_sha256=plan_digest,
                       input_sha256=before, source_sha256=source_hashes,
                       gpu_input_cache=cache,
                       initialization='M3 matching seed' if arm == 'V' else 'random matching seed',
                       baseline_sha256=None if baseline is None else sha(baseline),
                       amendment_sha256=sha(root/'PLAN_AMENDMENT.json') if (root/'PLAN_AMENDMENT.json').exists() else None,
                       input_validation='full SHA once then sealed size/mtime per seed; final root full hash audit required')
        write(folder/'training_receipt.json', receipt)
        return receipt
    finally:
        data.close(); del net, opt, scheduler; gc.collect()
        if device.startswith('cuda'): torch.cuda.empty_cache()


def infer(root=OUT, arm='V', seed=0, split='evaluation', device='cuda'):
    root = Path(root); configure(device); check_budget(root); start = time.monotonic()
    folder = root/'models'/arm/f'seed{seed}'
    receipt = json.loads((folder/'training_receipt.json').read_text(encoding='utf8'))
    if receipt['status'] != 'COMPLETE' or receipt['model_sha256'] != sha(folder/'model.pt'):
        raise ValueError('Incomplete/changed model')
    data = Inputs(root, split, arm, training=False)
    cache = data.enable_gpu_cache(device)
    net, _ = network(arm, seed, device, initialize=False)
    net.load_state_dict(torch.load(folder/'model.pt', map_location=device, weights_only=True)); net.eval()
    ambient = fixed_ambient(); out = root/'scores'/split; out.mkdir(parents=True, exist_ok=True)
    dest = out/f'{arm}_seed{seed}.npy'
    if dest.exists(): raise FileExistsError('Frozen score output exists')
    values = np.lib.format.open_memmap(dest, mode='w+', dtype=np.float32, shape=(data.n, 2))
    before = data.hashes()
    try:
        with torch.inference_mode():
            for offset in range(0, data.n, BATCH):
                check_budget(root); ids = np.arange(offset, min(offset+BATCH, data.n))
                batch = data.batch(ids, device)
                # Always float32 inference, including BF16-trained weights.
                scores = forward(net, arm, batch, ambient).float().cpu().numpy()
                if not np.isfinite(scores).all(): raise FloatingPointError('Nonfinite inference')
                values[ids] = scores
        values.flush()
        if data.hashes() != before: raise ValueError('Inference observation bytes changed')
        result = dict(status='COMPLETE', arm=arm, seed=seed, split=split, rows=data.n,
                      seconds=time.monotonic()-start, raw_sha256=sha(dest), input_sha256=before,
                      model_sha256=sha(folder/'model.pt'), label_access=False, gpu_input_cache=cache)
        write(dest.with_suffix('.json'), result)
        return result
    finally:
        del values, net; data.close(); gc.collect()
        if device.startswith('cuda'): torch.cuda.empty_cache()


def check_cache(root=OUT, device='cuda'):
    """Independent exact input/logit parity; never edit sealed smoke evidence."""
    configure(device); check_budget(root); began=time.monotonic(); checks={}
    for arm in ('V', 'T'):
        data=Inputs(root, 'smoke', arm, training=True)
        before=data.hashes(); ids=np.arange(min(recipe(root)['batch'], data.n))
        baseline=data.batch(ids, device); net,_=network(arm, 0, device); net.eval()
        with torch.inference_mode():
            old=forward(net, arm, baseline, fixed_ambient())
            cache=data.enable_gpu_cache(device)
            if not cache['used']: raise RuntimeError(f'Smoke GPU cache unavailable: {cache}')
            batch=data.batch(ids, device)
            equal={k: bool(torch.equal(baseline[k], batch[k])) for k in baseline}
            new=forward(net, arm, batch, fixed_ambient())
            logit_equal=bool(torch.equal(old, new))
            delta=float((old-new).abs().max())
            if not all(equal.values()) or not logit_equal:
                raise ValueError('GPU input cache changed input bytes or logits')
        if data.hashes()!=before: raise ValueError('Smoke input changed during cache check')
        checks[arm]=dict(input_bitwise_equal=equal, logits_bitwise_equal=logit_equal,
                         logits_max_abs=delta, gpu_input_cache=cache, input_sha256=before)
        data.close(); del net,baseline,batch,old,new; gc.collect(); torch.cuda.empty_cache()
    receipt=dict(status='PASS', seconds=time.monotonic()-began, checks=checks,
                 scope='engineering cache parity only; no updates, no repeated smoke epoch',
                 source_sha256={str(Path(p).resolve()): sha(p) for p in (__file__, M.__file__)})
    path=Path(root)/'gpu_cache_check.json'
    if path.exists(): raise FileExistsError('GPU cache check evidence already exists')
    write(path,receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('smoke', 'train', 'infer', 'check-cache'))
    parser.add_argument('--root', type=Path, default=OUT); parser.add_argument('--arm', choices=('V', 'T'), default='V')
    parser.add_argument('--seed', type=int, choices=(0, 1, 2), default=0); parser.add_argument('--split', default='evaluation')
    parser.add_argument('--device', default='cuda'); parser.add_argument('--precision', choices=('float32', 'bf16'), default='float32')
    args = parser.parse_args()
    if args.command == 'smoke': result = smoke(args.root, args.device, args.precision)
    elif args.command == 'check-cache': result = check_cache(args.root, args.device)
    elif args.command == 'train': result = train(args.root, args.arm, args.seed, args.device, args.precision)
    else: result = infer(args.root, args.arm, args.seed, args.split, args.device)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
