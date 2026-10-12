"""Task-labelled light scorers; observation-only inference, no work on import.

S is fixed shallow CPU HGB. B fine-tunes the inherited ordinary token network
on complete native bins; author geometry supplies supervision only. The fixed
bin/public-query feature transform is cached once and shared by all B seeds.
"""
import gc
import hashlib
import json
import math
from pathlib import Path
import pickle
import time

import numpy as np
from threadpoolctl import threadpool_limits

SEEDS = (2026100955, 2026100956, 2026100957)
FRAMES = np.arange(3, 16)
HGB_RECIPE = dict(max_iter=100, learning_rate=.1, max_leaf_nodes=7,
                  max_depth=3, min_samples_leaf=50, l2_regularization=1.,
                  early_stopping=False)
B_RECIPE = dict(epochs=8, lr=1e-4, batch=128, weight_decay=1e-4,
                seed_seconds=900., total_seconds=2700.,
                early_stopping='nonfinite or two successive epoch means >10x first epoch mean; wall limit stops seed',
                checkpoint='last completed fixed epoch; partial optimizer state retained diagnostic only')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve training evidence: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode())


def supervision(category, support, old5, lateral_gap):
    """Per slot labels, ignore mask, and v2 costs; balance separately per height.

    Support is author-only target support at each native frame. Once any target
    support has occurred, contact f3..13 is positive. Contact before support or
    after f13 is ignored. Every noncontact query is negative, including the
    other height in a contact scene. Strong old5 slots are always ignored.
    """
    cat, support, old5 = np.asarray(category), np.asarray(support, bool), np.asarray(old5, bool)
    if support.ndim == 3 and support.shape == (len(old5), 13, 2):
        support = np.broadcast_to(support[:, None], old5.shape)
    if support.shape != old5.shape or support.ndim != 4 or support.shape[2:] != (13, 2):
        raise ValueError('Expected matching [scene,K,13,2] support and old5')
    if cat.shape != (len(old5), 2) or len(lateral_gap) != len(old5):
        raise ValueError('Category/gap scene binding differs')
    contact = np.broadcast_to((cat == 'contact')[:, None, None], old5.shape)
    supported = np.maximum.accumulate(support, axis=2)
    positive = contact & supported & (FRAMES <= 13)[None, None, :, None]
    mask = (~contact | positive) & ~old5
    labels = positive.astype(np.float32)
    # Only authored pass at either height gets the reduced light-negative cost.
    near = (cat == 'pass').any(1) & (np.asarray(lateral_gap) <= .10+1e-10)
    weights = np.broadcast_to(np.where(near, .25, 1.)[:, None, None, None], old5.shape).copy()
    weights *= mask
    counts = {}
    for q, name in enumerate(('HEAD', 'BODY')):
        pos = positive[..., q] & mask[..., q]
        neg = ~positive[..., q] & mask[..., q]
        total_neg, total_pos = weights[..., q][neg].sum(), pos.sum()
        if not total_pos or not total_neg:
            raise ValueError('Task supervision needs both classes per height')
        weights[..., q][pos] = total_neg/total_pos
        counts[name] = dict(positive_slots=int(total_pos), negative_slots=int(neg.sum()),
                           ignored_slots=int((~mask[..., q]).sum()),
                           positive_weight=float(weights[..., q][pos].sum()),
                           negative_weight=float(total_neg))
    return labels, mask, weights.astype(np.float32), counts


def train_s(out, features, labels, weights, check=lambda: None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    out = Path(out)
    features = np.asarray(features)
    if features.shape[0] != 3 or features.shape[1:-1] != labels.shape or features.shape[-1] != 47:
        raise ValueError('Expected three seed features [3,scene,K,13,2,47]')
    began = time.monotonic()
    jobs = []
    with threadpool_limits(limits=2):
        for si, seed in enumerate(SEEDS):
            for q, height in enumerate(('HEAD', 'BODY')):
                check()
                path = out/'models'/f'S_{seed}_{height}.pickle'
                if path.exists():
                    raise FileExistsError('Preserve trained S model')
                start = time.monotonic()
                x = features[si, ..., q, :].reshape(-1, 47)
                y, w = labels[..., q].reshape(-1), weights[..., q].reshape(-1)
                active = w > 0
                model = HistGradientBoostingClassifier(**HGB_RECIPE, random_state=seed)
                model.fit(x[active], y[active], sample_weight=w[active])
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as f:
                    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
                jobs.append(dict(seed=seed, height=height, status='COMPLETE',
                                 seconds=time.monotonic()-start, active_slots=int(active.sum()),
                                 path=str(path), sha256=sha(path), backend='CPU sklearn HGB',
                                 iterations=int(model.n_iter_)))
                print('TASK_S', seed, height, round(jobs[-1]['seconds'], 3), flush=True)
    receipt = dict(status='COMPLETE', seconds=time.monotonic()-began, GPU_seconds=0.,
                   recipe=HGB_RECIPE, jobs=jobs)
    save_new(out/'S_training_receipt.json', receipt)
    return receipt


def predict_s(out, features):
    features = np.asarray(features)
    scores = np.empty(features.shape[:-1], np.float32)
    with threadpool_limits(limits=2):
        for si, seed in enumerate(SEEDS):
            for q, height in enumerate(('HEAD', 'BODY')):
                with (Path(out)/'models'/f'S_{seed}_{height}.pickle').open('rb') as f:
                    model = pickle.load(f)
                x = features[si, ..., q, :]
                scores[si, ..., q] = model.decision_function(x.reshape(-1, 47)).reshape(x.shape[:-1])
    return scores


def _cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('B requires CUDA')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def prepare_native_cache(out, split, z, sensor, query, check=lambda: None):
    """z=[clip,16,8,8,16] normalized photons, public poses only.

    Cached transform preserves every bin/zone; inherited ordinary architecture
    zeroes six signed boundary margins. No author box/category/support is read.
    Row order is clip major then native f3..15. One cache serves all three seeds.
    """
    import cnh_boundary_token_model as M
    import cnh_counterfactual_train_dev as T
    torch = _cuda()
    path = Path(out)/'native_cache'/f'{split}.npy'
    length_path = path.with_name(split+'_length.npy')
    if path.exists() or length_path.exists():
        raise FileExistsError('Preserve native cache')
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    n = len(z)*13
    cache = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16,
                                     shape=(n, 2, 16, 9, 8, 8))
    lengths = np.empty(n, np.int64)
    with torch.no_grad():
        for fj, frame in enumerate(FRAMES):
            start = max(0, int(frame)-7)
            tr = (query[frame]@np.linalg.inv(sensor[frame]))[None]@sensor[start:frame+1]
            length = len(tr)
            padded = np.repeat(np.eye(4)[None], 8, axis=0)
            padded[8-length:] = tr
            for offset in range(0, len(z), 64):
                check()
                stop = min(offset+64, len(z))
                history = np.zeros((stop-offset, 8, 8, 8, 16), np.float16)
                history[:, 8-length:] = z[offset:stop, start:frame+1]
                h = torch.as_tensor(history, device='cuda')
                t = torch.as_tensor(np.broadcast_to(padded, (stop-offset, 8, 4, 4)).copy(),
                                    device='cuda', dtype=torch.float32)
                le = torch.full((stop-offset,), length, device='cuda', dtype=torch.int64)
                g = M.feature_geometry(t, le, 'center')
                compact = M.build_features(h, g, le)[:, :, :, list(T.KEPT_CHANNELS)].half()
                if not bool(torch.isfinite(compact).all()):
                    raise ValueError('Nonfinite native cache')
                ix = np.arange(offset, stop)*13+fj
                cache[ix] = compact.cpu().numpy()
                lengths[ix] = length
            print('TASK_CACHE', split, int(frame), round(time.monotonic()-started, 3), flush=True)
    cache.flush()
    del cache, h, t, le, g, compact
    np.save(length_path, lengths)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    receipt = dict(status='COMPLETE', split=split, rows=n, shape=[n, 2, 16, 9, 8, 8],
                   seconds=time.monotonic()-started, backend='CUDA native bin feature transform',
                   cache_path=str(path), length_path=str(length_path),
                   kept_channels=list(T.KEPT_CHANNELS), author_geometry_read=False,
                   bytes=path.stat().st_size, sha256=sha(path), length_sha256=sha(length_path))
    save_new(path.with_suffix('.json'), receipt)
    return receipt


def _checkpoint_model(torch, checkpoint):
    import cnh_boundary_token_model as M
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state['arm'] != 'ordinary' or state['steps'] != 7488:
        raise ValueError('B init must be inherited ordinary checkpoint')
    model = M.BoundaryTokenReadout().cuda()
    model.load_state_dict(state['state_dict'])
    return model, state


def _restore(torch, compact):
    import cnh_counterfactual_train_dev as T
    return T.restore_features(compact)


def profile_b(out, cache_path, length_path, checkpoint, labels, weights, check=lambda: None):
    """Fixed first 256 active slots, eight steps, discarded; never chooses recipe."""
    torch = _cuda()
    began = time.monotonic()
    cache = np.load(cache_path, mmap_mode='r')
    lengths = np.load(length_path, mmap_mode='r')
    y, w = labels.reshape(-1, 2), weights.reshape(-1, 2)
    ix = np.flatnonzero(w.sum(1) > 0)[:256]
    model, state = _checkpoint_model(torch, checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=B_RECIPE['lr'], weight_decay=B_RECIPE['weight_decay'])
    losses = []
    try:
        step_started = time.monotonic()
        for step in range(8):
            check()
            start = (step*B_RECIPE['batch']) % len(ix)
            ids = ix[start:start+B_RECIPE['batch']]
            x = torch.as_tensor(np.array(cache[ids]), device='cuda')
            le = torch.as_tensor(np.array(lengths[ids]), device='cuda')
            label = torch.as_tensor(y[ids], device='cuda')
            weight = torch.as_tensor(w[ids], device='cuda')
            optimizer.zero_grad(set_to_none=True)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(model(_restore(torch, x), le), label,
                                                                         reduction='none')*weight).sum()/weight.sum()
            if not bool(torch.isfinite(loss)):
                raise ValueError('Profile loss is nonfinite')
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            losses.append(float(loss))
        elapsed = time.monotonic()-step_started
        steps = len(losses)
        projected = elapsed/max(1, steps)*math.ceil(int((w.sum(1) > 0).sum())/B_RECIPE['batch'])*B_RECIPE['epochs']
        receipt = dict(status='COMPLETE', seconds=time.monotonic()-began, timed_steps_seconds=elapsed,
                       steps=steps, slots=len(ix), losses=losses, projected_seconds_per_seed=projected,
                       fixed_recipe=B_RECIPE, discarded_model=True, init_seed=int(state['seed']),
                       cache_reuse='one bin-transform cache for all B seeds; no repeated model-independent transforms',
                       GPU_name=torch.cuda.get_device_name(), GPU_free_bytes=int(torch.cuda.mem_get_info()[0]))
        save_new(Path(out)/'B_profile_receipt.json', receipt)
        return receipt
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


def train_b(out, cache_path, length_path, checkpoints, labels, weights, check=lambda: None,
            total_seconds=2700., prior_seconds=0.):
    torch = _cuda()
    if not (Path(out)/'B_profile_receipt.json').exists():
        raise FileNotFoundError('Profile must precede long B training')
    began = time.monotonic()
    cache, lengths = np.load(cache_path, mmap_mode='r'), np.load(length_path, mmap_mode='r')
    y, w = labels.reshape(-1, 2), weights.reshape(-1, 2)
    active = np.flatnonzero(w.sum(1) > 0)
    if len(cache) != len(y) or len(lengths) != len(y):
        raise ValueError('Native sample supervision row binding differs')
    resident = None
    jobs = []
    # Sharing one resident cache is safe only with room for activations/optimizer.
    free = torch.cuda.mem_get_info()[0]
    resident_bytes = cache.nbytes
    try:
        if resident_bytes < free-1536*1024**2:
            resident = torch.empty(cache.shape, device='cuda', dtype=torch.float16)
            for start in range(0, len(cache), 512):
                check()
                resident[start:start+512].copy_(torch.from_numpy(np.array(cache[start:start+512])))
        for seed in SEEDS:
            check()
            seed_start = time.monotonic()
            if time.monotonic()-began+prior_seconds >= total_seconds:
                jobs.append(dict(seed=seed, status='NOT_STARTED_TOTAL_LIMIT', seconds=0.))
                continue
            path = Path(out)/'models'/f'B_{seed}.pt'
            if path.exists():
                raise FileExistsError('Preserve B model')
            torch.manual_seed(seed)
            model, inherited = _checkpoint_model(torch, checkpoints[seed])
            if int(inherited['seed']) != seed:
                raise ValueError('Ordinary seed binding differs')
            optimizer = torch.optim.AdamW(model.parameters(), lr=B_RECIPE['lr'], weight_decay=B_RECIPE['weight_decay'])
            records = []
            completed_epochs, steps, status = 0, 0, 'COMPLETE'
            completed_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            try:
                for epoch in range(B_RECIPE['epochs']):
                    order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 20261010])).permutation(active)
                    loss_sum = weight_sum = 0.
                    for start in range(0, len(order), B_RECIPE['batch']):
                        check()
                        if time.monotonic()-seed_start >= B_RECIPE['seed_seconds']:
                            status = 'PARTIAL_SEED_LIMIT'
                            break
                        if time.monotonic()-began+prior_seconds >= total_seconds:
                            status = 'PARTIAL_TOTAL_LIMIT'
                            break
                        ids = order[start:start+B_RECIPE['batch']]
                        ids_gpu = torch.as_tensor(ids, device='cuda')
                        x = resident[ids_gpu] if resident is not None else torch.as_tensor(np.array(cache[ids]), device='cuda')
                        le = torch.as_tensor(np.array(lengths[ids]), device='cuda')
                        label = torch.as_tensor(y[ids], device='cuda')
                        weight = torch.as_tensor(w[ids], device='cuda')
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(_restore(torch, x), le)
                        mass = weight.sum()
                        weighted = (torch.nn.functional.binary_cross_entropy_with_logits(logits, label, reduction='none')*weight).sum()
                        loss = weighted/mass
                        if not bool(torch.isfinite(loss)):
                            status = 'PARTIAL_DIVERGENCE'
                            break
                        loss.backward()
                        optimizer.step()
                        loss_sum += float(weighted.detach())
                        weight_sum += float(mass)
                        steps += 1
                    records.append(dict(epoch=epoch+1, weighted_loss=loss_sum/weight_sum if weight_sum else None,
                                        completed=(status == 'COMPLETE'), optimizer_steps=steps,
                                        seconds=time.monotonic()-seed_start))
                    print('TASK_B', seed, epoch+1, records[-1]['weighted_loss'], round(records[-1]['seconds'], 2), status, flush=True)
                    if status != 'COMPLETE':
                        break
                    completed_epochs += 1
                    completed_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    if (len(records) >= 3 and records[-1]['weighted_loss'] > 10*records[0]['weighted_loss']
                            and records[-2]['weighted_loss'] > 10*records[0]['weighted_loss']):
                        status = 'PARTIAL_DIVERGENCE'
                        break
                torch.cuda.synchronize()
                path.parent.mkdir(parents=True, exist_ok=True)
                partial_diagnostic = None
                if status != 'COMPLETE':
                    partial_path = path.with_name(path.stem+'_partial_step.pt')
                    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                    optimizer_state=optimizer.state_dict(), seed=seed, steps=steps,
                                    completed_epochs=completed_epochs, status=status,
                                    inference_eligible=False, diagnostic_only=True), partial_path)
                    partial_diagnostic = dict(path=str(partial_path), sha256=sha(partial_path))
                    model.load_state_dict(completed_state)
                torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                                arm='task_cost_B', seed=seed, steps=steps, epochs=completed_epochs,
                                status=status, recipe=B_RECIPE, initialization_sha256=sha(checkpoints[seed])), path)
                job = dict(seed=seed, status=status, seconds=time.monotonic()-seed_start,
                           completed_epochs=completed_epochs, steps=steps, loss_epochs=records,
                           path=str(path), sha256=sha(path), init_sha256=sha(checkpoints[seed]),
                           partial_step_diagnostic=partial_diagnostic,
                           backend='CUDA FP32 BoundaryTokenReadout', parameter_count=sum(p.numel() for p in model.parameters()))
                save_new(path.with_suffix('.json'), job)
                jobs.append(job)
            finally:
                del model, optimizer
                gc.collect()
                torch.cuda.empty_cache()
        receipt = dict(status='COMPLETE' if all(j['status'] == 'COMPLETE' for j in jobs) else 'PARTIAL',
                       seconds=time.monotonic()-began, charged_prior_seconds=prior_seconds,
                       GPU_seconds=time.monotonic()-began, recipe=B_RECIPE, jobs=jobs,
                       cache_mode='shared resident GPU' if resident is not None else 'disk batches',
                       cache_bytes=resident_bytes, active_rows=len(active), label_rows=len(y))
        save_new(Path(out)/'B_training_receipt.json', receipt)
        return receipt
    finally:
        if resident is not None:
            del resident
        gc.collect()
        torch.cuda.empty_cache()


def predict_b(out, cache_path, length_path, shape, check=lambda: None):
    import cnh_boundary_token_model as M
    torch = _cuda()
    cache, lengths = np.load(cache_path, mmap_mode='r'), np.load(length_path, mmap_mode='r')
    # Missing/incomplete fixed-recipe seeds remain NaN so the controller can
    # report partial completion and exclude them from full-recipe arms.
    results = np.full((3, len(cache), 2), np.nan, np.float32)
    model = None
    try:
        with torch.inference_mode():
            for si, seed in enumerate(SEEDS):
                check()
                checkpoint = Path(out)/'models'/f'B_{seed}.pt'
                if not checkpoint.exists():
                    continue
                state = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if state['status'] != 'COMPLETE':
                    continue
                model = M.BoundaryTokenReadout().cuda().eval()
                model.load_state_dict(state['state_dict'])
                for start in range(0, len(cache), 128):
                    check()
                    end = min(start+128, len(cache))
                    x = torch.as_tensor(np.array(cache[start:end]), device='cuda')
                    le = torch.as_tensor(np.array(lengths[start:end]), device='cuda')
                    results[si, start:end] = model(_restore(torch, x), le).cpu().numpy()
                del model
                model = None
                torch.cuda.empty_cache()
        return results.reshape((3, *shape))
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
