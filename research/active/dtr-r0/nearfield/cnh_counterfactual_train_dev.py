"""Matched three-arm training and label-blind inference for Development.

The unchanged 8,833-parameter BoundaryTokenReadout uses center geometry, with
six signed-margin inputs zero. Pair identities never enter the model or loss.
Each seed gives all arms one initialization and one slot permutation schedule.
Native data generation and evaluator-only category/threshold analysis live in
separate modules. This module performs no scientific work on import.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cnh_boundary_token_model as M
import cnh_counterfactual_common_dev as C


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009'
ARMS = ('base', 'ordinary', 'counterfactual')
SEEDS = (2026100955, 2026100956, 2026100957)
KEPT_CHANNELS = (0, 1, 2, 3, 4, 5, 12, 13, 14)


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def slot_schedule(seed, epoch, old_rows, new_rows):
    """One common permutation; base replaces new slots with old observations.

    Each old row occurs once per epoch before the extra slots. When possible,
    base's extra old rows are sampled without replacement within that epoch.
    Ordinary and counterfactual new rows each occur exactly once; their row
    bindings remain a data-builder responsibility, never a contrast loss.
    """
    if old_rows <= 0 or new_rows < 0:
        raise ValueError('Invalid old/new sample counts')
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 0])).permutation(old_rows+new_rows)
    extra = np.random.default_rng(np.random.SeedSequence([seed, epoch, 1])).choice(
        old_rows, size=new_rows, replace=new_rows > old_rows)
    base = np.concatenate((np.arange(old_rows), extra))[order]
    return order.astype(np.int64), base.astype(np.int64)


def restore_features(compact):
    """Expand nine cached channels; six signed margins are exactly zero."""
    if compact.ndim != 6 or compact.shape[1:] != (2, 16, 9, 8, 8):
        raise ValueError('Expected compact features [B,2,16,9,8,8]')
    features = compact.new_zeros((len(compact), 2, 16, 15, 8, 8), dtype=torch.float32)
    features[:, :, :, list(KEPT_CHANNELS)] = compact.float()
    return features


def _config(path):
    path = Path(path)
    cfg = json.loads(path.read_text(encoding='utf-8-sig'))
    out = Path(cfg.get('output', DEFAULT_OUT)).resolve()
    canonical = (ROOT/'artifacts.local').resolve()
    if not out.is_relative_to(canonical):
        raise ValueError('Outputs must stay under canonical artifacts.local')
    if out != C.OUT.resolve():
        raise ValueError('This runner must use the shared counterfactual run root')
    plan = Path(cfg.get('plan', out/'PLAN.json'))
    if not plan.is_file():
        raise FileNotFoundError('Scientific stages require the frozen main PLAN')
    cfg.update(output=str(out), plan=str(plan), config_path=str(path.resolve()))
    cfg.setdefault('seeds', list(SEEDS))
    cfg.setdefault('epochs', cfg.get('epochs_equivalent', 32))
    cfg.setdefault('batch', cfg.get('batch_size', 256))
    cfg.setdefault('lr', .001)
    cfg.setdefault('weight_decay', .0001)
    cfg.setdefault('feature_batch', 64)
    cfg.setdefault('steps_per_job', cfg.get('steps', 7488))
    data_root = out/cfg.get('data_subdir', 'data')
    cfg.setdefault('train_inputs', dict(old=str(ROOT/cfg['old_input']),
        ordinary=str(data_root/'ordinary_train'), counterfactual=str(data_root/'cf_train')))
    cfg.setdefault('inference_inputs', dict(cal=str(data_root/'cal'), validation=str(data_root/'validation')))
    cfg.setdefault('inference_shapes', {name: [cfg.get('eval_scenes_per_split', 384),
        cfg.get('eval_K', 4), 13, 2] for name in cfg['inference_inputs']})
    if cfg['epochs'] <= 0 or cfg['batch'] <= 0 or len(set(cfg['seeds'])) != len(cfg['seeds']):
        raise ValueError('Invalid training schedule')
    return cfg


def _inputs(spec, training=False):
    """Directory of .npy files or explicit per-array paths; no truth on infer.

    Keys are histories, transforms, length; training additionally requires
    labels, weights, mask. A directory can be given as a string or under
    directory. Explicit paths can override the names in that directory.
    """
    if isinstance(spec, str):
        spec = {'directory': spec}
    directory = Path(spec['directory']) if 'directory' in spec else None
    names = ('histories', 'transforms', 'length') + (('labels', 'weights', 'mask') if training else ())
    paths = {name: Path(spec[name]) if name in spec else directory/(name+'.npy') for name in names}
    arrays = {name: np.load(path, mmap_mode='r', allow_pickle=False) for name, path in paths.items()}
    n = len(arrays['length'])
    if not n or any(len(v) != n for v in arrays.values()):
        raise ValueError('Input row counts differ or are empty')
    if arrays['histories'].shape != (n, 8, 8, 8, 16) or arrays['transforms'].shape != (n, 8, 4, 4):
        raise ValueError('Native observation/transform axes differ')
    if training:
        if any(arrays[k].shape != (n, 2) for k in ('labels', 'weights', 'mask')):
            raise ValueError('Training supervision must be [N,2]')
        y, w, mask = (arrays[k] for k in ('labels', 'weights', 'mask'))
        if (not np.isin(y, [0., 1.]).all() or not np.isin(mask, [0., 1.]).all()
                or not np.isfinite(w).all() or (w < 0).any()
                or np.any(w[mask == 0] != 0) or not w.sum() > 0):
            raise ValueError('Training labels/valid-mask/weights violate BCE contract')
    return arrays, paths


def _configure():
    if not torch.cuda.is_available():
        raise RuntimeError('Scientific stages require CUDA')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _cache_name(out, name, yaw=0):
    suffix = '' if not yaw else '_yaw_plus3' if yaw == 3 else f'_yaw{yaw:+g}'
    return out/'features'/(name+suffix+'.npy')


def prepare_one(cfg, name, spec, budget, *, yaw=0):
    """Observation-only cache preparation; no training truth opened here."""
    out = Path(cfg['output'])
    path = _cache_name(out, name, yaw)
    receipt_path = path.with_suffix('.json')
    arrays, sources = _inputs(spec)
    provided_yaw = False
    if yaw == 3:
        yaw_path = sources['transforms'].with_name('transforms_yaw3.npy')
        if yaw_path.is_file():
            arrays['transforms'] = np.load(yaw_path, mmap_mode='r', allow_pickle=False)
            sources['transforms'] = yaw_path
            if arrays['transforms'].shape != (len(arrays['length']), 8, 4, 4):
                raise ValueError('Provided yaw transform axes differ')
            provided_yaw = True
    n = len(arrays['length'])
    signatures = {k: dict(path=str(p.resolve()), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                  for k, p in sources.items()}
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text(encoding='utf8'))
        if prior['rows'] != n or prior['sources'] != signatures or prior['yaw_degrees'] != yaw:
            raise ValueError('Completed feature cache binding changed')
        return prior
    if path.exists():
        raise FileExistsError('Partial feature cache preserved; use a fresh cache name for source repair')
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    maps = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=(n, 2, 16, 9, 8, 8))
    angle = math.radians(yaw)
    rotation = torch.eye(4, dtype=torch.float32, device='cuda')
    c, s = math.cos(angle), math.sin(angle)
    rotation[:3, :3] = rotation.new_tensor([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    with torch.no_grad():
        for start in range(0, n, cfg['feature_batch']):
            budget.check()
            ix = slice(start, min(start+cfg['feature_batch'], n))
            histories = torch.as_tensor(np.array(arrays['histories'][ix]), device='cuda')
            transforms = torch.as_tensor(np.array(arrays['transforms'][ix]), device='cuda', dtype=torch.float32)
            lengths = torch.as_tensor(np.array(arrays['length'][ix]), device='cuda')
            if yaw and not provided_yaw:
                transforms = rotation[None, None]@transforms
            geometry = M.feature_geometry(transforms, lengths, 'center')
            compact = M.build_features(histories, geometry, lengths)[:, :, :, list(KEPT_CHANNELS)]
            compact = compact.half()
            if not bool(torch.isfinite(compact).all()):
                raise ValueError('Nonfinite FP16 cached features')
            maps[ix] = compact.cpu().numpy()
            if start % (cfg['feature_batch']*100) == 0:
                print('FEATURES', name, yaw, start, '/', n, flush=True)
    maps.flush()
    del maps
    receipt = dict(rows=n, shape=[n, 2, 16, 9, 8, 8], dtype='float16',
        kept_channels=list(KEPT_CHANNELS), zero_channels=[6, 12], geometry='center',
        yaw_degrees=yaw, yaw_convention=('provided public-query/sensor rebuilt transforms_yaw3.npy'
            if provided_yaw else 'left multiply public query transforms with +yaw Y-axis rotation'),
        sources=signatures, seconds=time.monotonic()-started, bytes=path.stat().st_size)
    save(receipt_path, receipt)
    return receipt


def _load_shared_cache(cfg, names, budget):
    """One GPU allocation and chunked copies avoid duplicating resident caches."""
    out = Path(cfg['output'])
    maps = {n: np.load(_cache_name(out, n), mmap_mode='r', allow_pickle=False) for n in names}
    n_total = sum(len(v) for v in maps.values())
    needed = n_total*2*16*9*8*8*2
    free, _ = torch.cuda.mem_get_info()
    if needed > free-1536*1024**2:
        raise RuntimeError(f'Insufficient CUDA headroom for shared compact cache: {needed} bytes')
    features = torch.empty((n_total, 2, 16, 9, 8, 8), dtype=torch.float16, device='cuda')
    offsets, cursor = {}, 0
    for name, values in maps.items():
        offsets[name] = cursor
        for start in range(0, len(values), 1024):
            budget.check()
            end = min(start+1024, len(values))
            features[cursor+start:cursor+end].copy_(torch.from_numpy(np.array(values[start:end])))
        cursor += len(values)
    return features, offsets


def prepare(cfg, budget):
    receipts = {}
    for name in ('old', 'ordinary', 'counterfactual'):
        receipts[name] = prepare_one(cfg, name, cfg['train_inputs'][name], budget)
    for name, spec in cfg.get('inference_inputs', {}).items():
        for yaw in (0, 3):
            receipts[f'{name}_yaw{yaw}'] = prepare_one(cfg, name, spec, budget, yaw=yaw)
    return dict(features=receipts)


def train(cfg, *, only_seed=None, only_arm=None):
    out = Path(cfg['output'])
    data = {name: _inputs(cfg['train_inputs'][name], training=True)[0]
            for name in ('old', 'ordinary', 'counterfactual')}
    old_rows = len(data['old']['length'])
    new_rows = len(data['ordinary']['length'])
    if len(data['counterfactual']['length']) != new_rows:
        raise ValueError('Ordinary and counterfactual new row counts differ')
    if old_rows != cfg.get('old_rows', 39936) or new_rows != cfg.get('new_rows', 19968):
        raise ValueError('Frozen old/new row identity differs')
    steps_per_epoch = math.ceil((old_rows+new_rows)/cfg['batch'])
    expected_steps = cfg['epochs']*steps_per_epoch
    if expected_steps != cfg.get('steps_per_job', expected_steps):
        raise ValueError('Frozen shared optimizer step count differs')
    cache_stage = C.Stage('train_cache_load')
    try:
        features, offsets = _load_shared_cache(cfg, tuple(data), cache_stage)
        cache_stage.finish('COMPLETE', bytes=features.numel()*features.element_size())
    except Exception as error:
        cache_stage.finish('FAILED', error=repr(error))
        raise
    merged = {key: torch.as_tensor(np.concatenate([v[key] for v in data.values()]), device='cuda')
              for key in ('labels', 'weights', 'length')}
    models_folder = out/'models'
    models_folder.mkdir(exist_ok=True)
    jobs = []
    seeds = cfg['seeds'] if only_seed is None else [only_seed]
    arms = ARMS if only_arm is None else (only_arm,)
    if any(seed not in cfg['seeds'] for seed in seeds) or any(arm not in ARMS for arm in arms):
        raise ValueError('Unknown requested seed/arm')
    active_model = active_optimizer = None
    current_job = None
    completed_steps = 0
    loss_rows = []
    budget = None
    try:
        for seed in seeds:
            torch.manual_seed(seed)
            init = {k: v.clone() for k, v in M.BoundaryTokenReadout().state_dict().items()}
            for arm in arms:
                name = f'{arm}_seed{seed}'
                target = models_folder/(name+'.pt')
                job_receipt = models_folder/(name+'.json')
                if target.exists() or job_receipt.exists():
                    if not (target.exists() and job_receipt.exists()):
                        raise FileExistsError('Incomplete final-checkpoint binding preserved')
                    prior = json.loads(job_receipt.read_text(encoding='utf8'))
                    if prior['steps'] != expected_steps or prior['plan_sha256'] != sha(cfg['plan']):
                        raise ValueError('Completed job binding differs')
                    jobs.append(prior)
                    continue
                current_job, completed_steps, loss_rows = name, 0, []
                budget = C.Stage('train_'+name)
                job_start = time.monotonic()
                active_model = M.BoundaryTokenReadout().cuda()
                active_model.load_state_dict(init)
                active_optimizer = torch.optim.AdamW(active_model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
                active_model.train()
                for epoch in range(cfg['epochs']):
                    order, base = slot_schedule(seed, epoch, old_rows, new_rows)
                    if arm == 'base':
                        bound = base
                    else:
                        bound = order.copy()
                        new = bound >= old_rows
                        bound[new] = offsets[arm]+bound[new]-old_rows
                    mass_sum = bce_sum = 0.
                    for start in range(0, len(bound), cfg['batch']):
                        budget.check()
                        ix = torch.as_tensor(bound[start:start+cfg['batch']], device='cuda')
                        output = active_model(restore_features(features[ix]), merged['length'][ix])
                        bce = F.binary_cross_entropy_with_logits(output, merged['labels'][ix].float(), reduction='none')
                        mass = merged['weights'][ix].sum()
                        if not bool(mass > 0):
                            raise ValueError('A training batch has zero valid supervision mass')
                        weighted = (bce*merged['weights'][ix]).sum()
                        loss = weighted/mass
                        if not bool(torch.isfinite(loss)):
                            raise ValueError('Nonfinite weighted BCE')
                        active_optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        active_optimizer.step()
                        completed_steps += 1
                        bce_sum += float(weighted.detach())
                        mass_sum += float(mass)
                    row = dict(job=name, seed=seed, arm=arm, epoch=epoch+1,
                        steps=completed_steps, weighted_BCE=bce_sum/mass_sum,
                        supervised_mass=mass_sum, seconds=time.monotonic()-job_start)
                    loss_rows.append(row)
                    save(out/'progress'/(name+'.json'), row)
                    save(out/'loss_curves'/(name+'.json'), loss_rows)
                    print('TRAIN', name, epoch+1, completed_steps, round(row['weighted_BCE'], 6), flush=True)
                if completed_steps != expected_steps:
                    raise ValueError('Final optimizer step mismatch')
                torch.save(dict(state_dict=active_model.cpu().state_dict(), arm=arm,
                    seed=seed, epochs=cfg['epochs'], steps=completed_steps, parameters=M.parameter_count(),
                    geometry='center', zero_channels=[6, 12], plan_sha256=sha(cfg['plan'])), target)
                receipt = dict(job=name, arm=arm, seed=seed, epochs=cfg['epochs'], steps=completed_steps,
                    parameters=M.parameter_count(), old_rows=old_rows, new_slots=new_rows,
                    final_weighted_BCE=loss_rows[-1]['weighted_BCE'], seconds=time.monotonic()-job_start,
                    checkpoint_sha256=sha(target), plan_sha256=sha(cfg['plan']),
                    trainer_source_sha256=sha(__file__), model_source_sha256=sha(M.__file__),
                    supervision='ordinary masked weighted BCE only; no pair identity or contrast loss',
                    initialization='same exact seed state across all arms', sampling='shared epoch slot order; base repeats old rows')
                save(job_receipt, receipt)
                stage_metadata = {k: v for k, v in receipt.items() if k != 'seconds'}
                budget.finish('COMPLETE', job_seconds=receipt['seconds'], **stage_metadata)
                budget = None
                jobs.append(receipt)
                active_model = active_optimizer = None
        summary = dict(jobs=jobs, steps_per_job=expected_steps, shared_cache_bytes=features.numel()*features.element_size())
        save(out/'training_summary.json', summary)
        print('TRAIN_COMPLETE', len(jobs), 'jobs', expected_steps, 'steps/job', flush=True)
        return summary
    except Exception as error:
        if active_model is not None and current_job:
            torch.save(dict(state_dict=active_model.cpu().state_dict(), steps=completed_steps,
                optimizer_state_dict=active_optimizer.state_dict(), job=current_job,
                final=False, plan_sha256=sha(cfg['plan'])), out/'models'/(current_job+f'_partial_{time.time_ns()}.pt'))
            save(out/'loss_curves'/(current_job+f'_partial_{time.time_ns()}.json'), loss_rows)
        if budget is not None:
            budget.finish('FAILED', job=current_job, steps=completed_steps, error=repr(error))
        raise
    finally:
        del features, merged
        active_model = active_optimizer = None
        torch.cuda.empty_cache()


def infer(cfg, budget):
    out = Path(cfg['output'])
    checkpoints = [(seed, arm, out/'models'/f'{arm}_seed{seed}.pt')
                   for seed in cfg['seeds'] for arm in ARMS]
    if any(not p.is_file() for _, _, p in checkpoints):
        raise FileNotFoundError('Inference requires every declared final checkpoint')
    receipts = []
    for name, spec in cfg['inference_inputs'].items():
        arrays, _ = _inputs(spec)
        shape = tuple(cfg['inference_shapes'][name])
        if len(shape) != 4 or shape[-2:] != (13, 2) or int(np.prod(shape[:-1])) != len(arrays['length']):
            raise ValueError('Inference shape must bind all rows as [scene,K,13,2]')
        lengths = torch.as_tensor(np.array(arrays['length']), device='cuda')
        for yaw in (0, 3):
            cache_name = name if not yaw else f'{name}_yaw_plus3'
            # Load only the current inference split, shared across all nine jobs.
            features, _ = _load_shared_cache(cfg, (cache_name,), budget)
            for seed, arm, checkpoint in checkpoints:
                target = out/'scores'/f'{arm}_seed{seed}_{cache_name}.npz'
                receipt_path = target.with_suffix('.json')
                if target.exists() or receipt_path.exists():
                    if not (target.exists() and receipt_path.exists()):
                        raise FileExistsError('Incomplete final score binding preserved')
                    receipts.append(json.loads(receipt_path.read_text(encoding='utf8')))
                    continue
                started = time.monotonic()
                model = M.BoundaryTokenReadout().cuda().eval()
                state = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if not state['steps'] == cfg.get('steps_per_job', cfg['epochs']*math.ceil((cfg.get('old_rows', 39936)+cfg.get('new_rows', 19968))/cfg['batch'])):
                    raise ValueError('Inference checkpoint step count differs')
                model.load_state_dict(state['state_dict'])
                raw = np.empty((len(lengths), 2), np.float32)
                with torch.inference_mode():
                    for start in range(0, len(lengths), cfg['batch']):
                        budget.check()
                        ix = slice(start, min(start+cfg['batch'], len(lengths)))
                        raw[ix] = model(restore_features(features[ix]), lengths[ix]).cpu().numpy()
                if not np.isfinite(raw).all():
                    raise ValueError('Nonfinite raw inference logits')
                target.parent.mkdir(exist_ok=True)
                branch = 'ideal' if not yaw else 'yaw+3'
                np.savez_compressed(target, raw=raw.reshape(shape), arm=np.array(arm),
                    seed=np.array(seed), split=np.array(name), branch=np.array(branch))
                receipt = dict(seed=seed, arm=arm, split=name, yaw_degrees=yaw,
                    branch=branch, path=str(target.resolve()),
                    rows=len(lengths), shape=list(shape), seconds=time.monotonic()-started,
                    checkpoint_sha256=sha(checkpoint), scores_sha256=sha(target), plan_sha256=sha(cfg['plan']),
                    label_access=False, output='complete raw logits; thresholds belong to separate fixed-cal evaluator')
                save(receipt_path, receipt)
                save(out/'progress'/'infer.json', receipt)
                receipts.append(receipt)
                save(out/'scores'/'manifest.json', dict(outputs=receipts,
                    arm_names=list(ARMS), seed_ids=list(cfg['seeds'])))
                print('INFER', arm, seed, name, yaw, len(lengths), flush=True)
                del model, state
            del features
            torch.cuda.empty_cache()
        del lengths
    save(out/'scores'/'manifest.json', dict(outputs=receipts,
        arm_names=list(ARMS), seed_ids=list(cfg['seeds'])))
    return dict(inferences=receipts)


def execute(stage, config_path, *, seed=None, arm=None):
    cfg = _config(config_path)
    _configure()
    if stage == 'train':
        return train(cfg, only_seed=seed, only_arm=arm)
    budget = C.Stage('counterfactual_'+stage)
    try:
        budget.check()
        details = prepare(cfg, budget) if stage == 'prepare' else infer(cfg, budget)
        receipt = budget.finish('COMPLETE', **details, trainer_source_sha256=sha(__file__),
            model_source_sha256=sha(M.__file__), backend='CUDA', device=torch.cuda.get_device_name(0))
        print(json.dumps(dict(stage=stage, status='COMPLETE', seconds=receipt['seconds'],
                              cumulative_seconds=C.charged_seconds())), flush=True)
        return receipt
    except Exception as error:
        budget.finish('FAILED', error=repr(error))
        raise
    finally:
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'train', 'infer'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--arm', choices=ARMS)
    arguments = parser.parse_args()
    execute(arguments.stage, arguments.config, seed=arguments.seed, arm=arguments.arm)
