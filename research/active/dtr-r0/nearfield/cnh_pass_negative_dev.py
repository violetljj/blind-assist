"""Paired ordinary-checkpoint continuation with optional weak pass negatives.

Only new ordinary-training pass queries receive supervision changes. Old
supervision and all observations are preserved. Feature caches and outputs
belong to a new run; previous runs are immutable. No training, CUDA allocation
or result-based selection occurs on import. Evaluation thresholds are external.
"""
import argparse
import importlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cnh_boundary_token_model as M
import cnh_counterfactual_train_dev as T


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT/'artifacts.local/work/cnh-pass-boundary-dev-20261009'
PREVIOUS = ROOT/'artifacts.local/work/cnh-counterfactual-dev-20261009'
ARMS = ('control', 'weak_pass')
WEAK_RAW_FACTOR = .25


def weak_supervision(labels, mask, weights, pass_queries, far_factor, raw_weight_mass):
    """Modify pass slots only, then normalize new-array total mass to N.

    Restore original raw scale using the builder's exact FP32 N/raw_mass
    multiplier. New pass raw weights are .25*inherited far factor (1 or 2).
    The normalization rescales existing valid ordinary weights as well; this
    is a fixed loss-mass exchange, not a claim of class balancing.
    """
    labels, mask, weights = (np.asarray(x) for x in (labels, mask, weights))
    pass_queries = np.asarray(pass_queries, bool)
    far_factor = np.asarray(far_factor, np.float64)
    n = len(labels)
    if labels.shape != (n, 2) or mask.shape != labels.shape or weights.shape != labels.shape:
        raise ValueError('Expected new ordinary supervision [N,2]')
    if pass_queries.shape != labels.shape or far_factor.shape not in ((n,), (n, 2)):
        raise ValueError('Pass/row distance-factor axes differ')
    if not raw_weight_mass > 0 or not np.isin(far_factor, [1., 2.]).all():
        raise ValueError('Original raw mass or inherited far factor is invalid')
    far_factor = np.broadcast_to(far_factor[:, None] if far_factor.ndim == 1 else far_factor, labels.shape)
    if np.any(mask[pass_queries] != 0) or np.any(weights[pass_queries] != 0):
        raise ValueError('Expected original pass queries to be masked with zero weight')
    original_scale = float(np.float32(n/raw_weight_mass))
    raw = weights.astype(np.float64)/original_scale
    np.testing.assert_allclose(raw, mask.astype(np.float64)*far_factor, atol=1e-6, rtol=1e-6)
    new_labels, new_mask = labels.copy(), mask.copy()
    new_labels[pass_queries], new_mask[pass_queries] = 0., 1.
    raw[pass_queries] = WEAK_RAW_FACTOR*far_factor[pass_queries]
    raw_total = float(raw.sum(dtype=np.float64))
    normalized_scale = float(np.float32(n/raw_total))
    new_weights = (raw*normalized_scale).astype(np.float32)
    old_valid = ~pass_queries & (weights > 0)
    ratio = new_weights[old_valid].astype(np.float64)/weights[old_valid]
    receipt = dict(rows=n, pass_query_slots=int(pass_queries.sum()),
        original_raw_mass=float(raw_weight_mass), recovered_original_raw_mass=float((weights.astype(np.float64)/original_scale).sum()),
        original_normalized_mass=float(weights.sum(dtype=np.float64)),
        added_pass_raw_mass=float(raw[pass_queries].sum()), new_raw_mass=raw_total,
        target_normalized_mass=n, new_normalized_mass=float(new_weights.sum(dtype=np.float64)),
        weak_raw_factor=WEAK_RAW_FACTOR, original_normalization_scale=original_scale,
        new_normalization_scale=normalized_scale,
        normalized_pass_weights=np.unique(new_weights[pass_queries]).astype(float).tolist(),
        pass_normalized_mass=float(new_weights[pass_queries].sum(dtype=np.float64)),
        original_valid_normalized_mass_after=float(new_weights[old_valid].sum(dtype=np.float64)),
        original_valid_weight_ratio_min=float(ratio.min()) if len(ratio) else None,
        original_valid_weight_ratio_max=float(ratio.max()) if len(ratio) else None,
        nonpass_labels_unchanged=bool(np.array_equal(new_labels[~pass_queries], labels[~pass_queries])),
        nonpass_masks_unchanged=bool(np.array_equal(new_mask[~pass_queries], mask[~pass_queries])),
        interpretation='Pass supervision added at quarter inherited raw distance-weight; common new-array renormalization rescales original valid losses, old weights unchanged; not class balancing')
    return new_labels, new_mask, new_weights, receipt


def continuation_order(seed, offset, old_rows=39936, new_rows=19968):
    return T.slot_schedule(seed, 32+offset, old_rows, new_rows)[0]


def _common():
    return importlib.import_module('cnh_pass_common_dev')


def _config(path):
    path = Path(path)
    cfg = json.loads(path.read_text(encoding='utf-8-sig'))
    output = Path(cfg.get('output', OUT)).resolve()
    if output != OUT.resolve() or not output.is_relative_to((ROOT/'artifacts.local').resolve()):
        raise ValueError('Pass continuation outputs must stay in its new canonical run')
    cfg.update(output=str(output), plan=str(path.resolve()))
    cfg.setdefault('seeds', [2026100955, 2026100956, 2026100957])
    cfg.setdefault('epochs', cfg.get('epochs_equivalent', 12))
    cfg.setdefault('batch', cfg.get('batch_size', 256))
    cfg.setdefault('steps_per_job', cfg.get('steps', 2808))
    cfg.setdefault('lr', .001)
    cfg.setdefault('weight_decay', .0001)
    cfg.setdefault('feature_batch', 64)
    cfg.setdefault('train_inputs', dict(old=str(ROOT/'artifacts.local/work/cnh-temporal-readout-20261004/inputs/train'),
        ordinary=str(PREVIOUS/'data/ordinary_train')))
    cfg.setdefault('inference_inputs', dict(cal=str(PREVIOUS/'data/cal'), validation=str(PREVIOUS/'data/validation')))
    cfg.setdefault('inference_shapes', dict(cal=[384, 4, 13, 2], validation=[384, 4, 13, 2]))
    cfg.setdefault('checkpoint_dir', str(PREVIOUS/'models'))
    if (cfg['epochs'], cfg['batch'], cfg['steps_per_job'], cfg['lr'], cfg['weight_decay']) != (12, 256, 2808, .001, .0001):
        raise ValueError('Frozen pass continuation recipe differs')
    return cfg


def prepare_supervision(cfg):
    """CPU-only mapping from existing ordinary-training rows and authoring truth."""
    out = Path(cfg['output'])
    target = out/'supervision'
    if (target/'receipt.json').exists():
        prior = json.loads((target/'receipt.json').read_text(encoding='utf8'))
        if prior['plan_sha256'] != T.sha(cfg['plan']):
            raise ValueError('Existing pass-supervision PLAN binding differs')
        return prior
    if any((target/(k+'.npy')).exists() for k in ('labels', 'mask', 'weights')):
        raise FileExistsError('Partial new supervision preserved')
    data, paths = T._inputs(cfg['train_inputs']['ordinary'], training=True)
    folder = paths['labels'].parent
    with np.load(folder/'rows.npz', allow_pickle=False) as rows:
        scene, frame = rows['scene_id'], rows['frame']
    with np.load(folder/'physics.npz', allow_pickle=False) as physics:
        category, sensor = physics['frame_category'], physics['sensor']
    if scene.shape != (19968,) or frame.shape != scene.shape or not np.isin(frame, np.arange(3, 16)).all():
        raise ValueError('Ordinary training row binding differs')
    pass_queries = category[scene, frame-3] == 'pass'
    far = np.where((.65-sensor[frame, 2, 3] >= 1.6) & (.65-sensor[frame, 2, 3] < 2.6), 2., 1.)
    source_receipt = json.loads((folder/'receipt.json').read_text(encoding='utf8'))
    y, mask, w, summary = weak_supervision(data['labels'], data['mask'], data['weights'],
                                         pass_queries, far, source_receipt['raw_weight_mass'])
    target.mkdir(parents=True, exist_ok=True)
    for key, array in (('labels', y), ('mask', mask), ('weights', w)):
        np.save(target/(key+'.npy'), array)
    summary.update(plan_sha256=T.sha(cfg['plan']), source_sha256=T.sha(__file__),
        source_supervision_sha256={k: T.sha(paths[k]) for k in ('labels', 'mask', 'weights')},
        source_rows_sha256=T.sha(folder/'rows.npz'),
        source_ordinary_receipt_sha256=T.sha(folder/'receipt.json'),
        old_supervision='Not inferred or changed; inherited labels/mask/weights read only',
        features='No authoring category, pass flag or distance-weight enters cached model features')
    T.save(target/'receipt.json', summary)
    return summary


def prepare(cfg, budget):
    receipt = prepare_supervision(cfg)
    features = {}
    for split, spec in cfg['train_inputs'].items():
        features[split] = T.prepare_one(cfg, split, spec, budget)
    for split, spec in cfg['inference_inputs'].items():
        for yaw in (0, 3):
            features[f'{split}_yaw{yaw}'] = T.prepare_one(cfg, split, spec, budget, yaw=yaw)
    return dict(supervision=receipt, features=features)


def train(cfg, only_seed=None, only_arm=None):
    C = _common()
    out = Path(cfg['output'])
    old, _ = T._inputs(cfg['train_inputs']['old'], training=True)
    ordinary, _ = T._inputs(cfg['train_inputs']['ordinary'], training=True)
    if len(old['length']) != 39936 or len(ordinary['length']) != 19968:
        raise ValueError('Frozen old/ordinary identity differs')
    weak = {key: np.load(out/'supervision'/(key+'.npy'), allow_pickle=False) for key in ('labels', 'mask', 'weights')}
    load_stage = C.Stage('pass_train_cache_load')
    try:
        features, _ = T._load_shared_cache(cfg, ('old', 'ordinary'), load_stage)
        load_stage.finish('COMPLETE', bytes=features.numel()*features.element_size())
    except Exception as error:
        load_stage.finish('FAILED', error=repr(error))
        raise
    lengths = torch.as_tensor(np.concatenate((old['length'], ordinary['length'])), device='cuda')
    seeds = cfg['seeds'] if only_seed is None else [only_seed]
    arms = ARMS if only_arm is None else (only_arm,)
    if any(seed not in cfg['seeds'] for seed in seeds) or any(arm not in ARMS for arm in arms):
        raise ValueError('Unknown seed/arm')
    receipts = []
    model = optimizer = labels = weights = None
    try:
        for seed in seeds:
            checkpoint = Path(cfg['checkpoint_dir'])/f'ordinary_seed{seed}.pt'
            previous = torch.load(checkpoint, map_location='cpu', weights_only=True)
            if previous['seed'] != seed or previous['arm'] != 'ordinary' or previous['steps'] != 7488:
                raise ValueError('Continuation origin is not the ordinary final checkpoint')
            for arm in arms:
                name = f'{arm}_seed{seed}'
                target = out/'models'/(name+'.pt')
                receipt_path = target.with_suffix('.json')
                if target.exists() or receipt_path.exists():
                    if not (target.exists() and receipt_path.exists()):
                        raise FileExistsError('Incomplete pass continuation final checkpoint preserved')
                    prior = json.loads(receipt_path.read_text(encoding='utf8'))
                    if prior['steps'] != 2808 or prior['plan_sha256'] != T.sha(cfg['plan']):
                        raise ValueError('Completed continuation binding differs')
                    receipts.append(prior)
                    continue
                stage = C.Stage('pass_train_'+name)
                steps, trace = 0, []
                try:
                    torch.manual_seed(seed)
                    model = M.BoundaryTokenReadout().cuda()
                    model.load_state_dict(previous['state_dict'])
                    model.train()
                    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
                    source = ordinary if arm == 'control' else weak
                    labels = torch.as_tensor(np.concatenate((old['labels'], source['labels'])), device='cuda')
                    weights = torch.as_tensor(np.concatenate((old['weights'], source['weights'])), device='cuda')
                    started = time.monotonic()
                    for offset in range(12):
                        order = continuation_order(seed, offset)
                        bce_sum = mass_sum = 0.
                        for start in range(0, len(order), 256):
                            stage.check()
                            ix = torch.as_tensor(order[start:start+256], device='cuda')
                            logits = model(T.restore_features(features[ix]), lengths[ix])
                            bce = F.binary_cross_entropy_with_logits(logits, labels[ix].float(), reduction='none')
                            mass = weights[ix].sum()
                            weighted = (bce*weights[ix]).sum()
                            loss = weighted/mass
                            if not bool(mass > 0) or not bool(torch.isfinite(loss)):
                                raise ValueError('Invalid continuation weighted BCE')
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            optimizer.step()
                            steps += 1
                            bce_sum += float(weighted.detach())
                            mass_sum += float(mass)
                        row = dict(arm=arm, seed=seed, continuation_epoch=offset+1,
                            slot_schedule_epoch=32+offset, steps=steps, weighted_BCE=bce_sum/mass_sum,
                            supervised_mass=mass_sum, seconds=time.monotonic()-started)
                        trace.append(row)
                        T.save(out/'loss_curves'/(name+'.json'), trace)
                        T.save(out/'progress'/(name+'.json'), row)
                        print('PASS_TRAIN', name, offset+1, steps, round(row['weighted_BCE'], 6), flush=True)
                    if steps != 2808:
                        raise ValueError('Continuation final optimizer step mismatch')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(dict(state_dict=model.cpu().state_dict(), arm=arm, seed=seed,
                        steps=steps, epochs=12, parameters=M.parameter_count(),
                        origin_checkpoint_sha256=T.sha(checkpoint), plan_sha256=T.sha(cfg['plan'])), target)
                    receipt = dict(arm=arm, seed=seed, steps=steps, parameters=M.parameter_count(),
                        job_seconds=time.monotonic()-started, final_weighted_BCE=trace[-1]['weighted_BCE'],
                        origin_checkpoint_sha256=T.sha(checkpoint), checkpoint_sha256=T.sha(target),
                        plan_sha256=T.sha(cfg['plan']), source_sha256=T.sha(__file__),
                        optimizer='new AdamW lr0.001 wd0.0001; original optimizer state not resumed',
                        shared_order='Original slot_schedule(seed,32+offset,39936,19968)')
                    T.save(receipt_path, receipt)
                    stage.finish('COMPLETE', **receipt)
                    receipts.append(receipt)
                except Exception as error:
                    if model is not None:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(dict(state_dict=model.cpu().state_dict(), final=False, seed=seed,
                            arm=arm, steps=steps), target.with_name(name+f'_partial_{time.time_ns()}.pt'))
                    T.save(out/'loss_curves'/(name+f'_partial_{time.time_ns()}.json'), trace)
                    stage.finish('FAILED', arm=arm, seed=seed, steps=steps, error=repr(error))
                    raise
                finally:
                    model = optimizer = labels = weights = None
                    torch.cuda.empty_cache()
        T.save(out/'training_summary.json', dict(jobs=receipts))
        return dict(jobs=receipts)
    finally:
        model = optimizer = labels = weights = None
        del features, lengths
        torch.cuda.empty_cache()


def infer(cfg, budget):
    out = Path(cfg['output'])
    receipts = []
    features = lengths = model = None
    try:
        for split, spec in cfg['inference_inputs'].items():
            data, _ = T._inputs(spec)
            lengths = torch.as_tensor(np.array(data['length']), device='cuda')
            shape = tuple(cfg['inference_shapes'][split])
            if np.prod(shape[:-1]) != len(lengths) or shape[-2:] != (13, 2):
                raise ValueError('Complete continuation inference row binding differs')
            for yaw in (0, 3):
                cache_name = split+('_yaw_plus3' if yaw else '')
                features, _ = T._load_shared_cache(cfg, (cache_name,), budget)
                for seed in cfg['seeds']:
                    for arm in ARMS:
                        checkpoint = out/'models'/f'{arm}_seed{seed}.pt'
                        target = out/'scores'/f'{arm}_seed{seed}_{cache_name}.npz'
                        receipt_path = target.with_suffix('.json')
                        if target.exists() or receipt_path.exists():
                            if not (target.exists() and receipt_path.exists()):
                                raise FileExistsError('Incomplete continuation score binding preserved')
                            receipts.append(json.loads(receipt_path.read_text(encoding='utf8')))
                            continue
                        budget.check()
                        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
                        if state['steps'] != 2808 or state['plan_sha256'] != T.sha(cfg['plan']):
                            raise ValueError('Continuation inference checkpoint binding differs')
                        model = M.BoundaryTokenReadout().cuda().eval()
                        model.load_state_dict(state['state_dict'])
                        raw = np.empty((len(lengths), 2), np.float32)
                        with torch.inference_mode():
                            for start in range(0, len(lengths), 256):
                                budget.check()
                                ix = slice(start, min(start+256, len(lengths)))
                                raw[ix] = model(T.restore_features(features[ix]), lengths[ix]).cpu().numpy()
                        if not np.isfinite(raw).all():
                            raise ValueError('Nonfinite continuation raw scores')
                        branch = 'yaw+3' if yaw else 'ideal'
                        target.parent.mkdir(exist_ok=True)
                        np.savez_compressed(target, raw=raw.reshape(shape), arm=np.array(arm),
                            seed=np.array(seed), split=np.array(split), branch=np.array(branch))
                        receipt = dict(arm=arm, seed=seed, split=split, branch=branch,
                            path=str(target.resolve()), shape=list(shape), label_access=False,
                            checkpoint_sha256=T.sha(checkpoint), scores_sha256=T.sha(target))
                        T.save(receipt_path, receipt)
                        receipts.append(receipt)
                        T.save(out/'scores'/'manifest.json', dict(outputs=receipts,
                            arm_names=list(ARMS), seed_ids=cfg['seeds']))
                        print('PASS_INFER', arm, seed, split, yaw, flush=True)
                        model = None
                del features
                features = None
                torch.cuda.empty_cache()
            del lengths
            lengths = None
        T.save(out/'scores'/'manifest.json', dict(outputs=receipts, arm_names=list(ARMS), seed_ids=cfg['seeds']))
        return dict(outputs=receipts)
    finally:
        features = lengths = model = None
        torch.cuda.empty_cache()


def execute(stage_name, config, seed=None, arm=None):
    cfg = _config(config)
    if stage_name == 'supervision':
        return prepare_supervision(cfg)
    C = _common()
    T._configure()
    if stage_name == 'train':
        return train(cfg, seed, arm)
    stage = C.Stage('pass_'+stage_name)
    try:
        results = prepare(cfg, stage) if stage_name == 'prepare' else infer(cfg, stage)
        return stage.finish('COMPLETE', **results)
    except Exception as error:
        stage.finish('FAILED', error=repr(error))
        raise
    finally:
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('supervision', 'prepare', 'train', 'infer'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    result = execute(args.stage, args.config, args.seed, args.arm)
    print(json.dumps(result, ensure_ascii=False), flush=True)
