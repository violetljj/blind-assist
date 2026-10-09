"""Matched fixed-decoder control for one specific three-scalar bin summary.

The summary is first local peak position/amplitude plus the signed bin total.
Its decoder is a fixed encoding adapter, not physical return reconstruction.
This independent branch never changes main sources, arrays, feature caches or
metrics. An inferior result cannot by itself establish a bin-preserving paper
mechanism: it only evaluates this summary/decoder with the matched readout.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cnh_boundary_token_model as M
import cnh_counterfactual_common_dev as C
import cnh_counterfactual_train_dev as T


PEAK_THRESHOLD = 3.
ARM = 'bin_summary'


def decode_summary(histories, lengths):
    """Keep first peak and signed total, then fixed-decode sixteen radial bins.

    Input is native normalized_z [B,time8,zone_y8,zone_x8,bin16], before log.
    First peak: z>=3, z>left, z>=right, missing neighbors treated as -inf.
    This selects the leftmost member of a flat local maximum. With a peak,
    restore its amplitude at its bin and uniformly divide total-amplitude over
    the other fifteen bins. Without a peak, divide total over all sixteen bins.
    Padding is zero before peak/total extraction, including NaN left padding.
    """
    if histories.ndim != 5 or histories.shape[1:] != (8, 8, 8, 16):
        raise ValueError('Expected native histories [B,8,8,8,16]')
    lengths = torch.as_tensor(lengths, device=histories.device)
    if lengths.shape != (len(histories),) or bool(((lengths < 1) | (lengths > 8)
                                                | (lengths != lengths.round())).any()):
        raise ValueError('Expected integral lengths [B] in 1..8')
    valid = torch.arange(8, device=histories.device)[None] >= 8-lengths[:, None]
    z = torch.where(valid[:, :, None, None, None], histories.float(), 0.)
    missing = z.new_full((*z.shape[:-1], 1), -torch.inf)
    left = torch.cat((missing, z[..., :-1]), -1)
    right = torch.cat((z[..., 1:], missing), -1)
    peaks = (z >= PEAK_THRESHOLD) & (z > left) & (z >= right)
    found = peaks.any(-1)
    position = peaks.to(torch.int64).argmax(-1)
    amplitude = z.gather(-1, position[..., None]).squeeze(-1)
    total = z.sum(-1)
    residual = (total-amplitude)/15
    decoded = torch.where(found[..., None], residual[..., None], total[..., None]/16).expand_as(z).clone()
    replacement = torch.where(found, amplitude, total/16)
    decoded.scatter_(-1, position[..., None], replacement[..., None])
    return decoded


def _branch(out):
    return Path(out)/'bin_summary'


def _freeze_control(cfg):
    folder = _branch(cfg['output'])
    path = folder/'CONTROL_PLAN.json'
    expected = dict(control='first-local-peak three-scalar fixed-decoder matching control',
        arm=ARM, threshold_normalized_z=PEAK_THRESHOLD,
        peak_rule='first z>=3 and z>left and z>=right; endpoint neighbors -inf; leftmost plateau',
        decoder='peak amplitude at firstpeak bin, (signedtotal-amplitude)/15 elsewhere; missing peak signedtotal/16',
        input='native normalized_z before signedlog; valid history only',
        model='unchanged BoundaryTokenReadout 8833; center geometry; channels6:12 zero',
        paired_reference='main base checkpoint of same seed, full bin histories',
        seeds=cfg['seeds'], epochs=cfg['epochs'], batch=cfg['batch'], steps=cfg['steps_per_job'],
        lr=cfg['lr'], weight_decay=cfg['weight_decay'],
        train_rule='exact main base slot schedule; only inherited old supervision; final checkpoint',
        limits='Specific summary/decoder coding control, not physical reconstruction or isolated proof of native-bin mechanism',
        main_plan_sha256=T.sha(cfg['plan']), source_sha256=T.sha(__file__),
        model_source_sha256=T.sha(M.__file__), main_train_source_sha256=T.sha(T.__file__))
    if path.exists():
        if json.loads(path.read_text(encoding='utf8')) != expected:
            raise ValueError('Frozen bin-summary source/control binding changed')
    else:
        T.save(path, expected)
    return path


def _cache_path(out, split, yaw=0):
    return T._cache_name(Path(out), split+'_summary9', yaw)


def prepare_one(cfg, split, spec, budget, *, yaw=0):
    arrays, sources = T._inputs(spec)
    provided_yaw = False
    if yaw == 3:
        yaw_path = sources['transforms'].with_name('transforms_yaw3.npy')
        if yaw_path.is_file():
            arrays['transforms'] = np.load(yaw_path, mmap_mode='r', allow_pickle=False)
            sources['transforms'] = yaw_path
            provided_yaw = True
    n = len(arrays['length'])
    path = _cache_path(cfg['output'], split, yaw)
    receipt_path = path.with_suffix('.json')
    signatures = {name: dict(path=str(p.resolve()), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
                  for name, p in sources.items()}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf8'))
        if receipt['sources'] != signatures or receipt['rows'] != n:
            raise ValueError('Summary cache data binding changed')
        return receipt
    if path.exists():
        raise FileExistsError('Partial summary cache preserved')
    path.parent.mkdir(parents=True, exist_ok=True)
    maps = np.lib.format.open_memmap(path, mode='w+', dtype=np.float16, shape=(n, 2, 16, 9, 8, 8))
    angle = math.radians(yaw)
    c, s = math.cos(angle), math.sin(angle)
    rotation = torch.eye(4, dtype=torch.float32, device='cuda')
    rotation[:3, :3] = rotation.new_tensor([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, n, cfg['feature_batch']):
            budget.check()
            ix = slice(start, min(start+cfg['feature_batch'], n))
            z = torch.as_tensor(np.array(arrays['histories'][ix]), device='cuda')
            lengths = torch.as_tensor(np.array(arrays['length'][ix]), device='cuda')
            transforms = torch.as_tensor(np.array(arrays['transforms'][ix]), device='cuda', dtype=torch.float32)
            if yaw and not provided_yaw:
                transforms = rotation[None, None]@transforms
            decoded = decode_summary(z, lengths)
            geometry = M.feature_geometry(transforms, lengths, 'center')
            compact = M.build_features(decoded, geometry, lengths)[:, :, :, list(T.KEPT_CHANNELS)].half()
            if not bool(torch.isfinite(compact).all()):
                raise ValueError('Nonfinite summary compact features')
            maps[ix] = compact.cpu().numpy()
            if start % (cfg['feature_batch']*100) == 0:
                print('SUMMARY_FEATURES', split, yaw, start, '/', n, flush=True)
    maps.flush()
    del maps
    receipt = dict(rows=n, sources=signatures, split=split, yaw_degrees=yaw,
        seconds=time.monotonic()-started, shape=[n, 2, 16, 9, 8, 8],
        decoder='first peak threshold3 + signedtotal; fixed fifteen-bin residual', source_sha256=T.sha(__file__))
    T.save(receipt_path, receipt)
    return receipt


def prepare(cfg, budget):
    results = {'old': prepare_one(cfg, 'old', cfg['train_inputs']['old'], budget)}
    for split, spec in cfg['inference_inputs'].items():
        for yaw in (0, 3):
            results[f'{split}_yaw{yaw}'] = prepare_one(cfg, split, spec, budget, yaw=yaw)
    return dict(features=results)


def train(cfg, only_seed=None):
    out, folder = Path(cfg['output']), _branch(cfg['output'])
    data, _ = T._inputs(cfg['train_inputs']['old'], training=True)
    n_old, n_new = len(data['length']), cfg.get('new_rows', 19968)
    expected = cfg['epochs']*math.ceil((n_old+n_new)/cfg['batch'])
    if n_old != 39936 or expected != cfg['steps_per_job']:
        raise ValueError('Matched summary schedule differs from main base')
    load_stage = C.Stage('summary_cache_load')
    try:
        features, _ = T._load_shared_cache(cfg, ('old_summary9',), load_stage)
        load_stage.finish('COMPLETE', bytes=features.numel()*features.element_size())
    except Exception as error:
        load_stage.finish('FAILED', error=repr(error))
        raise
    labels = torch.as_tensor(np.array(data['labels']), device='cuda')
    weights = torch.as_tensor(np.array(data['weights']), device='cuda')
    lengths = torch.as_tensor(np.array(data['length']), device='cuda')
    seeds = cfg['seeds'] if only_seed is None else [only_seed]
    if any(seed not in cfg['seeds'] for seed in seeds):
        raise ValueError('Unknown summary seed')
    receipts = []
    for seed in seeds:
        target = folder/'models'/f'bin_summary_seed{seed}.pt'
        receipt_path = target.with_suffix('.json')
        if target.exists() or receipt_path.exists():
            if not (target.exists() and receipt_path.exists()):
                raise FileExistsError('Incomplete summary final checkpoint preserved')
            receipt = json.loads(receipt_path.read_text(encoding='utf8'))
            if receipt['steps'] != expected or receipt['main_plan_sha256'] != T.sha(cfg['plan']):
                raise ValueError('Summary completed job binding differs')
            receipts.append(receipt)
            continue
        stage = C.Stage(f'train_summary_seed{seed}')
        model = optimizer = None
        steps, trace = 0, []
        try:
            torch.manual_seed(seed)
            initial = {k: v.clone() for k, v in M.BoundaryTokenReadout().state_dict().items()}
            model = M.BoundaryTokenReadout().cuda()
            model.load_state_dict(initial)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
            started = time.monotonic()
            for epoch in range(cfg['epochs']):
                _, bound = T.slot_schedule(seed, epoch, n_old, n_new)
                bce_sum = mass_sum = 0.
                for start in range(0, len(bound), cfg['batch']):
                    stage.check()
                    ix = torch.as_tensor(bound[start:start+cfg['batch']], device='cuda')
                    logits = model(T.restore_features(features[ix]), lengths[ix])
                    bce = F.binary_cross_entropy_with_logits(logits, labels[ix].float(), reduction='none')
                    mass = weights[ix].sum()
                    weighted = (bce*weights[ix]).sum()
                    loss = weighted/mass
                    if not bool(mass > 0) or not bool(torch.isfinite(loss)):
                        raise ValueError('Invalid summary weighted BCE')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    steps += 1
                    bce_sum += float(weighted.detach())
                    mass_sum += float(mass)
                row = dict(seed=seed, arm=ARM, epoch=epoch+1, steps=steps,
                    weighted_BCE=bce_sum/mass_sum, supervised_mass=mass_sum, seconds=time.monotonic()-started)
                trace.append(row)
                T.save(folder/'loss_curves'/f'seed{seed}.json', trace)
                T.save(folder/'progress'/f'seed{seed}.json', row)
                print('SUMMARY_TRAIN', seed, epoch+1, steps, round(row['weighted_BCE'], 6), flush=True)
            if steps != expected:
                raise ValueError('Summary final optimizer step mismatch')
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(state_dict=model.cpu().state_dict(), arm=ARM, seed=seed, steps=steps,
                epochs=cfg['epochs'], parameters=M.parameter_count(), main_plan_sha256=T.sha(cfg['plan'])), target)
            receipt = dict(arm=ARM, seed=seed, steps=steps, parameters=M.parameter_count(),
                final_weighted_BCE=trace[-1]['weighted_BCE'], job_seconds=time.monotonic()-started,
                checkpoint_sha256=T.sha(target), main_plan_sha256=T.sha(cfg['plan']),
                control_plan_sha256=T.sha(folder/'CONTROL_PLAN.json'))
            T.save(receipt_path, receipt)
            stage.finish('COMPLETE', **receipt)
            receipts.append(receipt)
        except Exception as error:
            if model is not None:
                target.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(state_dict=model.cpu().state_dict(), seed=seed, steps=steps,
                    final=False), target.with_name(f'bin_summary_seed{seed}_partial_{time.time_ns()}.pt'))
                T.save(folder/'loss_curves'/f'seed{seed}_partial_{time.time_ns()}.json', trace)
            stage.finish('FAILED', seed=seed, steps=steps, error=repr(error))
            raise
        finally:
            model = optimizer = None
            torch.cuda.empty_cache()
    del features, labels, weights, lengths
    T.save(folder/'training_summary.json', dict(jobs=receipts))
    return dict(jobs=receipts)


def infer(cfg, budget):
    out, folder = Path(cfg['output']), _branch(cfg['output'])
    receipts = []
    for split, spec in cfg['inference_inputs'].items():
        data, _ = T._inputs(spec)
        lengths = torch.as_tensor(np.array(data['length']), device='cuda')
        shape = tuple(cfg['inference_shapes'][split])
        if np.prod(shape[:-1]) != len(lengths) or shape[-2:] != (13, 2):
            raise ValueError('Summary inference shape binding differs')
        for yaw in (0, 3):
            cache_name = split+'_summary9'+('_yaw_plus3' if yaw else '')
            features, _ = T._load_shared_cache(cfg, (cache_name,), budget)
            for seed in cfg['seeds']:
                checkpoint = folder/'models'/f'bin_summary_seed{seed}.pt'
                suffix = '_yaw_plus3' if yaw else ''
                target = out/'scores'/f'bin_summary_seed{seed}_{split}{suffix}.npz'
                receipt_path = target.with_suffix('.json')
                if target.exists() or receipt_path.exists():
                    if not (target.exists() and receipt_path.exists()):
                        raise FileExistsError('Incomplete summary scores preserved')
                    receipts.append(json.loads(receipt_path.read_text(encoding='utf8')))
                    continue
                budget.check()
                started = time.monotonic()
                state = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if state['steps'] != cfg['steps_per_job'] or state['main_plan_sha256'] != T.sha(cfg['plan']):
                    raise ValueError('Summary inference checkpoint binding differs')
                model = M.BoundaryTokenReadout().cuda().eval()
                model.load_state_dict(state['state_dict'])
                raw = np.empty((len(lengths), 2), np.float32)
                with torch.inference_mode():
                    for start in range(0, len(lengths), cfg['batch']):
                        budget.check()
                        ix = slice(start, min(start+cfg['batch'], len(lengths)))
                        raw[ix] = model(T.restore_features(features[ix]), lengths[ix]).cpu().numpy()
                if not np.isfinite(raw).all():
                    raise ValueError('Nonfinite summary logits')
                branch = 'yaw+3' if yaw else 'ideal'
                target.parent.mkdir(exist_ok=True)
                np.savez_compressed(target, raw=raw.reshape(shape), arm=np.array(ARM), seed=np.array(seed),
                    split=np.array(split), branch=np.array(branch))
                receipt = dict(arm=ARM, seed=seed, split=split, branch=branch, yaw_degrees=yaw,
                    path=str(target.resolve()), shape=list(shape), label_access=False,
                    seconds=time.monotonic()-started, checkpoint_sha256=T.sha(checkpoint), scores_sha256=T.sha(target))
                T.save(receipt_path, receipt)
                receipts.append(receipt)
                T.save(folder/'score_manifest.json', dict(outputs=receipts, arm_names=[ARM], seed_ids=cfg['seeds']))
                print('SUMMARY_INFER', seed, split, yaw, len(lengths), flush=True)
                del model, state
            del features
            torch.cuda.empty_cache()
        del lengths
    T.save(folder/'score_manifest.json', dict(outputs=receipts, arm_names=[ARM], seed_ids=cfg['seeds']))
    return dict(outputs=receipts)


def execute(stage_name, config, seed=None):
    cfg = T._config(config)
    T._configure()
    _freeze_control(cfg)
    if stage_name == 'train':
        return train(cfg, seed)
    stage = C.Stage('bin_summary_'+stage_name)
    try:
        details = prepare(cfg, stage) if stage_name == 'prepare' else infer(cfg, stage)
        return stage.finish('COMPLETE', **details)
    except Exception as error:
        stage.finish('FAILED', error=repr(error))
        raise
    finally:
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'train', 'infer'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    execute(args.stage, args.config, args.seed)
