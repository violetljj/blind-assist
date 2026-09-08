"""Zero-training replacement of step200 gates by cached same-image baseline gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F

from city_data import CityRGBDataset, _array
from city_pilot_metrics import evaluate
from decoupled_model import DecoupledModel
from whisker_model import SUPPORT_SIZE

EXPECTED = dict(baseline='0c14179486102993508dd2385a3b6f9e17d51a6d53e657c0240325e5f78d5c7b',
    finetuned='65a25a30ea0753e9692ad1766b3df72efa192bb0e0ad35b92328fbfcfd36d913')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def gated_near(model, deep, gate):
    if gate.shape != (deep.shape[0], 2, *deep.shape[-2:]):
        raise ValueError('Same-image two-head probability gate required')
    gated = deep[:, None] * gate[:, :, None]
    b, heads, channels, h, w = gated.shape
    pooled = model.near[0](gated.reshape(b * heads, channels, h, w)).reshape(b, heads, -1)
    return (pooled * model.near[2].weight[None]).sum(-1) + model.near[2].bias


@torch.inference_mode()
def run(a):
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; do not contend with another GPU job')
    out = a.output.resolve()
    artifacts = (Path(__file__).resolve().parents[4] / 'artifacts.local').resolve()
    if not out.is_relative_to(artifacts) or out == artifacts or out.exists():
        raise ValueError('Fresh canonical artifact output required')
    inputs = {}
    def checked(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f'Input hash mismatch: {path}')
        inputs[str(path)] = digest
        return path
    for arm in EXPECTED:
        checked(getattr(a, arm), EXPECTED[arm])
    out.mkdir(parents=True)
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        cm = read(checked(a.cache / 'manifest.json'))
        pilot = read(checked(a.test_predictions / 'protocol.json'))
        checked(Path(__file__).with_name('CITY_GATE_INTERVENTION_20260908.md'))
        for name in ('city_data.py', 'city_pilot_metrics.py', 'decoupled_model.py', 'representation_model.py'):
            checked(Path(__file__).with_name(name), pilot['source_sha256'][name])
        thin = {r['file']: r['sha256'] for r in read(checked(a.test_predictions / 'thin-manifest.json'))}
        checked(a.cache / 'manifest.json', pilot['cache_manifest_sha256'])
        if pilot['checkpoint_sha256'] != EXPECTED['baseline']:
            raise ValueError('Original pilot identity mismatch')
        train_receipt = read(checked(a.train_predictions / 'receipt.json'))
        if set(train_receipt['weights'].values()) != set(EXPECTED.values()):
            raise ValueError('TRAIN prediction checkpoint mismatch')
        original_result = read(checked(a.test_predictions / 'result.json'))
        checked(a.test_predictions / 'result.json', read(a.test_predictions / 'receipt.json')['result_sha256'])
        model = DecoupledModel(a.pretrained).cuda().eval()
        model.load_state_dict(torch.load(a.finetuned, map_location='cpu', weights_only=True), strict=True)
        for p in model.parameters():
            p.requires_grad_(False)
        result = {}
        for split in ('train', 'test'):
            dataset = CityRGBDataset(a.cache, split)
            labels_path = a.cache / ('supervision/train.json' if split == 'train' else 'evaluator/test.json')
            labels = read(checked(labels_path))
            if labels['sample_indices'] != dataset.ids:
                raise ValueError('Label and RGB sample order mismatch')
            y = np.array(_array(a.cache.resolve(), labels['near']), copy=True)
            gt = np.array(_array(a.cache.resolve(), labels['support']), copy=True)
            for entry in (cm['partitions'][split]['rgb'], labels['near'], labels['support']):
                checked(a.cache / entry['path'], entry['sha256'])
            pred = {}
            for arm in ('baseline', 'city_finetuned'):
                directory = a.train_predictions if split == 'train' else a.test_predictions
                filename = f'{arm}-train-predictions.npz' if split == 'train' else f'{arm}-city-predictions.npz'
                with np.load(checked(directory / filename, thin[filename] if split == 'test' else None), allow_pickle=False) as archive:
                    pred[arm] = {k: archive[k].copy() for k in ('near', 'support')}
                    if split == 'train' and archive['sample_indices'].tolist() != dataset.ids:
                        raise ValueError('TRAIN prediction identity/order mismatch')
                if pred[arm]['near'].shape != y.shape or pred[arm]['support'].shape != gt.shape:
                    raise ValueError('Cached prediction shape mismatch')
                if any(not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all() for p in pred[arm].values()):
                    raise ValueError('Invalid cached probabilities')
                metrics = evaluate(pred[arm]['near'], pred[arm]['support'], y, gt)
                expected = train_receipt['results'][arm]['ALL'] if split == 'train' else original_result['results'][arm]['city']
                if any(metrics['heads'][h]['near'][k] != expected['heads'][h]['near'][k]
                       for h in ('BODY', 'HEAD') for k in ('TP', 'FP', 'FN', 'TN')):
                    raise ValueError('Cached confusion differs from sealed receipt')
            normal, hybrid = [], []
            self_error = cached_near_error = cached_support_error = zero_gate_error = 0.
            for start in range(0, len(dataset), 32):
                stop = min(start + 32, len(dataset))
                rgb = torch.stack([dataset[i]['rgb'] for i in range(start, stop)]).cuda()
                deep, shallow = model.extract((rgb - model.image_mean) / model.image_std)
                deep = F.interpolate(model.deep_projection(deep), size=SUPPORT_SIZE, mode='bilinear', align_corners=False)
                own_gate = model.support(deep + model.detail(shallow)).sigmoid()
                own_logits = gated_near(model, deep, own_gate)
                public_logits, public_support = model(rgb)
                self_error = max(self_error, float((own_logits - public_logits).abs().max()))
                zero_gate_error = max(zero_gate_error, float((gated_near(model, deep, torch.zeros_like(own_gate)) - model.near[2].bias).abs().max()))
                n = own_logits.sigmoid().cpu().numpy()
                cached_near_error = max(cached_near_error, float(np.max(np.abs(n - pred['city_finetuned']['near'][start:stop]))))
                cached_support_error = max(cached_support_error, float(np.max(np.abs(public_support.sigmoid().cpu().numpy() - pred['city_finetuned']['support'][start:stop]))))
                baseline_gate = torch.from_numpy(pred['baseline']['support'][start:stop]).cuda()
                hybrid.append(gated_near(model, deep, baseline_gate).cpu().numpy())
                normal.append(n)
            normal = np.concatenate(normal)
            hybrid_logits = np.concatenate(hybrid)
            hybrid_probs = torch.from_numpy(hybrid_logits).sigmoid().numpy()
            parity = dict(self_gate_logit_max_abs=self_error, zero_gate_equals_bias_max_abs=zero_gate_error,
                cached_near_max_abs=cached_near_error, cached_support_max_abs=cached_support_error,
                cached_decision_flips=int(np.sum((normal >= .5) != (pred['city_finetuned']['near'] >= .5))))
            if max(self_error, cached_near_error, cached_support_error) > 1e-5 or zero_gate_error != 0 or parity['cached_decision_flips']:
                raise ValueError(f'Normal/self-gate parity failed: {parity}')
            np.savez_compressed(out / f'{split}-baseline-gate-predictions.npz', near=hybrid_probs,
                near_logits=hybrid_logits, sample_indices=np.array(dataset.ids))
            arms = {arm: evaluate(p['near'], p['support'], y, gt) for arm, p in pred.items()}
            arms['step200_baseline_gate'] = evaluate(hybrid_probs, pred['baseline']['support'], y, gt)
            result[split] = dict(parity=parity, arms=arms,
                gate_mean={arm: pred[arm]['support'].mean(axis=(0, 2, 3)).tolist() for arm in pred},
                hybrid_note='Support metrics describe the supplied original gate, not a repaired step200 support predictor')
        for path, digest in inputs.items():
            if sha(path) != digest:
                raise ValueError('Input changed during intervention')
        write(out / 'result.json', dict(status='PASS', threshold=.5, optimizer_steps=0,
            intervention='Only same-image sigmoid support gate replaced; step200 deep projection/backbone/near head fixed',
            results=result, scope='Consumed diagnostic hybrid; not a deployable one-model candidate or localization repair'))
        write(out / 'receipt.json', dict(status='PASS', optimizer_steps=0, backend='CUDA',
            device=torch.cuda.get_device_name(), seconds=time.perf_counter()-started,
            source_sha256=sha(__file__), input_sha256=inputs, result_sha256=sha(out/'result.json')))
    except BaseException as exc:
        write(out / 'failure.json', dict(status='FAIL', optimizer_steps=0, error=str(exc)))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'baseline', 'finetuned', 'pretrained', 'train-predictions', 'test-predictions', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    run(parser.parse_args())
