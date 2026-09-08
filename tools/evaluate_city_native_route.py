"""Fixed G10/G13 RGB-only spatial route inference and UNKNOWN-aware evaluation.

Labels are opened only after predictions have been cached. Optional label JSON:
{schema: city-native-route-labels-v1, sample_indices: [...], near: near.npy,
 support: support.npy, provenance: {...}}. Array paths are relative to that JSON.
Near is Nx2 and support Nx2x18x32 or Nx2x360x640, all values -1/0/1.
No fitting, calibration, temporal prediction, or navigation decisions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'research/active/dtr-r0/nearfield'
sys.path.insert(0, str(SOURCE))
import numpy as np
from PIL import Image
import torch
from city_pilot_metrics import evaluate
from decoupled_model import DecoupledModel
from diversity_model import DiversityModel

ARTIFACTS = ROOT / 'artifacts.local'
NF = ARTIFACTS / 'nearfield'
SEEDS = (17, 29, 43)
METHODS = {
    'g10': ('diversity-20260907', 'expanded_region',
            'representation-20260908', 'original_gate'),
    'g13': ('decoupled-20260908', 'decoupled',
            'decoupled-20260908', 'decoupled'),
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def within(root, relative):
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Path escapes input root')
    return path


def pooled(support):
    if support.shape[-2:] == (360, 640):
        n = len(support)
        cells = support.reshape(n, 2, 18, 20, 32, 20)
        positive = (cells == 1).any(axis=(3, 5))
        unknown = (cells == -1).any(axis=(3, 5))
        return np.where(positive, 1, np.where(unknown, -1, 0)).astype(np.int8)
    return support


def labels(path, ids):
    n = len(ids)
    if path is None or not path.exists():
        return (np.full((n, 2), -1, np.int8),
                np.full((n, 2, 18, 32), -1, np.int8),
                dict(status='UNKNOWN', reason='No evaluator labels supplied'))
    data = read(path)
    if data.get('schema') != 'city-native-route-labels-v1' or data['sample_indices'] != ids:
        raise ValueError('Evaluator schema/order mismatch')
    if not data.get('provenance'):
        raise ValueError('Evaluator label provenance is required')
    paths = {key: within(path.parent, data[key]) for key in ('near', 'support')}
    near, support = [np.load(paths[key], allow_pickle=False) for key in ('near', 'support')]
    if near.shape != (n, 2) or support.shape not in ((n, 2, 18, 32), (n, 2, 360, 640)):
        raise ValueError('Label dimensions mismatch')
    if not np.isin(near, [-1, 0, 1]).all() or not np.isin(support, [-1, 0, 1]).all():
        raise ValueError('Labels must preserve -1 UNKNOWN, 0 negative, 1 positive')
    support = pooled(support)
    return near, support, dict(status='SUPPLIED', provenance=data['provenance'],
        manifest_sha256=sha(path), array_sha256={key: sha(p) for key, p in paths.items()},
        support_pooling='20x20 positive wins; otherwise any UNKNOWN wins')


def target_metrics(path, ids, predictions, config):
    """Target masks must already mean reliable visible in-query support.

    Only EVALUABLE/PASS/RELIABLE/KNOWN status plus nonempty authority permits
    scoring. Missing/occluded/unreliable target visibility never creates a FN.
    """
    if path is None or not path.exists():
        return []
    results = []
    for target in read(path).get('targets', []):
        result = {key: target.get(key) for key in ('target_id', 'category', 'authority', 'status')}
        if not target.get('authority') or target.get('status') not in ('EVALUABLE', 'PASS', 'RELIABLE', 'KNOWN'):
            result.update(evaluation='UNKNOWN', reason='Target not declared reliable/evaluable', methods={})
            results.append(result)
            continue
        indices = target.get('frame_indices', ids)
        if not indices or len(set(indices)) != len(indices) or any(index not in ids for index in indices):
            raise ValueError('Invalid target frame indices')
        mask_path = within(path.parent, target['masks'])
        masks = np.load(mask_path, allow_pickle=False)
        if masks.ndim != 4 or masks.shape[1:] not in ((2, 18, 32), (2, 360, 640)):
            raise ValueError('Invalid target masks shape')
        if not np.isin(masks, [-1, 0, 1]).all():
            raise ValueError('Target masks must be -1/0/1')
        positions = [ids.index(index) for index in indices]
        if len(masks) == len(ids):
            masks = masks[positions]
        elif len(masks) != len(indices):
            raise ValueError('Target masks length mismatch')
        masks = pooled(masks)
        result.update(mask_sha256=sha(mask_path), methods={})
        has_positive = (masks == 1).any()
        result['evaluation'] = 'EVALUABLE' if has_positive else 'UNKNOWN'
        for name in METHODS:
            result['methods'][name] = {}
            for h, head in enumerate(('BODY', 'HEAD')):
                probability = predictions[name + '_near'][positions, h]
                maps = predictions[name + '_support'][positions, h]
                positive = masks[:, h] == 1
                visible = positive.reshape(len(indices), -1).any(axis=1)
                alarm = probability >= config[name]['near_thresholds'][h]['value']
                overlap = ((maps >= .5) & positive).reshape(len(indices), -1).any(axis=1)
                peak = [int(masks[i, h].reshape(-1)[int(maps[i].argmax())]) for i in range(len(indices))]
                result['methods'][name][head] = dict(
                    reliable_positive_frames=int(visible.sum()),
                    unknown_or_no_positive_frames=int((~visible).sum()),
                    alert_TP=int((visible & alarm).sum()), alert_FN=int((visible & ~alarm).sum()),
                    support_overlap_hits=int((visible & overlap).sum()),
                    support_overlap_misses=int((visible & ~overlap).sum()),
                    joint_alert_and_overlap_hits=int((visible & alarm & overlap).sum()),
                    peak_hits=sum(bool(v) and p == 1 for v, p in zip(visible, peak)),
                    peak_unknown_misses=sum(bool(v) and p == -1 for v, p in zip(visible, peak)),
                    false_alerts=None, false_alert_reason='Target mask does not label whole-query negatives')
        results.append(result)
    return results


def configuration():
    result = {}
    for name, (training, arm, evaluation, key) in METHODS.items():
        train = NF / training / 'main/learned'
        receipt = train / 'receipt.json'
        rows = read(receipt)['records']
        weights = []
        for seed in SEEDS:
            path = train / f'{arm}_seed{seed}.pt'
            expected = next(r['checkpoint_sha256'] for r in rows if r['arm'] == arm and r['seed'] == seed)
            if sha(path) != expected:
                raise ValueError('Frozen checkpoint hash mismatch')
            weights.append(dict(path=str(path), sha256=expected, seed=seed))
        threshold_path = NF / evaluation / 'main/evaluation/result.json'
        historical = read(threshold_path)
        thresholds = historical['primary']['evaluations'][key + '/ensemble/normal']['all_TEST']['thresholds'][:2]
        result[name] = dict(weights=weights, receipt_sha256=sha(receipt),
            near_thresholds=thresholds, support_threshold=.5,
            threshold_source=str(threshold_path), threshold_source_sha256=sha(threshold_path),
            threshold_policy=historical['threshold_policy'], ensemble='Arithmetic mean of per-seed sigmoid probabilities')
    return result


@torch.inference_mode()
def infer(paths, out, config):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; GPU required for this inference')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    arrays = []
    for path in paths:
        with Image.open(path) as image:
            if image.size != (640, 360):
                raise ValueError('Expected captured 640x360 RGB')
            arrays.append(np.array(image.convert('RGB').resize((256, 144), Image.Resampling.BOX)))
    # Match diversity_train.cache_inputs: uint8 transfer precedes conversion.
    # CPU versus CUDA division differs slightly and breaks strict replay parity.
    rgb = torch.from_numpy(np.stack(arrays)).cuda().permute(0, 3, 1, 2).float().div_(255.)
    predictions = {}
    timing = {}
    for name in METHODS:
        tick = time.perf_counter()
        for record in config[name]['weights']:
            model = (DiversityModel(region=True) if name == 'g10' else
                     DecoupledModel(NF / 'representation-20260908/pretrained'))
            model.load_state_dict(torch.load(record['path'], map_location='cpu', weights_only=True), strict=True)
            model = model.cuda().eval()
            near, support = [], []
            for batch in rgb.split(16):
                logits, masks = model(batch.cuda())
                near.append(logits.sigmoid().cpu().numpy())
                support.append(masks.sigmoid().cpu().numpy())
            predictions[f'{name}_seed{record["seed"]}_near'] = np.concatenate(near)
            predictions[f'{name}_seed{record["seed"]}_support'] = np.concatenate(support)
            del model
        # Historical evaluation averages Python floats (double precision).
        for kind in ('near', 'support'):
            predictions[f'{name}_{kind}'] = np.mean(
                [predictions[f'{name}_seed{seed}_{kind}'].astype(np.float64) for seed in SEEDS], axis=0)
        torch.cuda.synchronize()
        timing[name] = time.perf_counter() - tick
    np.savez_compressed(out / 'predictions.npz', **predictions)
    return predictions, dict(device=torch.cuda.get_device_name(), backend='cuda',
        per_method_load_and_inference_seconds=timing, frames=len(paths), optimizer_steps=0)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--predictions', type=Path, help='Reuse earlier output directory; no inference')
    args = parser.parse_args()
    if args.labels is not None and not args.labels.is_file():
        parser.error('Explicit --labels file does not exist')
    out = args.output.resolve()
    if out.exists() or out == ARTIFACTS.resolve() or not out.is_relative_to(ARTIFACTS.resolve()):
        parser.error('Use a fresh directory inside canonical artifacts.local')
    capture = args.capture.resolve(strict=True)
    model_root = capture / 'model'
    frames = read(model_root / 'dataset.json')['frames']
    ids = [int(f['sample_index']) for f in frames]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Empty or duplicate sample indices')
    paths = [within(model_root, f['rgb_path']) for f in frames]
    if any(p.suffix.lower() != '.png' for p in paths):
        raise ValueError('Only PNG model input supported')
    config = configuration()
    contract = dict(schema='city-native-route-inference-v1', sample_indices=ids,
        rgb_sha256={str(i): sha(p) for i, p in zip(ids, paths)}, methods=config,
        source_sha256={p.name: sha(p) for p in [Path(__file__)] +
            [SOURCE / n for n in ('diversity_model.py', 'whisker_model.py', 'decoupled_model.py',
                                  'representation_model.py', 'city_pilot_metrics.py')]},
        input='Current RGB only; BOX resize 256x144; float [0,1]; G13 internal normalization',
        scope='Ordered static-world spatial Development; approaching UNKNOWN; no route safety claim')
    out.mkdir(parents=True)
    write(out / 'protocol.json', contract)
    try:
        if args.predictions:
            prior = args.predictions.resolve(strict=True)
            if read(prior / 'protocol.json') != contract:
                raise ValueError('Cached prediction input/method/source contract mismatch')
            if read(prior / 'receipt.json')['predictions_sha256'] != sha(prior / 'predictions.npz'):
                raise ValueError('Prediction cache hash mismatch')
            with np.load(prior / 'predictions.npz', allow_pickle=False) as archive:
                predictions = {key: archive[key] for key in archive.files}
            np.savez_compressed(out / 'predictions.npz', **predictions)
            timing = dict(backend='CPU_CACHED', reason='TASK_NOT_GPU_SUITABLE_CACHED_METRICS', source=str(prior))
        else:
            predictions, timing = infer(paths, out, config)
        # All label and evaluator access begins after immutable RGB predictions.
        label_path = args.labels or capture / 'evaluator/native-route-labels.json'
        near, support, label_info = labels(label_path, ids)
        results, rows = {}, []
        for name in METHODS:
            prob = predictions[name + '_near']
            if prob.shape != (len(ids), 2) or not np.isfinite(prob).all() or not ((prob >= 0) & (prob <= 1)).all():
                raise ValueError('Invalid near probabilities in prediction cache')
            thresholds = np.array([r['value'] for r in config[name]['near_thresholds']])
            decision = prob >= thresholds
            results[name] = evaluate(decision.astype(float), predictions[name + '_support'], near, support)
            results[name]['thresholds'] = dict(near=thresholds.tolist(), support=.5, comparison='inclusive >=')
            for i, sid in enumerate(ids):
                rows.append(dict(method=name, sample_index=sid, near_probability=prob[i].tolist(),
                    alerts=decision[i].tolist(), truth=near[i].tolist(), approaching='UNKNOWN'))
        write(out / 'result.json', dict(schema='city-native-route-evaluation-v1', methods=results,
            labels=label_info, targets=target_metrics(label_path, ids, predictions, config),
            inference=timing, scope=contract['scope']))
        write(out / 'per-frame.json', rows)
        write(out / 'receipt.json', dict(status='PASS', predictions_sha256=sha(out / 'predictions.npz'),
            result_sha256=sha(out / 'result.json'), protocol_sha256=sha(out / 'protocol.json')))
        print(json.dumps(dict(status='PASS', frames=len(ids), output=str(out))))
    except Exception as error:
        write(out / 'failure.json', dict(status='FAIL', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
