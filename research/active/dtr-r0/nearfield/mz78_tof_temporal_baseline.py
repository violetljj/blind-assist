"""Run MZ78 strong pure-ToF temporal Development comparison on frozen MZ77."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

from mz77_sequence_metrics import analyze
from tof_geometry_temporal_expert import predict

TASK = 'mz78-tof-temporal-baseline-20260912'
PROFILES = ('CLEAN', 'CENTER_GAP')
METHODS = ('GEOMETRY_FRAME', 'GEOMETRY_TEMPORAL', 'DIVERSE', 'TEMPORAL_OR_DIVERSE')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def aggregate(result: dict) -> dict:
    out = {}
    for condition in PROFILES:
        out[condition] = {}
        for method in METHODS:
            total = dict(TP=0, FP=0, FN=0, TN=0)
            for clip in result['clips'].values():
                metrics = clip['conditions'][condition]['methods'][method]['full_scene']
                for query in ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR'):
                    for key, value in metrics[query]['confusion'].items():
                        total[key] += value
            out[condition][method] = total
    return out


def run(root: Path, output: Path) -> None:
    root = root.resolve()
    output = output.resolve()
    allowed = (root / 'artifacts.local' / 'work' / TASK).resolve()
    if output.parent != allowed or output.exists():
        raise ValueError('output must be a new direct child of the MZ78 artifact root')
    output.mkdir(parents=True)
    started = time.perf_counter()
    inputs = {}

    def bind(path: Path, expected: str | None = None) -> Path:
        path = path.resolve()
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError(f'hash mismatch: {path}')
        inputs[str(path)] = digest
        return path

    here = Path(__file__).resolve().parent
    for name in ('mz78_tof_temporal_baseline.py', 'tof_geometry_temporal_expert.py',
                 'mz77_sequence_metrics.py'):
        shutil.copyfile(bind(here / name), output / name)
    prior = root / 'artifacts.local' / 'work' / 'mz77-approach-sequence-20260911'
    sensor_receipt_path = bind(prior / 'sensor-v1' / 'receipt.json')
    inference_receipt_path = bind(prior / 'inference-v1' / 'receipt.json')
    score_receipt_path = bind(prior / 'score-v1' / 'receipt.json')
    sensor_receipt = json.loads(sensor_receipt_path.read_text(encoding='utf-8'))
    inference_receipt = json.loads(inference_receipt_path.read_text(encoding='utf-8'))
    score_receipt = json.loads(score_receipt_path.read_text(encoding='utf-8'))
    observed_path = bind(prior / 'sensor-v1' / 'observations.npz', sensor_receipt['outputs']['observations.npz'])
    learned_path = bind(prior / 'inference-v1' / 'predictions.npz', inference_receipt['outputs']['predictions.npz'])
    evaluator_path = prior / 'score-v1' / 'evaluator.npz'
    if score_receipt['outputs']['evaluator.npz'] != sha(evaluator_path):
        raise ValueError('frozen evaluator hash mismatch')

    with np.load(observed_path, allow_pickle=False) as z:
        obs = dict(z)
    with np.load(learned_path, allow_pickle=False) as z:
        learned = dict(z)
    predictions = {}
    diagnostic = {}
    for profile in PROFILES:
        geometry = predict(obs[profile + '/ranges'], obs[profile + '/valid'], obs['clip'], obs['time_s'])
        diverse = np.asarray(learned[profile + '/DIVERSE'])
        predictions[profile] = {
            'GEOMETRY_FRAME': geometry['GEOMETRY_FRAME'],
            'GEOMETRY_TEMPORAL': geometry['GEOMETRY_TEMPORAL'],
            'DIVERSE': diverse,
            'TEMPORAL_OR_DIVERSE': np.maximum(geometry['GEOMETRY_TEMPORAL'], diverse),
        }
        diagnostic[profile] = dict(
            direct_valid_slots=int(np.asarray(obs[profile + '/valid']).sum()),
            imputed_slots=int(geometry['imputed'].sum()),
            imputed_frames=int(geometry['imputed'].any(axis=(1, 2)).sum()),
            all_invalid_frames=int((~np.asarray(obs[profile + '/valid']).any(axis=(1, 2))).sum()),
            partial_frames=int(np.asarray(obs[profile + '/valid']).any(axis=(1, 2)).sum()),
        )
    sealed = {p + '/' + m: v for p, methods in predictions.items() for m, v in methods.items()}
    np.savez_compressed(output / 'predictions.npz', clip=obs['clip'], index=obs['index'], time_s=obs['time_s'], **sealed)
    write(output / 'predictor-receipt.json', dict(
        status='SEALED_BEFORE_EVALUATOR_OPEN', methods=METHODS, profiles=PROFILES,
        configuration=dict(history_frames=5, max_gap_frames=3, max_closing_speed_mps=3.0,
                           geometry='8x8 ordered two-return zone-center projection',
                           alert_support='at least one projected return in fixed query box',
                           fusion='logical OR of GEOMETRY_TEMPORAL and frozen DIVERSE'),
        training_steps=0, fitted_parameters=0, threshold_searches=0,
        evaluator_reads=0, predictor_inputs=inputs, diagnostics=diagnostic,
        predictions_sha256=sha(output / 'predictions.npz')))

    # Evaluator access begins only after the label-free predictions are sealed.
    bind(evaluator_path, score_receipt['outputs']['evaluator.npz'])
    with np.load(evaluator_path, allow_pickle=False) as z:
        evaluator = dict(z)
    arrays = {**obs, **evaluator}
    arrays['packets'] = {p: {k: obs[p + '/' + k] for k in ('ranges', 'valid')} for p in PROFILES}
    result = analyze(arrays, predictions)
    summary = aggregate(result)
    write(output / 'result.json', result)
    write(output / 'summary.json', dict(aggregate_four_query_bits=summary, packet_diagnostics=diagnostic))
    for path, digest in inputs.items():
        if sha(Path(path)) != digest:
            raise ValueError(f'input changed during run: {path}')
    write(output / 'receipt.json', dict(
        status='PASS', phase='EXPLORE_CONSUMED_DEVELOPMENT', frames=160,
        profiles=PROFILES, methods=METHODS, training_steps=0, fitted_parameters=0,
        threshold_searches=0, input_hashes=inputs,
        outputs={p.name: sha(p) for p in output.iterdir() if p.is_file()},
        scoring_backend='CPU TASK_NOT_GPU_SUITABLE', seconds=time.perf_counter() - started,
        claim='Controlled simulated replay only; missing returns remain UNKNOWN and no clearance, hardware, natural-scene, latency, user-benefit or safety claim follows.'))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('E:/linnan/linnan'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
