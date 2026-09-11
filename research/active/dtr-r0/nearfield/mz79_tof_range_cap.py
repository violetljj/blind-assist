"""MZ79 DS14161 range-cap sensitivity on the consumed MZ77 sequence."""
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
from tof_range_cap_sensitivity import (
    DATASHEET_REVISION, DATASHEET_TABLE, DATASHEET_URL, PROFILES, apply_range_cap,
)

TASK = 'mz79-tof-range-cap-20260912'
METHODS = ('GEOMETRY_TEMPORAL', 'RGB', 'TEMPORAL_OR_RGB')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def aggregate(result: dict) -> dict:
    summary = {}
    for profile in PROFILES:
        summary[profile] = {}
        for method in METHODS:
            total = dict(TP=0, FP=0, FN=0, TN=0)
            for clip in result['clips'].values():
                metrics = clip['conditions'][profile]['methods'][method]['full_scene']
                for query in ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR'):
                    for key, value in metrics[query]['confusion'].items():
                        total[key] += value
            summary[profile][method] = total
        geometry = summary[profile]['GEOMETRY_TEMPORAL']
        union = summary[profile]['TEMPORAL_OR_RGB']
        summary[profile]['RGB_INCREMENT_OVER_GEOMETRY'] = {
            'added_true_bits': union['TP'] - geometry['TP'],
            'added_false_bits': union['FP'] - geometry['FP'],
            'remaining_missed_bits': union['FN'],
        }
    return summary


def run(root: Path, output: Path) -> None:
    root, output = root.resolve(), output.resolve()
    allowed = (root / 'artifacts.local' / 'work' / TASK).resolve()
    if output.parent != allowed or output.exists():
        raise ValueError('output must be a new direct child of the MZ79 artifact root')
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
    for name in ('mz79_tof_range_cap.py', 'tof_range_cap_sensitivity.py',
                 'tof_geometry_temporal_expert.py', 'mz77_sequence_metrics.py'):
        shutil.copyfile(bind(here / name), output / name)
    prior = root / 'artifacts.local' / 'work' / 'mz77-approach-sequence-20260911'
    sensor_receipt_path = bind(prior / 'sensor-v1' / 'receipt.json')
    inference_receipt_path = bind(prior / 'inference-v1' / 'receipt.json')
    score_receipt_path = bind(prior / 'score-v1' / 'receipt.json')
    sensor_receipt = json.loads(sensor_receipt_path.read_text(encoding='utf-8'))
    inference_receipt = json.loads(inference_receipt_path.read_text(encoding='utf-8'))
    score_receipt = json.loads(score_receipt_path.read_text(encoding='utf-8'))
    observations_path = bind(prior / 'sensor-v1' / 'observations.npz', sensor_receipt['outputs']['observations.npz'])
    learned_path = bind(prior / 'inference-v1' / 'predictions.npz', inference_receipt['outputs']['predictions.npz'])
    evaluator_path = prior / 'score-v1' / 'evaluator.npz'
    if sha(evaluator_path) != score_receipt['outputs']['evaluator.npz']:
        raise ValueError('frozen evaluator hash mismatch')
    with np.load(observations_path, allow_pickle=False) as z:
        observations = dict(z)
    with np.load(learned_path, allow_pickle=False) as z:
        learned = dict(z)
    ranges = observations['CLEAN/ranges']
    valid = observations['CLEAN/valid']
    rgb = np.asarray(learned['CLEAN/rgb'])
    predictions, packets, diagnostics = {}, {}, {}
    for profile, (inner_m, corner_m) in PROFILES.items():
        stressed_ranges, stressed_valid, removed = apply_range_cap(ranges, valid, inner_m, corner_m)
        geometry = predict(stressed_ranges, stressed_valid, observations['clip'], observations['time_s'])
        predictions[profile] = {
            'GEOMETRY_TEMPORAL': geometry['GEOMETRY_TEMPORAL'],
            'RGB': rgb,
            'TEMPORAL_OR_RGB': np.maximum(geometry['GEOMETRY_TEMPORAL'], rgb),
        }
        packets[profile] = {'ranges': stressed_ranges, 'valid': stressed_valid}
        diagnostics[profile] = {
            'inner_cap_m': inner_m,
            'corner_cap_m': corner_m,
            'removed_direct_slots': int(removed.sum()),
            'retained_direct_slots': int(stressed_valid.sum()),
            'all_invalid_frames': int((~stressed_valid.any(axis=(1, 2))).sum()),
            'any_valid_frames': int(stressed_valid.any(axis=(1, 2)).sum()),
            'temporally_imputed_slots': int(geometry['imputed'].sum()),
            'temporally_imputed_frames': int(geometry['imputed'].any(axis=(1, 2)).sum()),
        }
    sealed = {profile + '/' + method: value for profile, methods in predictions.items()
              for method, value in methods.items()}
    np.savez_compressed(output / 'predictions.npz', clip=observations['clip'],
                        index=observations['index'], time_s=observations['time_s'], **sealed)
    write(output / 'predictor-receipt.json', {
        'status': 'SEALED_BEFORE_EVALUATOR_OPEN',
        'phase': 'EXPLORE_CONSUMED_DEVELOPMENT',
        'profiles': PROFILES,
        'methods': METHODS,
        'datasheet_source': {'url': DATASHEET_URL, 'revision': DATASHEET_REVISION,
                             'table': DATASHEET_TABLE, 'mode': 'continuous_8x8_15Hz'},
        'interpolation': 'radial center-four to corner endpoint; explicit sensitivity assumption',
        'limits': 'hard maximum-range caps, not a calibrated probability/noise/ambient sensor model; MZ77 poses remain nominal 10Hz',
        'training_steps': 0, 'fitted_parameters': 0, 'threshold_searches': 0,
        'evaluator_reads': 0, 'inputs': inputs, 'diagnostics': diagnostics,
        'predictions_sha256': sha(output / 'predictions.npz'),
    })
    bind(evaluator_path, score_receipt['outputs']['evaluator.npz'])
    with np.load(evaluator_path, allow_pickle=False) as z:
        evaluator = dict(z)
    arrays = {**observations, **evaluator, 'packets': packets}
    result = analyze(arrays, predictions)
    summary = aggregate(result)
    write(output / 'result.json', result)
    write(output / 'summary.json', {'aggregate_four_query_bits': summary,
                                    'packet_diagnostics': diagnostics})
    for path, digest in inputs.items():
        if sha(Path(path)) != digest:
            raise ValueError(f'input changed during run: {path}')
    write(output / 'receipt.json', {
        'status': 'PASS', 'phase': 'EXPLORE_CONSUMED_DEVELOPMENT',
        'frames': 160, 'profiles': list(PROFILES), 'methods': METHODS,
        'training_steps': 0, 'fitted_parameters': 0, 'threshold_searches': 0,
        'input_hashes': inputs,
        'outputs': {p.name: sha(p) for p in output.iterdir() if p.is_file()},
        'backend': 'CPU TASK_NOT_GPU_SUITABLE', 'seconds': time.perf_counter() - started,
        'claim': 'Datasheet-endpoint sensitivity on simulator-derived packets; no calibrated device, natural-scene, deployment, user-benefit or safety claim.',
    })
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('E:/linnan/linnan'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
