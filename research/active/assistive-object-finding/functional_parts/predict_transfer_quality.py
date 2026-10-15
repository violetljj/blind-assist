#!/usr/bin/env python3
"""Frozen visual quality models on observation-only independent-source inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import learn_visual_quality as visual

old, np = visual.old, visual.np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    tick, cpu = time.perf_counter(), time.process_time()
    training = args.training_root.resolve()
    seal_path = training / 'training_seal.json'
    seal = json.loads(seal_path.read_text(encoding='utf-8'))
    verified = {}
    for name, expected in seal['files'].items():
        actual = old.digest(training / name)
        if actual != expected:
            raise ValueError('Training seal mismatch: ' + name)
        verified[name] = actual
    observations = old.read_rows(args.predictions)
    if not observations or len({o['sample_id'] for o in observations}) != len(observations):
        raise ValueError('Empty or duplicate observations')
    # This interface deliberately has no ground-truth manifest or label argument.
    forbidden = {'gt_path', 'gt_sha256', 'mask_path', 'target_label', 'native_bbox'}
    if any(forbidden & set(o) for o in observations):
        raise ValueError('Evaluator-only fields in observations')
    matrix = np.load(training / 'projection.npy', allow_pickle=False)
    out = args.root / 'learned_predictions'
    out.mkdir(parents=True, exist_ok=False)
    old.write_json(out / 'model_source_seal.json', dict(
        training_root=str(training), training_seal_sha256=old.digest(seal_path), files=verified,
        protocol_sha256=old.digest(args.protocol), script_sha256=old.digest(__file__),
        visual_dependency_sha256=old.digest(visual.__file__),
        feature_dependency_sha256=old.digest(old.__file__),
        frozen_weights=True, new_training=False, fresh_GT_read=False))
    records = []
    for family in ('candidates', 'fixed'):
        X, points, _ = visual.assemble(observations, family, matrix)
        scores, seeds = {}, {}
        for variant in visual.VARIANTS:
            if time.process_time() - cpu > 600:
                raise TimeoutError('Transfer feature/prediction CPU budget 600s')
            np.savez(out / f'{family}_{variant}_features.npz', X=X[variant],
                     points=np.asarray(points), sample_ids=np.array([o['sample_id'] for o in observations]))
            values, per_seed = visual.predict_variant(X[variant], training / family / variant)
            scores.update({variant + '_' + k: v for k, v in values.items()})
            seeds.update({variant + '_' + k: v for k, v in per_seed.items()})
        records.extend(old.prediction_rows(observations, family, points, scores, seeds))
    if time.process_time() - cpu > 600:
        raise TimeoutError('Transfer feature/prediction CPU budget 600s')
    old.write_rows(out / 'fresh_predictions.jsonl', records)
    receipt = dict(status='SUCCESS', images=len(observations), records=len(records),
        fresh_GT_read=False, new_training=False, models=50, backbone_calls_added=0,
        observation_predictions_sha256=old.digest(args.predictions),
        training_seal_sha256=old.digest(seal_path), model_source_seal_sha256=old.digest(out / 'model_source_seal.json'),
        script_sha256=old.digest(__file__), prediction_sha256=old.digest(out / 'fresh_predictions.jsonl'),
        wall_s=time.perf_counter()-tick, cpu_s=time.process_time()-cpu,
        process_cpu_s=time.process_time(), backend='CPU', placement='FROZEN_PROTOCOL_CPU_ONLY')
    old.write_json(out / 'prediction_seal.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
