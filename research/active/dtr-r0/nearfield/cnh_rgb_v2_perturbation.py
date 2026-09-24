"""Frozen RGB-use diagnostic on the existing alley Development V2 checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from cnh_rgb_alley_v2 import AzimuthMaskedFrustumFusion, COLLECTION_SHA256, PARTITION_SHA256, SEEDS
from cnh_rgb_dev_comparison import read_inputs, sha


SHUFFLE_SEED = 20260925
BOOTSTRAP_SEED = 2026092501
BOOTSTRAP_REPLICATES = 1000
BATCH = 16


def within_layout_cycle(rows, dev):
    """One random full cycle in each dev layout; every RGB comes from another frame."""
    rng = np.random.default_rng(SHUFFLE_SEED)
    source = np.arange(len(rows))
    layouts = sorted({rows[i]['layout_id'] for i in np.flatnonzero(dev)})
    for layout in layouts:
        indices = np.array([i for i in np.flatnonzero(dev) if rows[i]['layout_id'] == layout])
        if len(indices) != 160:
            raise ValueError('Each dev layout must contain 160 frames')
        cycle = rng.permutation(indices)
        source[cycle] = np.roll(cycle, -1)
    if np.any(source[dev] == np.flatnonzero(dev)) or len(set(source[dev])) != int(dev.sum()):
        raise ValueError('RGB permutation must be a derangement and bijection')
    return source


def score(labels, logits):
    known = labels >= 0
    y, s = labels[known], logits[known]
    if len(np.unique(y)) != 2:
        return dict(auprc=None, auroc=None, positives=int((y == 1).sum()),
                    negatives=int((y == 0).sum()), unknown=int((~known).sum()))
    return dict(auprc=float(average_precision_score(y, s)),
                auroc=float(roc_auc_score(y, s)), positives=int((y == 1).sum()),
                negatives=int((y == 0).sum()), unknown=int((~known).sum()))


def bootstrap(labels, logits, groups, indices):
    """Paired frame bootstrap, stratified by fixed dev layout, conditional on layouts."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    members = [np.array([i for i in indices if groups[i] == group]) for group in sorted(set(groups[indices]))]
    samples = []
    for _ in range(BOOTSTRAP_REPLICATES):
        samples.append(np.concatenate([rng.choice(m, len(m), replace=True) for m in members]))
    result = {}
    for name, matrix in logits.items():
        actual = score(labels[indices], matrix[indices])
        distributions = {key: [] for key in ('auprc', 'auroc')}
        for sample in samples:
            estimate = score(labels[sample], matrix[sample])
            for key in distributions:
                if estimate[key] is not None:
                    distributions[key].append(estimate[key])
        result[name] = dict(**actual, ci95={
            key: [float(x) for x in np.quantile(values, [.025, .975])] if values else None
            for key, values in distributions.items()},
            valid_bootstrap_replicates={key: len(values) for key, values in distributions.items()})
    return result


def infer(data, checkpoint, rgb_source, device):
    model = AzimuthMaskedFrustumFusion().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    indices = np.flatnonzero(data['dev'])
    static = {name: torch.from_numpy(data[name]).to(device) for name in
              ('histogram', 'ambient', 'scalar', 'valid')}
    support = torch.from_numpy(data['support']).to(device)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH):
            batch = indices[start:start+BATCH]
            rgb = torch.from_numpy(data['rgb'][rgb_source[batch]].astype(np.float32) / 255).to(device)
            logits = model(rgb, static['histogram'][batch], static['ambient'][batch],
                           static['scalar'][batch], static['valid'][batch],
                           torch.zeros_like(static['ambient'][batch]), support,
                           mode='cnh')['occupancy_logits']
            chunks.append(logits.cpu().numpy())
    return np.concatenate(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection-overlay', type=Path, required=True)
    parser.add_argument('--partition-plan', type=Path, required=True)
    parser.add_argument('--v2-result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Output must be new or empty')
    if sha(args.collection_overlay) != COLLECTION_SHA256 or sha(args.partition_plan) != PARTITION_SHA256:
        raise ValueError('Frozen input identity mismatch')
    prior = json.loads(args.v2_result.read_text(encoding='utf-8'))
    if prior['status'] != 'COMPLETE_PROSPECTIVE_ALLEY_DEVELOPMENT_V2':
        raise ValueError('Expected completed frozen V2 receipt')
    data = read_inputs(args.collection_overlay, args.partition_plan)
    rows, dev = data['rows'], data['dev']
    source = within_layout_cycle(rows, dev)
    args.output.mkdir(parents=True, exist_ok=True)
    permutation = [dict(target_frame_key=rows[i]['frame_key'], source_frame_key=rows[source[i]]['frame_key'],
                        layout_id=rows[i]['layout_id']) for i in np.flatnonzero(dev)]
    (args.output / 'rgb-permutation.json').write_text(json.dumps(permutation, indent=2) + '\n', encoding='utf-8')
    # The permutation file, seed, rule and bootstrap settings are written before any new inference is read.
    plan = dict(status='FROZEN_BEFORE_PERTURBED_INFERENCE', shuffle_seed=SHUFFLE_SEED,
                rule='Within each dev layout, RNG permutation of the 160 ordered frames followed by one cyclic successor; no fixed points; one map shared by all seeds',
                bootstrap_seed=BOOTSTRAP_SEED, bootstrap_replicates=BOOTSTRAP_REPLICATES,
                bootstrap_unit='frame; six queries move together; resample 160 frames within each fixed dev layout',
                collection_sha256=sha(args.collection_overlay), partition_sha256=sha(args.partition_plan),
                v2_result_sha256=sha(args.v2_result), permutation_sha256=sha(args.output / 'rgb-permutation.json'),
                script_sha256=sha(__file__))
    (args.output / 'frozen-plan.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=plan['status'], permutation_sha256=plan['permutation_sha256'])), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA inference required for V2 reloaded-score parity')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')
    dev_indices = np.flatnonzero(dev)
    logits_by_seed = {}
    parity = {}
    checkpoints = {}
    for seed in SEEDS:
        entry = next(x for x in prior['results'] if x['arm'] == 'cnh_rgb' and x['seed'] == seed)
        checkpoint = Path(entry['model_path'])
        prediction_path = Path(entry['predictions_path'])
        if sha(checkpoint) != entry['model_sha256'] or sha(prediction_path) != entry['predictions_sha256']:
            raise ValueError('Frozen checkpoint or prediction hash mismatch')
        checkpoints[str(seed)] = entry['model_sha256']
        with np.load(prediction_path, allow_pickle=False) as saved:
            if list(saved['frame_key'].astype(str)[dev]) != [rows[i]['frame_key'] for i in dev_indices]:
                raise ValueError('Saved prediction frame keys differ')
            original_saved = saved['logits'][dev]
        arms = {}
        for name, rgb_source in [('original', np.arange(len(rows))), ('shuffled', source)]:
            arms[name] = infer(data, checkpoint, rgb_source, device)
        zero_data = dict(data, rgb=np.zeros_like(data['rgb']))
        arms['zero'] = infer(zero_data, checkpoint, np.arange(len(rows)), device)
        maximum = float(np.max(np.abs(arms['original'] - original_saved)))
        parity[str(seed)] = maximum
        if maximum > 1e-5:
            raise ValueError(f'Original prediction replay differs for seed {seed}: {maximum}')
        np.savez_compressed(args.output / f'seed-{seed}-logits.npz',
                            frame_key=np.array([rows[i]['frame_key'] for i in dev_indices]), **arms)
        logits_by_seed[seed] = arms
        print(json.dumps(dict(seed=seed, max_original_logit_difference=maximum)), flush=True)
    del zero_data
    torch.cuda.empty_cache()
    print(json.dumps(dict(status='GPU_INFERENCE_COMPLETE')), flush=True)
    labels = data['labels'][dev]
    groups = np.array([rows[i]['layout_id'] for i in dev_indices])
    all_indices = np.arange(len(dev_indices))
    results = {}
    for seed, arms in logits_by_seed.items():
        total = bootstrap(labels, arms, groups, all_indices)
        layouts = {layout: bootstrap(labels, arms, groups, all_indices[groups == layout])
                   for layout in sorted(set(groups))}
        original = arms['original']
        correlation = {name: float(np.corrcoef(original.flatten(), arms[name].flatten())[0, 1])
                       for name in ('shuffled', 'zero')}
        change = {name: dict(mean_abs_logit=float(np.mean(np.abs(original-arms[name]))),
                             max_abs_logit=float(np.max(np.abs(original-arms[name]))),
                             threshold_flip_queries=int(np.sum((original >= 0) != (arms[name] >= 0))))
                  for name in ('shuffled', 'zero')}
        results[str(seed)] = dict(total=total, layouts=layouts, correlation_with_original=correlation,
                                  perturbation_change=change,
                                  logits_sha256=sha(args.output / f'seed-{seed}-logits.npz'))
    report = dict(status='COMPLETE_RGB_USE_DEVELOPMENT_DIAGNOSTIC', data_role='Development',
                  plan=plan, checkpoints=checkpoints, original_replay_max_abs_logit_difference=parity,
                  dev_frames=480, queries_per_frame=6, results=results,
                  claim_limit='Bootstrap intervals condition on three fixed dev layouts and are not layout-generalization intervals; labels are geometry-derived and not independently precision-admitted; fixed V2 checkpoint only')
    (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], result_sha256=sha(args.output / 'result.json'))), flush=True)


if __name__ == '__main__':
    main()
