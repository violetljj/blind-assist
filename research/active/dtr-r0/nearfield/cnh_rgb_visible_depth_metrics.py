"""Frozen-layout, frame-resampled ranking metrics for the two depth controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from cnh_rgb_dev_comparison import read_inputs, sha


BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 20260925


def rank_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    y, score = np.asarray(y).ravel(), np.asarray(score).ravel()
    known = y >= 0
    y, score = y[known], score[known]
    if not np.isfinite(score).all() or not ((y == 1).any() and (y == 0).any()):
        raise ValueError('Finite scores and both known classes required')
    return dict(auprc=float(average_precision_score(y, score)),
                auroc=float(roc_auc_score(y, score)),
                positives=int((y == 1).sum()), negatives=int((y == 0).sum()),
                unknown=int((~known).sum()))


def draw_indices(layouts: np.ndarray) -> tuple[list[np.ndarray], dict[str, list[np.ndarray]]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    groups = {name: np.flatnonzero(layouts == name) for name in sorted(set(layouts))}
    if len(groups) != 3 or any(len(indices) != 160 for indices in groups.values()):
        raise ValueError('Exactly three 160-frame dev layouts required')
    total = []
    per_layout = {name: [] for name in groups}
    for _ in range(BOOTSTRAP_DRAWS):
        parts = []
        for name, indices in groups.items():
            sampled = rng.choice(indices, len(indices), replace=True)
            per_layout[name].append(sampled)
            parts.append(sampled)
        total.append(np.concatenate(parts))
    return total, per_layout


def summarize(y: np.ndarray, score: np.ndarray, indices: list[np.ndarray],
              point_indices: np.ndarray | None = None) -> dict:
    result = rank_metrics(y if point_indices is None else y[point_indices],
                          score if point_indices is None else score[point_indices])
    draws = [rank_metrics(y[i], score[i]) for i in indices]
    result['ci95_frame_bootstrap'] = {
        name: [float(x) for x in np.quantile([d[name] for d in draws], [.025, .975])]
        for name in ('auprc', 'auroc')
    }
    result['valid_bootstrap_draws'] = len(draws)
    return result


def run(collection: Path, partition: Path, learned: Path, direct: Path, output: Path) -> dict:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Fresh/empty output required')
    data = read_inputs(collection, partition)
    dev = data['dev']
    y = data['labels'][dev]
    keys = np.array([r['frame_key'] for r in data['rows']])
    layouts = np.array([r['layout_id'] for r in data['rows']])[dev]
    total_draws, layout_draws = draw_indices(layouts)
    report_in = json.loads((learned/'result.json').read_text(encoding='utf-8'))
    if report_in['status'] != 'COMPLETE_VISIBLE_DEPTH_LEARNED_CEILINGS_DEVELOPMENT' or len(report_in['results']) != 6:
        raise ValueError('Six complete learned controls required')
    arms = []
    for item in report_in['results']:
        path = Path(item['predictions_path'])
        if sha(path) != item['predictions_sha256']:
            raise ValueError('Frozen learned prediction hash differs')
        with np.load(path, allow_pickle=False) as saved:
            if (not np.array_equal(saved['frame_key'], keys) or
                    not np.array_equal(saved['dev'], dev) or saved['logits'].shape != (960, 6)):
                raise ValueError('Learned prediction frame identity differs')
            score = saved['logits'][dev]
        arm = dict(arm=item['arm'], seed=item['seed'], prediction_path=str(path),
                   prediction_sha256=sha(path), overall=summarize(y, score, total_draws),
                   layouts={name: summarize(y, score, draws, np.flatnonzero(layouts == name))
                            for name, draws in layout_draws.items()})
        arms.append(arm)
        print(json.dumps({'arm': arm['arm'], 'seed': arm['seed'],
                          'overall': arm['overall']}), flush=True)
    direct_path = direct/'direct-depth-rule-dev.npz'
    direct_report = json.loads((direct/'result.json').read_text(encoding='utf-8'))
    if sha(direct_path) != direct_report['files']['direct-depth-rule-dev.npz']:
        raise ValueError('Direct rule prediction hash differs')
    with np.load(direct_path, allow_pickle=False) as saved:
        if not np.array_equal(saved['frame_key'], keys[dev]) or saved['prediction'].shape != (480, 6):
            raise ValueError('Direct rule frame identity differs')
        score = saved['prediction'].astype(float)
    direct_result = dict(arm='direct_visible_depth_box_rule', prediction_path=str(direct_path),
                         prediction_sha256=sha(direct_path), overall=summarize(y, score, total_draws),
                         layouts={name: summarize(y, score, draws, np.flatnonzero(layouts == name))
                                  for name, draws in layout_draws.items()},
                         counts=dict(tp=int(((y == 1) & (score > 0)).sum()),
                                     fp=int(((y == 0) & (score > 0)).sum()),
                                     fn=int(((y == 1) & (score == 0)).sum()),
                                     tn=int(((y == 0) & (score == 0)).sum())))
    exceptions = []
    for row, query in zip(*np.where((y == 1) & (score == 0))):
        exceptions.append(dict(frame_key=str(keys[dev][row]), layout_id=str(layouts[row]),
                               query=str(data['query_names'][query])))
    direct_result['positive_without_visible_pixel'] = exceptions
    report = dict(status='COMPLETE_VISIBLE_DEPTH_CEILING_RANK_METRICS_DEVELOPMENT',
                  average_precision_definition='sklearn average_precision_score, no interpolation; direct rule binary score has ties',
                  bootstrap=dict(unit='frame, six queries kept together', stratification='resample 160 frames with replacement within each of three fixed dev layouts',
                                 seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS, interval='percentile 2.5/97.5'),
                  dev_frames=480, dev_queries=2880, base_positive_rate=float((y == 1).sum()/(y >= 0).sum()),
                  collection_overlay_sha256=sha(collection), partition_sha256=sha(partition),
                  learned_report_sha256=sha(learned/'result.json'), direct_report_sha256=sha(direct/'result.json'),
                  code_sha256=sha(__file__), learned=arms, direct=direct_result,
                  claim_limit='Three fixed dev layouts only; frame bootstrap conditions on these layouts and does not estimate unseen-layout uncertainty. Direct rule shares query-box geometry but no target triangles/instance IDs.')
    output.mkdir(parents=True, exist_ok=True)
    (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-overlay', type=Path, required=True)
    parser.add_argument('--partition-plan', type=Path, required=True)
    parser.add_argument('--learned', type=Path, required=True)
    parser.add_argument('--direct', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.collection_overlay, args.partition_plan, args.learned, args.direct, args.output)
    print(json.dumps({'status': result['status'], 'output': str(args.output/'result.json')}))
