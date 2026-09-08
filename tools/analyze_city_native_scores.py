"""Cached-only City score/ranking/localization diagnosis; never selects a model.

All threshold sweeps are POSTHOC_DIAGNOSTIC_ONLY on correlated consumed frames.
No inference, fitting, threshold recommendation, or promotion is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SCOPE = 'POSTHOC_DIAGNOSTIC_ONLY: one consumed static route; correlated frames; all-scene geometric reference, not independently checked obstacle accuracy; no threshold selection or promotion'
HEADS = ('BODY', 'HEAD')
METHODS = ('g10', 'g13')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8', newline='\n')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Input hash mismatch: {path}')
    return path


def distribution(values):
    return dict(n=len(values), mean=float(np.mean(values)) if len(values) else None,
        quantiles=dict(zip(('min', 'p10', 'p25', 'median', 'p75', 'p90', 'max'),
            [float(x) for x in np.quantile(values, [0, .1, .25, .5, .75, .9, 1])])) if len(values) else None)


def rank_auc(score, truth):
    """Pairwise rank AUC with half credit for exact ties; missing classes => None."""
    positive, negative = score[truth == 1], score[truth == 0]
    if not len(positive) or not len(negative):
        return None
    delta = positive[:, None] - negative[None, :]
    return float(((delta > 0) + .5 * (delta == 0)).mean())


def confusion(score, truth, threshold):
    alert = score >= threshold
    tp, fn = int((alert & (truth == 1)).sum()), int((~alert & (truth == 1)).sum())
    fp, tn = int((alert & (truth == 0)).sum()), int((~alert & (truth == 0)).sum())
    return dict(threshold=float(threshold), TP=tp, FN=fn, FP=fp, TN=tn,
        TPR=tp/(tp+fn) if tp+fn else None, FPR=fp/(fp+tn) if fp+tn else None,
        unknown=int((truth == -1).sum()))


def summarize(score, truth, threshold):
    if score.shape != truth.shape or not np.isin(truth, [-1, 0, 1]).all():
        raise ValueError('Score/label mismatch')
    if not np.isfinite(score).all() or not ((score >= 0) & (score <= 1)).all():
        raise ValueError('Invalid cached probability')
    # All exact observed decision breakpoints plus endpoints; no best row selected.
    grid = np.unique(np.concatenate(([0., 1., threshold], score[truth >= 0])))
    return dict(positive=distribution(score[truth == 1]), negative=distribution(score[truth == 0]),
        unknown=distribution(score[truth == -1]), rank_auc=rank_auc(score, truth),
        frozen=confusion(score, truth, threshold), sweep_scope=SCOPE,
        threshold_sweep=[confusion(score, truth, float(t)) for t in grid])


def draw_scores(predictions, near, ids, summary, path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for h, head in enumerate(HEADS):
        ax = axes[h, 0]
        for j, method in enumerate(METHODS):
            for offset, value, color in ((-.13, 0, '#2674ad'), (.13, 1, '#cc4d2a')):
                x = predictions[method + '_near'][:, h][near[:, h] == value]
                # Deterministic jitter is display-only and never changes scores.
                jitter = np.linspace(-.07, .07, len(x))
                ax.scatter(j+offset+jitter, x, s=24, color=color,
                    label=('negative' if value == 0 else 'positive') if j == 0 else None)
            ax.plot([j-.35, j+.35], [summary[method][head]['frozen']['threshold']]*2,
                    color='black', linestyle='--', linewidth=1.2)
        ax.set(xticks=[0, 1], xticklabels=['G10', 'G13'], ylim=(-.02, 1.02),
               ylabel=f'{head} near probability', title='Known-label scores; dashed = frozen threshold')
        ax.legend(fontsize=8)
        for method, color in zip(METHODS, ('#2674ad', '#cc4d2a')):
            rows = summary[method][head]['threshold_sweep']
            axes[h, 1].plot([r['FPR'] for r in rows], [r['TPR'] for r in rows], '.-',
                color=color, label=f'{method.upper()} AUC={summary[method][head]["rank_auc"]:.3f}')
            axes[h, 2].plot(ids, predictions[method + '_near'][:, h], '.-', color=color, label=method.upper())
        axes[h, 1].plot([0, 1], [0, 1], ':', color='gray')
        axes[h, 1].set(xlabel='False-positive rate', ylabel='Recall', xlim=(-.02, 1.02), ylim=(-.02, 1.02),
            title=f'{head}: all post-hoc sweep points; no selection')
        axes[h, 1].legend(fontsize=8)
        for i, truth in zip(ids, near[:, h]):
            if truth == 1:
                axes[h, 2].axvspan(i-.4, i+.4, color='#cc4d2a', alpha=.10)
            elif truth == -1:
                axes[h, 2].axvspan(i-.4, i+.4, color='gray', alpha=.35)
        axes[h, 2].set(xlabel='Spatial frame index (not time)', ylabel='Near probability', ylim=(-.02, 1.02),
            title='Red bands positive; gray UNKNOWN')
        axes[h, 2].legend(fontsize=8)
    fig.suptitle('Cached City route diagnosis | 39 known frames + 1 UNKNOWN | correlated Development', fontsize=13)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_localization(predictions, config, ids, capture, label_root, targets, selected, out):
    fig, axes = plt.subplots(len(selected), 5, figsize=(20, 3.5*len(selected)), constrained_layout=True, squeeze=False)
    rows = []
    for r, sid in enumerate(selected):
        i = ids.index(sid)
        rgb_path = capture / f'model/sample/{sid:04d}.png'
        checked(rgb_path, config['rgb_sha256'][str(sid)])
        rgb = np.array(Image.open(rgb_path).convert('RGB'))
        masks = np.zeros((2, 360, 640), bool)
        names = []
        for target in targets:
            if target.get('status') != 'EVALUABLE':
                continue
            data = np.load(label_root / target['masks'], mmap_mode='r', allow_pickle=False)
            if data.shape != (len(ids), 2, 360, 640):
                raise ValueError('Native target-mask shape mismatch')
            positive = data[i] == 1
            if positive.any():
                names.append(target['target_id'])
                masks |= positive
        axes[r, 0].imshow(rgb)
        for h, color in enumerate(('#00ff82', '#ff5adc')):
            if masks[h].any():
                axes[r, 0].contour(masks[h], levels=[.5], colors=[color], linewidths=.8)
        axes[r, 0].set_title(f'Frame {sid}: {", ".join(names) or "no reliable target"}\nOriginal RGB + reliable in-query outlines', fontsize=10)
        for c, (method, h) in enumerate((('g10', 0), ('g13', 0), ('g10', 1), ('g13', 1)), 1):
            support = predictions[method + '_support'][i, h]
            axes[r, c].imshow(rgb)
            im = axes[r, c].imshow(support, extent=(-.5, 639.5, 359.5, -.5), origin='upper',
                interpolation='nearest', vmin=0, vmax=1, cmap='inferno', alpha=.70)
            if masks[h].any():
                axes[r, c].contour(masks[h], levels=[.5], colors=['#00ff82' if h == 0 else '#ff5adc'], linewidths=.85)
            probability = float(predictions[method + '_near'][i, h])
            threshold = config['methods'][method]['near_thresholds'][h]['value']
            axes[r, c].set_title(f'{method.upper()} {HEADS[h]} | near={probability:.4f} / {threshold:.4f}\n'
                f'support max={support.max():.4f}; alert={probability >= threshold}', fontsize=10)
            # Exact native positives pooled only to measure heatmap overlap.
            pooled = masks[h].reshape(18, 20, 32, 20).any(axis=(1, 3))
            rows.append(dict(sample_index=sid, targets=names, method=method, head=HEADS[h],
                near_probability=probability, threshold=threshold, support_max=float(support.max()),
                mask_positive_cells=int(pooled.sum()),
                support_mean_in_target=float(support[pooled].mean()) if pooled.any() else None,
                support_mean_outside_target=float(support[~pooled].mean()),
                above_half_cells=int((support >= .5).sum()),
                overlapping_cells=int(((support >= .5) & pooled).sum())))
        for ax in axes[r]:
            ax.set(xlim=(-.5, 639.5), ylim=(359.5, -.5))
            ax.axis('off')
    bar = fig.colorbar(im, ax=axes[:, 1:].ravel().tolist(), fraction=.015, pad=.005)
    bar.set_label('Cached support probability (common 0–1 scale; alpha 0.70)', fontsize=9)
    fig.suptitle('Cached support localization | green BODY / magenta HEAD outlines = independently checked target pixels\n'
                 'Display enlargement only; original 18x32 probabilities; no inference or threshold selection', fontsize=13)
    fig.savefig(out / 'target-heatmaps.png', dpi=150)
    plt.close(fig)
    return rows


def run(args):
    started = time.perf_counter()
    out = args.output.resolve()
    artifacts = (ROOT / 'artifacts.local').resolve()
    if out.exists() or out == artifacts or not out.is_relative_to(artifacts):
        raise ValueError('Fresh canonical artifact output required')
    cache, labels, capture = args.predictions.resolve(), args.labels.resolve(), args.capture.resolve()
    receipt = read(cache / 'receipt.json')
    if receipt['status'] != 'PASS':
        raise ValueError('Completed prediction cache required')
    checked(cache / 'predictions.npz', receipt['predictions_sha256'])
    checked(cache / 'protocol.json', receipt['protocol_sha256'])
    config, label_manifest = read(cache / 'protocol.json'), read(labels / 'native-route-labels.json')
    ids = config['sample_indices']
    if ids != label_manifest['sample_indices']:
        raise ValueError('Prediction/label identity mismatch')
    validation = read(labels / 'label-validation.json')
    checked(labels / 'near.npy', validation['near_sha256'])
    checked(labels / 'native-route-labels.json', validation['label_manifest_sha256'])
    for target in label_manifest.get('targets', []):
        checked(labels / target['masks'], validation['target_mask_sha256'][target['target_id']])
    with np.load(cache / 'predictions.npz', allow_pickle=False) as data:
        predictions = {key: data[key] for key in data.files}
    near = np.load(labels / label_manifest['near'], allow_pickle=False)
    if near.shape != (len(ids), 2):
        raise ValueError('Invalid near labels')
    out.mkdir(parents=True)
    protocol = dict(scope=SCOPE, backend='CPU', backend_reason='TASK_NOT_GPU_SUITABLE_SMALL_CACHED_ARRAYS',
        inference_calls=0, optimizer_steps=0, selected_threshold=None, input_sha256=dict(
            predictions=sha(cache / 'predictions.npz'), protocol=sha(cache / 'protocol.json'),
            labels=sha(labels / 'native-route-labels.json'), label_validation=sha(labels / 'label-validation.json')),
        code_sha256=sha(Path(__file__)), display_frames=args.frames)
    write(out / 'protocol.json', protocol)
    summary = {method: {head: summarize(predictions[method + '_near'][:, h], near[:, h],
        config['methods'][method]['near_thresholds'][h]['value']) for h, head in enumerate(HEADS)} for method in METHODS}
    draw_scores(predictions, near, ids, summary, out / 'scores-and-ranking.png')
    localization = draw_localization(predictions, config, ids, capture, labels,
        label_manifest.get('targets', []), args.frames, out)
    write(out / 'result.json', dict(scope=SCOPE, scores=summary, selected_frame_localization=localization,
        unknown_indices={head: [sid for sid, value in zip(ids, near[:, h]) if value == -1] for h, head in enumerate(HEADS)}))
    write(out / 'receipt.json', dict(status='PASS', elapsed_seconds=time.perf_counter()-started,
        result_sha256=sha(out / 'result.json'), plot_sha256={n: sha(out / n) for n in ('scores-and-ranking.png', 'target-heatmaps.png')},
        inference_calls=0, selected_threshold=None))
    print(json.dumps({m:{h:dict(AUC=summary[m][h]['rank_auc'],positive=summary[m][h]['positive'],negative=summary[m][h]['negative'])
        for h in HEADS} for m in METHODS}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    for name in ('predictions', 'labels', 'capture', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--frames', type=int, nargs='+', default=[6, 14, 28])
    run(parser.parse_args())
