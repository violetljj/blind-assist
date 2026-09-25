"""Nontrained query-conditioned H3 geometry on frozen alley train/dev only.

Uses angular/radial cell overlap with predefined camera boxes. This is a
particular geometric readout, not an information-theoretic H3 ceiling.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from cnh_route_sensor import angular_rays, H3, RAW_BIN_M
from cnh_street_e2e_materialize import BOXES
from cnh_rgb_dev_comparison import read_inputs, sha, checked
from cnh_rgb_alley_v2 import COLLECTION_SHA256, PARTITION_SHA256
from cnh_alley_bottleneck_slices import target_context, distance_bin


def query_weights(boxes=BOXES, samples=16):
    """Solid-angle mean radial overlap fraction [query,zone,bin].

    Uniform-within-cell assumption only reconstructs the lost subcell location;
    neither scene depth nor labels enter this projection. Sensor/camera identity
    transform is verified separately because the cached observations assume it.
    """
    rays, area = angular_rays(samples)
    rays, area = rays.reshape(64, -1, 3), area.reshape(64, -1)
    area = area / area.sum(-1, keepdims=True)
    width = RAW_BIN_M * H3.sub_sample
    edges = (H3.start_bin * RAW_BIN_M + np.arange(H3.bins + 1) * width)
    result = []
    for low, high in boxes:
        a, b = low / rays, high / rays
        near = np.maximum(np.minimum(a, b).max(-1), 0)
        far = np.maximum(np.maximum(a, b).min(-1), 0)
        overlap = np.maximum(0, np.minimum(far[..., None], edges[1:]) -
                             np.maximum(near[..., None], edges[:-1])) / width
        result.append(np.einsum('zr,zrb->zb', area, overlap))
    return np.clip(np.asarray(result), 0, 1)


def summarize(labels, scores, threshold):
    known = labels >= 0
    y, s = labels[known], scores[known]
    pred = s >= threshold
    positive, negative = int((y == 1).sum()), int((y == 0).sum())
    return dict(positive=positive, negative=negative, unknown=int((~known).sum()),
                auprc=float(average_precision_score(y, s)) if positive and negative else None,
                auroc=float(roc_auc_score(y, s)) if positive and negative else None,
                tp=int(((y == 1) & pred).sum()), fp=int(((y == 0) & pred).sum()),
                fn=int(((y == 1) & ~pred).sum()), tn=int(((y == 0) & ~pred).sum()))


def train_threshold(labels, scores):
    mask = labels >= 0
    precision, recall, thresholds = precision_recall_curve(labels[mask], scores[mask])
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    # Deterministic tie: larger threshold, reducing alerts.
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def run(collection, partition, prepared, output):
    started = time.monotonic()
    if sha(collection) != COLLECTION_SHA256 or sha(partition) != PARTITION_SHA256:
        raise ValueError('Frozen collection/partition differs')
    if output.exists():
        raise FileExistsError(output)
    data = read_inputs(collection, partition)
    for item in data['collection']['layouts']:
        overlay = json.loads(checked({'path': item['overlay'], 'sha256': item['overlay_sha256']}).read_text())
        for frame in overlay['frames']:
            camera = json.loads(checked(frame['original_files']['camera.json']).read_text())
            if not np.allclose(camera['T_camera_tof'], np.eye(4), atol=1e-12):
                raise ValueError('Current cached H3 geometry requires collocated aligned camera/ToF')
    perfect_path = prepared / 'perfect-tof-h3.npz'
    receipt = json.loads((prepared / 'result.json').read_text())
    if sha(perfect_path) != receipt['files'][perfect_path.name]:
        raise ValueError('Perfect H3 cache hash differs')
    with np.load(perfect_path, allow_pickle=False) as cache:
        if list(cache['frame_key'].astype(str)) != [r['frame_key'] for r in data['rows']]:
            raise ValueError('Perfect H3 frame order differs')
        perfect = cache['histogram'].copy()
    weights = query_weights()
    assert weights.shape == (6, 64, 16) and np.all((weights >= 0) & (weights <= 1))
    context, source_hashes, _, _ = target_context(data['collection'])
    layouts = np.array([r['layout_id'] for r in data['rows']])
    buckets = np.array([distance_bin(context[r['frame_key']]['target_centre_distance_m'])
                        if data['dev'][i] else 'train' for i, r in enumerate(data['rows'])])
    output.mkdir(parents=True)
    results = {}
    scores_saved = {}
    for arm, histogram in [('noiseless_h3', perfect), ('simulated_sensor_h3', data['histogram'])]:
        # Signed sensor output is kept signed: clipping would create a noise bias.
        scores = np.einsum('nzb,qzb->nq', histogram.astype(np.float64), weights)
        threshold = train_threshold(data['labels'][data['train']], scores[data['train']])
        slices = {}
        for name, groups in [('layout', layouts), ('target_context_distance', buckets)]:
            slices[name] = {group: summarize(data['labels'][sel], scores[sel], threshold)
                            for group in sorted(set(groups[data['dev']]))
                            for sel in [data['dev'] & (groups == group)]}
        results[arm] = dict(threshold=threshold,
            train=summarize(data['labels'][data['train']], scores[data['train']], threshold),
            dev=summarize(data['labels'][data['dev']], scores[data['dev']], threshold), **slices)
        scores_saved[arm] = scores
    np.savez_compressed(output / 'scores.npz', frame_key=[r['frame_key'] for r in data['rows']],
                        weights=weights, train=data['train'], dev=data['dev'], **scores_saved)
    result = dict(status='COMPLETE_NONTRAINED_H3_GEOMETRY_DEVELOPMENT', results=results,
        definition='16x16 fixed angular rays per zone, exact ray/AABB radial interval intersection; each bin weighted by fractional radial overlap and zone solid angle; sum over zone/bin; signed sensor counts retained',
        threshold_selection='One threshold per arm maximizing pooled train F1 only; ties choose larger threshold. No dev tuning; AP independent of threshold.',
        distance_slices='Nominal inserted target centre 3D distance context; all six queries retained. Not object attribution or per-query obstacle range. Hidden-target context separate.',
        limits='Only 3 train/3 dev layouts with correlated frames; H3 rule is not an information upper bound. Sensor H3 is uncalibrated simulation with uniform reflectance/incidence, not real hardware. No test/City/capture.',
        collection_sha256=sha(collection), partition_sha256=sha(partition), perfect_h3_sha256=sha(perfect_path),
        code_sha256=sha(__file__), scores_sha256=sha(output / 'scores.npz'), source_spec_sha256=source_hashes,
        compute='NumPy CPU: tiny fixed 6x64x16 matrix and 960 observations, no training', wall_s=time.monotonic()-started)
    (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection-overlay', type=Path, required=True)
    p.add_argument('--partition-plan', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(run(args.collection_overlay, args.partition_plan, args.prepared, args.output), indent=2))
