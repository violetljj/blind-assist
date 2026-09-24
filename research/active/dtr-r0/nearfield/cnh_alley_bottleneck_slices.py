"""Read-only Development ranking and V2 failure slices from saved predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cnh_rgb_alley_v2 import COLLECTION_SHA256, PARTITION_SHA256, SEEDS
from cnh_rgb_dev_comparison import read_inputs, sha
from cnh_rgb_v2_perturbation import bootstrap, score


V2_RESULT_SHA256 = '719e224ce138f6bec92e7768a5e17dfe4bfebdc537507ca74fcf80e91cf0de85'
QUERY_NAMES = ('left_HEAD', 'left_BODY', 'centre_HEAD', 'centre_BODY', 'right_HEAD', 'right_BODY')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def target_context(collection):
    """Nominal target centre distance from frozen spec, not object attribution."""
    result, sources, target_sources, distractor_sources = {}, {}, set(), set()
    for item in collection['layouts']:
        if item['split'] != 'dev':
            continue
        overlay_path = Path(item['overlay'])
        if sha(overlay_path) != item['overlay_sha256']:
            raise ValueError('Overlay differs')
        overlay = read_json(overlay_path)
        capture = Path(overlay['frames'][0]['original_files']['camera.json']['path']).parents[2]
        spec_path = capture / 'source/spec.json'
        spec = read_json(spec_path)
        layout = next(x for x in spec['layouts'] if x['layout_id'] == item['layout_id'])
        clips = {x['id']: x for x in layout['clips']}
        assets = {x['id']: x['mesh_asset'] for x in spec['assets']}
        target_sources.add(assets[1])
        distractor_sources.add(assets[254])
        sources[str(spec_path)] = sha(spec_path)
        for frame in overlay['frames']:
            clip = clips[frame['clip_id']]
            camera = clip['poses'][frame['pose_index']]
            target = next(x for x in clip['insertions'] if x['id'] == 1)
            hidden = bool(target['hidden'])
            distance = None if hidden else float(np.linalg.norm(
                np.asarray(target['center_m'], dtype=np.float64) -
                np.asarray([camera[k] for k in ('x', 'y', 'z')], dtype=np.float64)))
            result[frame['frame_key']] = dict(layout_id=item['layout_id'],
                clip_id=frame['clip_id'], target_hidden=hidden,
                target_centre_distance_m=distance)
    return result, sources, sorted(target_sources), sorted(distractor_sources)


def distance_bin(value):
    if value is None:
        return 'target_hidden'
    if value < 1:
        return 'target_<1m'
    if value < 2:
        return 'target_1-2m'
    if value < 3:
        return 'target_2-3m'
    if value < 5:
        return 'target_3-5m'
    return 'target_>=5m'


def witness_depth_bin(value, count):
    if count == 0 or not np.isfinite(value):
        return 'no_visible_witness'
    if value < 1:
        return 'visible_<1m'
    if value < 2:
        return 'visible_1-2m'
    return 'visible_2-3m'


def edge_bin(value, count):
    if count == 0 or not np.isfinite(value):
        return 'no_visible_witness'
    if value < .1:
        return 'edge_fraction_<0.1'
    if value < .5:
        return 'edge_fraction_0.1-0.5'
    return 'edge_fraction_>=0.5'


def confusion(labels, logits, mask):
    known = labels >= 0
    positive = labels == 1
    negative = labels == 0
    predicted = logits >= 0
    return dict(tp=int(np.sum(mask & positive & predicted)),
                fp=int(np.sum(mask & negative & predicted)),
                fn=int(np.sum(mask & positive & ~predicted)),
                tn=int(np.sum(mask & negative & ~predicted)),
                unknown=int(np.sum(mask & ~known)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection-overlay', type=Path, required=True)
    p.add_argument('--partition-plan', type=Path, required=True)
    p.add_argument('--v2-result', type=Path, required=True)
    p.add_argument('--witness-stats', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if (sha(args.collection_overlay) != COLLECTION_SHA256 or
            sha(args.partition_plan) != PARTITION_SHA256 or
            sha(args.v2_result) != V2_RESULT_SHA256):
        raise ValueError('Frozen collection, partition or V2 result differs')
    data = read_inputs(args.collection_overlay, args.partition_plan)
    rows, labels, dev = data['rows'], data['labels'], data['dev']
    if tuple(data['query_names']) != QUERY_NAMES or labels.shape != (960, 6) or int(dev.sum()) != 480:
        raise ValueError('Expected six-query 960-frame alley data')
    with np.load(args.witness_stats, allow_pickle=False) as source:
        witness = {k: source[k].copy() for k in source.files}
    if (list(witness['frame_key'].astype(str)) != [row['frame_key'] for row in rows] or
            not np.array_equal(witness['dev'], dev) or
            witness['pixel_count'].shape != (960, 6)):
        raise ValueError('Witness stats do not align to Development frames')
    context, source_hashes, target_sources, distractor_sources = target_context(
        read_json(args.collection_overlay))
    dev_indices = np.flatnonzero(dev)
    if len(context) != 480 or any(rows[i]['frame_key'] not in context for i in dev_indices):
        raise ValueError('Missing context for dev frame')
    prior = read_json(args.v2_result)
    logits = {}
    prediction_hashes = {}
    for seed in SEEDS:
        logits[seed] = {}
        for arm in ('tof_only', 'cnh_rgb'):
            entry = next(x for x in prior['results'] if x['seed'] == seed and x['arm'] == arm)
            path = Path(entry['predictions_path'])
            if sha(path) != entry['predictions_sha256']:
                raise ValueError('Saved V2 prediction differs')
            prediction_hashes[f'{seed}:{arm}'] = sha(path)
            with np.load(path, allow_pickle=False) as saved:
                if (list(saved['frame_key'].astype(str)) != [row['frame_key'] for row in rows] or
                        not np.array_equal(saved['dev'], dev) or saved['logits'].shape != (960, 6)):
                    raise ValueError('V2 prediction identities differ')
                logits[seed][arm] = saved['logits'].copy()
            original = confusion(labels, logits[seed][arm], np.broadcast_to(dev[:, None], labels.shape))
            if any(original[k] != entry['dev'][k] for k in ('tp', 'fp', 'fn', 'tn')):
                raise ValueError('Recount differs from frozen V2 receipt')

    masks = {}
    matrix_dev = np.broadcast_to(dev[:, None], labels.shape)
    layout_group = np.array([row['layout_id'] for row in rows])
    clip_group = np.array([row['clip_id'] for row in rows])
    for layout in sorted(set(layout_group[dev])):
        masks[f'layout:{layout}'] = matrix_dev & (layout_group == layout)[:, None]
    for clip in sorted(set(clip_group[dev])):
        masks[f'clip:{clip}'] = matrix_dev & (clip_group == clip)[:, None]
    for query, name in enumerate(QUERY_NAMES):
        qmask = np.zeros_like(labels, dtype=bool)
        qmask[:, query] = True
        masks[f'query:{name}'] = matrix_dev & qmask
    target_distance = np.array([distance_bin(context[rows[i]['frame_key']]['target_centre_distance_m'])
                                if dev[i] else 'train' for i in range(len(rows))])
    for bucket in sorted(set(target_distance[dev])):
        masks[f'target_context_distance:{bucket}'] = matrix_dev & (target_distance == bucket)[:, None]
    positive_witness = matrix_dev & (labels == 1)
    depth_bucket = np.full(labels.shape, 'train', dtype=object)
    edge_bucket = np.full(labels.shape, 'train', dtype=object)
    for i in dev_indices:
        for query in range(6):
            count = int(witness['pixel_count'][i, query])
            depth_bucket[i, query] = witness_depth_bin(float(witness['min_z_m'][i, query]), count)
            edge_bucket[i, query] = edge_bin(float(
                witness['fraction_within_1deg_of_tof_zone_edge'][i, query]), count)
    for bucket in sorted(set(depth_bucket[positive_witness])):
        masks[f'positive_visible_depth:{bucket}'] = positive_witness & (depth_bucket == bucket)
    for bucket in sorted(set(edge_bucket[positive_witness])):
        masks[f'positive_zone_edge_fraction:{bucket}'] = positive_witness & (edge_bucket == bucket)
    slices = {}
    for name, mask in masks.items():
        slices[name] = dict(denominators=dict(positive=int(np.sum(mask & (labels == 1))),
                                              negative=int(np.sum(mask & (labels == 0))),
                                              unknown=int(np.sum(mask & (labels < 0)))),
                            seeds={str(seed): {arm: confusion(labels, logits[seed][arm], mask)
                                               for arm in ('tof_only', 'cnh_rgb')}
                                   for seed in SEEDS})
    dev_labels = labels[dev]
    groups = layout_group[dev]
    all_indices = np.arange(len(dev_labels))
    ranking = {}
    for seed in SEEDS:
        arms = {arm: logits[seed][arm][dev] for arm in ('tof_only', 'cnh_rgb')}
        ranking[str(seed)] = dict(total=bootstrap(dev_labels, arms, groups, all_indices),
                                 layouts={layout: bootstrap(dev_labels, arms, groups,
                                          all_indices[groups == layout]) for layout in sorted(set(groups))})
    result = dict(status='COMPLETE_READ_ONLY_ALLEY_V2_FAILURE_SLICES',
                  collection_sha256=sha(args.collection_overlay),
                  partition_sha256=sha(args.partition_plan),
                  v2_result_sha256=sha(args.v2_result),
                  witness_stats_sha256=sha(args.witness_stats),
                  source_spec_sha256=source_hashes,
                  prediction_sha256=prediction_hashes,
                  dev_frames=480, queries_per_frame=6,
                  target_source_families=target_sources,
                  distractor_source_families=distractor_sources,
                  category_limit='All dev inserted targets use one traffic-cone source; native and inserted triangle labels lack per-query object ID. Object category cannot be attributed for every FN/FP.',
                  size_limit='No independently varied target size family in three dev layouts; native object size has no per-query attribution. Thin-pole slice is NOT_ESTIMABLE.',
                  boundary_definition='For positive labels with visible witnesses only: fraction of witness pixels within 1 degree of any 8x8 ToF zone angular edge, bins <0.1, 0.1-0.5, >=0.5; negatives have no in-box witness and cannot be attributed this way.',
                  distance_definition='Positive visible depth uses nearest in-box visible witness axial Z; context distance is frozen target nominal 3D centre distance and does not assert target caused query label.',
                  bootstrap_limit='1000 frame resamples stratified within three fixed dev layouts; six queries resampled together; not independent-layout intervals.',
                  ranking=ranking, slices=slices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], slices=len(slices),
                          result_sha256=sha(args.output))))


if __name__ == '__main__':
    main()
