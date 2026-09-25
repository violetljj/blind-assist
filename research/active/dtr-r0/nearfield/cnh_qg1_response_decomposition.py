"""Frozen Development-only response decomposition; no training or capture.

This diagnoses one fixed geometric readout under an uncalibrated simulator.
It is not a hardware experiment or an additive information-loss decomposition.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from cnh_route_sensor import (
    H3, RAW_BINS, RAW_BIN_M, SensorParameters, _histogram, _pulse_matrix,
    _ray_inputs, synthesize_response, derive_readout,
)


COMPONENTS = ('RETURN_LAW', 'PULSE', 'NEIGHBOR', 'XTALK', 'SHOT_BACKGROUND')
CUMULATIVE = ('AREA4000',) + COMPONENTS
PARAMETERS = SensorParameters()


def exchange(signal, fraction):
    """Exactly simulator's symmetric nonwrapping four-neighbor exchange."""
    signal = signal.copy()
    old = signal.copy()
    leak = fraction / 4
    signal[..., 1:, :, :] += leak * (old[..., :-1, :, :] - old[..., 1:, :, :])
    signal[..., :-1, :, :] += leak * (old[..., 1:, :, :] - old[..., :-1, :, :])
    signal[..., :, 1:, :] += leak * (old[..., :, :-1, :] - old[..., :, 1:, :])
    signal[..., :, :-1, :] += leak * (old[..., :, 1:, :] - old[..., :, :-1, :])
    return signal


def aggregate(raw):
    """The implemented H3 bin sum; no quantizer, clipping or SNR mask."""
    return raw.reshape(8, 8, H3.bins, H3.sub_sample).sum(-1)


def raw_stage(radial, weights, enabled, seed, params=PARAMETERS):
    """Components always execute in actual simulator order, including isolates.

    Isolated XTALK retains its actual pulse-shaped residual kernel while scene
    returns remain impulses. Noise always draws ambient first, then counts.
    """
    enabled = frozenset(enabled)
    if not enabled.issubset(COMPONENTS):
        raise ValueError('Unknown response component')
    params.validate()
    distance, rho, cosine, area, valid = _ray_inputs(radial, .5, 1., weights)
    energy = params.signal_counts * area
    if 'RETURN_LAW' in enabled:
        safe = np.where(valid, distance, 1)
        energy = params.signal_counts * rho * cosine * area / np.maximum(safe, .05) ** 2
    histogram, _, _ = _histogram(distance, energy, valid, RAW_BINS, RAW_BIN_M,
                                  params.range_zero_m, 'numpy', 'cpu')
    matrix = _pulse_matrix(params)
    signal = histogram @ matrix if 'PULSE' in enabled else histogram
    if 'NEIGHBOR' in enabled:
        signal = exchange(signal, params.neighbour_leak)
    xtalk = np.zeros(RAW_BINS)
    if 'XTALK' in enabled:
        xtalk_bin = int(np.floor((params.crosstalk_range_m - params.range_zero_m) / RAW_BIN_M))
        if 0 <= xtalk_bin < RAW_BINS:
            xtalk = params.signal_counts * params.crosstalk_fraction * matrix[xtalk_bin]
    expectation = signal + xtalk
    if 'SHOT_BACKGROUND' in enabled:
        rng = np.random.default_rng(seed)
        ambient_estimate = rng.poisson(params.ambient_counts, size=expectation.shape)
        counts = rng.poisson(expectation + params.ambient_counts)
        signed = expectation + params.noise_scale * (counts - ambient_estimate - expectation)
    else:
        signed = expectation
    return signed * params.output_gain


def frame_stages(radial, weights, seed):
    result = {'AREA4000': aggregate(raw_stage(radial, weights, (), seed))}
    for end, component in enumerate(COMPONENTS, 1):
        result[component] = aggregate(raw_stage(radial, weights, COMPONENTS[:end], seed))
        result['ONLY_' + component] = aggregate(raw_stage(radial, weights, (component,), seed))
    return result


def checked_identity(manifest, row, frame):
    from cnh_street_e2e_materialize import frame_identity
    for name, field in [('camera.json', 'camera_sha256'), ('depth_left.exr', 'depth_sha256'),
                        ('depth_left_valid.npy', 'depth_valid_sha256')]:
        if row[field] != frame['original_files'][name]['sha256']:
            raise ValueError('Frozen row and overlay input hashes differ')
    if row['id'] != frame['frame_id']:
        raise ValueError('Raw frame id differs')
    identity, seed = frame_identity(manifest['source_manifest_sha256'], row,
                                    row['camera_sha256'], row['depth_sha256'])
    if identity != row['frame_key'] or identity != frame['frame_key'] or seed != row['seed']:
        raise ValueError('Frozen frame_identity or RNG seed differs')
    return seed


def metric_bundle(labels, scores, train, dev, layouts):
    from cnh_h3_geometry_diagnostic import summarize, train_threshold
    threshold = train_threshold(labels[train], scores[train])
    result = dict(threshold=threshold)
    for split, selection in [('train', train), ('dev', dev)]:
        result[split] = summarize(labels[selection], scores[selection], threshold)
        result[split + '_layouts'] = {
            name: summarize(labels[selection & (layouts == name)], scores[selection & (layouts == name)], threshold)
            for name in sorted(set(layouts[selection]))}
    return result


def scale_diagnostic(labels, original, factor, train, dev, layouts, baseline_metrics):
    """Report finite-precision rank/tie effects; never change scientific scores."""
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('Positive finite scale required')
    scaled = original * factor
    scaled_metrics = metric_bundle(labels, scaled, train, dev, layouts)
    result = dict(factor=factor, numerical_invariance_verified=True, splits={})
    for split, selection in [('train', train), ('dev', dev)]:
        details = {}
        for query in range(original.shape[1]):
            selected = selection & (labels[:, query] >= 0)
            base, changed = original[selected, query], scaled[selected, query]
            order = np.argsort(base, kind='stable')
            changed_order = np.argsort(changed, kind='stable')
            a, b = base[order], changed[order]
            original_ties = a[1:] == a[:-1]
            scaled_ties = b[1:] == b[:-1]
            details[str(query)] = dict(
                known=int(selected.sum()), raw_score_different=int(np.count_nonzero(base != changed)),
                roundtrip_score_different=int(np.count_nonzero(base != changed / factor)),
                stable_sort_positions_different=int(np.count_nonzero(order != changed_order)),
                adjacent_original_order_ties_changed=int(np.count_nonzero(original_ties != scaled_ties)),
                adjacent_original_order_inversions=int(np.count_nonzero(b[1:] < b[:-1])),
                original_unique_scores=int(len(np.unique(base))), scaled_unique_scores=int(len(np.unique(changed))))
        # Pooled metrics also compare scores across different queries, so pooled
        # ordering/ties must be checked in addition to per-query diagnostics.
        known = labels[selection] >= 0
        base, changed = original[selection][known], scaled[selection][known]
        order = np.argsort(base, kind='stable')
        changed_order = np.argsort(changed, kind='stable')
        pooled = dict(stable_sort_positions_different=int(np.count_nonzero(order != changed_order)),
                      adjacent_original_order_ties_changed=int(np.count_nonzero(
                          (base[order][1:] == base[order][:-1]) != (changed[order][1:] == changed[order][:-1]))))
        values = {}
        for metric in ('auprc', 'auroc'):
            before, after = baseline_metrics[split][metric], scaled_metrics[split][metric]
            values[metric] = dict(original=before, scaled=after, difference=after - before)
        invariant = (all(v['difference'] == 0 for v in values.values()) and
                     all(v == 0 for v in pooled.values()) and
                     all(d['stable_sort_positions_different'] == d['adjacent_original_order_ties_changed'] ==
                         d['adjacent_original_order_inversions'] == 0 for d in details.values()))
        result['splits'][split] = dict(metrics=values, per_query=details, pooled=pooled,
                                        numerical_invariance_verified=invariant)
        result['numerical_invariance_verified'] &= invariant
    return result


def run(collection, partition, prepared, output, protocol, protocol_sha256):
    from cnh_rgb_dev_comparison import read_inputs, checked, sha
    from cnh_rgb_alley_v2 import COLLECTION_SHA256, PARTITION_SHA256
    from cnh_rgb_visible_depth_audit import load_scene_depth, perfect_h3_histogram
    from cnh_street_development_baseline import sample_depth
    from cnh_h3_geometry_diagnostic import query_weights

    started = time.monotonic()
    if sha(protocol) != protocol_sha256:
        raise ValueError('Frozen protocol hash differs')
    if sha(collection) != COLLECTION_SHA256 or sha(partition) != PARTITION_SHA256:
        raise ValueError('Frozen six-layout Development inputs differ')
    if output.exists():
        raise FileExistsError(output)
    data = read_inputs(collection, partition)
    cache_path = prepared / 'perfect-tof-h3.npz'
    receipt_path = prepared / 'result.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8-sig'))
    if sha(cache_path) != receipt['files'][cache_path.name]:
        raise ValueError('Perfect area cache hash differs')
    keys = [row['frame_key'] for row in data['rows']]
    with np.load(cache_path, allow_pickle=False) as cache:
        if list(cache['frame_key'].astype(str)) != keys:
            raise ValueError('Perfect area frame order differs')
        perfect = cache['histogram'].copy()
    if perfect.shape != (960, 64, 16):
        raise ValueError('Unexpected perfect H3 shape')
    frames, manifests = [], []
    for item in data['collection']['layouts']:
        overlay = json.loads(checked(dict(path=item['overlay'], sha256=item['overlay_sha256'])).read_text(encoding='utf-8-sig'))
        manifest = json.loads(checked(overlay['materialized_manifest']).read_text(encoding='utf-8-sig'))
        frames.extend(overlay['frames'])
        manifests.extend([manifest] * len(overlay['frames']))
    if [frame['frame_key'] for frame in frames] != keys:
        raise ValueError('Overlay frame order differs')
    weights = query_weights()
    project = lambda histogram: np.einsum('zb,qzb->q', histogram.reshape(64, 16).astype(np.float64), weights)
    names = CUMULATIVE + tuple('ONLY_' + c for c in COMPONENTS)
    scores = {name: np.empty((960, 6), dtype=np.float64) for name in names}
    scores['REFERENCE_PERFECT_AREA'] = np.einsum('nzb,qzb->nq', perfect.astype(np.float64), weights)
    scores['REFERENCE_ORIGINAL_SIGNED'] = np.einsum('nzb,qzb->nq', data['histogram'].astype(np.float64), weights)
    scores['CONTROL_REGENERATED_AREA4000'] = np.empty((960, 6), dtype=np.float64)
    parity = []
    output.mkdir(parents=True)
    try:
        for index, (frame, manifest, row) in enumerate(zip(frames, manifests, data['rows'])):
            seed = checked_identity(manifest, row, frame)
            depth, camera = load_scene_depth(frame)
            if not np.allclose(camera['T_camera_tof'], np.eye(4), atol=1e-12):
                raise ValueError('Frozen geometric projection requires aligned collocated camera/ToF')
            radial, area = sample_depth(depth, camera, 16)
            regenerated_perfect, _, _ = perfect_h3_histogram(depth, camera)
            if not np.array_equal(regenerated_perfect, perfect[index]):
                raise ValueError('Exact perfect cache regeneration failed; source/sampling confound')
            stages = frame_stages(radial, area, seed)
            endpoint = stages['SHOT_BACKGROUND'].astype(np.float32).reshape(64, 16)
            original = data['histogram'][index]
            # Hard integrity gate before any decomposition dev metrics are computed.
            if not np.array_equal(endpoint, original):
                difference = float(np.max(np.abs(endpoint.astype(np.float64) - original)))
                unequal = int(np.count_nonzero(endpoint != original))
                raise ValueError(f'Original full sensor endpoint bit parity failed at frame {index}, '
                                 f'seed {seed}, unequal cells {unequal}, max abs {difference}; stop attribution')
            area_error = float(np.max(np.abs(stages['AREA4000'].reshape(64, 16) / PARAMETERS.signal_counts - perfect[index])))
            if area_error > 2e-7:
                raise ValueError('Raw128/area16 geometry bridge differs beyond frozen float32 tolerance')
            for name in names:
                # Keep sensor storage precision consistent with immutable endpoint.
                scores[name][index] = project(stages[name].astype(np.float32))
            scores['CONTROL_REGENERATED_AREA4000'][index] = scores['AREA4000'][index]
            # Exact preexisting area ranking; multiplication follows score reduction
            # so finite-precision summation cannot break tied scores.
            scores['AREA4000'][index] = scores['REFERENCE_PERFECT_AREA'][index] * PARAMETERS.signal_counts
            parity.append(dict(frame_key=keys[index], seed=seed, perfect_cache_bit_equal=True,
                               original_signed_h3_bit_equal=True, area_bridge_max_abs=area_error))
            if (index + 1) % 160 == 0:
                print(json.dumps(dict(part_b_frames=index + 1, total=960)), flush=True)
        layouts = np.asarray([row['layout_id'] for row in data['rows']])
        results = {name: metric_bundle(data['labels'], values, data['train'], data['dev'], layouts)
                   for name, values in scores.items()}
        # Persist fixed predictions and their metrics before optional numerical
        # diagnostics, so a reporting check cannot erase completed evaluation.
        np.savez_compressed(output / 'scores.npz', frame_key=keys, train=data['train'], dev=data['dev'], weights=weights, **scores)
        (output / 'metrics.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        scale_checks = {}
        for name, factor in [('positive_global_gain_7', 7.), ('uniform_rho_half_without_range_law', .5)]:
            scale_checks[name] = scale_diagnostic(data['labels'], scores['AREA4000'], factor,
                                                   data['train'], data['dev'], layouts, results['AREA4000'])
        drops = {}
        for split in ('train', 'dev'):
            base = results['AREA4000'][split]['auprc']
            isolated = {c: base - results['ONLY_' + c][split]['auprc'] for c in COMPONENTS}
            full = base - results['SHOT_BACKGROUND'][split]['auprc']
            drops[split] = dict(full_drop=full, isolated_drops=isolated,
                                interaction_residual=full - sum(isolated.values()),
                                cumulative_ap_changes={c: results[CUMULATIVE[i]][split]['auprc'] - results[c][split]['auprc']
                                                       for i, c in enumerate(COMPONENTS)})
        result = dict(status='COMPLETE_FROZEN_RESPONSE_DECOMPOSITION_DEVELOPMENT_ONLY', results=results,
                      parameters=asdict(PARAMETERS), cumulative_order=list(CUMULATIVE), isolated_components=list(COMPONENTS),
                      positive_scale_checks=scale_checks, ap_drop_diagnostics=drops,
                      endpoint_integrity=dict(frames=960, perfect_cache_bit_equal=True, original_signed_h3_bit_equal=True,
                                              area_bridge_max_abs=max(p['area_bridge_max_abs'] for p in parity), frames_detail=parity),
                      formula='E=4000*rho*cos*w/max(radial,.05)^2; rho=.5 cos=1; normalized solid-angle weights; pulse -> neighbor -> residual xtalk -> Poisson counts minus independent ambient estimate; H3 sums eight raw bins; signed outputs retained',
                      area_bridge='Cached perfect float32 score multiplied by 4000 preserves exact historical AP; raw128 regenerated area is a separate numerical parity control, using identical sample_depth16 inputs and normalization. Isolates use raw128 counts then float32 H3 storage.',
                      isolated_xtalk='Residual xtalk uses the actual default pulse kernel even when scene pulse component is disabled',
                      threshold_selection='Pooled known train F1 only, tie higher threshold, separately per stage; dev never tunes',
                      not_implemented=['QUANTIZATION', 'CLIPPING'],
                      limits='One fixed geometric score, uncalibrated simulator and three fixed layouts per split; component AP changes are order-dependent interactions, not additive causal information loss or information-theoretic limits. No training/capture/test/City.',
                      protocol_sha256=protocol_sha256, collection_sha256=sha(collection), partition_sha256=sha(partition),
                      mechanical_erratum_sha256=sha(Path(__file__).with_name('CNH_QG1_RESPONSE_ERRATUM_20260925.md')),
                      prepared_receipt_sha256=sha(receipt_path), perfect_h3_sha256=sha(cache_path),
                      source_code_sha256={name: sha(Path(__file__).with_name(name)) for name in
                                          ('cnh_qg1_response_decomposition.py', 'cnh_route_sensor.py', 'cnh_street_development_baseline.py',
                                           'cnh_rgb_visible_depth_audit.py', 'cnh_h3_geometry_diagnostic.py', 'cnh_street_e2e_materialize.py')},
                      scores_sha256=sha(output / 'scores.npz'), metrics_sha256=sha(output / 'metrics.json'),
                      compute='NumPy CPU; fixed 8x8x128 response arithmetic; no training', wall_s=time.monotonic()-started)
        (output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        return result
    except Exception as error:
        (output / 'failure.json').write_text(json.dumps(dict(status='STOP_ATTRIBUTION_INTEGRITY_FAILURE', error=str(error),
                                                             completed_frames=len(parity), protocol_sha256=protocol_sha256), indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection-overlay', type=Path, required=True)
    parser.add_argument('--partition-plan', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    result = run(args.collection_overlay, args.partition_plan, args.prepared, args.output, args.protocol, args.protocol_sha256)
    print(json.dumps(dict(status=result['status'], output=str(args.output)), indent=2))
