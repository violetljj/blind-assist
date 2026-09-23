"""Post-seal historical stereo geometry evaluation; no predictor sees truth.

Uses the exact historical common-FOV, ToF union, voxel support and 2-on/2-off
readout. Native depth is evaluator-only camera-forward depth, not range. Pixel
attribution is not voxel attribution; task annotations include occluded objects.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import mz101_spatial as m
from run_mz101_spatial import metrics, observation_contract, sha
from run_mz103_depth_frontend import PANELS, event_compare, write

ROOT = Path(__file__).resolve().parents[4]
WORK = ROOT / 'artifacts.local/work'
REFERENCE = WORK / 'mz103-depth-frontend-20260912/native-v1'
TAO = WORK / 'mz104-foundation-stereo-20260912/frontend-v1'
TAO_EVAL = WORK / 'mz104-foundation-stereo-20260912/evaluation-v1'
SGBM_SEAL = WORK / 'mz105-residual-matching-20260912/features-v3/seal.json'
CRITICAL = ('thin_left', 'thin_right', 'small_head', 'occluded_thin')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(folder):
    receipt = read(folder / 'receipt.json')
    assert receipt['status'] == 'PASS', folder
    for name, digest in receipt['hashes'].items():
        path = folder / name.replace('\\', '/')
        assert path.resolve().is_relative_to(folder.resolve()), name
        assert sha(path) == digest, path
    return receipt


def eligible(depth):
    assert depth.shape == (m.RIG['height'], m.RIG['width'])
    return np.where(np.isfinite(depth) & (depth >= m.MIN_DEPTH)
                    & (depth <= m.MAX_DEPTH), depth, np.nan)


def membership(depth, pose):
    """Pixel-space query mask, exactly matching readout's original points."""
    yy, xx = np.nonzero(np.isfinite(depth))
    points = m.depth_points(depth)
    fov = ((np.abs(np.degrees(np.arctan2(points[:, 1], points[:, 0]))) <= 22.5)
           & (np.abs(np.degrees(np.arctan2(points[:, 2], points[:, 0]))) <= 20.))
    body = points @ m.rotation(pose['yaw'], pose['pitch'], pose.get('roll', 0.)).T
    body += np.asarray(pose['camera_in_body_m'])
    result = np.zeros((*depth.shape, 2), bool)
    result[yy, xx] = np.stack([fov & ((body >= lo) & (body <= hi)).all(1)
                              for lo, hi in m.BOXES], 1)
    return result


def geometry_rows(predicted, native, pose):
    """All numerators retain a fixed native-query denominator incl missingness."""
    predicted = eligible(predicted)
    pin, nin = membership(predicted, pose), membership(eligible(native), pose)
    finite = np.isfinite(predicted)
    near = np.isfinite(native) & (native >= .5) & (native <= 4.)
    far = np.isfinite(native) & (native > 4.)
    error = np.abs(predicted - native)
    rows = []
    for j in range(2):
        p, n = pin[:, :, j], nin[:, :, j]
        origins = dict(pixels=int(p.sum()), native_in_query=int((p & n).sum()),
            native_far=int((p & far).sum()), native_near_outside=int((p & near & ~n).sum()),
            other=int((p & ~near & ~far).sum()))
        assert sum(v for k, v in origins.items() if k != 'pixels') == origins['pixels']
        coverage = dict(native_query_pixels=int(n.sum()), predicted_valid=int((n & finite).sum()),
            missing=int((n & ~finite).sum()), predicted_in_query=int((n & p).sum()))
        valid_error = error[n & finite]
        conditional = dict(count=int(valid_error.size), abs_z_sum_m=float(valid_error.sum(dtype=np.float64)))
        for cm in (5, 10, 20):
            hit = n & finite & (error <= cm / 100.)
            coverage[f'abs_z_{cm}cm'] = int(hit.sum())
            coverage[f'in_query_and_abs_z_{cm}cm'] = int((hit & p).sum())
        rows.append(dict(origin=origins, coverage=coverage, conditional_error=conditional))
    return rows


def aggregate_geometry(rows):
    out = {}
    for model in ('sgbm', 'tao', 'candidate'):
        selected = [r for r in rows if r['model'] == model]
        slices = {'all': selected}
        slices.update({f: [r for r in selected if r['family'] == f] for f in CRITICAL})
        slices.update({p: [r for r in selected if r['part'] == p] for p in m.PARTS})
        out[model] = {}
        for name, group in slices.items():
            origin, coverage = Counter(), Counter()
            query, conditional = Counter(), Counter()
            for r in group:
                origin.update(r['origin']); coverage.update(r['coverage'])
                conditional.update(r['conditional_error'])
                if r['origin']['pixels']:
                    label = 'task_tp' if r['truth'] else 'task_fp'
                    query[label] += 1
                    query[label + '_without_native_in_query'] += r['origin']['native_in_query'] == 0
                    query[label + '_all_far'] += r['origin']['native_far'] == r['origin']['pixels']
            denom = coverage['native_query_pixels']
            ratios = {k: v / denom if denom else None for k, v in coverage.items()
                      if k != 'native_query_pixels'}
            out[model][name] = dict(origin_pixels=dict(origin), coverage=dict(coverage),
                coverage_rates=ratios, supported_queries=dict(query),
                conditional_error=dict(conditional, mean_abs_z_m=(conditional['abs_z_sum_m']/conditional['count']
                    if conditional['count'] else None)))
    return out


def evaluate(candidate, output, inputs, reference=REFERENCE, tao=TAO, tao_eval=TAO_EVAL):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Fresh output required; preserve consumed evaluations')
    assert output.resolve().is_relative_to((ROOT / 'artifacts.local').resolve())
    receipts = {name: verify(path) for name, path in
                [('candidate', candidate), ('reference', reference), ('tao', tao), ('tao_eval', tao_eval)]}
    manifest = read(inputs)
    assert manifest['rig'] == m.RIG and len(manifest['frames']) == 576
    assert receipts['candidate']['frames'] == 576
    assert receipts['candidate']['input_manifest_sha256'] == sha(inputs)
    hashes = {k.replace('\\', '/'): v for k, v in receipts['candidate']['hashes'].items()}
    # MZ105 independently recomputed unchanged SGBM and sealed exact cache hashes.
    sgbm_hashes = {k.replace('\\', '/'): v for k, v in read(SGBM_SEAL)['inputs'].items()}
    output.mkdir(parents=True, exist_ok=True)
    frozen = {}
    # Freeze every candidate panel before opening task/native evaluator truth.
    for panel, (folder, old_dir, depth_dir, _) in PANELS.items():
        obs = read(reference / panel / 'observations.json')
        entries = [f for f in manifest['frames'] if f['panel'] == panel]
        assert [f['id'] for f in entries] == [o['id'] for o in obs]
        capture = WORK / folder / 'capture-v1'
        captured = read(capture / 'receipt.json')
        old = np.load(reference / panel / 'predictions.npz')
        historical = np.load(tao_eval / panel / 'predictions.npz')
        support = {k: [] for k in ('tof', 'sgbm', 'tao', 'candidate')}
        depth_paths = []
        for o, entry in zip(obs, entries):
            frame = capture / 'frame' / o['id']
            for view in ('left', 'right'):
                assert sha(Path(entry[view])) == entry[view + '_sha256'] == captured['hashes'][f"frame/{o['id']}/{view}.png"]
            for filename in ('tof-range.npy', 'tof-valid.npy'):
                assert sha(frame / filename) == captured['hashes'][f"frame/{o['id']}/{filename}"]
            depths = {'sgbm': WORK / folder / old_dir / depth_dir / (o['id'] + '.npy'),
                      'tao': tao / panel / 'depth' / (o['id'] + '.npy'),
                      'candidate': candidate / panel / 'depth' / (o['id'] + '.npy')}
            rel = f"{panel}/depth/{o['id']}.npy"
            assert rel in hashes and sha(depths['candidate']) == hashes[rel]
            assert sha(depths['sgbm']) == sgbm_hashes[depths['sgbm'].relative_to(ROOT).as_posix()]
            support['tof'].append(m.readout(m.tof_points(np.load(frame / 'tof-range.npy'),
                np.load(frame / 'tof-valid.npy')), o['pose'], common_fov=True)[0])
            for model, path in depths.items():
                support[model].append(m.readout(m.depth_points(eligible(np.load(path))), o['pose'], common_fov=True)[0])
            depth_paths.append(depths)
        support = {k: np.asarray(v) for k, v in support.items()}
        for model in ('sgbm', 'tao', 'candidate'):
            support[model + '_union'] = support[model] + support['tof']
        final = {k: m.hysteresis(v, [o['episode'] for o in obs]) for k, v in support.items()}
        for model in ('tof', 'sgbm', 'sgbm_union'):
            np.testing.assert_array_equal(support[model], old[model + '_support'])
            np.testing.assert_array_equal(final[model], old[model])
        for model, saved in [('tao', 'foundation'), ('tao_union', 'foundation_union')]:
            np.testing.assert_array_equal(support[model], historical[saved + '_support'])
            np.testing.assert_array_equal(final[model], historical[saved])
        dest = output / panel; dest.mkdir()
        np.savez_compressed(dest / 'predictions.npz', **final,
                            **{k + '_support': v for k, v in support.items()})
        write(dest / 'prediction-seal.json', dict(predictions_sha256=sha(dest / 'predictions.npz'),
            candidate_receipt_sha256=sha(candidate / 'receipt.json'), baseline_parity=True,
            input_manifest_sha256=sha(inputs), depth_hashes={str(p): sha(p)
                for paths in depth_paths for p in paths.values()}))
        frozen[panel] = (obs, capture, captured, support, final, depth_paths)
    panels, all_rows = {}, []
    for panel, (obs, capture, captured, support, final, paths) in frozen.items():
        spec_path = capture / 'spec.json'
        assert sha(spec_path) == captured['spec_sha256']
        spec = read(spec_path)
        assert observation_contract(spec) == obs
        gt = np.load(reference / panel / 'truth.npy')
        raw = {k: v > 0 for k, v in support.items()}
        scores = {stage: {k: metrics(v, gt, obs) for k, v in predictions.items()}
                  for stage, predictions in [('raw', raw), ('final', final)]}
        rows = []
        for i, o in enumerate(obs):
            native_path = capture / 'frame' / o['id'] / 'native-left-depth.npy'
            assert sha(native_path) == captured['hashes'][f"frame/{o['id']}/native-left-depth.npy"]
            native = np.load(native_path)
            for model, path in paths[i].items():
                for j, row in enumerate(geometry_rows(np.load(path), native, o['pose'])):
                    rows.append(dict(panel=panel, id=o['id'], model=model, part=m.PARTS[j],
                        family=spec['frames'][i]['family'], truth=bool(gt[i, j]),
                        tof_support=int(support['tof'][i, j]),
                        raw_union=bool(raw[model + '_union'][i, j]),
                        final_union=bool(final[model + '_union'][i, j]), **row))
        all_rows.extend(rows)
        summary = dict(scores=scores, geometry=aggregate_geometry(rows),
            event_comparison={k: event_compare(scores['final']['candidate_union'], scores['final'][k + '_union'])
                              for k in ('sgbm', 'tao')},
            slices={family: {stage: {k: metrics(v, gt, obs, [f['family'] == family for f in spec['frames']])
                for k, v in preds.items()} for stage, preds in [('raw', raw), ('final', final)]}
                for family in dict.fromkeys(f['family'] for f in spec['frames'])})
        write(output / panel / 'summary.json', summary)
        panels[panel] = summary
    pooled = {stage: {model: {key: sum(p['scores'][stage][model][key] for p in panels.values())
        for key in ('TP', 'FP', 'FN', 'events', 'missed_events', 'false_sessions', 'false_segments')}
        for model in ('tof', 'sgbm', 'sgbm_union', 'tao', 'tao_union', 'candidate', 'candidate_union')}
        for stage in ('raw', 'final')}
    for stage, model, expected in [('raw', 'sgbm_union', (435, 62, 11)), ('final', 'sgbm_union', (402, 40, 44)),
                                 ('raw', 'tao_union', (437, 273, 9)), ('final', 'tao_union', (399, 242, 47))]:
        assert tuple(pooled[stage][model][k] for k in ('TP', 'FP', 'FN')) == expected
    write(output / 'queries.json', all_rows)
    write(output / 'summary.json', dict(evidence='CONSUMED_RENDERED_DEVELOPMENT', frames=576,
        baseline_parity=True, pooled_descriptive=pooled, geometry=aggregate_geometry(all_rows),
        definitions=dict(coverage_denominator='All visible native in-query eligible pixels; missing predictions fail',
            abs_z='Camera-forward depth error in metres, not Euclidean range',
            attribution='Original pixels; support readout separately deduplicates 5cm voxels',
            supported_queries='Raw depth-only support positive queries; not final ToF-union alerts',
            conditional_error='MAE over native-query pixels with valid eligible predictions; read with fixed-denominator accuracy and coverage',
            missing='Nonfinite or outside historical 0.5..4m eligibility; UNKNOWN, not clear',
            task_truth='Historical object-box overlap; may be positive without visible native-query pixels'),
        sources={name: sha(path / 'receipt.json') for name, path in
                 [('candidate', candidate), ('reference', reference), ('tao', tao), ('tao_eval', tao_eval)]},
        sgbm_cache_seal_sha256=sha(SGBM_SEAL),
        backend=dict(device='CPU', reason='FROZEN_PROTOCOL_CPU_ONLY',
                     scope='unchanged historical geometry readout and post-seal evaluator'),
        evaluator_sha256=sha(Path(__file__))))
    write(output / 'receipt.json', dict(status='PASS', hashes={p.relative_to(output).as_posix(): sha(p)
        for p in output.rglob('*') if p.is_file()}))
    print(json.dumps(pooled, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=REFERENCE)
    parser.add_argument('--tao', type=Path, default=TAO)
    parser.add_argument('--tao-evaluation', type=Path, default=TAO_EVAL)
    args = parser.parse_args()
    evaluate(args.candidate, args.output, args.inputs, args.reference, args.tao, args.tao_evaluation)
