"""Fixed equal-logit ensemble of saved MZ1 outputs; no fitting or inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

EVENTS = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def masks(pred, truth):
    body, head = truth[:, :2].any(1), truth[:, 2:].any(1)
    return dict(wrong_far=pred[:, [1, 3]] & truth[:, [0, 2]] & ~truth[:, [1, 3]],
        cross_body=np.stack((pred[:, 2:].any(1) & body & ~head,
                             pred[:, :2].any(1) & head & ~body), axis=1))


def metric(pred, truth):
    body, head = truth[:, :2].any(1), truth[:, 2:].any(1)
    errors = masks(pred, truth)
    count = lambda n, d: dict(numerator=int(n), denominator=int(d))
    return dict(spatial_exact=count((pred == truth).all(1).sum(), len(truth)),
        wrong_far=count(errors['wrong_far'].sum(), (truth[:, [0, 2]] & ~truth[:, [1, 3]]).sum()),
        body_to_head=count(errors['cross_body'][:, 0].sum(), (body & ~head).sum()),
        head_to_body=count(errors['cross_body'][:, 1].sum(), (head & ~body).sum()),
        event_confusion={name: dict(tp=int((pred[:, k] & truth[:, k]).sum()),
            fp=int((pred[:, k] & ~truth[:, k]).sum()), fn=int((~pred[:, k] & truth[:, k]).sum()),
            tn=int((~pred[:, k] & ~truth[:, k]).sum())) for k, name in enumerate(EVENTS)})


def paired(candidate, baseline, truth, ids):
    c, b = (candidate == truth).all(1), (baseline == truth).all(1)
    ce, be = masks(candidate, truth), masks(baseline, truth)
    return dict(n=len(ids), gained=int((c & ~b).sum()), lost=int((~c & b).sum()),
        both_correct=int((c & b).sum()), both_wrong=int((~c & ~b).sum()),
        gained_frame_ids=ids[c & ~b].tolist(), lost_frame_ids=ids[~c & b].tolist(),
        errors={kind: dict(introduced=int((ce[kind] & ~be[kind]).sum()),
            removed=int((~ce[kind] & be[kind]).sum()), retained=int((ce[kind] & be[kind]).sum()))
            for kind in ce})


def run(root, output):
    started = time.perf_counter()
    root, output = Path(root).resolve(), Path(output).resolve()
    work = root/'artifacts.local/work'
    allowed = (work/'mz5-fixed-ensemble-20260910').resolve()
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError('Output must be a child of canonical MZ5 artifact root')
    if output.exists():
        raise FileExistsError('Fresh output required; preserve completed/failed attempts')
    paths = dict(predictions=work/'mz1-tiny-fusion-20260910/run-v1/predictions.npz',
        mz1_receipt=work/'mz1-tiny-fusion-20260910/run-v1/receipt.json',
        mz1_metrics=work/'mz1-tiny-fusion-20260910/run-v1/metrics.json',
        features=work/'mz1-tiny-fusion-20260910/cache-v1/features.npz',
        feature_receipt=work/'mz1-tiny-fusion-20260910/cache-v1/features-receipt.json',
        index=work/'body-query-5000-20260909/dataset-v1/index.json',
        mz0=work/'mz0-clean-20260910/run-v1/scored.npz',
        mz0_result=work/'mz0-clean-20260910/run-v1/result.json',
        mz3=work/'mz3-error-attribution-20260910/run-v1/result.json',
        source=Path(__file__).resolve())
    for arm in ('RGB_ONLY', 'TOF_ONLY', 'FUSION'):
        paths[arm] = work/f'mz1-tiny-fusion-20260910/run-v1/{arm}.pt'
    hashes = {key: sha(path) for key, path in paths.items()}
    output.mkdir(parents=True)
    # Saved before loading predictions or computing this diagnostic's outcomes.
    protocol = dict(status='FROZEN', utc=datetime.now(timezone.utc).isoformat(),
        method='Exactly 0.5 * (RGB_ONLY_logits + TOF_ONLY_logits); per-event threshold >= 0',
        weights=[0.5, 0.5], threshold=0, training_steps=0, model_forward_passes=0,
        scope='All 1500 consumed EVAL_ONLY rows; no tuning, selection, routing, or new information',
        comparisons=['FUSION', 'RGB_ONLY', 'TOF_ONLY', 'MZ0'],
        strata=['condition', 'condition_range', 'region_id', 'HEAD_NEAR_truth', 'negative_conditions'],
        known_consumed_expectation=dict(both_unimodal_correct_rows=1192),
        required_checks=['original hash/row/alert parity', 'original aggregate reproduction',
            'both-correct preservation', 'independent scalar scoring', 'actual checkpoint parameter counts'],
        backend='CPU', placement_reason='TASK_NOT_GPU_SUITABLE', input_sha256=hashes)
    write(output/'protocol.json', protocol)
    try:
        receipt, fr, m0 = read(paths['mz1_receipt']), read(paths['feature_receipt']), read(paths['mz0_result'])
        assert hashes['predictions'] == receipt['predictions_sha256']
        assert hashes['mz1_metrics'] == receipt['metrics_sha256']
        assert hashes['features'] == receipt['feature_sha256'] == fr['feature_sha256']
        assert hashes['index'] == fr['source_index_sha256'] == m0['source_index_sha256']
        assert hashes['mz0'] == m0['scored_sha256']
        for arm in ('RGB_ONLY', 'TOF_ONLY', 'FUSION'):
            assert hashes[arm] == receipt['arms'][arm]['checkpoint_sha256']
        data, features, a0 = load(paths['predictions']), load(paths['features']), load(paths['mz0'])
        rows = read(paths['index'])['frames']
        roles = np.array([row['source_role'] for row in rows])
        truth_all = np.array([np.array(row['counts']).reshape(4, 3).sum(1) >= 3 for row in rows])
        ev = roles == 'EVAL_ONLY'; selected = [row for row in rows if row['source_role'] == 'EVAL_ONLY']
        assert len(selected) == 1500
        np.testing.assert_array_equal(data['role'], roles)
        np.testing.assert_array_equal(data['truth'], truth_all)
        np.testing.assert_array_equal(data['original_alerts'], features['original_alerts'])
        truth = truth_all[ev]; ids = np.array([row['frame_id'] for row in selected])
        np.testing.assert_array_equal(a0['truth'], truth)
        np.testing.assert_array_equal(a0['original_alerts'], data['original_alerts'][ev])
        for arm in ('RGB_ONLY', 'TOF_ONLY', 'FUSION'):
            if arm+'_logits' not in data:
                raise ValueError('Saved logits unavailable; report before any inference')
            assert data[arm+'_logits'].shape == (5000, 4)
            assert np.isfinite(data[arm+'_logits']).all()
            np.testing.assert_array_equal(data[arm+'_logits'] >= 0, data[arm+'_flags'])
        rgb, tof = data['RGB_ONLY_logits'][ev], data['TOF_ONLY_logits'][ev]
        combine_started = time.perf_counter()
        logits = 0.5 * (rgb + tof)
        ensemble = logits >= 0
        combine_seconds = time.perf_counter()-combine_started
        pred = {arm: data[arm+'_flags'][ev] for arm in ('RGB_ONLY', 'TOF_ONLY', 'FUSION')}
        pred.update(MZ0=a0['z8_multi_surface'], ENSEMBLE=ensemble)
        mz1, mz3 = read(paths['mz1_metrics']), read(paths['mz3'])
        original_checks = 0
        for arm in ('RGB_ONLY', 'TOF_ONLY', 'FUSION'):
            for role in ('TRAIN_ONLY', 'DEV_ONLY', 'EVAL_ONLY'):
                result = metric(data[arm+'_flags'][roles == role], truth_all[roles == role])
                assert all(value == mz1['arms'][arm][role][key] for key, value in result.items())
                original_checks += 1
        for arm, key in (('RGB_ONLY', 'RGB'), ('TOF_ONLY', 'TOF'), ('FUSION', 'FUSION'), ('MZ0', 'MZ0')):
            result = metric(pred[arm], truth)
            assert all(value == mz3['aggregate'][key][name] for name, value in result.items())
        both_correct = (pred['RGB_ONLY'] == truth).all(1) & (pred['TOF_ONLY'] == truth).all(1)
        assert both_correct.sum() == 1192
        assert (ensemble[both_correct] == truth[both_correct]).all()
        identical = (pred['RGB_ONLY'] == pred['TOF_ONLY']).all(1)
        assert (ensemble[identical] == pred['RGB_ONLY'][identical]).all()
        strata = {}
        for key in ('condition', 'region_id'):
            vals = np.array([row[key] for row in selected])
            strata.update({key+'/'+v: vals == v for v in sorted(set(vals))})
        vals = np.array([row['condition']+'/'+row['declared_range'] for row in selected])
        strata.update({'condition_range/'+v: vals == v for v in sorted(set(vals))})
        strata['HEAD_NEAR_truth'] = truth[:, 2] & ~truth[:, 3]
        strata['negative_conditions'] = np.array([row['condition'] not in ('BODY_ONLY', 'HEAD_ONLY', 'BOTH') for row in selected])
        summary = dict(aggregate={arm: metric(p, truth) for arm, p in pred.items()},
            comparisons={arm: paired(ensemble, p, truth, ids) for arm, p in pred.items() if arm != 'ENSEMBLE'},
            strata={key: dict(n=int(mask.sum()), metrics={arm: metric(p[mask], truth[mask]) for arm, p in pred.items()},
                comparisons={arm: paired(ensemble[mask], p[mask], truth[mask], ids[mask])
                             for arm, p in pred.items() if arm != 'ENSEMBLE'}) for key, mask in strata.items()},
            both_unimodal_correct=dict(n=int(both_correct.sum()), ensemble_correct=int((ensemble[both_correct] == truth[both_correct]).all(1).sum()),
                fusion_correct=int((pred['FUSION'][both_correct] == truth[both_correct]).all(1).sum())),
            identical_unimodal_wrong=int((identical & ~both_correct).sum()))
        # Read weights only to count actual stored parameters; never build/forward a model.
        import torch
        counts = {}
        for arm in ('RGB_ONLY', 'TOF_ONLY', 'FUSION'):
            weights = torch.load(paths[arm], map_location='cpu', weights_only=True)
            counts[arm] = dict(parameters=sum(x.numel() for x in weights.values()),
                tensor_bytes=sum(x.numel()*x.element_size() for x in weights.values()),
                shapes={key: list(value.shape) for key, value in weights.items()})
            assert counts[arm]['parameters'] == 132228
        summary['cost'] = dict(checkpoint_counts=counts,
            ensemble_head_parameters=counts['RGB_ONLY']['parameters']+counts['TOF_ONLY']['parameters'],
            fusion_head_parameters=counts['FUSION']['parameters'],
            parameter_matched=False, head_parameter_ratio=2.0,
            unchanged_dense_head_MACs_per_row=dict(single_fusion=1028*128+128*4, two_heads=2*(1028*128+128*4)),
            existing_head_inference_passes_required=2, frozen_RGB_extraction_can_be_shared=True,
            inference_latency_status='NOT_MEASURED', inference_latency_seconds=None,
            combine_1500_rows_seconds=combine_seconds,
            limit='Array combination timing excludes feature extraction and both MLP forwards; no deployment latency claim. Zero-input weight pruning was not implemented.')
        saved = dict(frame_id=ids, truth=truth, original_alerts=data['original_alerts'][ev],
            RGB_ONLY_logits=rgb, TOF_ONLY_logits=tof, ENSEMBLE_logits=logits,
            **{arm+'_flags': p for arm, p in pred.items()},
            condition=np.array([row['condition'] for row in selected]),
            declared_range=np.array([row['declared_range'] for row in selected]),
            region_id=np.array([row['region_id'] for row in selected]))
        np.savez_compressed(output/'predictions.npz', **saved)
        stored = load(output/'predictions.npz')
        np.testing.assert_array_equal(stored['original_alerts'], a0['original_alerts'])
        # Separately expressed scalar reconstruction of signs, metrics and paired rows.
        exact = wf = bh = hb = 0; pair_audit = {arm: dict(gained=0, lost=0) for arm in summary['comparisons']}
        for i in range(len(truth)):
            scalar = [(float(rgb[i, k])+float(tof[i, k]))*0.5 >= 0 for k in range(4)]
            p, t = stored['ENSEMBLE_flags'][i].tolist(), stored['truth'][i].tolist()
            assert p == scalar
            correct = p == t; exact += correct
            wf += sum(p[k+1] and t[k] and not t[k+1] for k in (0, 2))
            bh += any(t[:2]) and not any(t[2:]) and any(p[2:])
            hb += any(t[2:]) and not any(t[:2]) and any(p[:2])
            for arm in pair_audit:
                base_correct = stored[arm+'_flags'][i].tolist() == t
                pair_audit[arm]['gained'] += correct and not base_correct
                pair_audit[arm]['lost'] += base_correct and not correct
        assert [exact, wf, bh, hb] == [summary['aggregate']['ENSEMBLE'][key]['numerator'] for key in
                                     ('spatial_exact', 'wrong_far', 'body_to_head', 'head_to_body')]
        assert all(all(value == summary['comparisons'][arm][key] for key, value in result.items()) for arm, result in pair_audit.items())
        write(output/'independent-audit.json', dict(status='PASS', scalar_counts=dict(exact=exact, wrong_far=wf, body_to_head=bh, head_to_body=hb),
            scalar_logit_signs=1500, paired_counts=pair_audit, original_split_metric_groups=original_checks,
            mz3_aggregate_parity_arms=4, all5000_original_alert_parity=True, eval_original_alert_parity=1500))
        write(output/'result.json', dict(status='PASS', scope=protocol['scope'], frames=1500, event_order=EVENTS, **summary))
        assert hashes == {key: sha(path) for key, path in paths.items()}
        write(output/'receipt.json', dict(status='PASS', backend='CPU', placement_reason='TASK_NOT_GPU_SUITABLE',
            input_sha256=hashes, input_paths={key: str(path) for key, path in paths.items()},
            protocol_sha256=sha(output/'protocol.json'), original_inputs_unchanged=True,
            output_sha256={path.name: sha(path) for path in output.iterdir()},
            seconds=time.perf_counter()-started, training_steps=0, model_forward_passes=0,
            numpy_version=np.__version__, torch_version=torch.__version__,
            alerts_modified=0, inference_latency_measured=False))
        print(json.dumps(dict(status='PASS', output=str(output), metrics=summary['aggregate']['ENSEMBLE'],
            pairs={arm: {k: value[k] for k in ('gained', 'lost')} for arm, value in summary['comparisons'].items()},
            both_correct=summary['both_unimodal_correct'], head_parameters=summary['cost']['ensemble_head_parameters']), indent=2))
    except Exception as exc:
        write(output/'failure.json', dict(status='FAILED', error=repr(exc), input_sha256=hashes))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
