"""Descriptive consumed-Development attribution of frozen MZ0/MZ1/MZ2 outputs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

EVENTS = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')
META = ('frame_id', 'name', 'group_id', 'region_id', 'condition', 'declared_range',
        'family', 'original_fixture_group_id')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def metrics(pred, truth):
    """Independent NumPy expression of all legacy aggregate metrics."""
    count = lambda n, d: dict(numerator=int(n), denominator=int(d))
    body = truth[:, :2].any(1)
    head = truth[:, 2:].any(1)
    nearonly = truth[:, [0, 2]] & ~truth[:, [1, 3]]
    return dict(
        spatial_exact=count(np.all(pred == truth, 1).sum(), len(truth)),
        body_head_accuracy=count(np.all(np.stack((pred[:, :2].any(1), pred[:, 2:].any(1)), 1)
            == np.stack((body, head), 1), 1).sum(), len(truth)),
        near_far_accuracy=count(np.all(np.stack((pred[:, [0, 2]].any(1), pred[:, [1, 3]].any(1)), 1)
            == np.stack((truth[:, [0, 2]].any(1), truth[:, [1, 3]].any(1)), 1), 1).sum(), len(truth)),
        wrong_far=count((pred[:, [1, 3]] & nearonly).sum(), nearonly.sum()),
        body_to_head=count((pred[:, 2:].any(1) & body & ~head).sum(), (body & ~head).sum()),
        head_to_body=count((pred[:, :2].any(1) & head & ~body).sum(), (head & ~body).sum()),
        event_confusion={name: dict(tp=int((pred[:, k] & truth[:, k]).sum()),
            fp=int((pred[:, k] & ~truth[:, k]).sum()), fn=int((~pred[:, k] & truth[:, k]).sum()),
            tn=int((~pred[:, k] & ~truth[:, k]).sum())) for k, name in enumerate(EVENTS)})


def errors(pred, truth):
    body, head = truth[:, :2].any(1), truth[:, 2:].any(1)
    return dict(wrong_far=pred[:, [1, 3]] & truth[:, [0, 2]] & ~truth[:, [1, 3]],
                cross_body=np.stack((pred[:, 2:].any(1) & body & ~head,
                                     pred[:, :2].any(1) & head & ~body), 1))


def compare(candidate, baseline, truth, ids):
    c, b = np.all(candidate == truth, 1), np.all(baseline == truth, 1)
    masks = dict(gained=c & ~b, lost=~c & b, both_correct=c & b, both_wrong=~c & ~b)
    result = dict(n=len(truth), **{k: int(v.sum()) for k, v in masks.items()},
                  gained_frame_ids=ids[masks['gained']].tolist(), lost_frame_ids=ids[masks['lost']].tolist())
    ce, be = errors(candidate, truth), errors(baseline, truth)
    result['errors'] = {kind: dict(introduced=int((ce[kind] & ~be[kind]).sum()),
        removed=int((~ce[kind] & be[kind]).sum()), retained=int((ce[kind] & be[kind]).sum()),
        candidate=int(ce[kind].sum()), baseline=int(be[kind].sum())) for kind in ce}
    return result


def run(root, output):
    started = time.perf_counter()
    root, output = Path(root).resolve(), Path(output).resolve()
    work = root/'artifacts.local/work'
    allowed = (work/'mz3-error-attribution-20260910').resolve()
    if not output.is_relative_to(allowed) or output == allowed:
        raise ValueError('Output must be a fresh child of canonical MZ3 artifact root')
    if output.exists():
        raise FileExistsError('Fresh output required; preserve prior evidence')
    protocol = Path(__file__).with_name('MZ3_ERROR_ATTRIBUTION_PROTOCOL_20260910.md')
    paths = dict(index=work/'body-query-5000-20260909/dataset-v1/index.json',
        manifest=work/'body-query-5000-20260909/dataset-v1/manifest.json',
        mz0=work/'mz0-clean-20260910/run-v1/scored.npz',
        depth=work/'mz0-clean-20260910/run-v1/depth.npz',
        mz0_result=work/'mz0-clean-20260910/run-v1/result.json',
        depth_receipt=work/'mz0-clean-20260910/run-v1/depth-receipt.json',
        features=work/'mz1-tiny-fusion-20260910/cache-v1/features.npz',
        features_receipt=work/'mz1-tiny-fusion-20260910/cache-v1/features-receipt.json',
        mz1=work/'mz1-tiny-fusion-20260910/run-v1/predictions.npz',
        mz1_receipt=work/'mz1-tiny-fusion-20260910/run-v1/receipt.json',
        mz1_metrics=work/'mz1-tiny-fusion-20260910/run-v1/metrics.json',
        mz2=work/'mz2-resolution-20260910/run-v1/predictions.npz',
        mz2_receipt=work/'mz2-resolution-20260910/run-v1/receipt.json',
        mz2_result=work/'mz2-resolution-20260910/run-v1/result.json', protocol=protocol,
        script=Path(__file__).resolve())
    hashes = {k: sha(v) for k, v in paths.items()}
    output.mkdir(parents=True)
    write(output/'start-receipt.json', dict(status='STARTED', started_utc=datetime.now(timezone.utc).isoformat(),
        backend='CPU', placement_reason='TASK_NOT_GPU_SUITABLE', input_sha256=hashes))
    try:
        m0, m1, m2 = read(paths['mz0_result']), read(paths['mz1_metrics']), read(paths['mz2_result'])
        r1, r2, fr = read(paths['mz1_receipt']), read(paths['mz2_receipt']), read(paths['features_receipt'])
        assert hashes['index'] == read(paths['manifest'])['index_sha256'] == fr['source_index_sha256'] == m0['source_index_sha256']
        assert hashes['mz0'] == m0['scored_sha256']
        assert hashes['depth'] == read(paths['depth_receipt'])['depth_sha256']
        assert hashes['features'] == fr['feature_sha256'] == r1['feature_sha256']
        assert hashes['mz1'] == r1['predictions_sha256'] and hashes['mz1_metrics'] == r1['metrics_sha256']
        assert hashes['mz2'] == r2['predictions_sha256'] and hashes['mz2_result'] == r2['result_sha256']
        allrows = read(paths['index'])['frames']
        rows = [r for r in allrows if r['source_role'] == 'EVAL_ONLY']
        assert len(rows) == 1500 and all(r['status'] == 'PASS' for r in allrows)
        a0, depth, features, a1, a2 = (load(paths[k]) for k in ('mz0', 'depth', 'features', 'mz1', 'mz2'))
        role = np.array([r['source_role'] for r in allrows]); ev = role == 'EVAL_ONLY'
        alltruth = np.array([np.array(r['counts']).reshape(4, 3).sum(1) >= 3 for r in allrows])
        for data in (features, a1, a2):
            np.testing.assert_array_equal(data['role'], role)
            np.testing.assert_array_equal(data['truth'], alltruth)
            np.testing.assert_array_equal(data['original_alerts'], features['original_alerts'])
        truth = alltruth[ev]; ids = np.array([r['frame_id'] for r in rows])
        np.testing.assert_array_equal(a0['truth'], truth)
        np.testing.assert_array_equal(a0['original_alerts'], features['original_alerts'][ev])
        np.testing.assert_array_equal(depth['D_full_events'], truth)
        original_checks = 0
        for name, expected in m0['scores'].items():
            assert metrics(a0[name], truth) == expected; original_checks += 1
        for name, expected in m0['depth_only_scores'].items():
            assert metrics(depth[name+'_events'], truth) == expected; original_checks += 1
        for name, expected in m1['arms'].items():
            for split, values in expected.items():
                mask = role == split
                assert metrics(a1[name+'_flags'][mask], alltruth[mask]) == values; original_checks += 1
        for name, expected in m2['resolution_scores'].items():
            for split, values in expected.items():
                mask = role == split
                assert metrics(a2[name+'_flags'][mask], alltruth[mask]) == values; original_checks += 1
        for name, expected in m2['stress_scores'].items():
            assert metrics(a2[name+'_flags'], truth) == expected; original_checks += 1
        pred = dict(MZ0=a0['z8_multi_surface'], RGB=a1['RGB_ONLY_flags'][ev],
            TOF=a1['TOF_ONLY_flags'][ev], FUSION=a1['FUSION_flags'][ev],
            Z1=a2['z1_flags'][ev], Z4=a2['z4_flags'][ev],
            NOISE=a2['gaussian05_flags'], DROPOUT=a2['dropout20_flags'], SHIFT=a2['right_shift1_flags'])
        pairs = [(c, b) for c in ('FUSION', 'Z1', 'Z4') for b in ('MZ0', 'TOF', 'RGB')]
        pairs += [(c, 'FUSION') for c in ('NOISE', 'DROPOUT', 'SHIFT')]
        strata = {key: np.array([r.get(key, 'NOT_AVAILABLE') for r in rows]) for key in
                  ('region_id', 'condition', 'declared_range', 'family', 'original_fixture_group_id')}
        strata['condition_range'] = np.array([r['condition']+'/'+r['declared_range'] for r in rows])
        comparisons = {}
        for c, b in pairs:
            comparisons[c+'_vs_'+b] = dict(overall=compare(pred[c], pred[b], truth, ids),
                strata={key: {value: compare(pred[c][mask], pred[b][mask], truth[mask], ids[mask])
                    for value in sorted(set(values)) for mask in [values == value]} for key, values in strata.items()})
        rgb, tof, fusion = pred['RGB'], pred['TOF'], pred['FUSION']
        rc, tc, fc = ((p == truth).all(1) for p in (rgb, tof, fusion))
        disagreement = dict(frames=int((rgb != tof).any(1).sum()),
            bits=int((rgb != tof).sum()), by_event={event: int((rgb[:, k] != tof[:, k]).sum()) for k, event in enumerate(EVENTS)},
            truth_groups={name: dict(n=int(mask.sum()), fusion_exact=int(fc[mask].sum()),
                fusion_equals_rgb=int((fusion[mask] == rgb[mask]).all(1).sum()),
                fusion_equals_tof=int((fusion[mask] == tof[mask]).all(1).sum()))
                for name, mask in dict(both_correct=rc & tc, rgb_only_correct=rc & ~tc,
                                       tof_only_correct=~rc & tc, both_wrong=~rc & ~tc).items()})
        center = depth['z8_multi_surface_events']
        full, crop = depth['D_full_counts'], depth['D_crop_counts']
        assert np.array_equal(full >= 3, truth)
        support = {}
        with (output/'false-events.jsonl').open('w', encoding='utf-8') as stream:
            for arm, p in pred.items():
                fp = p & ~truth
                nearonly = truth[:, [0, 2]] & ~truth[:, [1, 3]]
                wrongfar = np.zeros_like(fp); wrongfar[:, [1, 3]] = p[:, [1, 3]] & nearonly
                body, head = truth[:, :2].any(1), truth[:, 2:].any(1)
                cross = np.zeros_like(fp)
                cross[:, 2:] = p[:, 2:] & (body & ~head)[:, None]
                cross[:, :2] = p[:, :2] & (head & ~body)[:, None]
                support[arm] = {kind: dict(event_bits=int(mask.sum()), rows=int(mask.any(1).sum()),
                    clean_zone_center_supported=int((mask & center).sum()),
                    native_full_zero=int((mask & (full == 0)).sum()),
                    native_full_one_or_two=int((mask & (full > 0) & (full < 3)).sum()),
                    native_full_three_plus=int((mask & (full >= 3)).sum()),
                    native_crop_zero=int((mask & (crop == 0)).sum()),
                    native_crop_one_or_two=int((mask & (crop > 0) & (crop < 3)).sum()))
                    for kind, mask in dict(all_false_positive=fp, wrong_far_bits=wrongfar, cross_body_bits=cross).items()}
                for i, k in zip(*np.where(fp)):
                    record = dict(arm=arm, frame_id=int(ids[i]), event=EVENTS[k],
                        wrong_far=bool(wrongfar[i, k]), cross_body=bool(cross[i, k]),
                        clean_zone_center_support=bool(center[i, k]), native_full_count=int(full[i, k]),
                        native_crop_count=int(crop[i, k]))
                    stream.write(json.dumps(record)+'\n')
        with (output/'rows.jsonl').open('w', encoding='utf-8') as stream:
            for i, row in enumerate(rows):
                record = {k: row.get(k, 'NOT_AVAILABLE') for k in META}
                record.update(eval_index=i, truth=truth[i].tolist(),
                    predictions={k: v[i].tolist() for k, v in pred.items()},
                    exact={k: bool(np.all(v[i] == truth[i])) for k, v in pred.items()},
                    clean_zone_center_events=center[i].tolist(), native_full_counts=full[i].tolist(),
                    native_crop_counts=crop[i].tolist())
                stream.write(json.dumps(record)+'\n')
        result = dict(status='PASS', scope='Consumed controlled Development; descriptive attribution only',
            n=1500, event_order=EVENTS, aggregate={k: metrics(p, truth) for k, p in pred.items()},
            comparisons=comparisons, modality_disagreement=disagreement, geometric_support=support,
            original_metric_groups_verified=original_checks, original_alert_parity=5000,
            support_limits='Center support is an approximation; native >=3 defines truth. No object association; clean source support is counterfactual for stresses.')
        write(output/'result.json', result)
        # Different scalar implementation verifies counts from the durable export.
        saved = [json.loads(line) for line in (output/'rows.jsonl').read_text().splitlines()]
        audit = {}
        for key, values in comparisons.items():
            c, b = key.split('_vs_'); tally = dict(gained=0, lost=0, both_correct=0, both_wrong=0)
            for row in saved:
                ce = row['predictions'][c] == row['truth']; be = row['predictions'][b] == row['truth']
                label = 'both_correct' if ce and be else 'gained' if ce else 'lost' if be else 'both_wrong'
                tally[label] += 1
            assert all(tally[k] == values['overall'][k] for k in tally)
            audit[key] = tally
        for arm in pred:
            exact = wf = bh = hb = 0
            for row in saved:
                p, t = row['predictions'][arm], row['truth']; exact += p == t
                wf += sum(p[k+1] and t[k] and not t[k+1] for k in (0, 2))
                bh += any(t[:2]) and not any(t[2:]) and any(p[2:])
                hb += any(t[2:]) and not any(t[:2]) and any(p[:2])
            expected = result['aggregate'][arm]
            assert [exact, wf, bh, hb] == [expected[k]['numerator'] for k in ('spatial_exact', 'wrong_far', 'body_to_head', 'head_to_body')]
        write(output/'independent-audit.json', dict(status='PASS', scalar_row_reconstruction=True,
            original_metric_groups=original_checks, arms=9, pair_counts=audit))
        assert {k: sha(v) for k, v in paths.items()} == hashes
        write(output/'receipt.json', dict(status='PASS', backend='CPU', device='NumPy CPU',
            placement_reason='TASK_NOT_GPU_SUITABLE', numpy_version=np.__version__,
            seconds=time.perf_counter()-started, training_steps=0, model_forward_passes=0,
            input_sha256=hashes, input_paths={k: str(v) for k, v in paths.items()},
            original_inputs_unchanged=True, output_sha256={p.name: sha(p) for p in output.iterdir()},
            protocol_frozen_before_computation=True))
        print(json.dumps(dict(status='PASS', output=str(output), seconds=time.perf_counter()-started,
            pairs={k: {x: v['overall'][x] for x in ('gained', 'lost')} for k, v in comparisons.items()},
            disagreement=disagreement), indent=2))
    except Exception as exc:
        write(output/'failure.json', dict(status='FAILED', error=repr(exc), input_sha256=hashes))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
