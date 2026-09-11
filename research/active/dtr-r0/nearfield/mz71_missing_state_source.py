"""MZ71 CPU-only sealed prediction inheritance and observable missing-state masks.

No FeatureStore is instantiated. Schedules and source-role documents are inherited
unchanged. Packet state, never profile name or evaluator labels, defines missing.
"""
from pathlib import Path
import argparse
import time
import traceback
import numpy as np
from mz45_object_transfer import Bindings
from mz59_source import read, write_new, sha, load_npz
from mz64_geometry_source import exact
from mz70_diverse_source import prepare as prepare70, COHORTS
from mz63_geometry_transfer import packet, CONDITIONS

TASK = 'mz71-missing-state-calibration-20260911'
RUN_SHA = 'a3ddf2e8e6cca4742e9970eb7e3e1f9f2e1adf70f3284883882f15869ceaa1e8'
SCORE_SHA = '2fe35ce61155ba80c7197400d4331f60410159cc550270dca623153e2f23f84a'
ARMS = ('CONTROL', 'DIVERSE')
FIELDS = ('raw', 'support', 'winner', 'anchor_available', 'anchor_vector', 'candidate', 'UNION')


def reference(path, digest):
    return dict(path=str(Path(path).resolve()), sha256=digest)


def prepare(root, bind):
    started = time.perf_counter()
    root = Path(root)
    bundle = prepare70(root, bind)
    bind(Path(__file__))
    bind(Path(__file__).with_name('mz63_geometry_transfer.py'))
    work = root / 'artifacts.local/work'
    run = work / 'mz70-diverse-learning-20260911/run-v1'
    score_path = work / 'mz70-diverse-learning-20260911/score-v1/receipt.json'
    receipt_path = bind(run / 'receipt.json', RUN_SHA)
    rr = read(receipt_path)
    sr = read(bind(score_path, SCORE_SHA))
    assert rr['status'] == sr['status'] == 'PASS'
    assert rr['original_calibration_frames'] == 1256
    assert all(rr[key] == 0 for key in ('mz55_calibration_rows_used', 'mz61_calibration_rows_used', 'mz67_calibration_rows_used'))
    assert any(Path(p).resolve() == Path(receipt_path).resolve() and h == RUN_SHA for p, h in sr['inputs'].items())
    source_refs = [(p, h) for p, h in rr['inputs'].items() if Path(p).name == 'mz70_diverse_source.py']
    assert len(source_refs) == 1
    bind(*source_refs[0])
    pred_path = bind(run / 'predictions.npz', rr['outputs']['predictions.npz'])
    saved = load_npz(pred_path)
    inherited_count = len(bundle['predictions'])
    for key, value in bundle['predictions'].items():
        exact(saved[key], value, key)
    schedule_path = bind(run / 'schedule.npz', rr['outputs']['schedule.npz'])
    old_schedule = load_npz(schedule_path)
    assert set(old_schedule) == set(bundle['schedule'])
    for key, value in bundle['schedule'].items():
        exact(old_schedule[key], value, key)
    groups_path = bind(run / 'groups.json', rr['outputs']['groups.json'])
    groups = read(groups_path)
    for key in ('records', 'mz55_records', 'mz61_records', 'mz67_records'):
        assert groups[key] == bundle[key], key
    for key in ('groups', 'mz55_groups', 'mz61_groups', 'mz67_groups'):
        assert groups[key] == {k: np.asarray(v).tolist() for k, v in bundle[key].items()}, key
    bundle['predictions'] = saved
    bundle['original_groups_document'] = groups
    bundle['frozen_prediction_keys'] = tuple(sorted(saved))
    new = COHORTS[:7]
    assert sum(len(bundle['cohorts'][c]['frame_ids']) for c in new) == 9544
    checks = {}
    for cohort in COHORTS:
        count = len(bundle['cohorts'][cohort]['frame_ids'])
        np.testing.assert_array_equal(saved[cohort + '/frame_ids'], bundle['cohorts'][cohort]['frame_ids'])
        null_present = cohort in ('mz61', 'mz67')
        for arm in ARMS:
            prefix = cohort + '/ALL_INVALID/MZ70/' + arm + '/'
            if null_present:
                for field in FIELDS:
                    assert prefix + field in saved
                    assert saved[prefix + field].shape[0] == count
                assert not saved[prefix + 'anchor_available'].any()
                assert (saved[prefix + 'anchor_vector'] == 0).all()
            else:
                assert not any(key.startswith(prefix) for key in saved), prefix
        checks[cohort] = dict(frames=count, saved_MZ70_ALL_INVALID_both_arms=null_present,
                             new_ALL_INVALID_frames=0 if null_present else count)
    cal48 = np.asarray(groups['groups']['calibration'], np.int64)
    assert cal48.shape == (256,) and len(np.unique(cal48)) == 256
    for key, ids in groups['groups'].items():
        if key != 'calibration':
            assert not np.isin(cal48, ids).any(), key
    assert len(bundle['cohorts']['DEV']['frame_ids']) == 1000
    assert saved['DEV/known'].all() and saved['mz48/known'][cal48].all()
    calibration_ids = dict(DEV=np.arange(1000, dtype=np.int64), mz48=cal48.copy())
    bundle['calibration_ids'] = calibration_ids
    parity = {c: np.sort(np.asarray(bundle[c + '_groups']['fit'], np.int64))[:16] for c in ('mz61', 'mz67')}
    for cohort, ids in parity.items():
        assert ids.shape == (16,) and len(np.unique(ids)) == 16
        assert all(bundle[cohort + '_records'][int(i)]['role'] == 'TRAIN_CANDIDATE' for i in ids)
        for arm in ARMS:
            for profile in CONDITIONS:
                prefix = cohort + '/' + profile + '/MZ70/' + arm + '/'
                assert saved[prefix + 'raw'][ids].shape == (16, 4)
                assert saved[prefix + 'support'][ids].shape == (16, 4)
    bundle['initial_parity_ids'] = parity
    observed, missing_masks, missing_counts = {}, {}, {}
    for cohort in COHORTS:
        source = bundle['cohorts'][cohort]
        for profile in CONDITIONS:
            prefix = cohort + '/' + profile
            obs = packet(source['ranges'], source['valid'], profile)
            assert obs['ranges'].dtype == np.float32 and obs['valid'].dtype == bool
            assert obs['ranges'].shape == obs['valid'].shape == (len(source['frame_ids']), 64, 2)
            mask = ~obs['valid'].any((1, 2))
            checked = []
            for field in ('ranges', 'valid'):
                key = prefix + '/' + field
                if key in saved:
                    exact(saved[key], obs[field], key)
                    checked.append(key)
            if profile == 'ALL_INVALID':
                assert mask.all() and not obs['ranges'].any() and not obs['valid'].any()
            observed[prefix], missing_masks[prefix] = obs, mask
            missing_counts[prefix] = dict(frames=len(mask), missing=int(mask.sum()),
                nonmissing=int((~mask).sum()), existing_packet_arrays_checked=checked)
    bundle['observed_packets'] = observed
    bundle['missing_masks'] = missing_masks
    assert set(bundle['source_info']['label_refs']) == {'mz48', 'mz55', 'mz61', 'mz67'}
    assert set(bundle['source_info']['source_index_refs']) >= {'mz55', 'mz61', 'mz67'}
    bundle['source_info'].update(adapter=Path(__file__).name,
        prior70_receipt_ref=reference(receipt_path, RUN_SHA), prior70_score_ref=reference(score_path, SCORE_SHA),
        prior70_predictions_ref=reference(pred_path, rr['outputs']['predictions.npz']),
        prior70_groups_ref=reference(groups_path, rr['outputs']['groups.json']),
        prior70_schedule_ref=reference(schedule_path, rr['outputs']['schedule.npz']),
        prior70_prediction_keys=list(bundle['frozen_prediction_keys']), prior70_predictions_preserved=len(saved),
        original_adapter_arrays_preserved=inherited_count, new_ALL_INVALID_frames=9544,
        ALL_INVALID_coverage=checks, missing_state_counts=missing_counts,
        missing_state_definition='~observed_valid.any((1,2)); applies to naturally missing rows under every profile',
        original_calibration=dict(DEV=1000, mz48=256, total=1256, other_sources=0,
            mz48_role_authority='Original MZ70 groups/calibration; do not reinterpret record training_role'),
        initial_parity_ids={c: ids.tolist() for c, ids in parity.items()},
        MZ71_feature_cache_allocations=0, MZ71_schedule_changes=0, prepare_seconds=time.perf_counter()-started)
    assert bundle['old_maps'] is None
    return bundle


def smoke(root, output):
    import torch
    output = Path(output).resolve()
    task = (Path(root) / 'artifacts.local/work' / TASK).resolve()
    assert output == task / 'source-preparation-v1'
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    bind = Bindings()
    bundle = None
    assert not torch.cuda.is_initialized()
    try:
        bundle = prepare(root, bind)
        packets = {}
        for prefix, observed in bundle['observed_packets'].items():
            for field, value in observed.items():
                packets[prefix + '/' + field] = value
            packets[prefix + '/missing_mask'] = bundle['missing_masks'][prefix]
        np.savez_compressed(output / 'observed-packets.npz', **packets)
        np.savez_compressed(output / 'initial-parity-ids.npz', **bundle['initial_parity_ids'])
        np.savez_compressed(output / 'calibration-ids.npz', **bundle['calibration_ids'])
        write_new(output / 'source-info.json', bundle['source_info'])
        write_new(output / 'prediction-keys.json', bundle['frozen_prediction_keys'])
        bind.check()
        assert not torch.cuda.is_initialized()
        write_new(output / 'receipt.json', dict(status='PASS', inputs=bind.inputs,
            outputs={p.name: sha(p) for p in output.iterdir() if p.is_file()},
            preserved_prediction_arrays=len(bundle['predictions']), new_ALL_INVALID_frames=9544,
            observed_packet_prefixes=len(bundle['observed_packets']),
            missing_counts=bundle['source_info']['missing_state_counts'], original_calibration_frames=1256,
            original_groups_unchanged=True, original_schedule_unchanged=True, parity_frames=32,
            source_native_labels_and_UNKNOWN_preserved=True, training_steps=0, feature_cache_allocations=0,
            model_reads=0, model_inference_frames=0, RGB_decodes=0, raw_native_depth_reads=0,
            cuda_initialized=False, mmap_handles_closed=True, backend='FROZEN_PROTOCOL_CPU_ONLY',
            seconds=time.perf_counter()-started))
        print('SOURCE PASS', dict(arrays=len(bundle['predictions']), new_ALL_INVALID_frames=9544,
            missing_counts={k: v['missing'] for k, v in bundle['source_info']['missing_state_counts'].items()},
            seconds=time.perf_counter()-started), flush=True)
    except BaseException:
        write_new(output / 'failure.json', dict(status='FAIL', error=traceback.format_exc(), inputs=bind.inputs))
        raise
    finally:
        if bundle is not None and bundle.get('old_maps') is not None:
            bundle['old_maps']._mmap.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    smoke(args.root.resolve(), args.output.resolve())
