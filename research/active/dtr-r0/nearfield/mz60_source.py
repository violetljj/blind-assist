"""CPU-only MZ60 source continuation; explicit inherited feature-build boundary."""
from __future__ import annotations

import argparse
from pathlib import Path
import time
import numpy as np

from mz59_source import (prepare as prepare59, FeatureStore as FeatureStore59,
                        feature_index, read, write_new, sha, load_npz)

TASK = 'mz60-native-query-20260911'


def prepare(root, bind):
    """Preserve complete MZ59 outputs and labels; no model or RGB work."""
    started = time.perf_counter()
    root = Path(root)
    prior = root / 'artifacts.local/work/mz59-training-diversity-20260911/run-v1'
    receipt = read(bind(prior / 'receipt.json'))
    assert receipt['status'] == 'PASS'
    frozen = [(path, digest) for path, digest in receipt['inputs'].items()
              if Path(path).name == 'mz59_source.py']
    assert len(frozen) == 1
    bind(*frozen[0]); bind(Path(__file__))
    bundle = prepare59(root, bind)
    predictions = load_npz(bind(prior / 'predictions.npz', receipt['outputs']['predictions.npz']))
    for key, expected in bundle['predictions'].items():
        actual = predictions[key]
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, key
        np.testing.assert_array_equal(actual, expected, err_msg=key)
        assert actual.tobytes() == expected.tobytes(), key
    original_count = len(bundle['predictions'])
    schedule = load_npz(bind(prior / 'schedule.npz', receipt['outputs']['schedule.npz']))
    assert set(schedule) == set(bundle['schedule'])
    for key, expected in bundle['schedule'].items():
        assert schedule[key].dtype == expected.dtype and schedule[key].shape == expected.shape
        np.testing.assert_array_equal(schedule[key], expected, err_msg=key)
    metadata = read(bind(prior / 'groups.json', receipt['outputs']['groups.json']))
    assert metadata['records'] == bundle['records']
    assert metadata['mz55_records'] == bundle['mz55_records']
    for source_key in ('groups', 'mz55_groups'):
        assert set(metadata[source_key]) == set(bundle[source_key])
        for group, ids in bundle[source_key].items():
            np.testing.assert_array_equal(metadata[source_key][group], ids)
    bundle['predictions'] = predictions
    bundle['schedule'] = schedule
    bundle['native_labels'] = {
        'mz48': dict(cell_truth=bundle['cell_truth'], cell_known=bundle['cell_known']),
        'mz55': dict(cell_truth=bundle['mz55_cell_truth'], cell_known=bundle['mz55_cell_known'])}
    plan = dict(bundle['source_info']['feature_plan'], scratch_owner=TASK)
    assert plan['training_unique'] == dict(mz48=1250, old=3533, mz55=1231)
    assert plan['training_unique_total'] == 6014 and plan['expected_file_bytes'] == 5542502528
    bundle['source_info'].update(prepare_seconds=time.perf_counter()-started,
        adapter='mz60_source.py', feature_plan=plan, scratch_owner=TASK,
        pre_mz59_arrays_byte_preserved=original_count, mz59_arrays_preserved=len(predictions),
        mz59_predictions_sha256=receipt['outputs']['predictions.npz'],
        mz59_schedule_sha256=receipt['outputs']['schedule.npz'], mz59_groups_sha256=receipt['outputs']['groups.json'],
        exact_mz59_schedule=True, exact_mz59_groups=True, native_labels_authority='loss/evaluator only')
    assert bundle['old_maps'] is None
    return bundle


class FeatureStore(FeatureStore59):
    """Only current-task ownership changes; all feature arithmetic is inherited."""
    def __init__(self, base, store, bundle, task, bind):
        self.base, self.store, self.bundle, self.bind = base, store, bundle, bind
        self.task = Path(task).resolve()
        assert self.task == Path(bundle['source_info']['artifact_work']) / TASK
        self.index, self.plan = feature_index(bundle)
        self.plan['scratch_owner'] = TASK
        for key, value in self.index.items():
            np.testing.assert_array_equal(value, bundle['feature_index'][key])
        assert self.plan == bundle['source_info']['feature_plan']
        self.old_ids, self.mz48_ids, self.mz55_ids = (self.index[s + '_ids'] for s in ('old', 'mz48', 'mz55'))
        self.path = self.task / 'scratch-v1/training-full.npy'
        self.out = self.task / 'feature-plan-v1'
        self.full = None
        self.stats = dict(training_extracted_frames=0, full_extracted_frames=0, full_cache_eval_hits=0,
                          evaluation_extracted_frames=0, encoder_seconds=0., build_seconds=0., evaluation_seconds=0.)


def smoke(root, output):
    """One CPU check including zero-allocation constructor and owner guard."""
    from mz45_object_transfer import Bindings
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    bind = Bindings(); features = None
    try:
        bundle = prepare(root, bind)
        task = Path(bundle['source_info']['artifact_work']) / TASK
        features = FeatureStore(None, None, bundle, task, bind)
        for name in ('build', 'extract', 'training', 'evaluation', 'close'):
            assert getattr(FeatureStore, name) is getattr(FeatureStore59, name)
        try:
            FeatureStore(None, None, bundle, task.with_name('mz59-training-diversity-20260911'), bind)
        except AssertionError:
            rejects_other_owner = True
        else:
            raise AssertionError('Constructor accepted a different experiment owner')
        assert not features.path.exists() and features.full is None
        np.savez_compressed(output / 'schedule.npz', **bundle['schedule'])
        np.savez_compressed(output / 'feature-cache-index.npz', **bundle['feature_index'])
        write_new(output / 'plan.json', bundle['source_info'])
        write_new(output / 'receipt.json', dict(status='PASS', scope='CPU source/baseline/schedule/owner check only',
            source_info=bundle['source_info'], inputs=bind.inputs,
            outputs={name: sha(output / name) for name in ('schedule.npz', 'feature-cache-index.npz', 'plan.json')},
            inherited_feature_methods_exact=True, wrong_task_owner_rejected=rejects_other_owner,
            training_steps=0, model_inference_frames=0, rgb_decodes=0, native_depth_reads=0,
            temporary_dense_bytes_created=0, mmap_handles_closed=True, GPU_feature_store_not_executed=True))
        print('PASS', bundle['source_info'], flush=True)
    except BaseException:
        import traceback
        write_new(output / 'failure.json', dict(status='FAIL', error=traceback.format_exc(), inputs=bind.inputs))
        raise
    finally:
        if features is not None:
            features.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    smoke(args.root.resolve(), args.output.resolve())
