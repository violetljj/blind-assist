"""MZ62 CPU source binding, exact profile coverage schedule and owned cache."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time
import numpy as np

from mz60_source import prepare as prepare60
from mz59_source import FeatureStore as FeatureStore59, feature_index as index59
from mz59_source import read, write_new, sha, load_npz

TASK = 'mz62-profile-coverage-20260911'
DESIGN = 'mz62-exposure-design-20260911'
SCHEDULE_SHA = '9424f5e85ef985ab999541cf75f8c4725cbff5237e0245bf4fa1b5946867d672'
ARMS = ('CONTROL', 'COVERAGE')
STEPS = 1200


def validate_schedule(schedule, previous, bundle):
    """Reject schema, material, profile, partition or replay drift on CPU."""
    expected = {'profile_index', 'mz48', 'OLD_NEG', 'query', 'CONTROL', 'COVERAGE',
                'fit_ids', 'mz48_fit_ids', 'old_train_ids'}
    assert set(schedule) == expected
    fit = np.asarray(bundle['mz55_groups']['fit'])
    assert len(fit) == 1600 and len(bundle['groups']['fit']) == 1280
    np.testing.assert_array_equal(schedule['fit_ids'], fit)
    np.testing.assert_array_equal(schedule['mz48_fit_ids'], bundle['groups']['fit'])
    np.testing.assert_array_equal(schedule['old_train_ids'], previous['train_ids'])
    assert all(value.dtype == np.int64 for value in schedule.values())
    np.testing.assert_array_equal(schedule['profile_index'], np.arange(STEPS) % 3)
    forbidden = np.r_[bundle['mz55_groups']['calibration'], bundle['mz55_groups']['heldout_site']]
    for arm in ARMS:
        ids = schedule[arm]
        assert ids.shape == (STEPS, 4) and not np.isin(ids, forbidden).any()
        assert Counter(ids.ravel()) == {i: 3 for i in fit}
    for profile in range(3):
        np.testing.assert_array_equal(np.sort(schedule['COVERAGE'][profile::3].ravel()), np.sort(fit))
    for key, old in [('mz48', previous['shared'][:, :4]), ('OLD_NEG', previous['OLD_NEG']), ('query', previous['query'])]:
        np.testing.assert_array_equal(schedule[key], np.tile(old, (2, 1)))
    assert np.isin(schedule['mz48'], bundle['groups']['fit']).all()
    assert np.isin(schedule['OLD_NEG'], previous['train_ids']).all()
    for source, records, ids in [('mz48', bundle['records'], bundle['groups']['fit']),
                                 ('mz55', bundle['mz55_records'], fit)]:
        assert all(records[i]['role'] == 'TRAIN_CANDIDATE' and records[i]['source_valid'] for i in ids)
        assert bundle['predictions'][source + '/known'][ids].all()
    return dict(status='PASS', eligible_mz48=1280, eligible_mz55=1600,
                total_presentations_per_mz55_frame=3, coverage_per_frame_per_profile=1,
                old_replay_exact_twice=True, query_replay_exact_twice=True,
                mz48_first_four_exact_twice=True, no_calibration_or_heldout=True)


def feature_index(bundle):
    """Reuse encoder index arithmetic with the actual four-slot native IDs."""
    schedule = bundle['schedule']
    view = dict(bundle, schedule=dict(shared=schedule['mz48'], OLD_NEG=schedule['OLD_NEG'],
                                     train_ids=schedule['old_train_ids'], mz55_diverse=schedule['COVERAGE']))
    index, plan = index59(view)
    plan['scratch_owner'] = TASK
    assert plan['training_unique'] == dict(mz48=1095, old=3533, mz55=1600)
    assert plan['training_unique_total'] == 6228 and plan['expected_file_bytes'] == 5739724928
    return index, plan


def prepare(root, bind):
    """Load compact observations/labels and sealed predictions; no RGB/model work."""
    started = time.perf_counter()
    root = Path(root)
    work = root / 'artifacts.local/work'
    prior = work / 'mz60-native-query-20260911/run-v1'
    receipt = read(bind(prior / 'receipt.json'))
    assert receipt['status'] == 'PASS'
    for name in ('mz60_source.py', 'mz59_source.py'):
        frozen = [(p,h) for p,h in receipt['inputs'].items() if Path(p).name == name]
        assert len(frozen) == 1
        bind(*frozen[0])
    bind(Path(__file__))
    bundle = prepare60(root, bind)
    previous_schedule = bundle['schedule']
    predictions = load_npz(bind(prior / 'predictions.npz', receipt['outputs']['predictions.npz']))
    for key, expected in bundle['predictions'].items():
        actual = predictions[key]
        assert actual.dtype == expected.dtype and actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes(), key
    groups_path = bind(prior / 'groups.json', receipt['outputs']['groups.json'])
    groups = read(groups_path)
    for name in ('records', 'mz55_records'):
        assert groups[name] == bundle[name]
    for name in ('groups', 'mz55_groups'):
        assert set(groups[name]) == set(bundle[name])
        for group, ids in bundle[name].items():
            np.testing.assert_array_equal(groups[name][group], ids)
    schedule_path = bind(work / DESIGN / 'schedule.npz', SCHEDULE_SHA)
    schedule = load_npz(schedule_path)
    bundle['predictions'] = predictions
    checks = validate_schedule(schedule, previous_schedule, bundle)
    bundle['schedule'] = schedule
    bundle['candidate_schedule_path'] = schedule_path
    bundle['inherited_groups_path'] = groups_path
    bundle['previous_schedule'] = previous_schedule
    bundle['feature_index'], plan = feature_index(bundle)
    bundle['source_info'].update(prepare_seconds=time.perf_counter()-started, adapter='mz62_exposure_source.py',
        feature_plan=plan, scratch_owner=TASK, training_unique=plan['training_unique'], training_unique_total=6228,
        feature_file_bytes=plan['expected_file_bytes'], schedule_seed159=False,
        schedule_design='SeedSequence162; same per-ID total3; assignment across profiles only',
        mz60_arrays_preserved=len(predictions), mz60_predictions_sha256=receipt['outputs']['predictions.npz'],
        mz60_groups_sha256=receipt['outputs']['groups.json'], exact_mz60_groups=True,
        exact_mz59_schedule=False, mz59_replay_blocks_exact_twice=True,
        candidate_schedule_sha256=SCHEDULE_SHA, schedule_checks=checks,
        native_labels_authority='loss/evaluator only; original MZ59 loss',
        original_baseline_cohort_inferences=0)
    assert bundle['old_maps'] is None
    return bundle


class FeatureStore(FeatureStore59):
    """New task owner and6228-row index; feature arithmetic unchanged."""
    def __init__(self, base, store, bundle, task, bind):
        self.base, self.store, self.bundle, self.bind = base, store, bundle, bind
        self.task = Path(task).resolve()
        assert self.task == Path(bundle['source_info']['artifact_work']) / TASK
        self.index, self.plan = feature_index(bundle)
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
    """Full CPU assembly plus five corruption checks and zero-allocation guard."""
    from mz45_object_transfer import Bindings
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bind = Bindings()
    features = None
    try:
        bundle = prepare(root, bind)
        task = Path(bundle['source_info']['artifact_work']) / TASK
        features = FeatureStore(None, None, bundle, task, bind)
        for name in ('build', 'extract', 'training', 'evaluation', 'close'):
            assert getattr(FeatureStore, name) is getattr(FeatureStore59, name)
        try:
            FeatureStore(None, None, bundle, task.with_name('mz60-native-query-20260911'), bind)
        except AssertionError:
            pass
        else:
            raise AssertionError('Wrong cache owner accepted')
        failures = []
        for key in ('COVERAGE', 'mz48', 'OLD_NEG', 'query', 'profile_index'):
            corrupted = {k:v.copy() for k,v in bundle['schedule'].items()}
            corrupted[key].flat[0] = -12345
            try:
                validate_schedule(corrupted, bundle['previous_schedule'], bundle)
            except AssertionError:
                failures.append(key)
            else:
                raise AssertionError('Failed to reject corrupted ' + key)
        assert len(failures) == 5 and features.full is None and not features.path.exists()
        (output / 'schedule.npz').write_bytes(Path(bundle['candidate_schedule_path']).read_bytes())
        (output / 'groups.json').write_bytes(Path(bundle['inherited_groups_path']).read_bytes())
        np.savez_compressed(output / 'feature-cache-index.npz', **bundle['feature_index'])
        write_new(output / 'plan.json', bundle['source_info'])
        bind.check()
        write_new(output / 'receipt.json', dict(status='PASS', inputs=bind.inputs, source_info=bundle['source_info'],
            outputs={n:sha(output/n) for n in ('schedule.npz','groups.json','feature-cache-index.npz','plan.json')},
            inherited_feature_methods_exact=True, wrong_task_owner_rejected=True, corruption_rejections=failures,
            training_steps=0, model_inference_frames=0, rgb_decodes=0, native_depth_reads=0,
            temporary_dense_bytes_created=0, mmap_handles_closed=True, GPU_feature_store_not_executed=True))
        print('PASS', dict(training_unique=bundle['source_info']['training_unique'],
              preserved_arrays=bundle['source_info']['mz60_arrays_preserved'], corruption_rejections=failures), flush=True)
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
