"""Reuse sealed MZ64 source/sequence; cache only one geometry arm's fit rows."""
from pathlib import Path
import time
import numpy as np

from mz45_object_transfer import Bindings
from mz59_source import read, write_new, sha, load_npz
from mz64_geometry_source import prepare as prepare64, feature_index as feature_index64
from mz64_geometry_source import FeatureStore as FeatureStore64, exact

TASK = 'mz66-replay-projection-20260911'
MZ64_RECEIPT_SHA = '3e917c4e2d6d56f134ab52633e572ae528e75487bc68405d2ce5a6a247d2508e'


def feature_index(bundle):
    # The fit schedule itself stays byte-identical. The old MZ55 CONTROL arm
    # is not fitted again; its images are encoded on demand for evaluation.
    cache_schedule = dict(bundle['schedule'], CONTROL=np.empty((0, 4), np.int64))
    index, plan = feature_index64(dict(bundle, schedule=cache_schedule))
    plan.update(scratch_owner=TASK, fit_arm='PROJECT', target_source='mz61',
                reference_only_mz55_training_cache=False)
    assert plan['training_unique']['mz55'] == 0
    return index, plan


def prepare(root, bind):
    started = time.perf_counter()
    bundle = prepare64(root, bind)
    prior = Path(root) / 'artifacts.local/work/mz64-geometry-learning-20260911/run-v1'
    receipt_path = Path(bind(prior / 'receipt.json', MZ64_RECEIPT_SHA))
    receipt = read(receipt_path)
    assert receipt['status'] == 'PASS' and receipt['steps_per_arm'] == 1536
    for name in ('mz64_geometry_learning.py', 'mz64_geometry_source.py'):
        matches = [(p, h) for p, h in receipt['inputs'].items() if Path(p).name == name]
        assert len(matches) == 1
        bind(*matches[0])
    saved = load_npz(bind(prior / 'predictions.npz', receipt['outputs']['predictions.npz']))
    for key, value in bundle['predictions'].items():
        exact(saved[key], value, key)
    schedule_path = Path(bind(prior / 'schedule.npz', receipt['outputs']['schedule.npz']))
    previous_schedule = load_npz(schedule_path)
    assert set(previous_schedule) == set(bundle['schedule'])
    for key, value in bundle['schedule'].items():
        exact(previous_schedule[key], value, key)
    groups_path = Path(bind(prior / 'groups.json', receipt['outputs']['groups.json']))
    groups = read(groups_path)
    for name in ('records', 'mz55_records', 'mz61_records'):
        assert groups[name] == bundle[name]
    for name in ('groups', 'mz55_groups', 'mz61_groups'):
        assert groups[name] == {k: np.asarray(v).tolist() for k, v in bundle[name].items()}
    bundle['predictions'] = saved
    bundle['candidate_schedule_path'] = schedule_path.resolve()
    bundle['feature_index'], plan = feature_index(bundle)
    bundle['source_info'].update(adapter=Path(__file__).name, scratch_owner=TASK,
        feature_plan=plan, training_unique=plan['training_unique'],
        training_unique_total=plan['training_unique_total'], feature_file_bytes=plan['expected_file_bytes'],
        candidate_schedule_path=str(schedule_path.resolve()), candidate_schedule_sha256=sha(schedule_path),
        exact_mz64_geometry_sequence=True, mz64_receipt_sha256=MZ64_RECEIPT_SHA,
        mz64_predictions_sha256=receipt['outputs']['predictions.npz'], mz64_arrays_preserved=len(saved),
        geometry_presentations=6144, mz48_presentations=6144, old_negative_presentations=12288,
        mz55_fit_presentations=0, prepare_seconds=time.perf_counter()-started)
    assert bundle['old_maps'] is None
    return bundle


class FeatureStore(FeatureStore64):
    """Same extraction/build/read arithmetic, with an owned smaller cache."""
    def __init__(self, base, store, bundle, task, bind):
        self.base, self.store, self.bundle, self.bind = base, store, bundle, bind
        self.task = Path(task).resolve()
        assert self.task == Path(bundle['source_info']['artifact_work']) / TASK
        self.index, self.plan = feature_index(bundle)
        assert self.plan == bundle['source_info']['feature_plan']
        for key, value in self.index.items():
            exact(value, bundle['feature_index'][key], key)
        self.old_ids, self.mz48_ids, self.mz55_ids, self.mz61_ids = (
            self.index[s + '_ids'] for s in ('old', 'mz48', 'mz55', 'mz61'))
        self.path = self.task / 'scratch-v1/training-full.npy'
        self.out = self.task / 'feature-plan-v1'
        self.full = None
        self.stats = dict(training_extracted_frames=0, full_extracted_frames=0,
            full_cache_eval_hits=0, evaluation_extracted_frames=0, encoder_seconds=0.,
            build_seconds=0., evaluation_seconds=0.)


def smoke(root, output):
    output = Path(output).resolve()
    task = (Path(root) / 'artifacts.local/work' / TASK).resolve()
    assert output == task / 'source-check-v1'
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    bind = Bindings()
    bundle = prepare(root, bind)
    feature = FeatureStore(None, None, bundle, task, bind)
    assert feature.full is None and not feature.path.exists()
    assert len(feature.mz55_ids) == 0 and len(feature.mz61_ids) == 2048
    for source, roles in (('mz48', ('calibration', 'heldout_site')),
                          ('mz61', ('calibration', 'heldout_geometry'))):
        groups = bundle['groups'] if source == 'mz48' else bundle[source+'_groups']
        for role in roles:
            if role in groups:
                assert (feature.index[source+'_lookup'][groups[role]] < 0).all()
    try:
        FeatureStore(None, None, bundle, task.parent / 'mz64-geometry-learning-20260911', bind)
    except AssertionError:
        pass
    else:
        raise AssertionError('Wrong cache owner accepted')
    assert FeatureStore.build is FeatureStore64.build
    assert FeatureStore.extract is FeatureStore64.extract
    assert FeatureStore.evaluation is FeatureStore64.evaluation
    np.savez_compressed(output / 'feature-cache-index.npz', **feature.index)
    write_new(output / 'plan.json', feature.plan)
    write_new(output / 'source-info.json', bundle['source_info'])
    bind.check()
    feature.close()
    write_new(output / 'receipt.json', dict(status='PASS', inputs=bind.inputs,
        source_info=bundle['source_info'], outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()},
        unchanged_mz64_schedule=True, old_predictions_preserved=len(bundle['predictions']),
        wrong_owner_rejected=True, no_calibration_heldout_in_fit=True,
        model_inference_frames=0, rgb_decodes=0, dense_cache_bytes_created=0,
        backend='TASK_NOT_GPU_SUITABLE source identity and array validation', seconds=time.perf_counter()-started))
    print('SOURCE PASS', dict(unique_frames=feature.plan['training_unique_total'],
        cache_bytes=feature.plan['expected_file_bytes'], arrays=len(bundle['predictions'])), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    smoke(args.root.resolve(), args.output)
