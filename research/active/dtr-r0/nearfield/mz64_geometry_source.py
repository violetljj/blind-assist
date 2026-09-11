"""MZ64 source-only design: sealed baselines, balanced schedule and owned cache.

prepare(root, bind) is CPU-only. Predictor cohorts contain frame IDs, RGBRef
objects and observed ranges/validity; records and native labels are loss/eval
only. FeatureStore.build() is the explicit encoder/allocation boundary. Call
evaluation(cohort, ids) once outside arm/profile loops, with at most16 rows.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import io
from pathlib import Path
import shutil
import time

import numpy as np

from mz62_exposure_source import prepare as prepare62
from mz59_source import FeatureStore as FeatureStore59, FEATURE_SHAPE, FRAME_BYTES
from mz59_source import read, write_new, sha, load_npz
from mz63_geometry_transfer import load_predictor, bound_ref

TASK = 'mz64-geometry-learning-20260911'
DESIGN = 'mz64-design-20260911'
ARMS = ('CONTROL', 'GEOMETRY')
STEPS = 1536
SEED = 164
PROFILES = ('IDEAL', 'MERGE_CLOSE', 'DROP_CLOSE')
COHORTS = ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48', 'mz55', 'mz61')
SOURCES = ('mz48', 'old', 'mz55', 'mz61')
SOURCE_INDEX_SHA = '6f9e4be717b7fe23d66af58a9002a667d7acff304ec0c0af58497dfc502c72c7'
PREVIOUS = {
    'mz62_run': ('mz62-profile-coverage-20260911/run-v1', 'c423418b7b54583b19a334d542d5f832780d7f40e23781b0d4ceb0eb52f985c8'),
    'mz62_score': ('mz62-profile-coverage-20260911/score-v1', '0ae3f93b82f33f0ddaebb9b98ccd988e2712490d20247da95b30bac9da60122d'),
    'mz63_run': ('mz63-geometry-transfer-20260911/run-v1', '42023213ba290cc98cf80d8f7ccc0db5146754ee3c01c037d490bc23a1c5b3da'),
    'mz63_score': ('mz63-geometry-transfer-20260911/score-v1', '1135e68db083f89688fb5d93d3f021cf09b0b63f9fc71501abb23fee5fa55305'),
}


def exact(actual, expected, label='array'):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, label
    assert actual.tobytes() == expected.tobytes(), label


def fit_pairs(records, fit):
    pairs = defaultdict(list)
    for i in fit:
        row = records[int(i)]
        assert row['role'] == 'TRAIN_CANDIDATE' and row['source_valid']
        pairs[row['pair_id']].append(int(i))
    for key, ids in pairs.items():
        assert len(ids) == 2 and {records[i]['support_context'] for i in ids} == {'unsupported', 'supported'}, key
    return np.array([sorted(pairs[key]) for key in sorted(pairs)], np.int64)


def make_schedule(previous, bundle):
    """Labels/margins never select repeats; three disjoint224-pair extras."""
    fit55 = np.asarray(bundle['mz55_groups']['fit'], np.int64)
    fit61 = np.asarray(bundle['mz61_groups']['fit'], np.int64)
    pairs = fit_pairs(bundle['mz55_records'], fit55)
    assert pairs.shape == (800, 2) and len(fit61) == 2048
    rng = np.random.default_rng(SEED)
    extras = pairs[rng.permutation(len(pairs))[:672]].reshape(3, 224, 2)
    schedule = dict(profile_index=np.arange(STEPS, dtype=np.int64) % 3,
        mz48=previous['shared'][np.arange(STEPS) % 600, :4].copy(),
        OLD_NEG=previous['OLD_NEG'][np.arange(STEPS) % 600].copy(),
        query=previous['query'][np.arange(STEPS) % 600].copy(),
        CONTROL=np.empty((STEPS, 4), np.int64), GEOMETRY=np.empty((STEPS, 4), np.int64),
        mz55_fit_ids=fit55.copy(), mz61_fit_ids=fit61.copy(),
        mz48_fit_ids=np.asarray(bundle['groups']['fit'], np.int64).copy(),
        old_train_ids=previous['train_ids'].copy())
    # Frozen RNG order: pair permutation once; CONTROL then GEOMETRY per profile.
    for profile in range(3):
        control = np.r_[fit55, extras[profile].ravel()]
        schedule['CONTROL'][profile::3] = rng.permutation(control).reshape(512, 4)
        schedule['GEOMETRY'][profile::3] = rng.permutation(fit61).reshape(512, 4)
    return schedule


def validate_schedule(schedule, previous, bundle):
    expected = make_schedule(previous, bundle)
    assert set(schedule) == set(expected)
    for key in expected:
        assert schedule[key].dtype == np.int64
        exact(schedule[key], expected[key], key)
    extras = []
    for arm, source in (('CONTROL', 'mz55'), ('GEOMETRY', 'mz61')):
        groups = bundle[source + '_groups']
        fit = groups['fit']
        forbidden = np.concatenate([v for k, v in groups.items()
                                    if k in ('calibration', 'heldout_site', 'heldout_geometry', 'nonfit_family')])
        assert np.isin(schedule[arm], fit).all() and not np.isin(schedule[arm], forbidden).any()
        assert bundle['predictions'][source + '/known'][fit].all()
        assert all(bundle[source + '_records'][int(i)]['role'] == 'TRAIN_CANDIDATE' for i in fit)
        assert schedule[arm].shape == (1536, 4)
    for profile in range(3):
        counts = Counter(schedule['CONTROL'][profile::3].ravel())
        assert len(counts) == 1600 and Counter(counts.values()) == {1: 1152, 2: 448}
        extra = {int(i) for i, count in counts.items() if count == 2}
        for pair in fit_pairs(bundle['mz55_records'], bundle['mz55_groups']['fit']):
            assert (int(pair[0]) in extra) == (int(pair[1]) in extra)
        extras.append(extra)
        assert Counter(schedule['GEOMETRY'][profile::3].ravel()) == {i: 1 for i in bundle['mz61_groups']['fit']}
    assert all(not extras[i] & extras[j] for i in range(3) for j in range(i))
    assert Counter(Counter(schedule['CONTROL'].ravel()).values()) == {4: 1344, 3: 256}
    assert Counter(Counter(schedule['GEOMETRY'].ravel()).values()) == {3: 2048}
    assert np.isin(schedule['mz48'], bundle['groups']['fit']).all()
    assert all(bundle['records'][int(i)]['role'] == 'TRAIN_CANDIDATE' for i in np.unique(schedule['mz48']))
    assert np.isin(schedule['OLD_NEG'], previous['train_ids']).all()
    assert not bundle['old_truth'][schedule['OLD_NEG'], schedule['query']].any()
    for cohort in COHORTS[:3]:
        assert not np.isin(schedule['OLD_NEG'], bundle['cohorts'][cohort]['global_ids']).any()
    return dict(status='PASS', steps=1536, target_presentations_per_arm=6144,
        target_presentations_per_profile=2048, seed=164,
        CONTROL_unique=1600, GEOMETRY_unique=2048,
        CONTROL_total_repeat_histogram={'3': 256, '4': 1344}, GEOMETRY_total_repeat_histogram={'3': 2048},
        CONTROL_extra_pairs_per_profile=224, CONTROL_extra_pairs_disjoint=True,
        old_negative_query_truth=False, replay_cycle_steps=600, replay_steps=1536,
        mz48_first_four_exact=True, old_negative_and_query_replay_exact=True,
        ALL_INVALID_fit_presentations=0, no_calibration_or_heldout=True)


def feature_index(bundle):
    schedule = bundle['schedule']
    ids = dict(mz48=np.unique(schedule['mz48']), old=np.unique(schedule['OLD_NEG']),
               mz55=np.unique(schedule['CONTROL']), mz61=np.unique(schedule['GEOMETRY']))
    allowed = dict(mz48=bundle['groups']['fit'], old=schedule['old_train_ids'],
                   mz55=bundle['mz55_groups']['fit'], mz61=bundle['mz61_groups']['fit'])
    index = {}; offset = 0
    for source in SOURCES:
        assert np.isin(ids[source], allowed[source]).all()
        size = len(bundle['old_truth']) if source == 'old' else len(bundle['cohorts'][source]['frame_ids'])
        lookup = np.full(size, -1, np.int32)
        lookup[ids[source]] = np.arange(len(ids[source]), dtype=np.int32) + offset
        index[source + '_ids'], index[source + '_lookup'] = ids[source], lookup
        offset += len(ids[source])
    shape = (offset, *FEATURE_SHAPE)
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, dict(descr='<f4', fortran_order=False, shape=shape))
    frames, hits = {}, {}
    for cohort in COHORTS:
        rows = bundle['cohorts'][cohort]
        frames[cohort] = len(rows['frame_ids'])
        if cohort in COHORTS[:3]:
            cached = index['old_lookup'][rows['global_ids']]
        elif cohort in ('mz48', 'mz55', 'mz61'):
            cached = index[cohort + '_lookup']
        else:
            cached = np.full(frames[cohort], -1)
        hits[cohort] = int((cached >= 0).sum())
    misses = sum(frames.values()) - sum(hits.values())
    plan = dict(training_unique={source: len(ids[source]) for source in SOURCES},
        training_unique_total=offset, shape=list(shape), dtype='float32', bytes_per_frame=FRAME_BYTES,
        data_bytes=offset * FRAME_BYTES, npy_header_bytes=len(header.getvalue()),
        expected_file_bytes=offset * FRAME_BYTES + len(header.getvalue()), fixed_encoder_batch=16,
        evaluation_frames=frames, evaluation_cache_hits=hits, uncached_evaluation_frames=misses,
        planned_full_rgb_encodes=offset + misses, normalization='Original encoder values; runner applies frozen mean/std',
        scratch_owner=TASK, prior_full_cache_reused=False, calibration_or_heldout_in_training=False,
        baseline_replay_frames=0, extraction_unit='One per unique registered source/frame RGB reference',
        evaluation_call_contract='One batch outside arm/profile loops; reuse returned values for every arm/profile')
    return index, plan


def prepare(root, bind):
    started = time.perf_counter()
    root = Path(root); work = root / 'artifacts.local/work'
    refs = {}; receipts = {}
    for name, (relative, digest) in PREVIOUS.items():
        path = Path(bind(work / relative / 'receipt.json', digest)).resolve()
        receipts[name] = read(path); assert receipts[name]['status'] == 'PASS'
        refs[name] = dict(path=str(path), sha256=digest)
    for name in ('mz62_exposure_source.py', 'mz60_source.py', 'mz59_source.py'):
        matches = [(p, h) for p, h in receipts['mz62_run']['inputs'].items() if Path(p).name == name]
        assert len(matches) == 1; bind(*matches[0])
    frozen63 = [(p, h) for p, h in receipts['mz63_run']['inputs'].items() if Path(p).name == 'mz63_geometry_transfer.py']
    assert len(frozen63) == 1; bind(*frozen63[0]); bind(Path(__file__))
    bundle = prepare62(root, bind)
    previous = bundle['previous_schedule']
    prior62 = work / PREVIOUS['mz62_run'][0]
    r62 = receipts['mz62_run']
    saved62 = load_npz(bind(prior62 / 'predictions.npz', r62['outputs']['predictions.npz']))
    for key, expected in bundle['predictions'].items():
        exact(saved62[key], expected, key)
    inherited62 = load_npz(bind(prior62 / 'schedule.npz', r62['outputs']['schedule.npz']))
    assert set(inherited62) == set(bundle['schedule'])
    for key, expected in bundle['schedule'].items():
        exact(inherited62[key], expected, key)
    groups_path = Path(bind(prior62 / 'groups.json', r62['outputs']['groups.json']))
    groups62 = read(groups_path)
    for name in ('records', 'mz55_records'):
        assert groups62[name] == bundle[name]
    for name in ('groups', 'mz55_groups'):
        assert set(groups62[name]) == set(bundle[name])
        for key, ids in bundle[name].items():
            np.testing.assert_array_equal(groups62[name][key], ids)

    source = work / 'mz61-geometry-source-20260911'
    predictor, index_path = load_predictor(source, bind, SOURCE_INDEX_SHA)
    assert set(predictor) == {'frame_ids', 'ranges', 'valid', 'rgb_refs'}
    index = read(index_path); combined = index['combined']
    sr = read(bound_ref(source, combined['receipt.json'], bind))
    paths = {}
    for name in ('metadata.json', 'evaluator.npz', 'fullframe-cells.npz'):
        assert sr['outputs'][name] == combined[name]['sha256']
        paths[name] = bound_ref(source, combined[name], bind)
    meta = read(paths['metadata.json']); records, pairs = meta['records'], meta['pairs']
    labels, cells = load_npz(paths['evaluator.npz']), load_npz(paths['fullframe-cells.npz'])
    assert meta['source_role'] == 'CONSUMED_DEVELOPMENT' and len(records) == 4096
    for actual in ([r['frame_id'] for r in records], labels['frame_ids'], cells['frame_ids']):
        np.testing.assert_array_equal(actual, predictor['frame_ids'])
    np.testing.assert_array_equal([r['index'] for r in records], np.arange(4096))
    np.testing.assert_array_equal(cells['global_indices'], np.arange(4096))
    truth, known = labels['truth'], labels['known']
    counts, valid_counts = cells['fullframe_event_counts'], cells['valid_counts']
    assert truth.shape == known.shape == (4096, 4) and truth.dtype == known.dtype == bool
    assert counts.shape == (4096, 45, 80, 4) and valid_counts.shape == (4096, 45, 80)
    assert counts.dtype == valid_counts.dtype == np.uint8
    assert (valid_counts <= 64).all() and (counts <= valid_counts[..., None]).all()
    np.testing.assert_array_equal(counts.sum((1, 2)), [r['event_counts'] for r in records])
    np.testing.assert_array_equal(counts.sum((1, 2)) >= 3, truth)
    np.testing.assert_array_equal(truth, [r['event_truth'] for r in records])
    np.testing.assert_array_equal(known, [[r['source_valid']] * 4 for r in records])
    roles = np.array([r['role'] for r in records])
    assert Counter(roles) == meta['roles'] == dict(TRAIN_CANDIDATE=2048, CALIBRATION=1024, HELDOUT_GEOMETRY=1024)
    groups = dict(fit=np.flatnonzero(roles == 'TRAIN_CANDIDATE'),
        calibration=np.flatnonzero(roles == 'CALIBRATION'), heldout_geometry=np.flatnonzero(roles == 'HELDOUT_GEOMETRY'))
    assert sorted(i for pair in pairs.values() for i in pair) == list(range(4096)) and len(pairs) == 2048
    geometry_roles = defaultdict(set)
    for i, row in enumerate(records):
        ref = predictor['rgb_refs'][i]
        assert row['rgb'] == ref.member and row['rgb_sha256'] == ref.sha256
        geometry_roles[row['geometry_id']].add(row['role'])
    assert all(len(assigned) == 1 for assigned in geometry_roles.values())
    for name, ids in pairs.items():
        assert len(ids) == 2 and {records[i]['pair_id'] for i in ids} == {name}
        assert len({roles[i] for i in ids}) == 1
        assert {records[i]['support_context'] for i in ids} == {'unsupported', 'supported'}
        np.testing.assert_array_equal(counts[ids[0]], counts[ids[1]])
        np.testing.assert_array_equal(truth[ids[0]], truth[ids[1]])
    for old in ('mz48', 'mz55'):
        assert not set(predictor['frame_ids']) & set(bundle['cohorts'][old]['frame_ids'])

    r63 = receipts['mz63_run']; prior63 = work / PREVIOUS['mz63_run'][0]
    assert r63['source_index_sha256'] == SOURCE_INDEX_SHA and r63['frames'] == 4096
    saved63 = load_npz(bind(prior63 / 'predictions.npz', r63['outputs']['predictions.npz']))
    exact(saved63['frame_ids'], predictor['frame_ids'])
    predictions = dict(saved62)
    for key, value in saved63.items():
        assert 'mz61/' + key not in predictions
        predictions['mz61/' + key] = value
    predictions['mz61/truth'], predictions['mz61/known'] = truth, known
    bundle['predictions'] = predictions
    bundle['cohorts']['mz61'] = predictor
    bundle.update(mz61_records=records, mz61_groups=groups, mz61_pairs=pairs,
        mz61_truth=truth, mz61_known=known, mz61_cell_counts=counts, mz61_valid_counts=valid_counts,
        mz61_cell_truth=counts > 0, mz61_cell_known=valid_counts > 0)
    bundle['native_labels'] = dict(bundle['native_labels'],
        mz61=dict(cell_truth=bundle['mz61_cell_truth'], cell_known=bundle['mz61_cell_known']))
    bundle['schedule'] = make_schedule(previous, bundle)
    checks = validate_schedule(bundle['schedule'], previous, bundle)
    schedule_path = work / DESIGN / 'source-check-v1/schedule.npz'
    if schedule_path.exists():
        stored = load_npz(bind(schedule_path))
        assert set(stored) == set(bundle['schedule'])
        for key in stored: exact(stored[key], bundle['schedule'][key], key)
    bundle['candidate_schedule_path'] = schedule_path.resolve()
    bundle['inherited_groups_path'] = groups_path.resolve()
    bundle['inherited_mz62_schedule'] = inherited62
    bundle['feature_index'], plan = feature_index(bundle)
    index52 = read(bind(work / 'mz52-full-frame-supervision-20260911/source-index.json'))
    label48 = work / 'mz52-full-frame-supervision-20260911' / index52['array']['path']
    label48 = Path(bind(label48, index52['array']['sha256'])).resolve()
    source55 = Path(bundle['source_info']['source55_index'])
    index55 = read(source55)
    def reference(path, digest): return dict(path=str(Path(path).resolve()), sha256=digest)
    bundle['source_info'].update(adapter=Path(__file__).name, prepare_seconds=time.perf_counter()-started,
        scratch_owner=TASK, feature_plan=plan, training_unique=plan['training_unique'],
        training_unique_total=plan['training_unique_total'], feature_file_bytes=plan['expected_file_bytes'],
        candidate_schedule_path=str(schedule_path.resolve()),
        candidate_schedule_sha256=sha(schedule_path) if schedule_path.exists() else None,
        schedule_seed=SEED, schedule_checks=checks, schedule_seed159=False,
        schedule_design='seed164:800 sorted support pairs permuted once; disjoint224-pair extras; CONTROL then GEOMETRY permutation per profile',
        exact_mz59_schedule=False, mz59_replay_blocks_exact_twice=False,
        replay_description='Original600-step MZ59/MZ51 shared first4, OLD_NEG and query cyclically indexed to1536 steps',
        mz62_arrays_preserved=len(saved62), mz63_arrays_prefixed=len(saved63),
        mz62_predictions_sha256=r62['outputs']['predictions.npz'], mz63_predictions_sha256=r63['outputs']['predictions.npz'],
        previous_receipts=refs,
        label_refs=dict(mz48=reference(label48, index52['array']['sha256']),
            mz55=reference(source55.parent / index55['combined']['fullframe-cells.npz']['path'], index55['combined']['fullframe-cells.npz']['sha256']),
            mz61=reference(paths['fullframe-cells.npz'], combined['fullframe-cells.npz']['sha256'])),
        source_index_refs=dict(mz55=reference(source55, bundle['source_info']['source55_index_sha256']),
            mz61=reference(index_path, SOURCE_INDEX_SHA)),
        mz61_metadata_ref=reference(paths['metadata.json'], combined['metadata.json']['sha256']),
        mz61_evaluator_ref=reference(paths['evaluator.npz'], combined['evaluator.npz']['sha256']),
        mz61_groups={key:len(ids) for key,ids in groups.items()}, mz61_pairs=len(pairs),
        mz61_unknown_query_bits=int((~known).sum()), mz61_unknown_cells=int((valid_counts == 0).sum()),
        native_labels_authority='Three-source fullframe counts/known are loss/evaluator only',
        inherited_old_crop_access='prepare54 streams existing MZ16 crop SHA and opens header-only readonly mmap; prepare59 closes it without feature row reads',
        deleted_full_cache_reads=0, initialization_rgb_decodes=0, model_inference_frames=0,
        source61_native_depth_reads=0, temporary_dense_bytes_created=0,
        ALL_INVALID='Inherited MZ63 saved eval only; no fit or calibration presentations')
    assert bundle['old_maps'] is None
    return bundle


class FeatureStore(FeatureStore59):
    """Four-source owner; inherited encoder arithmetic and mmap close, no patching."""
    def __init__(self, base, store, bundle, task, bind):
        self.base, self.store, self.bundle, self.bind = base, store, bundle, bind
        self.task = Path(task).resolve()
        assert self.task == Path(bundle['source_info']['artifact_work']) / TASK
        self.index, self.plan = feature_index(bundle)
        assert self.plan == bundle['source_info']['feature_plan']
        for key, value in self.index.items(): exact(value, bundle['feature_index'][key], key)
        self.old_ids, self.mz48_ids, self.mz55_ids, self.mz61_ids = (self.index[s + '_ids'] for s in ('old', 'mz48', 'mz55', 'mz61'))
        self.path = self.task / 'scratch-v1/training-full.npy'
        self.out = self.task / 'feature-plan-v1'
        self.full = None
        self.stats = dict(training_extracted_frames=0, full_extracted_frames=0, full_cache_eval_hits=0,
                          evaluation_extracted_frames=0, encoder_seconds=0., build_seconds=0., evaluation_seconds=0.)

    def build(self):
        assert self.full is None and not self.path.exists() and not self.out.exists()
        assert not self.path.parent.exists(), 'Preserve prior scratch/failure evidence'
        self.task.mkdir(parents=True, exist_ok=True)
        assert shutil.disk_usage(self.task).free > self.plan['expected_file_bytes'] + 1024 ** 3
        self.out.mkdir(); self.path.parent.mkdir(); tick = time.perf_counter()
        try:
            np.savez_compressed(self.out / 'feature-cache-index.npz', **self.index)
            write_new(self.out / 'plan.json', self.plan)
            self.full = np.lib.format.open_memmap(self.path, mode='w+', dtype=np.float32, shape=tuple(self.plan['shape']))
            refs = []
            for source in SOURCES:
                original = self.bundle['old_rgb_refs'] if source == 'old' else self.bundle['cohorts'][source]['rgb_refs']
                refs.extend(original[int(i)] for i in self.index[source + '_ids'])
            for start in range(0, len(refs), 16):
                images = []
                try:
                    for ref in refs[start:start+16]:
                        images.append(self.store.load(ref))
                    self.full[start:start+len(images)] = self.extract(images)
                    self.stats['training_extracted_frames'] += len(images)
                finally:
                    for image in images: image.close()
                if start % 512 == 0 or start + len(images) == len(refs):
                    print('FEATURES', dict(frames=start+len(images), total=len(refs), seconds=time.perf_counter()-tick), flush=True)
            self.full.flush(); self.full._mmap.close(); self.full = None
            assert self.path.stat().st_size == self.plan['expected_file_bytes']
            self.full = np.load(self.path, mmap_mode='r', allow_pickle=False)
            assert self.full.mode == 'r' and not self.full.flags.writeable
            assert self.full.shape == tuple(self.plan['shape']) and self.full.dtype == np.float32
            assert self.stats['training_extracted_frames'] == self.plan['training_unique_total']
            self.cache_evidence = dict(path=str(self.path), bytes=self.path.stat().st_size, sha256=sha(self.path),
                feature_index_sha256=sha(self.out / 'feature-cache-index.npz'), owner=self.task.name,
                prior_full_cache_reused=False, readonly_after_build=True)
            self.stats['build_seconds'] = time.perf_counter() - tick
            write_new(self.out / 'receipt.json', dict(status='PASS', plan=self.plan, cache=self.cache_evidence,
                stats=self.stats, code_sha256=sha(__file__), inherited_encoder_code_sha256=sha(Path(__file__).with_name('mz59_source.py')),
                cleanup_owner='root after terminal score'))
            self.bind(self.out / 'feature-cache-index.npz', self.cache_evidence['feature_index_sha256'])
            self.bind(self.out / 'receipt.json')
            return self.stats['build_seconds']
        except BaseException:
            import traceback
            self.close()
            write_new(self.out / 'failure.json', dict(status='FAIL', error=traceback.format_exc(), stats=self.stats))
            raise

    def training(self, source, ids):
        assert self.full is not None and source in SOURCES
        ids = np.asarray(ids); lookup = self.index[source + '_lookup']
        assert ids.ndim == 1 and ids.dtype.kind in 'iu' and 0 < len(ids) <= 16
        assert ((ids >= 0) & (ids < len(lookup))).all()
        rows = lookup[ids]
        assert (rows >= 0).all(), 'Only registered training unique IDs may be read'
        return np.array(self.full[rows])

    def evaluation(self, cohort, ids):
        if cohort != 'mz61': return super().evaluation(cohort, ids)
        assert self.full is not None
        ids = np.asarray(ids); rows = self.bundle['cohorts'][cohort]
        assert ids.ndim == 1 and ids.dtype.kind in 'iu' and 0 < len(ids) <= 16
        assert ((ids >= 0) & (ids < len(rows['frame_ids']))).all()
        tick = time.perf_counter(); dense = np.empty((len(ids), *FEATURE_SHAPE), np.float32)
        images, missing = [], []
        try:
            for j, i in enumerate(ids):
                cached = self.index['mz61_lookup'][i]
                if cached >= 0:
                    dense[j] = self.full[cached]; self.stats['full_cache_eval_hits'] += 1
                else:
                    missing.append(j); images.append(self.store.load(rows['rgb_refs'][int(i)]))
            if missing:
                dense[missing] = self.extract(images)
                self.stats['evaluation_extracted_frames'] += len(missing)
            return dense
        finally:
            for image in images: image.close()
            self.stats['evaluation_seconds'] += time.perf_counter() - tick


def smoke(root, output):
    from mz45_object_transfer import Bindings
    output = Path(output).resolve()
    work = (Path(root) / 'artifacts.local/work').resolve()
    assert output == work / DESIGN / 'source-check-v1'
    output.mkdir(parents=True, exist_ok=False)
    bind = Bindings(); features = None; started = time.perf_counter()
    try:
        bundle = prepare(root, bind)
        task = work / TASK
        features = FeatureStore(None, None, bundle, task, bind)
        assert FeatureStore.extract is FeatureStore59.extract and FeatureStore.close is FeatureStore59.close
        rejections = []
        for key in ('CONTROL', 'GEOMETRY', 'mz48', 'OLD_NEG', 'query', 'profile_index'):
            corrupt = {k:v.copy() for k,v in bundle['schedule'].items()}
            corrupt[key].flat[0] = -12345
            try: validate_schedule(corrupt, bundle['previous_schedule'], bundle)
            except AssertionError: rejections.append(key)
            else: raise AssertionError('Corrupt schedule accepted: ' + key)
        try: FeatureStore(None, None, bundle, work / 'mz62-profile-coverage-20260911', bind)
        except AssertionError: rejections.append('cache_owner')
        else: raise AssertionError('Wrong cache owner accepted')
        for source, key in (('mz55', 'calibration'), ('mz55', 'heldout_site'), ('mz61', 'calibration'), ('mz61', 'heldout_geometry')):
            forbidden = bundle[source + '_groups'][key]
            assert (features.index[source + '_lookup'][forbidden] == -1).all()
        role_bad = dict(bundle)
        role_bad['mz61_records'] = list(bundle['mz61_records'])
        i = int(bundle['mz61_groups']['fit'][0])
        role_bad['mz61_records'][i] = dict(role_bad['mz61_records'][i], role='CALIBRATION')
        try: validate_schedule(bundle['schedule'], bundle['previous_schedule'], role_bad)
        except AssertionError: rejections.append('fit_role_leakage')
        else: raise AssertionError('Role leakage accepted')
        assert features.full is None and not features.path.exists()
        np.savez_compressed(output / 'schedule.npz', **bundle['schedule'])
        np.savez_compressed(output / 'feature-cache-index.npz', **bundle['feature_index'])
        bundle['source_info']['candidate_schedule_sha256'] = sha(output / 'schedule.npz')
        bind(output / 'schedule.npz', bundle['source_info']['candidate_schedule_sha256'])
        groups = {k:bundle[k] for k in ('records', 'mz55_records', 'mz61_records')}
        for key in ('groups', 'mz55_groups', 'mz61_groups'):
            groups[key] = {k:v.tolist() for k,v in bundle[key].items()}
        write_new(output / 'groups.json', groups)
        write_new(output / 'plan.json', bundle['source_info'])
        bind.check()
        features.close(); features = None
        write_new(output / 'receipt.json', dict(status='PASS', inputs=bind.inputs,
            outputs={name:sha(output/name) for name in ('schedule.npz', 'feature-cache-index.npz', 'groups.json', 'plan.json')},
            source_info=bundle['source_info'], corruption_rejections=rejections,
            forbidden_group_cache_entries=0, source_cohort_predictor_keys=['frame_ids', 'ranges', 'valid', 'rgb_refs'],
            inherited_encoder_arithmetic=True, global_module_monkeypatches=0,
            training_steps=0, encoder_frames=0, model_inference_frames=0, rgb_decodes=0,
            raw_native_depth_reads=0, deleted_full_cache_reads=0, temporary_dense_bytes_created=0,
            mmap_handles_closed=True, backend='FROZEN_PROTOCOL_CPU_ONLY', seconds=time.perf_counter()-started))
        print('PASS', dict(training_unique=bundle['source_info']['training_unique'],
            expected_cache_bytes=bundle['source_info']['feature_file_bytes'], arrays=len(bundle['predictions']),
            corruption_rejections=rejections, seconds=time.perf_counter()-started), flush=True)
    except BaseException:
        import traceback
        write_new(output / 'failure.json', dict(status='FAIL', error=traceback.format_exc(), inputs=bind.inputs))
        raise
    finally:
        if features is not None: features.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    smoke(args.root.resolve(), args.output.resolve())
