"""MZ59 CPU source assembly and explicitly started temporary full-RGB features.

``prepare(root, bind)`` has no encoder work. Legacy flat records/groups/cell
labels remain MZ48; their MZ55 counterparts use an ``mz55_`` prefix. Only
``cohorts`` and RGB references are observable predictor inputs. Labels and
metadata stay on the training/evaluator side. FeatureStore.build() is the
explicit GPU boundary; the constructor only computes its CPU index/plan.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import shutil
import time

import numpy as np

from mz54_full_rgb_source import prepare as prepare54, RGBStore
from mz58_diverse_transfer import load_predictor, bound_ref

COHORTS = ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48', 'mz55')
SOURCES = ('mz48', 'old', 'mz55')
FEATURE_SHAPE = (64, 45, 80)
FRAME_BYTES = int(np.prod(FEATURE_SHAPE)) * np.dtype(np.float32).itemsize


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def feature_index(bundle):
    """One global cache row space; no calibration/heldout row can enter it."""
    schedule = bundle['schedule']
    ids = dict(mz48=np.unique(schedule['shared']), old=np.unique(schedule['OLD_NEG']),
               mz55=np.unique(schedule['mz55_diverse']))
    assert np.isin(ids['mz48'], bundle['groups']['fit']).all()
    assert np.isin(ids['old'], schedule['train_ids']).all()
    assert np.isin(ids['mz55'], bundle['mz55_groups']['fit']).all()
    index = {}; offset = 0
    for source in SOURCES:
        size = len(bundle['old_truth']) if source == 'old' else 2560
        lookup = np.full(size, -1, np.int32)
        lookup[ids[source]] = np.arange(len(ids[source]), dtype=np.int32) + offset
        index[source + '_ids'] = ids[source]
        index[source + '_lookup'] = lookup
        offset += len(ids[source])
    shape = (offset, *FEATURE_SHAPE)
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, dict(descr='<f4', fortran_order=False, shape=shape))
    hits = {}
    for cohort in COHORTS:
        rows = bundle['cohorts'][cohort]
        if cohort in COHORTS[:3]:
            cached = index['old_lookup'][rows['global_ids']]
        elif cohort in ('mz48', 'mz55'):
            cached = index[cohort + '_lookup']
        else:
            cached = np.full(len(rows['frame_ids']), -1)
        hits[cohort] = int((cached >= 0).sum())
    frames = {cohort: len(bundle['cohorts'][cohort]['frame_ids']) for cohort in COHORTS}
    misses = sum(frames.values()) - sum(hits.values())
    plan = dict(training_unique={source: len(ids[source]) for source in SOURCES},
        training_unique_total=offset, shape=list(shape), dtype='float32', bytes_per_frame=FRAME_BYTES,
        data_bytes=offset * FRAME_BYTES, npy_header_bytes=len(header.getvalue()),
        expected_file_bytes=offset * FRAME_BYTES + len(header.getvalue()), fixed_encoder_batch=16,
        evaluation_frames=frames, evaluation_cache_hits=hits, uncached_evaluation_frames=misses,
        planned_full_rgb_encodes=offset + misses, normalization='Original encoder values; runner applies frozen mean/std',
        scratch_owner='mz59-training-diversity-20260911', prior_full_cache_reused=False,
        calibration_or_heldout_in_training=False, baseline_replay_frames=0)
    return index, plan


def prepare(root, bind):
    """CPU only. Preserve sealed baselines and construct the registered schedule.

    ``bind(path, expected_sha=None) -> Path`` verifies every consumed file.
    MZ54's original MZ16 crop mmap is immediately closed without reading its
    feature rows. No deleted MZ54/MZ56 full-cache path is opened or referenced.
    """
    started = time.perf_counter()
    root = Path(root); work = root / 'artifacts.local/work'
    for name in ('mz59_source.py', 'mz54_full_rgb_source.py', 'mz58_diverse_transfer.py'):
        bind(Path(__file__).with_name(name))
    bundle = prepare54(root, bind)
    bundle['old_maps']._mmap.close()
    bundle['old_maps'] = None
    prior = work / 'mz56-global-anchor-20260911/run-v2'
    receipt56 = read(bind(prior / 'receipt.json'))
    assert receipt56['status'] == 'PASS'
    predictions = load_npz(bind(prior / 'predictions.npz', receipt56['outputs']['predictions.npz']))
    for key, expected in bundle['predictions'].items():
        actual = predictions[key]
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, key
        np.testing.assert_array_equal(actual, expected, err_msg=key)
    preserved = len(bundle['predictions'])
    prior_schedule = load_npz(bind(prior / 'schedule.npz', receipt56['outputs']['schedule.npz']))
    assert set(prior_schedule) == set(bundle['schedule'])
    for key, expected in bundle['schedule'].items():
        np.testing.assert_array_equal(prior_schedule[key], expected, err_msg=key)

    source = work / 'mz55-diverse-mesh-source-20260911'
    predictor, index_path = load_predictor(source, bind)
    index = read(index_path); combined = index['combined']
    receipt55 = read(bound_ref(source, combined['receipt.json'], bind))
    paths = {}
    for name in ('evaluator.npz', 'fullframe-cells.npz', 'metadata.json'):
        assert receipt55['outputs'][name] == combined[name]['sha256']
        paths[name] = bound_ref(source, combined[name], bind)
    labels = load_npz(paths['evaluator.npz'])
    cells = load_npz(paths['fullframe-cells.npz'])
    metadata = read(paths['metadata.json']); records = metadata['records']
    assert metadata['source_role'] == 'CONSUMED_DEVELOPMENT'
    assert len(records) == 2560 and len(set(predictor['frame_ids'])) == 2560
    for frame_ids in (labels['frame_ids'], cells['frame_ids'], [r['frame_id'] for r in records]):
        np.testing.assert_array_equal(frame_ids, predictor['frame_ids'])
    np.testing.assert_array_equal(cells['global_indices'], np.arange(2560))
    np.testing.assert_array_equal([r['index'] for r in records], np.arange(2560))
    truth, known = labels['truth'], labels['known']
    counts, valid_counts = cells['fullframe_event_counts'], cells['valid_counts']
    assert truth.shape == known.shape == (2560, 4) and truth.dtype == known.dtype == bool
    assert counts.shape == (2560, 45, 80, 4) and counts.dtype == np.uint8
    assert valid_counts.shape == (2560, 45, 80) and valid_counts.dtype == np.uint8
    assert counts.max() <= 64 and valid_counts.max() <= 64 and (counts <= valid_counts[..., None]).all()
    np.testing.assert_array_equal(counts.sum((1, 2)), [r['event_counts'] for r in records])
    np.testing.assert_array_equal(counts.sum((1, 2)) >= 3, truth)
    np.testing.assert_array_equal(truth, [r['event_truth'] for r in records])
    np.testing.assert_array_equal(known, [[r['source_valid']] * 4 for r in records])
    for row, ref in zip(records, predictor['rgb_refs']):
        assert row['rgb_sha256'] == ref.sha256 and row['rgb'] == ref.member

    roles = np.array([r['role'] for r in records]); families = np.array([r['family'] for r in records])
    assert Counter(roles) == metadata['roles'] == dict(TRAIN_CANDIDATE=1600, CALIBRATION=320, HELDOUT_SITE=640)
    assert Counter(families) == {name: 640 for name in ('retained_rod', 'shallow_awning', 'sign_panel', 'square_grille')}
    sites = Counter(r['site_id'] for r in records)
    assert len(sites) == 8 and set(sites.values()) == {320}
    groups = dict(fit=np.flatnonzero(roles == 'TRAIN_CANDIDATE'),
        calibration=np.flatnonzero(roles == 'CALIBRATION'), heldout_site=np.flatnonzero(roles == 'HELDOUT_SITE'),
        heldout_new_forms=np.flatnonzero((roles == 'HELDOUT_SITE') & (families != 'retained_rod')),
        new_forms=np.flatnonzero(families != 'retained_rod'))
    assert [len(groups[k]) for k in groups] == [1600, 320, 640, 480, 1920]
    np.testing.assert_array_equal(np.sort(np.concatenate([groups[k] for k in ('fit', 'calibration', 'heldout_site')])), np.arange(2560))
    pairs = {}
    for i, row in enumerate(records):
        pairs.setdefault(row['pair_id'], []).append(i)
    assert pairs == metadata['pairs'] and len(pairs) == 1280
    for pair, ids in pairs.items():
        assert len(ids) == 2, pair
        a, b = ids
        assert roles[a] == roles[b] and families[a] == families[b]
        assert {records[i]['support_context'] for i in ids} == {'unsupported', 'supported'}
        for key in ('site_id', 'relation', 'range', 'setting'):
            assert records[a][key] == records[b][key]
        np.testing.assert_array_equal(counts[a], counts[b], err_msg=pair)
    assert not set(predictor['frame_ids']) & set(bundle['cohorts']['mz48']['frame_ids'])

    prior58 = work / 'mz58-diverse-transfer-20260911/run-v1'
    receipt58 = read(bind(prior58 / 'receipt.json'))
    assert receipt58['status'] == 'PASS'
    saved58 = load_npz(bind(prior58 / 'predictions.npz', receipt58['outputs']['predictions.npz']))
    np.testing.assert_array_equal(saved58['frame_ids'], predictor['frame_ids'])
    for key, value in saved58.items():
        assert 'mz55/' + key not in predictions
        predictions['mz55/' + key] = value
    predictions['mz55/truth'], predictions['mz55/known'] = truth, known
    bundle['predictions'] = predictions
    bundle['cohorts']['mz55'] = predictor
    bundle.update(mz55_records=records, mz55_groups=groups, mz55_truth=truth, mz55_known=known,
                  mz55_cell_truth=counts > 0, mz55_cell_known=valid_counts > 0)
    schedule = dict(bundle['schedule'])
    schedule['mz55_diverse'] = np.random.default_rng(159).choice(groups['fit'], (600, 4), replace=True)
    assert np.isin(schedule['mz55_diverse'], groups['fit']).all()
    assert not np.isin(schedule['mz55_diverse'], np.r_[groups['calibration'], groups['heldout_site']]).any()
    bundle['schedule'] = schedule
    bundle['feature_index'], plan = feature_index(bundle)
    bundle['source_info'].update(prepare_seconds=time.perf_counter()-started,
        artifact_work=str(work.resolve()), source55_index=str(Path(index_path).resolve()), source55_index_sha256=sha(index_path),
        source55_loaded_label_file_bytes={name: Path(path).stat().st_size for name, path in paths.items()},
        mz51_arrays_preserved=preserved, mz56_arrays_preserved=len(predictions) - len(saved58) - 2,
        mz58_arrays_prefixed=len(saved58), mz55_pairs=1280, mz55_groups={k: len(v) for k, v in groups.items()},
        mz55_unknown_query_bits=int((~known).sum()), mz55_unknown_cells=int((valid_counts == 0).sum()),
        source55_native_depth_loads=0, initialization_rgb_decodes=0, model_inference_frames=0,
        old_crop_mmap_closed=True, old_crop_feature_rows_read=0, prior_full_cache_reads=0,
        schedule_seed159=True, feature_plan=plan, training_unique=plan['training_unique'],
        training_unique_total=plan['training_unique_total'], feature_file_bytes=plan['expected_file_bytes'])
    return bundle


class FeatureStore:
    """Task-owned float32 cache; caller owns encoder and RGBStore lifetimes.

    Constructor is CPU only. ``build()`` explicitly encodes training unique
    rows; call once after GPU GO. ``training(source, ids)`` accepts old/mz48/
    mz55 IDs. ``evaluation(cohort, ids)`` takes cohort row indices, at most16,
    and must be called once outside the arm/profile loops. It returns raw
    encoder float32 values; the runner keeps the existing mean/std contract.
    ``close()`` releases our mmap; root removes scratch after terminal scoring.
    """
    def __init__(self, base, store, bundle, task, bind):
        self.base, self.store, self.bundle, self.bind = base, store, bundle, bind
        self.task = Path(task).resolve()
        assert self.task == Path(bundle['source_info']['artifact_work']) / 'mz59-training-diversity-20260911'
        self.index, self.plan = feature_index(bundle)
        for key, value in self.index.items():
            np.testing.assert_array_equal(value, bundle['feature_index'][key])
        self.old_ids, self.mz48_ids, self.mz55_ids = (self.index[s + '_ids'] for s in ('old', 'mz48', 'mz55'))
        self.path = self.task / 'scratch-v1/training-full.npy'
        self.out = self.task / 'feature-plan-v1'
        self.full = None
        self.stats = dict(training_extracted_frames=0, full_extracted_frames=0, full_cache_eval_hits=0,
                          evaluation_extracted_frames=0, encoder_seconds=0., build_seconds=0., evaluation_seconds=0.)

    def extract(self, images):
        import torch
        from mz36_frozen_inference import fixed_batch_dense
        assert 0 < len(images) <= 16
        tick = time.perf_counter()
        rgb = torch.from_numpy(np.stack([np.asarray(image, dtype=np.uint8) for image in images]))
        assert rgb.shape == (len(images), 360, 640, 3)
        rgb = rgb.permute(0, 3, 1, 2).cuda().float() / 255
        with torch.inference_mode():
            dense = fixed_batch_dense(self.base, rgb).cpu().numpy()
        assert dense.shape == (len(images), *FEATURE_SHAPE) and dense.dtype == np.float32
        assert np.isfinite(dense).all()
        self.stats['full_extracted_frames'] += len(images)
        self.stats['encoder_seconds'] += time.perf_counter() - tick
        return dense

    def build(self):
        assert self.full is None and not self.path.exists() and not self.out.exists()
        assert not self.path.parent.exists(), 'Preserve prior scratch/failure evidence'
        self.task.mkdir(parents=True, exist_ok=True)
        assert shutil.disk_usage(self.task).free > self.plan['expected_file_bytes'] + 1024 ** 3
        self.out.mkdir(); self.path.parent.mkdir()
        tick = time.perf_counter()
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
                    for image in images:
                        image.close()
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
                stats=self.stats, code_sha256=sha(__file__), cleanup_owner='root after terminal score'))
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
        ids = np.asarray(ids)
        lookup = self.index[source + '_lookup']
        assert ids.ndim == 1 and ids.dtype.kind in 'iu' and 0 < len(ids) <= 16
        assert ((ids >= 0) & (ids < len(lookup))).all()
        rows = lookup[ids]
        assert (rows >= 0).all(), 'Only registered training unique IDs may be read as training features'
        return np.array(self.full[rows])

    def evaluation(self, cohort, ids):
        assert self.full is not None and cohort in COHORTS
        ids = np.asarray(ids); rows = self.bundle['cohorts'][cohort]
        assert ids.ndim == 1 and ids.dtype.kind in 'iu' and 0 < len(ids) <= 16
        assert ((ids >= 0) & (ids < len(rows['frame_ids']))).all()
        tick = time.perf_counter(); dense = np.empty((len(ids), *FEATURE_SHAPE), np.float32)
        images, missing = [], []
        try:
            for j, i in enumerate(ids):
                cached = -1
                if cohort in COHORTS[:3]:
                    cached = self.index['old_lookup'][int(rows['global_ids'][i])]
                elif cohort in ('mz48', 'mz55'):
                    cached = self.index[cohort + '_lookup'][i]
                if cached >= 0:
                    dense[j] = self.full[cached]
                    self.stats['full_cache_eval_hits'] += 1
                else:
                    missing.append(j); images.append(self.store.load(rows['rgb_refs'][int(i)]))
            if missing:
                dense[missing] = self.extract(images)
                self.stats['evaluation_extracted_frames'] += len(missing)
            return dense
        finally:
            for image in images:
                image.close()
            self.stats['evaluation_seconds'] += time.perf_counter() - tick

    def close(self):
        if self.full is not None:
            self.full._mmap.close()
            self.full = None


def smoke(root, output):
    """A single CPU assembly check; no RGB decode, encoder or cache allocation."""
    from mz45_object_transfer import Bindings
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    bind = Bindings()
    try:
        bundle = prepare(root, bind)
        np.savez_compressed(output / 'schedule.npz', **bundle['schedule'])
        np.savez_compressed(output / 'feature-cache-index.npz', **bundle['feature_index'])
        write_new(output / 'plan.json', bundle['source_info'])
        write_new(output / 'receipt.json', dict(status='PASS', scope='CPU source/split/schedule/hash verification only',
            source_info=bundle['source_info'], inputs=bind.inputs,
            outputs={name: sha(output / name) for name in ('schedule.npz', 'feature-cache-index.npz', 'plan.json')},
            training_steps=0, model_inference_frames=0, rgb_decodes=0, temporary_dense_bytes_created=0,
            mmap_handles_closed=True, GPU_feature_store_not_executed=True))
        print(json.dumps(dict(status='PASS', **bundle['source_info'])), flush=True)
    except BaseException:
        import traceback
        write_new(output / 'failure.json', dict(status='FAIL', error=traceback.format_exc(), inputs=bind.inputs))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    smoke(args.root.resolve(), args.output.resolve())
