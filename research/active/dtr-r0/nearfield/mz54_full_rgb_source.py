"""CPU-only MZ54 metadata adapter; no native depth, encoder, fit or calibration.

prepare(root, bind) returns saved MZ51 predictions/schedule, MZ52 local labels,
old read-only crop mmap and six observable cohorts. Use ``with RGBStore() as s``
and ``s.load(ref)`` for SHA-checked, detached original 640x360 RGB images.
Records and labels are training/evaluator authorities, never model inputs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from data_lightweight import CompactSource

COHORTS = ('DEV', 'relation10000', 'distance5000', 'rich', 'mz36', 'mz48')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _npz(path, keys=None):
    # Read only named members; never materialize old local supervision/visuals.
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in (z.files if keys is None else keys)}


def _key(path):
    return str(Path(path).resolve()).casefold()


@dataclass(frozen=True)
class RGBRef:
    path: str
    sha256: str
    member: str | None = None

    @property
    def archive(self):
        return self.path if self.member is not None else None


class RGBStore:
    """Cached compact handles, no image cache; every requested PNG is hashed."""
    def __init__(self):
        self.handles = {}
        self.loads = 0
        self.bytes_read = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        for source in self.handles.values():
            source.close()
        self.handles.clear()

    def load(self, ref):
        assert isinstance(ref, RGBRef)
        if ref.member is None:
            assert Path(ref.path).suffix.lower() == '.png'
            data = Path(ref.path).read_bytes()
        else:
            assert ref.member.startswith('model/') and ref.member.endswith('.png')
            key = _key(ref.path)
            if key not in self.handles:
                self.handles[key] = CompactSource(ref.path)
            source = self.handles[key]
            assert source.entries[ref.member]['sha256'] == ref.sha256
            data = source.read_bytes(ref.member)
        assert hashlib.sha256(data).hexdigest() == ref.sha256, ref
        with Image.open(io.BytesIO(data)) as image:
            assert image.size == (640, 360), (ref, image.size)
            result = image.convert('RGB').copy()
        self.loads += 1
        self.bytes_read += len(data)
        return result


def prepare(root, bind):
    """Bind and verify inputs, retaining small arrays and one read-only mmap.

    ``bind(path, expected_sha=None)->Path`` follows mz45 Bindings. No setup or
    model imports occur here. Whole-file SHA verification streams archives and
    the existing mmap once; no PNG is decoded during source initialization.
    Caller owns ``bundle['old_maps']._mmap.close()`` after use.
    """
    started = time.perf_counter()
    root = Path(root); work = root / 'artifacts.local/work'
    cache = {}; expected_hashes = {}

    def absorb(mapping):
        if not isinstance(mapping, dict):
            # Older cache receipts have per-frame records in inputs, not a
            # file->SHA map. Their named output hashes bind consumed arrays.
            return
        for path, digest in mapping.items():
            if isinstance(digest, str) and len(digest) == 64:
                key = _key(path)
                assert key not in expected_hashes or expected_hashes[key] == digest, path
                expected_hashes[key] = digest

    def bound(path, expected=None):
        path = Path(path); key = _key(path)
        expected = expected or expected_hashes.get(key)
        if key not in cache:
            cache[key] = (Path(bind(path, expected)), _sha(path) if expected is None else expected)
        if expected is not None:
            assert cache[key][1] == expected, path
        return cache[key][0]

    def receipt(folder):
        value = _read(bound(folder / 'receipt.json'))
        assert value['status'] == 'PASS', folder
        absorb(value.get('inputs', {}))
        return value

    def output(folder, name, rec=None):
        rec = receipt(folder) if rec is None else rec
        hashes = rec.get('outputs', rec.get('files', {}))
        return bound(folder / name, hashes[name])

    bind(Path(__file__))
    design = work / 'mz53-full-rgb-design-20260911'
    bound(design / 'proposal.md')
    inventory = _read(bound(design / 'inventory.json'))
    absorb({str(root / p): h for p, h in inventory['inspected_bindings'].items()})
    prior = work / 'mz51-training-coverage-20260911/run-v1'
    r51 = receipt(prior)
    predictions = _npz(output(prior, 'predictions.npz', r51))
    metadata = _read(output(prior, 'groups.json', r51))
    records = metadata['records']
    groups = {k: np.asarray(v, dtype=np.int64) for k, v in metadata['groups'].items()}
    schedule = _npz(output(prior, 'schedule.npz', r51))
    assert len(records) == 2560
    np.testing.assert_array_equal(predictions['mz48/frame_ids'], [r['frame_id'] for r in records])
    assert len(set(predictions['mz48/frame_ids'])) == 2560
    # Independently restate the frozen split; do not import the training runner.
    assigned = {}
    split = {k: [] for k in ('fit', 'calibration', 'heldout_site', 'nonfit_family')}
    for i, r in enumerate(records):
        group = ('heldout_site' if r['role'] == 'HELDOUT_SITE' else
                 'nonfit_family' if r['family'] == 'birch' else
                 'calibration' if r['site_id'] == 'mz36_dense_candidate_06_site_002' else 'fit')
        split[group].append(i)
        assert assigned.setdefault(r['pair_id'], group) == group
    assert len(assigned) == 1280
    for name, size in zip(split, (1280, 256, 640, 384)):
        np.testing.assert_array_equal(groups[name], split[name]); assert len(groups[name]) == size

    s52 = work / 'mz52-full-frame-supervision-20260911'
    r52 = receipt(s52); absorb(r52['source_inputs'])
    index52 = _read(output(s52, 'source-index.json', r52))
    assert index52['status'] == 'PASS' and index52['frames'] == 2560
    paths52 = {k: bound(s52 / index52[k]['path'], index52[k]['sha256'])
               for k in ('array', 'schema_definition', 'records', 'validation')}
    assert _read(paths52['validation'])['status'] == 'PASS'
    manifest52 = _read(paths52['records']); absorb(manifest52['source_inputs'])
    a52 = _npz(paths52['array'])
    np.testing.assert_array_equal(a52['frame_ids'], predictions['mz48/frame_ids'])
    np.testing.assert_array_equal(a52['global_indices'], np.arange(2560))
    counts, valid_counts = a52['fullframe_event_counts'], a52['valid_counts']
    assert counts.shape == (2560, 45, 80, 4) and counts.dtype == np.uint8
    assert valid_counts.shape == (2560, 45, 80) and valid_counts.dtype == np.uint8
    np.testing.assert_array_equal(counts.sum((1, 2)) >= 3, predictions['mz48/truth'])
    cell_truth, cell_known = counts > 0, valid_counts > 0
    assert not (cell_truth & ~cell_known[..., None]).any()
    schema52 = _read(paths52['schema_definition'])
    for name in groups:
        np.testing.assert_array_equal(schema52['partitions'][name], groups[name])

    s48 = work / 'mz48-rich-kilotier-20260911'
    total48 = _read(bound(s48 / 'source-total-receipt.json'))
    assert total48['status'] == 'PASS' and total48['source_valid_frames'] == 2560
    index48 = _read(bound(s48 / 'source-index.json', total48['source_index']['sha256']))
    assert index48['status'] == 'COMPLETE' and len(index48['shards']) == 10
    frozen = _read(bound(s48 / index48['spec_manifest']['path'], index48['spec_manifest']['sha256']))
    assert frozen['total_frames'] == 2560
    packet_by_frame = {}
    for shard in index48['shards']:
        ref = shard['archive']; path = bound(s48 / ref['path'], ref['sha256'])
        with CompactSource(path) as source:
            def member(name):
                data = source.read_bytes(name)
                assert hashlib.sha256(data).hexdigest() == source.entries[name]['sha256']
                return data
            meta = json.loads(member('evaluator/metadata.json'))
            p = _npz(io.BytesIO(member('model/packets.npz')))
            for j, r in enumerate(meta['records']):
                assert r['frame_id'] not in packet_by_frame and r['index'] == j
                assert source.entries[r['rgb']]['sha256'] == r['rgb_sha256']
                packet_by_frame[r['frame_id']] = (r, p['ranges'][j], p['valid'][j],
                    RGBRef(str(path), r['rgb_sha256'], r['rgb']))
    rr, vv, refs = [], [], []
    assert len(packet_by_frame) == 2560
    for i, r in enumerate(records):
        row, ranges, valid, ref = packet_by_frame[r['frame_id']]
        for key in ('frame_id', 'pair_id', 'site_id', 'family', 'role', 'rgb_sha256', 'event_counts', 'event_truth'):
            assert row[key] == r[key], (i, key)
        np.testing.assert_array_equal(manifest52['rows'][i]['event_counts'], r['event_counts'])
        assert manifest52['rows'][i]['frame_id'] == r['frame_id']
        np.testing.assert_array_equal(predictions['mz48/known'][i], [row['source_valid']] * 4)
        rr.append(ranges); vv.append(valid); refs.append(ref)
    cohorts = {'mz48': dict(frame_ids=predictions['mz48/frame_ids'], ranges=np.stack(rr), valid=np.stack(vv), rgb_refs=refs)}

    old_parts = []
    for name in ('mz8-attribution-20260910/cache-v5', 'mz15-shared-support-20260910/cache-v1'):
        folder = work / name; rec = receipt(folder)
        obs = _npz(output(folder, 'observations.npz', rec), ('ranges', 'valid'))
        truth = _npz(output(folder, 'evaluator.npz', rec), ('truth',))['truth']
        old_parts.append(dict(**obs, truth=truth))
    old = {k: np.concatenate([p[k] for p in old_parts]) for k in ('ranges', 'valid', 'truth')}
    old_cache = work / 'mz16-visual-detail-20260910/cache-v2'; rec = receipt(old_cache)
    selected = _read(output(old_cache, 'selected.json', rec))
    map_ids = np.load(output(old_cache, 'ids.npy', rec), allow_pickle=False)
    maps_path = output(old_cache, 'dense_HIGH_DETAIL.npy', rec)
    lookup = np.full(len(old['truth']), -1, np.int64); lookup[map_ids] = np.arange(len(map_ids))
    np.testing.assert_array_equal(map_ids, [r['global_id'] for r in selected])
    old_refs = {r['global_id']: RGBRef(r['rgb'], r['rgb_sha']) for r in selected}
    for name in COHORTS[:3]:
        ids = predictions[name + '/frame_ids']
        assert np.issubdtype(ids.dtype, np.integer) and (lookup[ids] >= 0).all()
        np.testing.assert_array_equal(old['truth'][ids], predictions[name + '/truth'])
        assert not np.isin(ids, schedule['train_ids']).any()
        cohorts[name] = dict(frame_ids=ids, global_ids=ids, ranges=old['ranges'][ids],
                             valid=old['valid'][ids], rgb_refs=[old_refs[int(i)] for i in ids])
    assert len(schedule['train_ids']) == 7562 and (lookup[schedule['train_ids']] >= 0).all()
    assert schedule['shared'].shape == schedule['OLD_NEG'].shape == schedule['query'].shape == (600, 8)
    np.testing.assert_array_equal(schedule['fit_ids'], groups['fit'])
    np.testing.assert_array_equal(schedule['query'], np.tile(np.repeat(np.arange(4), 2), (600, 1)))
    assert np.isin(schedule['shared'], groups['fit']).all()
    assert np.isin(schedule['OLD_NEG'], schedule['train_ids']).all()
    assert not old['truth'][schedule['OLD_NEG'], schedule['query']].any()
    assert len(np.unique(schedule['OLD_NEG'])) == 3533 and len(np.unique(schedule['shared'])) == 1250

    rich = {}
    for eid, take in (('mz45-object-transfer-20260911', slice(None)), ('mz46-scaffold-context-20260911', slice(4, 8))):
        folder = work / eid / 'prepared-v1'; rec = receipt(folder)
        rows = _read(output(folder, 'predictor.json', rec))['frames']
        pp = _npz(output(folder, 'packets.npz', rec), ('frame_ids', 'ranges', 'valid'))
        np.testing.assert_array_equal(pp['frame_ids'], [r['frame_id'] for r in rows])
        for i in np.arange(len(rows))[take]:
            r = rows[i]; path = bound(r['archive'])
            with CompactSource(path) as archive:
                ref = RGBRef(str(path), archive.entries[r['rgb']]['sha256'], r['rgb'])
            rich[r['frame_id']] = (pp['ranges'][i], pp['valid'][i], ref)
    ids = predictions['rich/frame_ids']; assert set(ids) == set(rich) and len(ids) == 44
    cohorts['rich'] = dict(frame_ids=ids, ranges=np.stack([rich[i][0] for i in ids]),
                          valid=np.stack([rich[i][1] for i in ids]), rgb_refs=[rich[i][2] for i in ids])
    s36 = work / 'mz36-new-source-20260910'; rec = receipt(s36 / 'inference-v1')
    p36 = _npz(output(s36 / 'inference-v1', 'predictions.npz', rec), ('frame_ids', 'ranges', 'valid'))
    rows = _read(bound(s36 / 'admission-v1/predictor-manifest.json'))['frames']
    np.testing.assert_array_equal(p36['frame_ids'], predictions['mz36/frame_ids'])
    np.testing.assert_array_equal(p36['frame_ids'], [r['frame_id'] for r in rows])
    refs = [RGBRef(r['rgb_path'], expected_hashes[_key(r['rgb_path'])]) for r in rows]
    cohorts['mz36'] = dict(**p36, rgb_refs=refs)
    assert len(refs) == 380 and predictions['mz36/attempted_known'].shape == (400, 4)
    assert int((~predictions['mz36/attempted_known']).sum()) == 80
    for name, data in cohorts.items():
        n = len(predictions[name + '/frame_ids'])
        assert data['ranges'].shape == data['valid'].shape == (n, 64, 2)
        assert data['ranges'].dtype == np.float32 and data['valid'].dtype == bool
        assert len(data['rgb_refs']) == n
        np.testing.assert_array_equal(data['frame_ids'], predictions[name + '/frame_ids'])
    maps = np.load(maps_path, mmap_mode='r', allow_pickle=False)
    assert maps.shape == (11562, 64, 28, 28) and maps.dtype == np.float32
    return dict(predictions=predictions, records=records, groups=groups, schedule=schedule,
                cell_truth=cell_truth, cell_known=cell_known, old_maps=maps, old_lookup=lookup,
                old_rgb_refs=old_refs, old_ranges=old['ranges'], old_valid=old['valid'], old_truth=old['truth'],
                cohorts=cohorts, rgb_store_class=RGBStore,
                source_info=dict(prepare_seconds=time.perf_counter()-started, metadata_bound_files=len(cache),
                    initialization_rgb_decodes=0, native_depth_loads=0, old_local_label_arrays_loaded=0,
                    model_inference_frames=0, training_steps=0, old_dense_resident_copy=False))


def smoke(root, output):
    """Metadata/ordering verification plus exactly two original RGB decodes."""
    class Bindings:
        def __init__(self): self.inputs = {}
        def __call__(self, path, expected=None):
            path = Path(path); digest = _sha(path)
            assert expected is None or digest == expected, path
            self.inputs[str(path)] = digest
            return path
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    bind = Bindings(); bundle = None
    try:
        bundle = prepare(root, bind)
        with RGBStore() as store:
            samples = [bundle['old_rgb_refs'][int(bundle['schedule']['OLD_NEG'][0, 0])],
                       bundle['cohorts']['mz48']['rgb_refs'][0]]
            sizes = []
            for ref in samples:
                with store.load(ref) as image:
                    sizes.append(list(image.size))
            rgb_bytes = store.bytes_read
        result = dict(status='PASS', scope='Metadata and exactly two SHA-checked original RGB decodes; no model or native depth',
            **bundle['source_info'], cohort_frames={k: len(v['frame_ids']) for k, v in bundle['cohorts'].items()},
            partitions={k: len(v) for k, v in bundle['groups'].items()}, rgb_decodes=2, rgb_bytes=rgb_bytes,
            rgb_sizes=sizes, samples=[vars(r) for r in samples], unknown_cells=int((~bundle['cell_known']).sum()),
            inputs=bind.inputs, resources_released=True)
        (output / 'receipt.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k not in ('inputs', 'samples')}), flush=True)
    except BaseException:
        import traceback
        (output / 'failure.json').write_text(json.dumps(dict(status='FAIL', error=traceback.format_exc()), indent=2), encoding='utf-8')
        raise
    finally:
        if bundle is not None:
            bundle['old_maps']._mmap.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    smoke(args.root.resolve(), args.output.resolve())
