"""CPU reader for the complete, explicitly bound MZ48 compact source.

``data = load_source(mz48_task_root, source_index_dict_or_json_path)`` requires
all 2,560 frames. Only ``data['predictor']`` belongs on the prediction side:
it contains ranges, validity and original RGB archive/member references.
Records, global/local labels and split indices are training/evaluator data.

The index schema is ``mz48-source-index-v1`` with status ``COMPLETE``, a
``spec_manifest`` and ten ``shards`` in any order. Each shard has ``shard_id``
and refs named ``archive``, ``dataset_receipt``, ``dataset_result``,
``auxiliary``, ``auxiliary_receipt``. A ref is {"path": relative_path,
"sha256": lowercase_hex}. ``capture_spec`` is required only when the capture
uses the primary host's map-file substitution; no other change is accepted.
Index order does not choose data order: the frozen spec manifest does.

``load_shard_for_check`` is a separate engineering check for an unfinished
delivery, never a reduced-denominator version of ``load_source``. No native
depth, dense features, torch, GPU, fit or inference is used by this module.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import re
import stat

import numpy as np

from data_lightweight import CompactSource, relative_path


SCHEMA = 'mz48-source-index-v1'
EVENT_ORDER = ('BODY_NEAR', 'BODY_FAR', 'HEAD_NEAR', 'HEAD_FAR')
SITES = frozenset('mz36_dense_candidate_' + value for value in (
    '05_site_001', '05_site_002', '05_site_004', '05_site_007',
    '06_site_001', '06_site_002', '06_site_003', '06_site_004'))
HOLDOUT = frozenset(('mz36_dense_candidate_05_site_002',
                     'mz36_dense_candidate_06_site_004'))
FAMILIES = frozenset(('pipe', 'ladder', 'pouch', 'birch', 'oblique_rod'))
RELATIONS = frozenset(('HEAD_ONLY', 'BODY_ONLY', 'BOTH', 'CLEAR'))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _path(root, relative):
    path = root
    for part in relative_path(relative).split('/'):
        path = path / part
        info = path.stat(follow_symlinks=False)
        _require(not stat.S_ISLNK(info.st_mode) and
                 not getattr(info, 'st_file_attributes', 0) & 1024,
                 f'Reparse/symlink source is not admitted: {path}')
    _require(path.is_file(), f'Expected an ordinary source file: {path}')
    return path


def _bound(root, ref):
    _require(isinstance(ref, dict) and set(ref) >= {'path', 'sha256'},
             'Expected a source reference with path and sha256')
    _require(re.fullmatch('[0-9a-f]{64}', ref['sha256']) is not None,
             f'Invalid SHA-256 binding: {ref["path"]}')
    path = _path(root, ref['path'])
    before = path.stat()
    _require(_sha(path) == ref['sha256'], f'Hash mismatch: {path}')
    after = path.stat()
    _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
             f'Source changed during hashing: {path}')
    return path


def _json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _member(source, name):
    data = source.read_bytes(name)
    _require(hashlib.sha256(data).hexdigest() == source.entries[name]['sha256'],
             f'Package member hash mismatch: {name}')
    return data


def _array(value, shape, dtype, name):
    _require(value.shape == shape and value.dtype == np.dtype(dtype),
             f'{name}: expected {shape}/{np.dtype(dtype)}, got {value.shape}/{value.dtype}')
    return value


def _index(root, bindings, complete):
    # Canonicalize the explicit root once. Relative payload paths may not traverse
    # junctions, and nothing below the root is enumerated recursively.
    root = Path(root).resolve(strict=True)
    if not isinstance(bindings, dict):
        path = Path(bindings)
        bindings = _json(path if path.is_absolute() else _path(root, str(path)))
    _require(bindings.get('schema') == SCHEMA, 'Unsupported source-index schema')
    if complete:
        _require(bindings.get('status') == 'COMPLETE',
                 'MZ48 source is incomplete: load_source requires all 2,560 frames')
    manifest_path = _bound(root, bindings['spec_manifest'])
    manifest = _json(manifest_path)
    _require(manifest['status'] == 'FROZEN_BEFORE_CAPTURE' and
             manifest['total_frames'] == 2560 and manifest['canary_frames'] == 64 and
             manifest['main_frames'] == 2496, 'Unexpected frozen source denominator')
    _require({r['site_id'] for r in manifest['sites']} == SITES and
             len(manifest['sites']) == 8 and set(manifest['families']) == FAMILIES and
             set(manifest['relations']) == RELATIONS, 'Frozen design differs from MZ48')
    rows = manifest['shards']
    ids = [r['shard_id'] for r in rows]
    expected_ids = {'canary-dense_candidate_05', 'canary-dense_candidate_06'} | {
        'main-' + site for site in SITES}
    _require(len(ids) == 10 and set(ids) == expected_ids, 'Expected the ten frozen MZ48 shards')
    _require(sorted(r['frames'] for r in rows) == [32, 32] + [312] * 8,
             'Expected two 32-frame and eight 312-frame shards')
    entries = bindings['shards']
    selected = {entry['shard_id']: entry for entry in entries}
    _require(len(selected) == len(entries) and set(selected) <= expected_ids,
             'Duplicate or unexpected source-index shard')
    if complete:
        _require(set(selected) == expected_ids,
                 f'Missing MZ48 shards: {sorted(expected_ids - set(selected))}')
    return root, bindings, manifest, manifest_path, selected


def _original_spec(root, manifest_path, entry):
    # The frozen manifest stores old absolute host paths. Its basename and hash
    # bind the file in the observed spec-v1/shards layout, making the index movable.
    name = Path(entry['path']).name
    _require(name == entry['shard_id'] + '.json', 'Frozen shard filename mismatch')
    rel = (manifest_path.parent / 'shards' / name).relative_to(root).as_posix()
    spec = _json(_bound(root, dict(path=rel, sha256=entry['sha256'])))
    _require(len(spec['cases']) == entry['frames'], 'Frozen shard frame count mismatch')
    return spec


@dataclass(frozen=True)
class RGBReference:
    """File transport only; do not encode its path or hash as model features."""
    archive: str
    member: str
    sha256: str


def read_rgb(reference):
    """Read one original RGB image, hash-check it, and return a detached PIL image."""
    from PIL import Image
    _require(isinstance(reference, RGBReference), 'read_rgb requires an RGBReference')
    member = relative_path(reference.member)
    _require(member.startswith('model/rgb/') and member.endswith('.png'),
             'Only original model/rgb PNG members are observable inputs')
    with CompactSource(reference.archive) as source:
        data = _member(source, member)
        _require(hashlib.sha256(data).hexdigest() == reference.sha256, 'RGB binding mismatch')
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        return image.copy()


def _pairs(records, pairs):
    ids = [r['frame_id'] for r in records]
    _require(len(ids) == len(set(ids)), 'Duplicate frame IDs')
    seen, pair_ids = set(), set()
    for pair in pairs:
        indices = pair['indices']
        _require(len(indices) == 2 and all(type(i) is int and 0 <= i < len(records)
                                         for i in indices), 'Invalid pair indices')
        a, b = indices
        _require(a != b and not seen.intersection(indices) and pair['pair_id'] not in pair_ids,
                 'Duplicate pair/member assignment')
        seen.update(indices)
        pair_ids.add(pair['pair_id'])
        x, y = records[a], records[b]
        _require(x['pair_id'] == y['pair_id'] == pair['pair_id'], 'Pair identity mismatch')
        _require(x['role'] == y['role'] == pair['role'] and x['site_id'] == y['site_id'],
                 'Pair crosses site or split')
        _require({x['support_context'], y['support_context']} == {'unsupported', 'supported'},
                 'Pair must contain both support contexts')
        for name in ('family', 'relation', 'range', 'setting', 'target_actors'):
            _require(x[name] == y[name], f'Paired target mismatch: {name}')
        _require(pair['target_dictionary_equal'] and pair['actual_target_receipt_equal'] and
                 pair['native_pair_invariant'] and pair['changed_event_mask_pixels'] == [0] * 4 and
                 pair['event_depth_max_abs_m'] == 0, 'Recorded pair-source checks did not pass')
    _require(seen == set(range(len(records))), 'Unpaired or missing source frames')


def _load_shard(root, entry, frozen, manifest_path):
    paths = {key: _bound(root, entry[key]) for key in (
        'archive', 'dataset_receipt', 'dataset_result', 'auxiliary', 'auxiliary_receipt')}
    receipt, result, aux_receipt = (_json(paths[key]) for key in
                                   ('dataset_receipt', 'dataset_result', 'auxiliary_receipt'))
    spec = _original_spec(root, manifest_path, frozen)
    capture_sha = frozen['sha256']
    if 'capture_spec' in entry:
        captured = _json(_bound(root, entry['capture_spec']))
        _require({k: v for k, v in captured.items() if k != 'map_file'} ==
                 {k: v for k, v in spec.items() if k != 'map_file'},
                 'Capture spec changes more than the declared map_file substitution')
        capture_sha = entry['capture_spec']['sha256']
    _require(receipt['status'] == aux_receipt['status'] == 'PASS' and
             result['status'] == 'PASS_SOURCE_CHECKS', 'Source receipts are not PASS')
    _require(receipt['source_spec_sha256'] == aux_receipt['source_spec_sha256'] == capture_sha,
             'Capture/auxiliary/frozen spec hashes differ')
    _require(receipt['outputs']['result.json'] == entry['dataset_result']['sha256'],
             'Dataset result is not bound by its source receipt')
    _require(aux_receipt['schema'] == 'mz48-echo-independent-native-supervision-v1' and
             aux_receipt['output_sha256'] == entry['auxiliary']['sha256'] and
             aux_receipt['output_bytes'] == paths['auxiliary'].stat().st_size,
             'Independent native cell supervision binding mismatch')
    with CompactSource(paths['archive']) as source:
        meta = json.loads(_member(source, 'evaluator/metadata.json'))
        predictor = json.loads(_member(source, 'model/predictor.json'))
        source_binding = json.loads(_member(source, 'evaluator/source-bindings.json'))
        _require(meta['schema'] == 'mz48-training-source-v1' and
                 predictor['schema'] == 'mz48-observable-input-v1', 'Unexpected package schemas')
        _require(source_binding['source_spec_sha256'] == capture_sha and
                 source_binding['code_sha256'] == receipt['code_sha256'],
                 'Package/source receipt binding mismatch')
        dependencies = {Path(path).name: digest for path, digest in source_binding['dependencies'].items()}
        _require(all(dependencies.get(name) == digest
                     for name, digest in aux_receipt['geometry_dependencies'].items()),
                 'Packet/native cell geometry dependencies differ')
        records, pairs = meta['records'], meta['pairs']
        n = len(records)
        _require(n == frozen['frames'] == result['frames'] == aux_receipt['frames'] == predictor['frames'],
                 'Shard frame count mismatch')
        with np.load(io.BytesIO(_member(source, 'model/packets.npz')), allow_pickle=False) as p:
            _require(set(p.files) == {'ranges', 'valid'}, 'Unexpected predictor packet fields')
            ranges = _array(p['ranges'], (n, 64, 2), np.float32, 'ranges')
            valid = _array(p['valid'], (n, 64, 2), bool, 'valid')
        _require(np.isfinite(ranges).all() and (ranges[valid] > 0).all(), 'Invalid range values')
        with np.load(io.BytesIO(_member(source, 'evaluator/labels.npz')), allow_pickle=False) as labels:
            truth = _array(labels['truth'], (n, 4), bool, 'truth')
            known = _array(labels['known'], (n, 4), bool, 'known')
            original_known = _array(labels['cell_known'], (n, 64, 49), bool, 'original cell_known')
            selected = _array(labels['query_presence'], (n, 64, 2, 49, 4), bool, 'selected query_presence')
        with np.load(paths['auxiliary'], allow_pickle=False) as aux:
            _require(set(aux.files) == {'cell_event_presence', 'cell_known'}, 'Unexpected auxiliary fields')
            presence = _array(aux['cell_event_presence'], (n, 64, 49, 4), bool, 'cell_event_presence')
            local_known = _array(aux['cell_known'], (n, 64, 49), bool, 'cell_known')
        _require(np.array_equal(local_known, original_known), 'Auxiliary known mask differs from original')
        _require(not (presence & ~local_known[..., None]).any(), 'UNKNOWN local cells have positive labels')
        _require(not (selected.any(2) & ~presence).any(), 'Selected-echo label outside independent native label')
        _require(len(aux_receipt['native_inputs']) == n and len(source_binding['native_files']) == n,
                 'Native provenance row count mismatch')
        _require(predictor['rgb_files'] == [r['rgb'] for r in records], 'Predictor/RGB order mismatch')
        refs = []
        for i, (record, case, native, original_native) in enumerate(zip(
                records, spec['cases'], aux_receipt['native_inputs'], source_binding['native_files'])):
            _require(record['index'] == native['index'] == original_native['index'] == i,
                     'Source row order mismatch')
            _require(record['frame_id'] == case['name'] == native['frame_id'], 'Frame/spec/auxiliary mismatch')
            for key, expected in dict(pair_id=case['pair_id'], site_id=case['site_id'],
                    family=case['condition']['family'], relation=case['variant_id'], range=case['declared_range'],
                    support_context=case['support_context'], setting=case['profile_id']).items():
                _require(record[key] == expected, f'Record/frozen design mismatch: {key}')
            role = 'HELDOUT_SITE' if record['site_id'] in HOLDOUT else 'TRAIN_CANDIDATE'
            _require(record['role'] == role, 'Source split was changed')
            _require(record['native_sha256'] == native['native_sha256'] == original_native['sha256'],
                     'Native provenance mismatch')
            _require(np.array_equal(truth[i], record['event_truth']) and
                     np.array_equal(known[i], [record['source_valid']] * 4), 'Global label mismatch')
            rgb = relative_path(record['rgb'])
            _require(rgb.startswith('model/rgb/') and rgb.endswith('.png') and
                     source.entries[rgb]['sha256'] == record['rgb_sha256'], 'RGB provenance mismatch')
            refs.append(RGBReference(str(paths['archive']), rgb, record['rgb_sha256']))
        _pairs(records, pairs)
        _require(all(np.array_equal(truth[a], truth[b]) and np.array_equal(known[a], known[b])
                     for a, b in (pair['indices'] for pair in pairs)), 'Paired global labels differ')
        _require(np.array_equal(truth.sum(0), result['positive_frames_by_query']) and
                 int((~known).sum()) == result['unknown_bits'] and
                 int(known.all(1).sum()) == result['source_valid_frames'] and
                 len(pairs) == result['pairs'] == result['invariant_pairs'], 'Dataset count receipt mismatch')
    return dict(records=records, pairs=pairs, truth=truth, known=known,
                predictor=dict(ranges=ranges, valid=valid, rgb_refs=tuple(refs)),
                cell_event_presence=presence, cell_known=local_known)


def _complete_design(records, pairs):
    _pairs(records, pairs)
    _require(len(records) == 2560 and len(pairs) == 1280, 'Full source must have 2,560 frames/1,280 pairs')
    _require(Counter(r['site_id'] for r in records) == Counter({site: 320 for site in SITES}),
             'Full source must contain all eight sites with 320 frames each')
    combinations = [(r['site_id'], r['family'], r['relation'], r['range'], r['setting'], r['support_context'])
                    for r in records]
    expected = {(s, f, rel, distance, setting, context)
                for s in SITES for f in FAMILIES for rel in RELATIONS
                for distance in ('near', 'far') for setting in range(4)
                for context in ('unsupported', 'supported')}
    _require(len(set(combinations)) == 2560 and set(combinations) == expected,
             'Missing/duplicate Cartesian design cells')
    for record in records:
        _require(record['role'] == ('HELDOUT_SITE' if record['site_id'] in HOLDOUT else 'TRAIN_CANDIDATE'),
                 'Source split was changed')
    train = np.array([i for i, r in enumerate(records) if r['role'] == 'TRAIN_CANDIDATE'], dtype=np.int64)
    heldout = np.array([i for i, r in enumerate(records) if r['role'] == 'HELDOUT_SITE'], dtype=np.int64)
    _require(len(train) == 1920 and len(heldout) == 640, 'Expected the fixed 1,920/640 split')
    return train, heldout


def load_source(root, bindings):
    """Load all frozen MZ48 frames or fail; never infer a subset from available files."""
    root, index, manifest, manifest_path, entries = _index(root, bindings, complete=True)
    chunks, records, pairs, refs = [], [], [], []
    for frozen in manifest['shards']:
        chunk = _load_shard(root, entries[frozen['shard_id']], frozen, manifest_path)
        offset = len(records)
        for record in chunk['records']:
            records.append(dict(record, index=offset + record['index'], source_index=record['index'],
                                shard_id=frozen['shard_id']))
        pairs.extend(dict(pair, indices=[i + offset for i in pair['indices']]) for pair in chunk['pairs'])
        refs.extend(chunk['predictor']['rgb_refs'])
        chunks.append(chunk)
    train, heldout = _complete_design(records, pairs)
    result = {name: np.concatenate([chunk[name] for chunk in chunks])
              for name in ('truth', 'known', 'cell_event_presence', 'cell_known')}
    predictor = {name: np.concatenate([chunk['predictor'][name] for chunk in chunks])
                 for name in ('ranges', 'valid')}
    predictor['rgb_refs'] = tuple(refs)
    return dict(result, records=records, pairs=pairs, predictor=predictor,
                train_candidate=train, heldout_site=heldout, event_order=EVENT_ORDER,
                source_status='COMPLETE', source_index=index)


def load_shard_for_check(root, bindings, shard_id):
    """Read one bound shard for CPU engineering validation, even with a draft index.

    No global split or complete-source status is returned. This does not admit a
    partial training cohort and cannot be requested through load_source().
    """
    root, _, manifest, manifest_path, entries = _index(root, bindings, complete=False)
    frozen = next((row for row in manifest['shards'] if row['shard_id'] == shard_id), None)
    _require(frozen is not None and shard_id in entries, f'Unknown/unbound shard: {shard_id}')
    result = _load_shard(root, entries[shard_id], frozen, manifest_path)
    return dict(result, source_status='SINGLE_SHARD_ENGINEERING_CHECK_ONLY')
