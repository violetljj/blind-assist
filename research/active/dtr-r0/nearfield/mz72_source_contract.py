"""Bound MZ72 physical source admission. No depth, image or model inputs."""
from collections import Counter
import hashlib
import json
from pathlib import Path

MANIFEST_SHA = '6687a5434a07c0afcdbb8e99a9afdbdc6718e1a298d207d5d150f2a6e2a09233'
MAP_SHA = '02b3f0909049b022d2b62275d48ed61a54d0e4ed515561e0723bdbf86c99f1b3'
MZ55_DATASET_SHA = '61d06662453f9db36a3a36a038e37c03a57779e4053d654b75052efdaf72470a'
ROLES = ('TRAIN_CANDIDATE', 'CALIBRATION', 'HELDOUT_GEOMETRY')
FIELDS = ('geometry_id', 'scene_geometry_id', 'geometry_recipe_id', 'recipe_id',
          'recipe_group', 'regime', 'angular_pair_id', 'placement_focus',
          'required_positive_bits', 'off_path_expected_zero', 'split_unit', 'site_replica')


def intent_matches(case, truth):
    """Fixture intent never replaces actual labels or erases extra positives."""
    actual = [bool(v) for v in truth]
    required = case['required_positive_bits']
    assert len(actual) == len(required) == 4
    assert all(type(v) is bool for v in required)
    if case['off_path_expected_zero']:
        assert not any(required), 'OFF_PATH must declare no required event'
        return not any(actual)
    assert any(required), 'Non-OFF_PATH must declare a required positive event'
    return all(not expected or observed for expected, observed in zip(required, actual))


def shared_objects(case):
    """All actual assembly actors, including mounts and fixed occluders."""
    for obj in case['objects']:
        assert all(type(obj[k]) is bool for k in ('target_part', 'mandatory_mount', 'optional_context'))
        assert not (obj['mandatory_mount'] and obj['optional_context'])
        assert obj['physical_role'] in ('target', 'mount', 'occluder', 'context')
        assert obj['target_part'] == (obj['physical_role'] == 'target')
        assert not (obj['target_part'] and obj['optional_context'])
        if obj['mandatory_mount']:
            assert obj['physical_role'] == 'mount'
    return [obj for obj in case['objects'] if not obj['optional_context']]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source_binding_fields(contract, captured_spec_path):
    return dict(original_spec_sha256=contract['frozen_spec_sha256'],
                captured_spec_sha256=sha(captured_spec_path), manifest_sha256=MANIFEST_SHA,
                shard_id=contract['shard_id'], map_sha256=MAP_SHA,
                frozen_parent=dict(file='mz67_dataset.py', sha256='d86c300079a2badd3bb44a7885acf2fb3f5b8fa5ea24f7a40e6bf7ba350500be',
                                   reuse='Source copy; physical metadata, mounts and subset intent admission changes only; no runtime import'),
                native_kernel_ancestor=dict(file='mz55_dataset.py', sha256=MZ55_DATASET_SHA))


def verify_map_file(captured):
    assert captured['map_sha256'] == MAP_SHA
    path = Path(captured['map_file'])
    assert sha(path) == MAP_SHA, 'Host map bytes differ from frozen map'
    return {str(path): MAP_SHA}


def geometry_metadata(case):
    assert case['training_role'] in ROLES
    assert case['split_unit'] == 'recipe_group'
    assert type(case['recipe_id']) is int and case['recipe_id'] >= 0
    assert case['geometry_recipe_id'] == case['recipe_group']
    assert case['regime'] in ('ACTUAL_SIZE', 'ANGULAR_CONTROL')
    assert case['placement_focus'] in ('HEAD_PATH', 'BODY_PATH', 'BOTH_PATH', 'VISIBLE_OFF_PATH')
    assert type(case['off_path_expected_zero']) is bool
    assert case['off_path_expected_zero'] == (case['placement_focus'] == 'VISIBLE_OFF_PATH')
    assert case['expected_events'] == case['required_positive_bits']
    assert intent_matches(case, case['required_positive_bits'])
    assert (case['angular_pair_id'] is None) == (case['regime'] == 'ACTUAL_SIZE')
    if case['angular_pair_id'] is not None:
        assert isinstance(case['angular_pair_id'], str) and case['angular_pair_id']
    for key in ('geometry_id', 'scene_geometry_id'):
        assert len(case[key]) == 64 and all(ch in '0123456789abcdef' for ch in case[key])
    assert case['site_replica'] in (0, 1)
    shared_objects(case)
    return {key: case[key] for key in FIELDS}


def admit_spec(manifest, specs, captured):
    manifest, specs = Path(manifest), Path(specs)
    assert sha(manifest) == MANIFEST_SHA, 'Not frozen MZ72 spec-v1 manifest'
    m = read(manifest)
    assert captured['map_sha256'] == MAP_SHA
    assert (m['frames'], m['canary_frames'], m['main_frames']) == (4096, 64, 4032)
    inputs = {str(manifest): MANIFEST_SHA, str(Path(__file__)): sha(__file__)}
    all_cases, matched = [], []
    for shard in m['shards']:
        path = specs / (shard['shard_id'] + '.json')
        assert sha(path) == shard['sha256'], str(path)
        inputs[str(path)] = shard['sha256']
        spec = read(path); cases = spec['cases']
        assert spec['map_sha256'] == MAP_SHA
        assert len(cases) == shard['frames'] and len(cases) % 2 == 0
        assert all(case['canary'] == shard['canary'] for case in cases)
        for case in cases:
            geometry_metadata(case)
        for a, b in zip(cases[::2], cases[1::2]):
            assert a['support_context'] == 'unsupported' and b['support_context'] == 'supported'
            assert a['pair_id'] == b['pair_id'] and a['training_role'] == b['training_role']
            # Angular IDs pair near/far within a context, not the two background contexts.
            assert {k:v for k,v in geometry_metadata(a).items() if k != 'angular_pair_id'} == {k:v for k,v in geometry_metadata(b).items() if k != 'angular_pair_id'}
            assert not any(o['optional_context'] for o in a['objects'])
            assert a['objects'] == shared_objects(a) == shared_objects(b)
        if captured['cases'][0]['name'] == cases[0]['name']:
            assert {k:v for k,v in captured.items() if k != 'map_file'} == {k:v for k,v in spec.items() if k != 'map_file'}, 'Only map_file host remapping allowed'
            matched.append(shard)
        all_cases.extend(cases)
    assert len(matched) == 1
    assert len(all_cases) == len({c['name'] for c in all_cases}) == 4096
    assert Counter(c['training_role'] for c in all_cases) == dict(TRAIN_CANDIDATE=2048, CALIBRATION=1024, HELDOUT_GEOMETRY=1024)
    assert sum(c['canary'] for c in all_cases) == 64
    assert sum(c['native_audit_sample'] for c in all_cases) == 80
    assert Counter(c['regime'] for c in all_cases) == dict(ACTUAL_SIZE=3072, ANGULAR_CONTROL=1024)
    assert len({c['condition']['family'] for c in all_cases}) == 4
    angular = {}
    for c in all_cases:
        if c['angular_pair_id'] is not None:
            angular.setdefault(c['angular_pair_id'], []).append(c)
    assert len(angular) == 512
    for rows in angular.values():
        assert len(rows) == 2 and {c['declared_range'] for c in rows} == {'near', 'far'}
        for field in ('recipe_group', 'training_role', 'site_id', 'support_context', 'placement_focus'):
            assert len({c[field] for c in rows}) == 1, (field, 'angular-pair mismatch')
    for field in ('recipe_group', 'geometry_id', 'scene_geometry_id'):
        grouped = {}
        for c in all_cases:
            grouped.setdefault(c[field], set()).add(c['training_role'])
        assert all(len(v) == 1 for v in grouped.values()), (field, 'cross-role collision')
    return dict(manifest_sha256=MANIFEST_SHA, shard_id=matched[0]['shard_id'], frames=matched[0]['frames'],
                frozen_spec_sha256=matched[0]['sha256'], inputs=inputs,
                allowed_host_remap=['map_file'], geometry_metadata_authority='evaluator only')
