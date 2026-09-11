"""Bound MZ67 metadata admission. No depth, image or model inputs."""
from collections import Counter
import hashlib
import json
from pathlib import Path

MANIFEST_SHA = '851482eba0c6bc6c201897ef090e2b29d223542be94fef30b7735e24ab10b99a'
MAP_SHA = '02b3f0909049b022d2b62275d48ed61a54d0e4ed515561e0723bdbf86c99f1b3'
MZ55_DATASET_SHA = '61d06662453f9db36a3a36a038e37c03a57779e4053d654b75052efdaf72470a'
ROLES = ('TRAIN_CANDIDATE', 'CALIBRATION', 'HELDOUT_GEOMETRY')
FIELDS = ('geometry_id', 'geometry_recipe_id', 'geometry_joint_factors', 'split_unit', 'site_replica')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source_binding_fields(contract, captured_spec_path):
    return dict(original_spec_sha256=contract['frozen_spec_sha256'],
                captured_spec_sha256=sha(captured_spec_path), manifest_sha256=MANIFEST_SHA,
                shard_id=contract['shard_id'], map_sha256=MAP_SHA,
                frozen_parent=dict(file='mz55_dataset.py', sha256=MZ55_DATASET_SHA,
                                   reuse='Source copy with explicit admission/metadata changes; no runtime import'))


def verify_map_file(captured):
    assert captured['map_sha256'] == MAP_SHA
    path = Path(captured['map_file'])
    assert sha(path) == MAP_SHA, 'Host map bytes differ from frozen map'
    return {str(path): MAP_SHA}


def geometry_metadata(case):
    assert case['training_role'] in ROLES and case['split_unit'] == 'geometry_recipe'
    wi, di, ti = case['geometry_joint_factors']
    assert wi in range(4) and di in range(4) and ti in range(2)
    fold = (wi + di + ti) % 4
    assert case['training_role'] == ROLES[0 if fold < 2 else fold - 1]
    assert case['geometry_recipe_id'] == case['condition']['family'] + f'-g{wi+4*di+16*ti:02d}'
    assert len(case['geometry_id']) == 64 and case['site_replica'] in (0, 1)
    return {key: case[key] for key in FIELDS}


def admit_spec(manifest, specs, captured):
    manifest, specs = Path(manifest), Path(specs)
    assert sha(manifest) == MANIFEST_SHA, 'Not frozen MZ67 spec-v1 manifest'
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
            assert geometry_metadata(a) == geometry_metadata(b)
            assert a['objects'] == [o for o in b['objects'] if o['target_part']]
        if captured['cases'][0]['name'] == cases[0]['name']:
            assert {k:v for k,v in captured.items() if k != 'map_file'} == {k:v for k,v in spec.items() if k != 'map_file'}, 'Only map_file host remapping allowed'
            matched.append(shard)
        all_cases.extend(cases)
    assert len(matched) == 1
    assert len(all_cases) == len({c['name'] for c in all_cases}) == 4096
    assert Counter(c['training_role'] for c in all_cases) == dict(TRAIN_CANDIDATE=2048, CALIBRATION=1024, HELDOUT_GEOMETRY=1024)
    assert sum(c['canary'] for c in all_cases) == 64
    assert sum(c['native_audit_sample'] for c in all_cases) == 80
    by_geometry = {}
    for c in all_cases:
        by_geometry.setdefault(c['geometry_id'], set()).add(c['training_role'])
    assert len(by_geometry) == 1024 and all(len(v) == 1 for v in by_geometry.values())
    return dict(manifest_sha256=MANIFEST_SHA, shard_id=matched[0]['shard_id'], frames=matched[0]['frames'],
                frozen_spec_sha256=matched[0]['sha256'], inputs=inputs,
                allowed_host_remap=['map_file'], geometry_metadata_authority='evaluator only')
