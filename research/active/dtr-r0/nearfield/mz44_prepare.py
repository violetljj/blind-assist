"""Place one20-frame far block using unchanged MZ42 meshes and camera."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(root, output):
    prior = root / 'artifacts.local/work/mz42-rich-objects-20260910'
    source_path = prior / 'spec-v2/spec.json'
    manifest_path = prior / 'spec-v2/manifest.json'
    metadata_path = prior / 'returned-v1/metadata-v1/metadata.json'
    manifest = read(manifest_path)
    assert sha(source_path) == manifest['spec_sha256']
    source = read(source_path)
    assert read(metadata_path)['status'] == 'PASS'
    assert len(source['cases']) == 20
    spec = copy.deepcopy(source)
    spec.update(schema='mz44-rich-far-engineering-v1',
        scope='Same-source far real-mesh controlled Development; no model inference or sensor claim')
    expected = {'HEAD_ONLY': [0, 0, 0, 1], 'BODY_ONLY': [0, 1, 0, 0],
                'BOTH': [0, 1, 0, 1], 'CLEAR': [0, 0, 0, 0]}
    variant = {'HEAD_ONLY': 'HEAD_FAR', 'BODY_ONLY': 'BODY_FAR', 'BOTH': 'BOTH_FAR', 'CLEAR': 'NONINTRUDING'}
    checks = []
    for case in spec['cases']:
        assert case['site_id'] == 'mz36_dense_candidate_05_site_001' and case['camera']['yaw'] == 0
        original = copy.deepcopy(case)
        family = case['condition']['family']
        if family != 'oblique_rod':
            extent = manifest['transforms'][family]['rotated_extents_m'][0]
            for i, obj in enumerate(case['objects']):
                assert obj['placement'] == 'bounds_front_center_floor'
                obj['center_m'][0] = case['camera']['x'] + 2.2 + .1 * i
                assert 1.68 < 2.2 + .1*i < 2.2 + .1*i + extent < 3.13
            checks.append(dict(frame=case['name'], target_x_min_m=2.2, target_x_max_m=2.3+extent,
                               authority='Loaded mesh bounds for placement only'))
        else:
            target = next(obj for obj in case['objects'] if obj['name'] == 'adjustable_cross_member')
            delta = case['camera']['x'] + 2.25 - target['center_m'][0]
            for obj in case['objects']:
                assert obj['rotation_deg']['pitch'] == obj['rotation_deg']['yaw'] == 0
                obj['center_m'][0] += delta
            front = min(obj['center_m'][0] - case['camera']['x'] - obj['size_m'][0]/2 for obj in case['objects'])
            assert front > 1.68
            assert 2.25-target['size_m'][0]/2 > 1.68 and 2.25+target['size_m'][0]/2 < 3.13
            checks.append(dict(frame=case['name'], all_constituent_x_min_m=front, target_x_center_m=2.25,
                rear_support='Preserved lateral support extends beyond far range; not target labels'))
        for obj, old in zip(case['objects'], original['objects']):
            assert {k: v for k, v in obj.items() if k != 'center_m'} == {k: v for k, v in old.items() if k != 'center_m'}
            assert obj['center_m'][1:] == old['center_m'][1:]
        case['original_mz42_case'] = original['name']
        case['name'] = 'mz44-' + case['site_id'] + '-' + family + '-' + variant[case['variant_id']]
        case['group_id'] = 'mz44-' + case['site_id'] + '-' + family
        case['declared_range'] = 'far'
        case['range_variant'] = variant[case['variant_id']]
        case['expected_events'] = expected[case['variant_id']]
        case['condition']['condition_id'] = case['name']
    bindings = {str(path): sha(path) for path in (source_path, manifest_path, metadata_path, Path(__file__))}
    spec['provenance'] = dict(inputs=bindings, frames=20, source_site='mz36_dense_candidate_05_site_001',
        change='Only object X translations; same camera/mesh/material/scale/rotation/height/lateral placement',
        expected_events_authority='INTENT_ONLY_NOT_LABELS')
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'spec.json'
    path.write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
    result = dict(status='FROZEN_BEFORE_CAPTURE', frames=20, spec_sha256=sha(path), inputs=bindings,
                  placement_checks=checks, site_unchanged=True, reused_actual_mesh_metadata=True,
                  runtime_asset_hash_recheck='REQUIRED_BEFORE_CAPTURE', training_steps=0, model_inference_frames=0)
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'frames', 'spec_sha256', 'site_unchanged')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root.resolve(), args.output.resolve())
