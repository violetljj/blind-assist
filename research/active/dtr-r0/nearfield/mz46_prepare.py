"""Freeze four MZ44 rod cases with the twelve ancillary actors removed."""
import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, output):
    expected = 'bfb20744c146f1e87061d2196b481d1f9700bcb07f0283d6136393d1a70b30a2'
    assert sha(source) == expected
    original = json.loads(source.read_text(encoding='utf-8-sig'))
    selected = [c for c in original['cases'] if c['condition']['family'] == 'oblique_rod']
    assert len(selected) == 4
    assert {c['variant_id'] for c in selected} == {'HEAD_ONLY', 'BODY_ONLY', 'BOTH', 'CLEAR'}
    spec = copy.deepcopy(original)
    spec['cases'] = []
    mapping = []
    for previous in selected:
        case = copy.deepcopy(previous)
        target = [o for o in case['objects'] if o['name'] == 'adjustable_cross_member']
        assert len(case['objects']) == 13 and len(target) == 1 and target[0]['target_part']
        removed = [o for o in case['objects'] if o['name'] != 'adjustable_cross_member']
        assert len(removed) == 12 and not any(o['target_part'] for o in removed)
        case['objects'] = target
        assert {k: v for k, v in case.items() if k != 'objects'} == {k: v for k, v in previous.items() if k != 'objects'}
        assert target[0] == next(o for o in previous['objects'] if o['target_part'])
        spec['cases'].append(case)
        mapping.append(dict(original_case_id=previous['name'], case_id=case['name'],
                            removed_names=[o['name'] for o in removed], retained_object=target[0]))
    # All top-level render/light/runtime settings remain byte-value equivalent.
    assert {k: v for k, v in spec.items() if k != 'cases'} == {k: v for k, v in original.items() if k != 'cases'}
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'spec.json'
    path.write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8', newline='\n')
    receipt = dict(status='FROZEN_BEFORE_CAPTURE', frames=4, original_spec_sha256=expected,
                   spec_sha256=sha(path), prepare_sha256=sha(__file__), mapping=mapping,
                   invariant='Only selected case subset and removal of twelve non-target actors; original case IDs and complete target dictionaries retained',
                   inherited_provenance_note='Top-level MZ44 provenance is unchanged source metadata; this manifest owns the four-frame intervention',
                   capture_attempt_budget=1, recapture_budget=0)
    (output / 'manifest.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(status=receipt['status'], frames=4, spec_sha256=receipt['spec_sha256'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve())
