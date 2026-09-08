"""Compare controlled target sweeps with all-scene visible support; never relabel."""
import argparse
import collections
import json
import sys
from pathlib import Path
from make_city_obstacle_suite import REPO, read, sha, under_artifacts

sys.path.insert(0, str(REPO / 'research/active/dtr-r0/nearfield'))
from headspace_spec import RELATIONS, target_contact


def check(root):
    root = under_artifacts(root)
    spec_path, native_path = root/'source/spec.json', root/'world-verification.json'
    spec, native = read(spec_path), read(native_path)
    if spec['schema'] != 'city-headspace-controls-v1' or native['status'] != 'PASS':
        raise ValueError('Expected headspace spec and passing native verification')
    if read(root/'receipt.json')['spec_sha256'] != sha(spec_path):
        raise ValueError('Captured spec changed')
    cases, observations = spec['cases'], native['rows']
    if len(cases) != 64 or len(observations) != len(cases):
        raise ValueError('Incomplete fixed suite')
    relation_map = {(0,0): 'CLEAR', (1,0): 'BODY_ONLY', (0,1): 'HEAD_ONLY', (1,1): 'BOTH'}
    rows, groups = [], collections.defaultdict(list)
    for case, observed in zip(cases, observations):
        if case['name'] != observed['name']:
            raise ValueError('Native row identity mismatch')
        actual = target_contact(case['objects'], case['wearer'])
        if actual != case['expected_target_contact']:
            raise ValueError('Target geometry expectation changed')
        visible = relation_map.get(tuple(observed['body_head_visible_targets']), 'UNKNOWN')
        row = dict(name=case['name'], family=case['family'], group_id=case['group_id'],
                   target_relation=actual['relation'], visible_relation=visible,
                   agrees=visible == actual['relation'],
                   pixels=observed['visible_pixels_per_height'],
                   distance_evidence=observed.get('distance_evidence'),
                   unknown_depth_pixels=observed['unknown_depth_pixels'])
        rows.append(row)
        groups[case['group_id']].append(row)
    if len(groups) != 16 or any(len(g) != 4 or {r['target_relation'] for r in g} != set(RELATIONS)
                                for g in groups.values()):
        raise ValueError('Incomplete quartet grouping')
    result = dict(status='PASS' if all(r['agrees'] for r in rows) else 'VISIBILITY_OR_SCENE_GAP',
        frames=len(rows), agreement_frames=sum(r['agrees'] for r in rows),
        complete_visible_quartets=sum(all(r['agrees'] for r in g) for g in groups.values()),
        spec_sha256=sha(spec_path), native_report_sha256=sha(native_path), rows=rows,
        scope='Target cuboid sweep versus all-scene visible depth diagnostic. CLEAR is target-only; '
              'a mismatch is retained, not relabeled or rejected. No mesh, independent world, model or safety claim.')
    with (root/'headspace-check.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return {k:v for k,v in result.items() if k != 'rows'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', required=True, type=Path)
    print(json.dumps(check(parser.parse_args().capture)))
