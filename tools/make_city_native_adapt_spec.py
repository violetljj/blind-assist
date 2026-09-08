"""Geometry-only native TRAIN/DEV pose recipe, fixed before model outcomes."""
import argparse
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    artifact = (Path(__file__).resolve().parents[1] / 'artifacts.local').resolve()
    assert root.is_relative_to(artifact)
    old = read(artifact / 'nearfield/city-native-validation-20260908/final-route-spec.json')
    specs = {}
    for split in ('train', 'dev'):
        spec = read(root / 'scout-spec.json')
        spec.pop('inventory_indices', None)
        spec['export_native_inventory'] = False
        spec['native_targets'] = read(root / f'{split}-target-candidates.json')['native_targets']
        cases = []
        distances = (.8, 1.5, 2.4) if split == 'train' else (1., 1.8, 2.6)
        offsets = (-.9, -.08, 0., .08, .9) if split == 'train' else (-1., -.08, 0., .08, 1.)
        for target in spec['native_targets']:
            yaw = target['capture_heading_deg']
            angle = math.radians(yaw)
            forward, right = (math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))
            for distance in distances:
                for offset in offsets:
                    pos = target['position_m']
                    cases.append(dict(name=f'{split}_{len(cases):04d}',
                        camera=dict(x=pos[0]-distance*forward[0]+offset*right[0],
                                    y=pos[1]-distance*forward[1]+offset*right[1],
                                    z=2.43, yaw=yaw, pitch=-5., roll=0.),
                        objects=[], probe_native_floor=True,
                        source_target=target['target_id'], approach_distance_m=distance, lateral_offset_m=offset))
        spec['cases'] = cases
        spec['selection'] = 'Original geometry-only target approaches; no model scores; split by street segment and exact instance'
        specs[split] = spec
    identity = lambda t: (t['component_path'], t.get('instance_index'))
    sets = {key: {identity(t) for t in value['native_targets']} for key, value in specs.items()}
    old_ids = {identity(t) for t in old['native_targets']}
    assert not sets['train'] & sets['dev'] and not (sets['train'] | sets['dev']) & old_ids
    xy = lambda spec: [(c['camera']['x'], c['camera']['y']) for c in spec['cases']]
    separation = min(math.dist(a, b) for a in xy(specs['train']) for b in xy(specs['dev']))
    regression_distance = {k: min(math.dist(a, b) for a in xy(s) for b in xy(old)) for k, s in specs.items()}
    assert separation > 30 and min(regression_distance.values()) > 10
    assert sum(len(s['cases']) for s in specs.values()) <= 96
    receipt = dict(status='PASS_GEOMETRY_PLACEMENT_ONLY', train_dev_camera_distance_m=separation,
        regression_camera_distance_m=regression_distance, exact_target_identities_disjoint=True,
        counts={k: len(s['cases']) for k, s in specs.items()},
        label_admission='PENDING_NATIVE_CAPTURE', scope='Same-map different instances and segments, not unseen-world evidence')
    for key, spec in specs.items():
        with (root / f'{key}-spec.json').open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(spec, stream, indent=2, allow_nan=False)
    with (root / 'source-separation.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
