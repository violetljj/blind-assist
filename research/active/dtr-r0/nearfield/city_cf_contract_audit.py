"""Read-only City triplet/label audit; no rendering, inference or training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(capture):
    paths = {name: capture / relative for name, relative in (
        ('spec', 'source/spec.json'), ('verification', 'world-verification.json'),
        ('receipt', 'receipt.json'))}
    data = {k: json.loads(p.read_text(encoding='utf-8-sig')) for k, p in paths.items()}
    if data['verification']['status'] != 'PASS' or data['receipt']['status'] != 'PASS':
        raise ValueError('Completed capture and truth required')
    if sha(paths['spec']) != data['receipt']['spec_sha256']:
        raise ValueError('Capture specification hash mismatch')
    cases, rows = data['spec']['cases'], data['verification']['rows']
    if len(cases) != 1500 or len(rows) != len(cases):
        raise ValueError('Expected original 1500-frame cohort')
    groups = defaultdict(list)
    for i, (case, row) in enumerate(zip(cases, rows)):
        if row['sample_index'] != i or row['name'] != case['name']:
            raise ValueError('Case/truth ordering mismatch')
        if any((count >= 3) != bool(label) for count, label in zip(
                row['visible_pixels_per_height'], row['body_head_visible_targets'])):
            raise ValueError('Native support and near truth disagree')
        groups[case['group_id']].append((case, row))
    result = dict(schema='city-cf-contract-audit-v1', backend='CPU',
        backend_reason='TASK_NOT_GPU_SUITABLE_METADATA', optimizer_steps=0,
        input_sha256={k: sha(p) for k, p in paths.items()}, regions={})
    for region in ('west_sidewalk', 'plaza'):
        subset = [g for key, g in groups.items() if key.startswith(region + '__')]
        if len(subset) != 250:
            raise ValueError('Unexpected region coverage')
        for g in subset:
            if [c['variant_id'] for c, _ in g] != ['clear', 'center', 'right_clearance']:
                raise ValueError('Triplet order mismatch')
            if any(c['camera'] != g[0][0]['camera'] or c['floor_z_m'] != g[0][0]['floor_z_m'] for c, _ in g):
                raise ValueError('Nonmatched triplet geometry')
            if g[0][0]['objects']:
                raise ValueError('Clear control contains injected object')
        heads = {}
        for h, name in enumerate(('BODY', 'HEAD')):
            patterns = Counter(tuple(r['body_head_visible_targets'][h] for _, r in g) for g in subset)
            negatives = [r for g in subset for _, r in g if r['body_head_visible_targets'][h] == 0]
            heads[name] = dict(label_patterns={str(k): v for k, v in patterns.items()},
                mixed_label_groups=sum(v for k, v in patterns.items() if len(set(k)) == 2),
                positive_negative_pairs=sum(v * sum(k) * (3 - sum(k)) for k, v in patterns.items()),
                negative_frames=len(negatives), negative_frames_with_positive_native_support=sum(
                    r['visible_pixels_per_height'][h] > 0 for r in negatives))
        result['regions'][region] = dict(groups=len(subset), matched_camera_and_floor=len(subset), heads=heads)
    result['interpretation'] = (
        'Support truth is visible surface inside body-relative forward corridors, not all visible pixels. '
        'Negative native support has zero positive pixels in this cohort. Binary positive-minus-negative '
        'support therefore cannot add positive locations; unknown correspondence still requires care. '
        'Equal camera/floor permits image correspondence but does not certify identical rendered background '
        'or isolate shadows, occlusion and temporal rendering effects. Ranking requires actual mixed labels.')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.capture)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
