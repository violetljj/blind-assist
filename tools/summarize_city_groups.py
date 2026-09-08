"""Summarize completed clear/center/right-clearance groups without outcome labels."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
VARIANTS = ('clear', 'center', 'right_clearance')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired_groups(cases):
    groups = defaultdict(dict)
    for index, case in enumerate(cases):
        group, variant = case['group_id'], case['variant_id']
        require(isinstance(group, str) and group, 'Missing group identity')
        require(variant in VARIANTS and variant not in groups[group], 'Invalid/duplicate group variant: ' + group)
        groups[group][variant] = index
    require(bool(groups), 'No groups')
    for group, indices in groups.items():
        require(set(indices) == set(VARIANTS), 'Incomplete triplet: ' + group)
        clear, center, lateral = [cases[indices[v]] for v in VARIANTS]
        require(all(c['camera'] == clear['camera'] and c['group_type'] == clear['group_type']
                    and c.get('floor_z_m') == clear.get('floor_z_m') for c in (center, lateral)),
                'Camera/group type/floor mismatch: ' + group)
        require(not clear.get('objects') and len(center['objects']) == len(lateral['objects']) == 1,
                'Expected empty clear and one object per placement: ' + group)
        require(all(c.get('baseline_index') == indices['clear'] for c in (clear, center, lateral)),
                'Baseline reference mismatch: ' + group)
        a, b = center['objects'][0], lateral['objects'][0]
        require(a.get('mesh_asset') and a.get('mesh_asset') == b.get('mesh_asset'), 'Mesh mismatch: ' + group)
        require({k: v for k, v in a.items() if k not in ('center_m', 'placement')} ==
                {k: v for k, v in b.items() if k not in ('center_m', 'placement')},
                'Objects differ beyond origin/placement: ' + group)
        delta = np.asarray(b['center_m'], dtype=float) - np.asarray(a['center_m'], dtype=float)
        require(delta.shape == (3,) and np.isfinite(delta).all(), 'Invalid placement origin: ' + group)
        yaw = math.radians(float(clear['camera']['yaw']))
        forward = delta[0]*math.cos(yaw) + delta[1]*math.sin(yaw)
        sideways = -delta[0]*math.sin(yaw) + delta[1]*math.cos(yaw)
        require(abs(forward) < 1e-6 and abs(delta[2]) < 1e-6 and sideways > 0,
                'Placement difference must be wearer-right only: ' + group)
    return groups


def summarize(capture):
    root = capture.resolve(strict=True)
    require(root.is_relative_to((REPO/'artifacts.local').resolve()), 'Capture must be under artifacts.local')
    output = root/'group-summary.json'
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    report = dict(status='FAIL', schema='city-group-quality-v1', backend='CPU',
                  backend_reason='TASK_NOT_GPU_SUITABLE_METADATA_AND_SMALL_RGB_QA',
                  scope='Descriptive static simulator QA; support is not object identity or certified free space. No expected center/lateral labels.',
                  code_sha256=sha(Path(__file__)), rows=[], aggregates=[], observations=[])
    try:
        spec_path = root/'source/spec.json'
        spec, receipt, completion, verification, release = [read(root/p) for p in
            ('source/spec.json', 'receipt.json', 'completion.json', 'world-verification.json', 'process-release.json')]
        require(all(x['status'] == 'PASS' for x in (receipt, completion, verification)), 'Capture or native verification failed')
        require(receipt['source_unchanged'] is True and read(root/'source-integrity.json')['unchanged'] is True,
                'Source integrity failed')
        require(release['released'] is True and not release.get('survivors'), 'Processes not released')
        require(sha(spec_path) == receipt['spec_sha256'] == verification['source_spec_sha256'], 'Spec hash mismatch')
        require(sha(root/'receipt.json') == verification['receipt_sha256'], 'Verification receipt mismatch')
        cases = spec['cases']
        require(len(cases) == receipt['frame_count'] == len(verification['rows']), 'Frame count mismatch')
        groups = paired_groups(cases)
        manifest = root/'payload-hashes.json'
        require(sha(manifest) == completion['payload_hashes_sha256'], 'Payload manifest mismatch')
        for relative, expected in read(manifest).items():
            path = (root/relative).resolve(strict=True)
            require(path.is_relative_to(root) and sha(path) == expected, 'Payload hash mismatch: ' + relative)
        report.update(group_count=len(groups), frame_count=len(cases), source_spec_sha256=sha(spec_path),
                      world_verification_sha256=sha(root/'world-verification.json'))
        buckets = defaultdict(list)
        for index, (case, native_row) in enumerate(zip(cases, verification['rows'])):
            require(native_row['sample_index'] == index and native_row['name'] == case['name']
                    and native_row['camera'] == case['camera'], 'Native row mismatch')
            rgb_path, depth_path = root/f'model/sample/{index:04d}.png', root/f'evaluator/native/{index:04d}.npy'
            mask_path = (root/native_row['mask_path']).resolve(strict=True)
            require(mask_path.is_relative_to(root), 'Mask path escapes capture')
            require(sha(rgb_path) == native_row['rgb_sha256'] and sha(depth_path) == native_row['native_sha256']
                    and sha(mask_path) == native_row['mask_sha256'], 'Native verification payload changed')
            depth = np.load(depth_path, allow_pickle=False)
            require(depth.shape == (360,640) and depth.dtype == np.dtype('<f4') and
                    np.isfinite(depth).all() and (depth >= 0).all() and (depth < 100).all(), 'Invalid native depth')
            unknown = int((depth == 0).sum())
            require(unknown == native_row['unknown_depth_pixels'] and depth.size-unknown == native_row['valid_depth_pixels'],
                    'Native validity counts mismatch')
            with Image.open(rgb_path) as image:
                require(image.format == 'PNG' and image.size == (640,360), 'Invalid RGB')
                rgb = np.asarray(image.convert('RGB'))
            gray = rgb.astype(np.float32) @ np.array([.2126,.7152,.0722], dtype=np.float32)
            asset = cases[groups[case['group_id']]['center']]['objects'][0]['mesh_asset']
            row = dict(index=index, group_id=case['group_id'], variant=case['variant_id'], asset=asset,
                       body_head=native_row['body_head_visible_targets'],
                       distance_states=[v['state'] for v in native_row.get('distance_evidence', [])],
                       unknown_fraction=unknown/depth.size, rgb_luma_mean=float(gray.mean()),
                       near_black_fraction=float((rgb.max(axis=2)<=5).mean()),
                       near_white_fraction=float((rgb.min(axis=2)>=250).mean()))
            report['rows'].append(row)
            buckets[asset, case['variant_id']].append(row)
            if row['near_black_fraction'] > .95 or row['near_white_fraction'] > .95:
                report['observations'].append(dict(index=index, kind='NEAR_UNIFORM_EXTREME_RGB_REVIEW'))
            if row['unknown_fraction'] > .5:
                report['observations'].append(dict(index=index, kind='MAJORITY_DEPTH_UNKNOWN_REVIEW'))
        for (asset, variant), rows in sorted(buckets.items()):
            report['aggregates'].append(dict(asset=asset, variant=variant, count=len(rows),
                body_head_counts=[dict(Counter(str(r['body_head'][h]) for r in rows)) for h in (0,1)],
                distance_state_counts=[dict(Counter(r['distance_states'][h] if len(r['distance_states']) == 2
                    else 'UNAVAILABLE' for r in rows)) for h in (0,1)],
                means={k:sum(r[k] for r in rows)/len(rows) for k in
                    ('unknown_fraction','rgb_luma_mean','near_black_fraction','near_white_fraction')}))
        report.update(status='PASS', timing=dict(capture_wall_s=release.get('wall_elapsed_s'),
                      native_verification_s=verification.get('elapsed_s')))
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
    report['summary_elapsed_s'] = time.perf_counter()-started
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    result = summarize(parser.parse_args().capture)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','aggregates')}, indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
