"""Independent Pillow validation of the three-cold-start camera engineering probe."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate(root):
    root = Path(root).resolve()
    output = root / 'pixel-validation.json'
    if output.exists():
        raise FileExistsError(output)
    protocol = read(root / 'protocol.json')
    if (protocol.get('authority') != 'ENGINEERING_DEVELOPMENT_ONLY'
            or protocol.get('cold_starts') != 3
            or protocol.get('pairs_per_start') != 100
            or protocol.get('resolution') != [1280, 720]):
        raise ValueError('unexpected_probe_protocol')
    for name, digest in protocol['code_sha256'].items():
        source = root / 'code' / name
        if source.resolve().parent != root / 'code' or sha(source) != digest:
            raise ValueError('code_snapshot_hash_mismatch')
    result = read(root / 'result.json')
    starts = result.get('starts', [])
    if (result.get('status') != 'CAPTURE_PASS_PENDING_INDEPENDENT_PIXELS'
            or len(starts) != 3 or [s.get('start') for s in starts] != [0, 1, 2]):
        raise ValueError('incomplete_probe')
    expected = {(scene, sample, mode) for scene in range(2)
                for sample in range(50) for mode in ('rgb', 'depth')}
    reference_poses = None
    reference_bases = None
    reports = []
    for item in starts:
        target = root / str(item['start'])
        if (item.get('status') != 'PASS' or item.get('ports_released') is not True
                or item.get('images') != 200 or read(target / 'outcome.json') != item):
            raise ValueError('incomplete_start_or_outcome_mismatch')
        bases = item['poses']
        if len(bases) != 2:
            raise ValueError('camera_base_poses')
        rows = read(target / 'receipt.json')['records']
        if len(rows) != 200 or {(r['scene'], r['sample'], r['mode']) for r in rows} != expected:
            raise ValueError('incomplete_or_duplicate_images')
        pairs = {}
        paths = set()
        poses = {}
        byte_count = 0
        for row in rows:
            relative = Path(row['path'])
            path = target / relative
            if (relative.is_absolute() or len(relative.parts) != 1
                    or path.resolve().parent != target or path.suffix.lower() != '.png'
                    or row['path'] in paths):
                raise ValueError('payload_path_escape_or_duplicate')
            paths.add(row['path'])
            if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
                raise ValueError('payload_hash_mismatch')
            with Image.open(path) as image:
                if image.format != 'PNG' or image.size != (1280, 720) or image.mode != 'RGBA':
                    raise ValueError('png_format_dimensions_or_mode')
                if hashlib.sha256(image.convert('RGBA').tobytes()).hexdigest() != row['pixel_sha256']:
                    raise ValueError('decoded_pixels_differ_from_sensor_bytes')
            if (type(row['frame']) is not int or row['frame'] < 0
                    or not math.isfinite(row['timestamp']) or len(row['pose']) != 6
                    or not all(math.isfinite(v) for v in row['pose'])):
                raise ValueError('invalid_frame_timestamp_pose')
            planned = list(bases[row['scene']])
            if len(planned) != 6 or not all(math.isfinite(v) for v in planned):
                raise ValueError('invalid_base_pose')
            planned[0] += row['sample'] * .04
            if max(abs(a - b) for a, b in zip(planned, row['pose'])) > 1e-3:
                raise ValueError('camera_pose_differs_from_planned_path')
            pairs.setdefault((row['scene'], row['sample']), []).append(row)
            poses[(row['scene'], row['sample'], row['mode'])] = row['pose']
            byte_count += row['bytes']
        if {p.name for p in target.rglob('*.png')} != paths or len(list(target.rglob('*.png'))) != 200:
            raise ValueError('unexpected_png_membership')
        previous = None
        for key in sorted(pairs):
            a, b = pairs[key]
            if any(a[k] != b[k] for k in ('frame', 'timestamp', 'pose')):
                raise ValueError('unaligned_pair')
            if previous is not None and (a['frame'] <= previous['frame'] or a['timestamp'] <= previous['timestamp']):
                raise ValueError('nonmonotonic_capture')
            previous = a
        if reference_poses is not None and (reference_poses != poses or reference_bases != bases):
            raise ValueError('cold_start_camera_paths_differ')
        reference_poses, reference_bases = poses, bases
        reports.append({'start': item['start'], 'verified_images': 200, 'verified_pairs': 100,
                        'encoded_bytes': byte_count, 'ports_released': True,
                        'receipt_sha256': sha(target / 'receipt.json'),
                        'outcome_sha256': sha(target / 'outcome.json')})
    receipt = {'status': 'PASS', 'authority': 'ENGINEERING_DEVELOPMENT_ONLY',
               'independent_decoder': 'Pillow', 'verified_images': 600, 'verified_pairs': 300,
               'raw_pixel_mismatches': 0, 'all_pair_frame_timestamp_pose_matches': True,
               'all_three_camera_paths_identical': True, 'all_three_ports_released': True,
               'ports_check_basis': 'Recorded probe outcomes; no live port checks',
               'cvar_application': protocol.get('cvar_application'),
               'claim_scope': 'Camera startup and pixel integrity only; no method score or long-run reliability claim',
               'protocol_sha256': sha(root / 'protocol.json'),
               'result_sha256': sha(root / 'result.json'),
               'validator_sha256': sha(Path(__file__)), 'starts': reports}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(json.dumps(validate(parser.parse_args().root)))
