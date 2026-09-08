"""Check controlled cube axial depth in a City PCG capture on CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'research/active/dtr-r0/nearfield'))
from worlds_verify import camera_points, visible_support


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cube_depth(camera, obj, device='cuda'):
    """Slab t is axial depth: the unnormalized ray has camera-forward dot 1."""
    origin = torch.tensor([camera[k] for k in ('x', 'y', 'z')], device=device, dtype=torch.float32)
    rays = camera_points(torch.ones((360, 640), device=device), camera) - origin
    center = torch.tensor(obj['center_m'], device=device, dtype=torch.float32)
    size = torch.tensor(obj['size_m'], device=device, dtype=torch.float32)
    require(center.shape == size.shape == (3,) and bool(torch.isfinite(center).all())
            and bool(torch.isfinite(size).all()) and bool((size > 0).all()), 'Invalid cube bounds')
    low, high = center - size / 2, center + size / 2
    parallel = rays.abs() < 1e-8
    safe = torch.where(parallel, torch.ones_like(rays), rays)
    a, b = (low - origin) / safe, (high - origin) / safe
    near = torch.where(parallel, -torch.inf, torch.minimum(a, b)).amax(-1)
    far = torch.where(parallel, torch.inf, torch.maximum(a, b)).amin(-1)
    outside = (parallel & ((origin < low) | (origin > high))).any(-1)
    distance = torch.where(near > 0, near, far)
    hit = (~outside) & (far >= near) & (distance > 0) & (distance < 100)
    return distance, hit


def score_cube(clear, observed, camera, obj):
    expected, hit = cube_depth(camera, obj, clear.device)
    # Exclude a two-pixel silhouette boundary, including image boundaries.
    padded = F.pad(hit[None, None].float(), (2, 2, 2, 2), value=0)
    interior = (-F.max_pool2d(-padded, 5, stride=1))[0, 0].bool()
    known = torch.isfinite(clear) & (clear > 0) & (clear < 100)
    eligible = interior & known & (expected < clear - .03)
    count = int(eligible.sum())
    row = dict(name=obj['name'], projected_pixels=int(hit.sum()), interior_pixels=int(interior.sum()),
               unknown_baseline_interior_pixels=int((interior & ~known).sum()), eligible_pixels=count,
               median_error_m=None, within_3cm_fraction=None, status='FAIL')
    if count:
        error = (observed[eligible] - expected[eligible]).abs()
        valid = torch.isfinite(observed[eligible]) & (observed[eligible] > 0) & (observed[eligible] < 100)
        error = torch.where(valid, error, torch.inf)
        median = float(error.median())
        fraction = float((error <= .03).float().mean())
        row.update(median_error_m=median if np.isfinite(median) else None, within_3cm_fraction=fraction,
                   valid_observed_pixels=int(valid.sum()))
        if count >= 20 and fraction >= .95 and median <= .01:
            row['status'] = 'PASS'
    return row


@torch.inference_mode()
def verify(root):
    root = root.resolve(strict=True)
    require(root.is_relative_to((REPO / 'artifacts.local').resolve()), 'Capture must be under artifacts.local')
    output = root / 'verification.json'
    if output.exists():
        raise FileExistsError('Refuse overwrite verification.json')
    report = dict(status='FAIL', schema='city-pcg-controlled-depth-verification-v1', rows=[],
                  scope='Controlled static cube capture engineering only; no realism, dynamic synchronization, model or safety claim.',
                  thresholds=dict(minimum_eligible_pixels=20, within_3cm_fraction=.95, median_error_m=.01,
                                  silhouette_erosion_pixels=2, baseline_clearance_m=.03),
                  calibration=dict(width=640, height=360, horizontal_fov_degrees=100., principal_point=[319.5,179.5]),
                  depth_semantics='Axial forward metres; zero baseline is UNKNOWN and excluded.',
                  code_sha256=sha(Path(__file__)))
    started = time.perf_counter()
    try:
        receipt = read(root / 'receipt.json')
        require(receipt['status'] == 'PASS' and receipt['source_unchanged'] is True, 'Capture receipt failed')
        require(read(root / 'completion.json')['status'] == 'PASS', 'Host capture completion failed')
        require(read(root / 'process-release.json')['released'] is True, 'Capture process not released')
        source = root / 'source/spec.json'
        require(sha(source) == receipt['spec_sha256'], 'Spec hash differs from engine receipt')
        spec = read(source)
        cases = spec['cases']
        require(len(cases) >= 2 and receipt['frame_count'] == len(cases), 'Expected clear baseline and controlled cases')
        require(cases[0]['objects'] == [], 'First case must be empty clear baseline')
        camera = cases[0]['camera']
        require(all(np.isfinite(float(camera[k])) for k in ('x','y','z','pitch','yaw','roll')), 'Invalid camera')
        require(abs(float(camera['roll'])) < 1e-8, 'Projection requires zero camera roll')
        require(torch.cuda.is_available(), 'CUDA required for dense cube verification')
        report.update(backend='CUDA', device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                      source_spec_sha256=sha(source), receipt_sha256=sha(root / 'receipt.json'))
        depths = []
        for index, case in enumerate(cases):
            require(case['camera'] == camera, 'Every case must use the clear baseline camera')
            path = root / f'evaluator/native/{index:04d}.npy'
            native = np.load(path, allow_pickle=False)
            require(native.shape == (360,640) and native.dtype == np.dtype('<f4') and np.isfinite(native).all()
                    and (native >= 0).all() and (native < 100).all(), 'Invalid native depth')
            rgb = root / f'model/sample/{index:04d}.png'
            with Image.open(rgb) as image:
                image.load()
                require(image.format == 'PNG' and image.size == (640,360), 'Invalid RGB image')
            depths.append(torch.from_numpy(native).cuda())
            report['rows'].append(dict(sample_index=index, name=case['name'], rgb_sha256=sha(rgb),
                                       native_sha256=sha(path), valid_depth_pixels=int((native > 0).sum())))
        require(bool((depths[0] > 0).any()), 'All clear baseline pixels are UNKNOWN')
        if 'floor_z_m' in cases[0]:
            points = camera_points(depths[0], camera)
            patch = points[280:340,280:360,2]
            valid = depths[0][280:340,280:360] > 0
            require(int(valid.sum()) >= 100, 'Insufficient native floor support')
            floor_error = (patch[valid] - float(cases[0]['floor_z_m'])).abs()
            median = float(floor_error.median())
            fraction = float((floor_error <= .05).float().mean())
            report['floor_check'] = dict(expected_z_m=cases[0]['floor_z_m'], valid_pixels=int(valid.sum()),
                median_error_m=median, within_5cm_fraction=fraction,
                eye_height_m=float(camera['z'])-float(cases[0]['floor_z_m']))
            require(median <= .02 and fraction >= .95, 'Native floor differs from declared walking surface')
        for index, case in enumerate(cases[1:], 1):
            require(len(case['objects']) == 1, 'Each controlled case must contain exactly one axis-aligned cube')
            metric = score_cube(depths[0], depths[index], camera, case['objects'][0])
            report['rows'][index]['cube'] = metric
        report['status'] = 'PASS' if all(row['cube']['status'] == 'PASS' for row in report['rows'][1:]) else 'FAIL'
        if report['status'] == 'PASS' and 'floor_z_m' in cases[0]:
            require(abs(float(camera['yaw'])) < 1e-8, 'Visible body query currently requires zero wearer yaw')
            wearer = dict(x=camera['x'], y=camera['y'], z=cases[0]['floor_z_m'], yaw=0., pitch=0., roll=0.)
            mask_dir = root / 'evaluator/visible_support'
            mask_dir.mkdir(exist_ok=False)
            support_rows = []
            for index, native in enumerate(depths):
                target, pooled, masks, valid = visible_support(native, camera, wearer)
                mask_path = mask_dir / f'{index:04d}.npy'
                np.save(mask_path, masks.byte().cpu().numpy(), allow_pickle=False)
                support_rows.append(dict(sample_index=index, body_head_visible_targets=target.tolist(),
                    visible_pixels_per_height=masks.sum((-2,-1)).tolist(),
                    unknown_depth_pixels=int((~valid).sum()), mask_sha256=sha(mask_path)))
            report['visible_support'] = dict(authority='EVALUATOR_ONLY_ALL_VISIBLE_NATIVE_SURFACES_NOT_HIDDEN_FREE_SPACE',
                mask_order=['BODY','HEAD'], query_range_m=3., wearer=wearer, rows=support_rows)
        torch.cuda.synchronize()
    except Exception as exc:
        report['status'] = 'FAIL'
        report['error'] = f'{type(exc).__name__}: {exc}'
    report['elapsed_s'] = time.perf_counter() - started
    with output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    result = verify(parser.parse_args().capture)
    print(json.dumps(result, indent=2, allow_nan=False))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
