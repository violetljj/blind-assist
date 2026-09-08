"""CUDA visible native-surface support for City PCG views; no AABB ground truth."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from verify_city_pcg_capture import REPO, camera_points, read, require, score_cube, sha, visible_support
from contact_retina_spec import BODY_BOXES


def distance_evidence(native, camera, wearer, policy):
    """Observed straight-ahead surface gaps; not motion risk or certified free space."""
    danger, warning, review = (float(policy[k]) for k in ('danger_m','warning_m','review_m'))
    require(0 < danger < warning < review <= 100, 'Invalid ordered distance policy')
    points = camera_points(native, camera)
    valid = torch.isfinite(native) & (native > 0) & (native < 100)
    rows=[]
    for low, high in BODY_BOXES:
        gap=points[...,0]-high[0]
        eligible=valid & (gap>=0) & (gap<=review)
        eligible &= (points[...,1]>=low[1]) & (points[...,1]<=high[1])
        eligible &= (points[...,2]>=wearer['z']+low[2]) & (points[...,2]<=wearer['z']+high[2])
        gaps=gap[eligible]
        count=int(gaps.numel())
        third=float(torch.topk(gaps,3,largest=False).values[-1]) if count>=3 else None
        state='NO_VISIBLE_SUPPORT'
        if third is not None:
            state='DANGER' if third<=danger else 'WARNING' if third<=warning else 'OBSERVE'
        rows.append(dict(state=state,visible_pixels=count,minimum_visible_gap_m=float(gaps.min()) if count else None,
            three_pixel_gap_m=third,within_danger_pixels=int((gaps<=danger).sum()),
            within_warning_pixels=int((gaps<=warning).sum())))
    return rows


def local_query(camera, floor_z_m=None):
    require(all(np.isfinite(float(camera[k])) for k in ('x', 'y', 'z', 'pitch', 'yaw', 'roll')), 'Invalid camera pose')
    require(abs(float(camera['roll'])) < 1e-8, 'Projection requires zero camera roll')
    floor = float(camera['z']) - 1.70 if floor_z_m is None else float(floor_z_m)
    require(np.isfinite(floor) and floor < float(camera['z']), 'Invalid wearer floor height')
    # Translate camera to origin and undo wearer yaw. Axial depth and pitch stay unchanged.
    local_camera = dict(x=0., y=0., z=0., pitch=float(camera['pitch']), yaw=0., roll=0.)
    wearer = dict(x=0., y=0., z=floor-float(camera['z']), pitch=0., yaw=0., roll=0.)
    return local_camera, wearer


@torch.inference_mode()
def verify(root):
    root = root.resolve(strict=True)
    require(root.is_relative_to((REPO / 'artifacts.local').resolve()), 'Capture must be under artifacts.local')
    output, masks_dir = root / 'world-verification.json', root / 'evaluator/world_support'
    if output.exists() or masks_dir.exists():
        raise FileExistsError('Refuse overwrite world report or support directory')
    report = dict(status='FAIL', schema='city-pcg-world-support-v1', rows=[], code_sha256=sha(Path(__file__)),
                  authority='EVALUATOR_ONLY_VISIBLE_NATIVE_SURFACES', mask_order=['BODY', 'HEAD'],
                  mask_values={'UNKNOWN': -1, 'NO_VISIBLE_SUPPORT': 0, 'VISIBLE_SUPPORT': 1}, query_range_m=3.,
                  scope='Static simulator geometry engineering. Missing, occluded and unobserved space is not certified free. No realism, dynamic, model or safety claim.',
                  calibration=dict(width=640, height=360, horizontal_fov_degrees=100., principal_point=[319.5,179.5]))
    started = time.perf_counter()
    try:
        receipt = read(root / 'receipt.json')
        require(receipt['status'] == 'PASS' and receipt['source_unchanged'] is True, 'Capture receipt failed')
        completion = read(root / 'completion.json')
        require(completion['status'] == 'PASS', 'Host capture completion failed')
        require(read(root / 'process-release.json')['released'] is True, 'Capture process not released')
        spec_path = root / 'source/spec.json'
        require(sha(spec_path) == receipt['spec_sha256'], 'Source spec hash mismatch')
        spec = read(spec_path)
        for key in ('map_sha256_before', 'map_sha256_after'):
            if key in receipt:
                require(receipt[key] == spec['map_sha256'], 'Source map hash mismatch')
        if 'payload_hashes_sha256' in completion:
            hashes_path = root / 'payload-hashes.json'
            require(sha(hashes_path) == completion['payload_hashes_sha256'], 'Payload manifest hash mismatch')
            for relative, expected in read(hashes_path).items():
                path = (root / relative).resolve(strict=True)
                require(path.is_relative_to(root), 'Payload path escapes capture')
                require(sha(path) == expected, 'Payload changed: ' + relative)
        cases = spec['cases']
        require(bool(cases) and len(cases) == receipt['frame_count'], 'Capture case count mismatch')
        require(torch.cuda.is_available(), 'CUDA required for dense native support verification')
        report.update(backend='CUDA', device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                      source_spec_sha256=sha(spec_path), receipt_sha256=sha(root / 'receipt.json'))
        masks_dir.mkdir(parents=True, exist_ok=False)
        for index, case in enumerate(cases):
            camera = case['camera']
            local_camera, wearer = local_query(camera, case.get('floor_z_m'))
            native_path = root / f'evaluator/native/{index:04d}.npy'
            data = np.load(native_path, allow_pickle=False)
            require(data.shape == (360,640) and data.dtype == np.dtype('<f4') and np.isfinite(data).all()
                    and (data >= 0).all() and (data < 100).all(), 'Invalid native axial depth')
            rgb_path = root / f'model/sample/{index:04d}.png'
            with Image.open(rgb_path) as image:
                image.load()
                require(image.format == 'PNG' and image.size == (640,360), 'Invalid RGB payload')
            native = torch.from_numpy(data).cuda()
            target, pooled, masks, valid = visible_support(native, local_camera, wearer)
            row = dict(sample_index=index, name=case['name'], camera=camera,
                       wearer_world=dict(x=camera['x'], y=camera['y'], z=float(camera['z'])+wearer['z'], yaw=camera['yaw']),
                       floor_authority='DECLARED' if 'floor_z_m' in case else 'ASSUMED_1_70M_EYE_HEIGHT',
                       valid_depth_pixels=int(valid.sum()), unknown_depth_pixels=int((~valid).sum()),
                       body_head_visible_targets=target.tolist(), visible_pixels_per_height=masks.sum((-2,-1)).tolist(),
                       native_sha256=sha(native_path), rgb_sha256=sha(rgb_path))
            report['rows'].append(row)
            if 'distance_policy' in spec:
                row['distance_evidence']=distance_evidence(native,local_camera,wearer,spec['distance_policy'])
                report['distance_policy']=dict(**spec['distance_policy'],
                    authority='CONFIGURED_STATIC_DEVELOPMENT_ONLY_NOT_SAFETY_THRESHOLDS',
                    no_support_semantics='NO_OBSERVED_RELEVANT_SURFACE_NOT_OBJECT_ABSENCE_OR_CERTIFIED_FREE_SPACE')
                if 'expected_distance_states' in case:
                    require([v['state'] for v in row['distance_evidence']]==case['expected_distance_states'],
                            'Distance scenario geometry differs from expected states: '+case['name'])
            if not row['valid_depth_pixels']:
                row.update(body_head_visible_targets=[None,None], support_status='UNKNOWN')
                raise ValueError('All native pixels UNKNOWN: ' + case['name'])
            row['support_status'] = ['PRESENT' if int(v) else 'NO_VISIBLE_SUPPORT' for v in target.tolist()]
            floor_check = case.get('floor_check', True)
            require(type(floor_check) is bool, 'floor_check must be boolean')
            if 'floor_z_m' in case and floor_check:
                points = camera_points(native, local_camera)
                valid_patch = valid[280:340,280:360]
                count = int(valid_patch.sum())
                require(count >= 100, 'Insufficient known floor patch pixels')
                error = (points[280:340,280:360,2][valid_patch] - wearer['z']).abs()
                median, fraction = float(error.median()), float((error <= .05).float().mean())
                row['floor_check'] = dict(status='PASS' if median <= .02 and fraction >= .95 else 'FAIL',
                    valid_pixels=count, median_error_m=median, within_5cm_fraction=fraction)
                require(row['floor_check']['status'] == 'PASS', 'Native floor patch differs from declared floor')
            else:
                row['floor_check'] = dict(status='SKIPPED', reason='EXPLICIT_FLOOR_CHECK_FALSE' if not floor_check else 'FLOOR_UNDECLARED')
            encoded = masks.to(torch.int8).masked_fill(~valid[None], -1)
            path = masks_dir / f'{index:04d}.npy'
            with path.open('xb') as stream:
                np.save(stream, encoded.cpu().numpy(), allow_pickle=False)
            row.update(mask_path=path.relative_to(root).as_posix(), mask_sha256=sha(path))
        for index, case in enumerate(cases):
            if not case.get('analytic_control', False):
                continue
            baseline = case['baseline_index']
            require(type(baseline) is int and 0 <= baseline < len(cases) and baseline != index, 'Invalid analytic baseline index')
            require(cases[baseline]['camera'] == case['camera'] and not cases[baseline].get('objects'), 'Analytic baseline must be empty at the same camera')
            objects = case.get('objects', [])
            require(len(objects) == 1 and 'size_m' in objects[0] and not any(k in objects[0] for k in ('mesh_asset','primitive_asset')), 'Analytic control requires one legacy cube')
            require(all(abs(float(v)) < 1e-8 for v in objects[0].get('rotation_deg', {}).values()), 'Analytic cube must be axis aligned')
            # The first pass validated every payload. Reload only this analytic
            # pair so GPU memory does not grow with the number of captured views.
            baseline_depth = torch.from_numpy(np.load(root / f'evaluator/native/{baseline:04d}.npy', allow_pickle=False)).cuda()
            control_depth = torch.from_numpy(np.load(root / f'evaluator/native/{index:04d}.npy', allow_pickle=False)).cuda()
            metric = score_cube(baseline_depth, control_depth, case['camera'], objects[0])
            del baseline_depth, control_depth
            report['rows'][index]['analytic_control'] = metric
            require(metric['status'] == 'PASS', 'Analytic cube depth check failed')
        torch.cuda.synchronize()
        report['status'] = 'PASS'
    except Exception as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
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
