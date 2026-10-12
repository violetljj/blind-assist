"""Camera-local visible-surface readout of estimated optical-Z depth.

Pixels are observed surfaces, not complete free-space coverage or collision
labels. Camera-right/down/forward is not a wearer/body frame. All query bounds
are closed, so points on shared boundaries may support more than one query.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from rgb_body_query_input_diagnostic import sha, write


def camera_queries():
    return [dict(name=f'{lateral}_{distance}', low=[lo, -.55, near],
                 high=[hi, .55, far])
            for distance, near, far in [('near', .3, 3.), ('far', 3., 6.)]
            for lateral, lo, hi in [('center', -.3, .3), ('left', -.9, -.3),
                                    ('right', .3, .9)]]


def resize_intrinsics(matrix, source_shape, target_shape):
    """Integer pixel centers; half-pixel resize matches PIL/bilinear sampling.

    Shape order is height,width. u'=(u+.5)*width_ratio-.5 and likewise v.
    """
    k = np.asarray(matrix, dtype=np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError('Expected finite 3x3 public K')
    if not np.allclose(k[2], [0., 0., 1.]) or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Expected pinhole K with positive focal lengths')
    if any(len(s) != 2 or any(int(x) != x or x <= 0 for x in s)
           for s in (source_shape, target_shape)):
        raise ValueError('Expected positive integer height,width shapes')
    sy, sx = np.asarray(target_shape, dtype=float) / np.asarray(source_shape)
    affine = np.array([[sx, 0., (sx-1)/2], [0., sy, (sy-1)/2], [0., 0., 1.]])
    return affine @ k


def visible_readout(optical_z, matrix, image_shape, queries=None, min_support=1):
    """Accept axial depth in metres. Radial range must be converted by caller.

    Support threshold is a pipeline presence check, not a confidence threshold.
    Invalid or non-positive values contribute no support; no support is UNKNOWN.
    All surface types are counted, without semantic or object attribution.
    """
    depth = np.asarray(optical_z)
    if depth.ndim != 2 or tuple(depth.shape) != tuple(image_shape):
        raise ValueError('Depth shape must equal public K image shape')
    k = resize_intrinsics(matrix, image_shape, image_shape)
    if int(min_support) != min_support or min_support < 1:
        raise ValueError('min_support must be a positive integer')
    qs = camera_queries() if queries is None else queries
    if not qs:
        raise ValueError('At least one public query required')
    yy, xx = np.indices(depth.shape, dtype=np.float64)
    # Inverse public K accepts even nonzero skew. Third ray coordinate is 1.
    inv = np.linalg.inv(k)
    ray_x = inv[0, 0]*xx + inv[0, 1]*yy + inv[0, 2]
    ray_y = inv[1, 0]*xx + inv[1, 1]*yy + inv[1, 2]
    valid = np.isfinite(depth) & (depth > 0)
    z = np.where(valid, depth, 0.).astype(np.float64)
    x, y = ray_x*z, ray_y*z
    rows = []
    for query in qs:
        low, high = np.asarray(query['low'], dtype=float), np.asarray(query['high'], dtype=float)
        if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or (low > high).any():
            raise ValueError('Query requires finite ordered xyz bounds')
        mask = valid.copy()
        for axis, lo, hi in zip((x, y, z), low, high):
            # 1e-12 m is arithmetic roundoff protection for closed boundaries,
            # not a sensor/depth tolerance or a tunable acceptance margin.
            mask &= (axis >= lo-1e-12) & (axis <= hi+1e-12)
        count = int(mask.sum())
        rows.append(dict(name=query['name'], low=low.tolist(), high=high.tolist(),
                         visible_support_pixels=count, support_fraction_of_image=count/depth.size,
                         estimated_optical_z_min_m=float(z[mask].min()) if count else None,
                         estimated_optical_z_max_m=float(z[mask].max()) if count else None,
                         state='ESTIMATED_VISIBLE_SUPPORT' if count >= min_support else 'UNKNOWN'))
    return dict(image_shape=list(depth.shape), valid_depth_pixels=int(valid.sum()),
                invalid_depth_pixels=int((~valid).sum()), total_pixels=int(depth.size),
                minimum_support_pixels=int(min_support), queries=rows)


def run(repo, output, budget_s=120):
    start = time.perf_counter()
    selection_path = output/'baseline_selection.json'
    selection = json.loads(selection_path.read_text('utf-8-sig'))
    smoke_path = output/'depthpro_smoke.json'
    smoke = json.loads(smoke_path.read_text('utf-8-sig'))
    if smoke['status'] != 'COMPLETE' or len(selection['rows']) != len(smoke['rows']):
        raise ValueError('Need complete matched smoke rows')
    rows = []
    for index, (selected, predicted) in enumerate(zip(selection['rows'], smoke['rows'])):
        if time.perf_counter()-start >= budget_s:
            raise TimeoutError('Geometry CPU budget reached')
        if any(selected[key] != predicted[key] for key in ('session', 'camera', 'frame', 'rgb_sha256')):
            raise ValueError('Smoke/selection identity mismatch')
        rgb_path, desc_path = repo/selected['rgb'], repo/selected['description']
        if sha(rgb_path) != selected['rgb_sha256'] or sha(desc_path) != selected['description_sha256']:
            raise ValueError('Public input identity changed')
        desc = json.loads(desc_path.read_text('utf-8-sig'))
        params = desc['session_camera_details'][desc['session_camera_location'].index(selected['camera'])]['left_camera_params']
        if any(float(x) != 0 for x in params['distortion']):
            raise ValueError('Only rectified public pinhole calibration is supported')
        public_shape = (params['image_height'], params['image_width'])
        public_k = np.array([[params['fx'], 0., params['cx']], [0., params['fy'], params['cy']], [0., 0., 1.]])
        with Image.open(rgb_path) as image:
            native_shape = (image.height, image.width)
        native_k = resize_intrinsics(public_k, public_shape, native_shape)
        arms = {}
        for arm, shape in [('native', native_shape), ('low128', (72, 128))]:
            path = output/f'depthpro_{index:02d}_{arm}.npz'
            with np.load(path, allow_pickle=False) as payload:
                depth = payload['depth']
            k = resize_intrinsics(native_k, native_shape, shape)
            expected_fx = predicted['fx_native'] if arm == 'native' else predicted['fx_low128']
            if not np.isclose(k[0, 0], expected_fx, rtol=1e-12, atol=1e-12):
                raise ValueError('Focal length differs from smoke inference')
            artifact_relative = path.resolve().relative_to((repo/'artifacts.local').resolve())
            arms[arm] = dict(depth_path=str(Path('artifacts.local')/artifact_relative), depth_sha256=sha(path),
                             public_K=k.tolist(), readout=visible_readout(depth, k, shape))
        rows.append(dict(session=selected['session'], camera=selected['camera'], frame=selected['frame'], arms=arms))
    supported = {arm: sum(q['state'] == 'ESTIMATED_VISIBLE_SUPPORT' for r in rows
                         for q in r['arms'][arm]['readout']['queries']) for arm in ('native', 'low128')}
    query_total = len(rows)*len(camera_queries())
    result = dict(status='COMPLETE', frames=len(rows), queries_per_arm=query_total,
                  supported_queries=supported, unknown_queries={arm: query_total-n for arm, n in supported.items()},
                  coordinate_contract='camera-right x, camera-down y, camera-forward z; estimated optical-Z metres',
                  resize_convention='integer pixel centers; u_new=(u_old+.5)*scale-.5; full-frame no crop',
                  boundaries='closed with 1e-12m arithmetic roundoff only; shared-boundary pixels can support adjacent queries',
                  source='already-consumed real official-train Development; frozen Depth Pro predictions, no new inference',
                  evidence='visible estimated surface pixels; all surface types, no object attribution',
                  labels='NOT_EVALUABLE: no independent depth/body extrinsic/obstacle truth or event timeline',
                  limits=['No support is UNKNOWN, never negative or clear passage',
                          'Native and low count denominators differ; count differences are not accuracy differences',
                          'Depth Pro inference accepted fx only; principal-point effects on depth predictions were not corrected',
                          'Camera-local boxes are not body envelopes; no collision, detection, real safety or mobile result'],
                  selection_sha256=sha(selection_path), smoke_sha256=sha(smoke_path),
                  seconds=time.perf_counter()-start, rows=rows)
    write(output/'geometry_readouts.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--budget-s', default=120., type=float)
    args = parser.parse_args()
    run(args.repo.resolve(), args.output.resolve(), args.budget_s)
