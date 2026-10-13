"""Consumed Development: floor homography -> translation/height -> pole line.

This is a conditional geometry diagnostic. The deterministic lower-side image
mask is a presumed-floor proposal, not a semantic floor label. Keep every
homography decomposition. Automatic selection requires exactly one solution
meeting fixed normal/gravity/cheirality checks; never select using pole depth.
No DAV2 values or annotation coordinates participate in this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from atom_camera import AtomCamera, ROOT
import atom_tof_extrinsic_dev_20261012 as X
import real_rgb_query_feasibility_dev_20261012 as F
from thin_metric_geometry_dev_20261013 import CACHE, OUT, rays, line_gravity_error

TRACKS = ROOT / 'artifacts.local/work/thin-local-geometry-dev-20261013/camera_chain_frames.jsonl'
KEYS = ('073554', '073635', '073821')
CFG = dict(pair_positions=[.2, .4, .6, .8], target_gap_s=.5,
           mask_lower_fraction=.55, mask_side_fraction=.25, min_ground_depression_deg=10.,
           sift_features=2500, sift_contrast=.02, descriptor_ratio=.75, ransac_px=2.,
           min_inliers=12, min_x_span_fraction=.35, min_y_span_fraction=.15,
           max_floor_normal_error_deg=10., max_gravity_transport_error_deg=5.,
           pole_gravity_angle_deg=15., line_shift_sensitivity_px=1.)


def angle(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b / den, -1., 1.)))) if den > 1e-12 else None


def line_plane(c, K):
    p = rays([[c['x1'], c['y1']], [c['x2'], c['y2']]], K)
    n = np.cross(p[0], p[1])
    return n / np.linalg.norm(n)


def shifted(c, shift):
    delta = np.array([c['x2'] - c['x1'], c['y2'] - c['y1']], float)
    normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
    return dict(c, x1=c['x1'] + shift * normal[0], y1=c['y1'] + shift * normal[1],
                x2=c['x2'] + shift * normal[0], y2=c['y2'] + shift * normal[1])


def triangulate_vertical(c1, c2, K, camera1_to_corridor, relative_R, relative_t):
    """Solve the intersection of two ray planes with a gravity-vertical line.

    relative_t is in camera-height units; output has the same units. Segment
    endpoints need not identify the same physical points on the vertical pole.
    """
    n1, n2 = line_plane(c1, K), line_plane(c2, K)
    a = np.vstack((n1 @ camera1_to_corridor.T,
                   n2 @ relative_R @ camera1_to_corridor.T))
    matrix = a[:, [0, 2]]
    condition = float(np.linalg.cond(matrix))
    plane_angle = float(np.degrees(np.arccos(np.clip(abs(n1 @ (relative_R.T @ n2)), 0., 1.))))
    result = dict(status='PARALLEL_OR_ILL_CONDITIONED', condition=condition if np.isfinite(condition) else None,
                  plane_angle_deg=plane_angle, x_over_height=None, z_over_height=None,
                  height_gauge='Inherited camera-to-floor height', distance_verified=False)
    if not np.isfinite(condition) or condition > 1e8:
        return result
    x, z = np.linalg.solve(matrix, [0., -float(n2 @ relative_t)])
    camera2_point = relative_R @ camera1_to_corridor.T @ np.array([x, 0., z]) + relative_t
    front = bool(z > 0 and camera2_point[2] > 0)
    result.update(status='CONDITIONAL_FINITE_FRONT' if front else 'BEHIND_OR_INVALID',
                  x_over_height=float(x), z_over_height=float(z),
                  plane_vertical_residuals=a[:, 1].tolist())
    return result


def with_sensitivity(c1, c2, K, B, R, t):
    r = triangulate_vertical(c1, c2, K, B, R, t)
    variants = [triangulate_vertical(shifted(c1, a), shifted(c2, b), K, B, R, t)
                for a in (-1., 1.) for b in (-1., 1.)]
    zz = [v['z_over_height'] for v in variants if v['z_over_height'] is not None]
    r['one_pixel_line_shift_z_over_height_minmax'] = [min(zz), max(zz)] if zz else None
    r['one_pixel_line_shift_front_count'] = sum(v['status'] == 'CONDITIONAL_FINITE_FRONT' for v in variants)
    r['sensitivity_note'] = 'Line offsets only; excludes homography, floor, pose, timing and height uncertainty.'
    return r


def synthetic_checks():
    K = np.array([[200., 0., 240.], [0., 200., 320.], [0., 0., 1.]])
    def project_line(x, z, R, t):
        p = np.array([[x, -.2, z], [x, .8, z]]) @ R.T + t
        uv = p @ K.T; uv = uv[:, :2] / uv[:, 2, None]
        return dict(zip(('x1', 'y1', 'x2', 'y2'), uv.ravel()))
    I = np.eye(3); t = np.array([-.1, 0., 0.]); c1 = project_line(.2, 2., I, np.zeros(3))
    c2 = project_line(.2, 2., I, t)
    r = triangulate_vertical(c1, c2, K, I, I, t)
    assert r['status'] == 'CONDITIONAL_FINITE_FRONT'
    assert np.allclose([r['x_over_height'], r['z_over_height']], [.2, 2.])
    scaled = triangulate_vertical(c1, c2, K, I, I, t / 2)
    assert np.allclose([scaled['x_over_height'], scaled['z_over_height']], [.1, 1.])
    yaw = cv2.Rodrigues(np.array([0., .1, 0.]))[0]
    pure = triangulate_vertical(c1, project_line(.2, 2., yaw, np.zeros(3)), K, I, yaw, np.zeros(3))
    assert pure['status'] == 'PARALLEL_OR_ILL_CONDITIONED'
    t = np.array([0., 0., -.1]); center = project_line(0., 2., I, np.zeros(3))
    forward = triangulate_vertical(center, project_line(0., 2., I, t), K, I, I, t)
    assert forward['status'] == 'PARALLEL_OR_ILL_CONDITIONED'
    side = triangulate_vertical(c1, project_line(.2, 2., I, t), K, I, I, t)
    assert np.isclose(side['z_over_height'], 2.)
    return dict(lateral_translation_recovers_height_normalized_line=True,
                metric_gauge_scales_with_baseline=True, pure_rotation_unobservable=True,
                forward_center_unobservable=True, forward_side_observable=True)


def floor_mask(shape, K, B):
    h, w = shape; yy, xx = np.indices(shape)
    q = rays(np.column_stack((xx.ravel(), yy.ravel())), K) @ B.T
    dep = np.arctan2(q[:, 1], np.hypot(q[:, 0], q[:, 2])).reshape(shape)
    use = (yy >= CFG['mask_lower_fraction'] * h) & ((xx < CFG['mask_side_fraction'] * w) |
          (xx >= (1 - CFG['mask_side_fraction']) * w)) & (dep > np.deg2rad(CFG['min_ground_depression_deg']))
    return use.astype(np.uint8) * 255


def select_pairs(manifest, tracks):
    pairs = []
    for key in KEYS:
        unique = {}; available = []
        for m in manifest['sessions'][key]['frames']:
            if not m.get('cache') or not m['pose_ok'] or not np.isfinite([m['pitch'], m['roll']]).all():
                continue
            if m['rgb_index'] not in unique:
                unique[m['rgb_index']] = True; available.append(m)
        indices = np.rint(np.asarray(CFG['pair_positions']) * (len(available) - 1)).astype(int)
        for bidx in indices:
            b = available[bidx]; earlier = available[:bidx]
            if not earlier:
                continue
            a = min(earlier, key=lambda m: abs((b['rgb_t_ns'] - m['rgb_t_ns']) / 1e9 - CFG['target_gap_s']))
            pairs.append((key, a, b))
    return pairs


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path)
    parser.add_argument('--correspondence', choices=('sift', 'lk'), default='sift')
    args = parser.parse_args()
    if args.output is None:
        args.output = OUT / ('multiview' if args.correspondence == 'sift' else 'multiview-lk')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'results.json').exists():
        raise FileExistsError('Do not overwrite evidence; use --output')
    start = time.perf_counter(); cpu_start = time.process_time(); cv2.setNumThreads(2)
    manifest = json.loads((CACHE / 'cache_manifest.json').read_text())
    tracks = {(r['key'], r['i']): r for r in map(json.loads, TRACKS.read_text().splitlines()) if r['key'] in KEYS}
    pairs = select_pairs(manifest, tracks); checks = synthetic_checks()
    plan = dict(evidence='Consumed Development, conditional geometry only', cpu_branch_limit_s=900,
                mechanism='Presumed-floor homography -> t/h -> vertical line-plane triangulation',
                configuration=CFG, correspondence=args.correspondence,
                lk_configuration=dict(max_corners=250, quality_level=.01, min_distance_px=6,
                                      block_size=7, win_size=[21,21], pyramid_level=3,
                                      max_forward_backward_error_px=1.5),
                pairs=[dict(key=k, i0=a['i'], i1=b['i']) for k, a, b in pairs],
                fixed_before_results=True, annotations_read=False, depth_values_read=False,
                height_source=manifest['height_source'], height_measured=False,
                stop='One fixed configuration; preserve failures; no thresholds or masks retuned using outcomes.')
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf8')
    cam = AtomCamera(); K = cam.K_upright
    Rct = np.asarray(json.loads((X.OUT / 'camera_tof_extrinsic.json').read_text())['rotation'])
    sift = cv2.SIFT_create(nfeatures=CFG['sift_features'], contrastThreshold=CFG['sift_contrast'])
    rgb = {k: list(F.lines(F.RAW / manifest['sessions'][k]['session'] / 'rgb.jsonl')) for k in KEYS}
    payloads = {}; results = []; counts = Counter()
    def image_features(key, m):
        identity = (key, m['rgb_index'])
        if identity not in payloads:
            path = F.RAW / manifest['sessions'][key]['session'] / rgb[key][m['rgb_index']]['file']
            im = cam.undistort_upright(cv2.imread(str(path)))
            B = F.rot(m['pitch'], m['roll']) @ Rct.T
            mask = floor_mask(im.shape[:2], K, B)
            kp, des = sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), mask)
            payloads[identity] = (im, B, mask, kp, des)
        return payloads[identity]
    for pair_index, (key, a, b) in enumerate(pairs):
        if time.perf_counter() - start > 300:
            raise TimeoutError('300-second per-script ceiling')
        im1, B1, mask1, kp1, des1 = image_features(key, a)
        im2, B2, mask2, kp2, des2 = image_features(key, b)
        h, w = im1.shape[:2]
        pair = dict(key=key, i0=a['i'], i1=b['i'], rgb0=a['rgb_index'], rgb1=b['rgb_index'],
                    gap_s=(b['rgb_t_ns'] - a['rgb_t_ns']) / 1e9,
                    source='Automatic lower-side presumed-floor mask, unverified floor semantics',
                    floor_verified=False, depth_verified=False, match_count=0, inliers=0,
                    status='INSUFFICIENT_MATCHES', solutions=[], selected_solution=None, candidates=[])
        canvas = np.hstack((im1.copy(), im2.copy()))
        for mask, offset in ((mask1, 0), (mask2, w)):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour[:, :, 0] += offset; cv2.polylines(canvas, [contour], True, (255, 160, 0), 1)
        good = []
        if des1 is not None and des2 is not None:
            for mm in cv2.BFMatcher().knnMatch(des1, des2, k=2):
                if len(mm) == 2 and mm[0].distance < CFG['descriptor_ratio'] * mm[1].distance:
                    good.append(mm[0])
        p1 = np.array([kp1[m.queryIdx].pt for m in good], np.float32).reshape(-1,2)
        p2 = np.array([kp2[m.trainIdx].pt for m in good], np.float32).reshape(-1,2)
        if args.correspondence == 'lk':
            gray1, gray2 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(gray1, 250, .01, 6, mask=mask1, blockSize=7)
            p1 = p2 = np.empty((0,2), np.float32)
            pair['raw_corners'] = 0 if corners is None else len(corners)
            if corners is not None:
                moved, st1, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners, None, winSize=(21,21), maxLevel=3)
                if moved is not None:
                    back, st2, _ = cv2.calcOpticalFlowPyrLK(gray2, gray1, moved, None, winSize=(21,21), maxLevel=3)
                    if back is not None:
                        original, forward = corners[:,0], moved[:,0]
                        fb = np.linalg.norm(back[:,0] - original, axis=1)
                        ij = np.rint(forward).astype(int)
                        bounds = (ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
                        safe = np.clip(ij, [0,0], [w-1,h-1])
                        retained = st1.ravel().astype(bool) & st2.ravel().astype(bool) & (fb<=1.5) & bounds & (mask2[safe[:,1],safe[:,0]]>0)
                        p1, p2 = original[retained], forward[retained]
                        pair['lk_forward_backward_median_px'] = float(np.median(fb[retained])) if retained.any() else None
        pair['match_count'] = len(p1)
        if len(p1) >= 4:
            H, inlier = cv2.findHomography(p1, p2, cv2.RANSAC, CFG['ransac_px'], maxIters=5000, confidence=.999)
            if H is not None and inlier is not None:
                keep = inlier.ravel().astype(bool); pair['inliers'] = int(keep.sum())
                span = np.ptp(p1[keep], axis=0) / [w, h]
                pair['inlier_span_fraction_xy'] = span.tolist()
                floor_support = bool(keep.sum() >= CFG['min_inliers'] and span[0] >= CFG['min_x_span_fraction'] and span[1] >= CFG['min_y_span_fraction'])
                pair['homography_support_pass'] = floor_support
                mapped = cv2.perspectiveTransform(p1[keep, None], H)[:, 0]
                pair['reprojection_median_px'] = float(np.median(np.linalg.norm(mapped - p2[keep], axis=1)))
                pair['homography'] = H.tolist()
                n, rotations, translations, normals = cv2.decomposeHomographyMat(H, K)
                expected1, expected2 = B1.T @ [0., 1., 0.], B2.T @ [0., 1., 0.]
                q1 = rays(p1[keep], K)
                eligible = []
                for sidx, (R, t_raw, normal_raw) in enumerate(zip(rotations, translations, normals, strict=True)):
                    t, normal = t_raw.ravel(), normal_raw.ravel()
                    normal_error, gravity_error = angle(normal, expected1), angle(R @ expected1, expected2)
                    den = q1 @ normal
                    positive = den > 1e-9
                    points = np.zeros_like(q1); points[positive] = q1[positive] / den[positive, None]
                    points2 = points @ R.T + t
                    front = positive & (points2[:, 2] > 0)
                    fraction = float(front.mean())
                    passed = bool(floor_support and normal_error is not None and normal_error <= CFG['max_floor_normal_error_deg']
                                  and gravity_error <= CFG['max_gravity_transport_error_deg'] and front.all())
                    translation_corridor = B1 @ (-R.T @ t)
                    solution = dict(index=sidx, R=R.tolist(), t_over_plane_distance=t.tolist(), normal=normal.tolist(),
                                    normal_error_deg=normal_error, gravity_transport_error_deg=gravity_error,
                                    positive_ground_fraction=fraction, eligible=passed,
                                    camera_translation_over_plane_distance_corridor=translation_corridor.tolist(),
                                    translation_norm_over_plane_distance=float(np.linalg.norm(t)))
                    pair['solutions'].append(solution)
                    if passed: eligible.append(sidx)
                pair['status'] = ('CONDITIONAL_UNIQUE_SOLUTION' if len(eligible) == 1 else
                                  'AMBIGUOUS_SOLUTIONS' if len(eligible) > 1 else 'NO_ADMISSIBLE_FLOOR_SOLUTION')
                if len(eligible) == 1: pair['selected_solution'] = eligible[0]
                aa = {c['track_id']: c for c in tracks[key, a['i']]['candidates']}
                bb = {c['track_id']: c for c in tracks[key, b['i']]['candidates']}
                for track in sorted(aa.keys() & bb.keys()):
                    c1, c2 = aa[track], bb[track]
                    errs = [line_gravity_error(c1, K, B1), line_gravity_error(c2, K, B2)]
                    vertical = all(e is not None and e <= CFG['pole_gravity_angle_deg'] for e in errs)
                    triangulations = []
                    for solution, R, t in zip(pair['solutions'], rotations, translations, strict=True):
                        result = with_sensitivity(c1, c2, K, B1, R, t.ravel())
                        result['solution_index'] = solution['index']; triangulations.append(result)
                    c = dict(track_id=track, c0={n:c1[n] for n in ('x1','y1','x2','y2','rank','stable')},
                             c1={n:c2[n] for n in ('x1','y1','x2','y2','rank','stable')},
                             gravity_errors_deg=errs, vertical_hypothesis_compatible=vertical,
                             all_solution_triangulations=triangulations, selected_triangulation=None)
                    if pair['selected_solution'] is not None and vertical:
                        c['selected_triangulation'] = triangulations[pair['selected_solution']]
                    pair['candidates'].append(c)
                    for line, offset in ((c1, 0), (c2, w)):
                        color = (230, 0, 230) if vertical else (80, 80, 180)
                        cv2.line(canvas, (round(line['x1']) + offset, round(line['y1'])),
                                 (round(line['x2']) + offset, round(line['y2'])), color, 2)
                for j, (u, v) in enumerate(zip(p1, p2)):
                    color = (20, 220, 20) if keep[j] else (60, 60, 200)
                    cv2.circle(canvas, tuple(np.rint(u).astype(int)), 3, color, 1)
                    cv2.circle(canvas, tuple(np.rint(v + [w, 0]).astype(int)), 3, color, 1)
                    if keep[j]: cv2.line(canvas, tuple(np.rint(u).astype(int)), tuple(np.rint(v + [w, 0]).astype(int)), color, 1)
        header = np.zeros((48, canvas.shape[1], 3), np.uint8)
        label = f"{key} {a['i']}->{b['i']} match={pair['match_count']} inlier={pair['inliers']} {pair['status']}"
        cv2.putText(header, label, (8, 29), 0, .51, (255, 255, 255), 1)
        panel = args.output / f'pair_{pair_index:02d}_{key}_{a["i"]}_{b["i"]}.jpg'
        cv2.imwrite(str(panel), np.vstack((header, canvas)))
        pair['panel'] = str(panel.relative_to(ROOT)); results.append(pair); counts[pair['status']] += 1
        counts['pairs'] += 1; counts['common_track_candidates'] += len(pair['candidates'])
        counts['selected_finite_vertical_lines'] += sum(c['selected_triangulation'] is not None and c['selected_triangulation']['status'] == 'CONDITIONAL_FINITE_FRONT' for c in pair['candidates'])
        (args.output / 'results.partial.json').write_text(json.dumps(results, indent=2), encoding='utf8')
        print(label, flush=True)
    result = dict(plan=plan, checks=checks, counts=counts, pairs=results,
                  interpretation='Plane distance is treated as camera height only under the assumed floor normal; all output remains conditional. No metric-ground-truth comparison exists.',
                  provisional_height_m=manifest['height'], camera_height_measured=False,
                  wall_s=time.perf_counter()-start, process_cpu_s=time.process_time()-cpu_start,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    print(json.dumps(dict(counts=counts, checks=checks, wall_s=result['wall_s'], process_cpu_s=result['process_cpu_s'])))


if __name__ == '__main__':
    main()
