"""Height-normalized ray geometry on consumed thin-structure Development.

No labels, depth maps, threshold fitting, or candidate removal. A visible line
endpoint is not a ground contact. Ground bounds are conditional on points being
above the camera's horizontal floor plane; a shared forward distance further
assumes a vertical/frontoparallel line. Neither assumption is certified here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from atom_camera import AtomCamera, ROOT
import atom_tof_extrinsic_dev_20261012 as X
import real_rgb_query_feasibility_dev_20261012 as F

CACHE = ROOT / 'artifacts.local/work/fusion-refinement-dev-20261013'
SOURCE = ROOT / 'artifacts.local/work/thin-confidence-dev-20261013/distance_states.jsonl'
OUT = ROOT / 'artifacts.local/evidence/useful-quota-20261013/thin-metric'
SAMPLES = 40


def rays(xy, K):
    xy = np.asarray(xy, float)
    return np.column_stack(((xy[:, 0] - K[0, 2]) / K[0, 0],
                            (xy[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(xy))))


def segment_points(c):
    p = np.array([[c['x1'], c['y1']], [c['x2'], c['y2']]], float)
    return p[0] + np.linspace(0., 1., SAMPLES)[:, None] * (p[1] - p[0])


def ray_box_intervals(directions, lower, upper):
    """Exact positive camera-optical-z slabs, one interval per observed ray."""
    d = np.asarray(directions, float)
    lo, hi = np.zeros(len(d)), np.full(len(d), np.inf)
    for axis in range(3):
        nonzero = np.abs(d[:, axis]) > 1e-12
        a = np.full(len(d), -np.inf); b = np.full(len(d), np.inf)
        v = d[nonzero, axis]
        u, w = lower[axis] / v, upper[axis] / v
        a[nonzero], b[nonzero] = np.minimum(u, w), np.maximum(u, w)
        outside = ~nonzero & ((lower[axis] > 0) | (upper[axis] < 0))
        a[outside], b[outside] = np.inf, -np.inf
        lo, hi = np.maximum(lo, a), np.minimum(hi, b)
    valid = np.isfinite(lo) & np.isfinite(hi) & (hi >= lo) & (hi > 0)
    return lo, hi, valid


def line_gravity_error(c, K, rotation):
    p = np.array([[c['x1'], c['y1']], [c['x2'], c['y2']]], float)
    g = rotation.T @ np.array([0., 1., 0.])
    m = rays(p.mean(axis=0, keepdims=True), K)[0]
    tangent = np.array([K[0, 0] * (g[0] - m[0] * g[2]),
                        K[1, 1] * (g[1] - m[1] * g[2])])
    delta = p[1] - p[0]
    den = np.linalg.norm(tangent) * np.linalg.norm(delta)
    if den < 1e-12:
        return None
    return float(np.degrees(np.arccos(np.clip(abs(delta @ tangent) / den, 0., 1.))))


def geometry(c, K, rotation, provisional_height_m):
    q = rays(segment_points(c), K) @ rotation.T
    forward = q[:, 2] > 1e-12
    down = forward & (q[:, 1] > 1e-12)
    box = {}
    for name, (ylo, yhi) in F.BOXES.items():
        lo, hi, ok = ray_box_intervals(q, [-.30, ylo, .30], [.30, yhi, 2.00])
        box[name] = dict(possible_ray_points=int(ok.sum()), sampled_ray_points=SAMPLES,
                         camera_z_envelope_m=[float(lo[ok].min()), float(hi[ok].max())] if ok.any() else None,
                         status='ANGULAR_POSSIBILITY_ONLY' if ok.any() else 'SAMPLED_RAYS_MISS_BOX',
                         detection_or_safety_decision=None)
    upper = np.full(SAMPLES, np.inf)
    upper[down] = q[down, 2] / q[down, 1]
    finite = np.isfinite(upper)
    r = dict(geometry_status='CONDITIONAL_BOUND' if finite.any() else 'NO_FORWARD_DOWNWARD_RAY',
             ground_contact_status='UNKNOWN_NOT_OBSERVED', ground_plane_status='ASSUMED_SAME_LEVEL',
             gravity_alignment_error_deg=line_gravity_error(c, K, rotation),
             conditional_ground_upper_z_over_height=None, provisional_upper_z_m=None,
             lowest_ray_xy=None, lowest_ray_depression_deg=None,
             horizontal_range_over_height_elevation_plus_minus_2deg=None,
             pointwise_finite_ground_bounds=int(finite.sum()), box_ray_geometry=box,
             distance_verified=False, safety_decision=None)
    if finite.any():
        k = int(np.argmin(upper))
        alpha = float(np.arctan2(q[k, 1], np.hypot(q[k, 0], q[k, 2])))
        a0, a1 = alpha - np.deg2rad(2.), alpha + np.deg2rad(2.)
        r.update(conditional_ground_upper_z_over_height=float(upper[k]),
                 provisional_upper_z_m=float(upper[k] * provisional_height_m),
                 lowest_ray_xy=segment_points(c)[k].tolist(),
                 lowest_ray_depression_deg=float(np.degrees(alpha)),
                 horizontal_range_over_height_elevation_plus_minus_2deg=[
                     float(1 / np.tan(a1)) if a1 < np.pi / 2 else 0.,
                     float(1 / np.tan(a0)) if a0 > 0 else None])
    return r


def checks():
    K = np.array([[200., 0., 240.], [0., 200., 320.], [0., 0., 1.]])
    # A pole at z=2h, visible from y=0 to y=h: the true foot gives equality.
    c = dict(x1=240., y1=320., x2=240., y2=420.)
    g = geometry(c, K, np.eye(3), 1.5)
    assert abs(g['conditional_ground_upper_z_over_height'] - 2.) < 1e-12
    assert g['provisional_upper_z_m'] == 3.
    assert geometry(dict(c, y2=370.), K, np.eye(3), 1.5)['conditional_ground_upper_z_over_height'] == 4.
    assert geometry(dict(c, y1=220., y2=280.), K, np.eye(3), 1.5)['geometry_status'] == 'NO_FORWARD_DOWNWARD_RAY'
    rng = np.random.default_rng(20261013)
    d = rng.normal(size=(500, 3)); d[:, 2] = np.abs(d[:, 2]) + .01
    lo, hi, ok = ray_box_intervals(d, [-.3, .42, .3], [.3, .9, 2.])
    mid = (lo[ok] + hi[ok]) / 2
    p = d[ok] * mid[:, None]
    assert np.all((p >= [-.3, .42, .3]) & (p <= [.3, .9, 2.]))
    assert line_gravity_error(c, K, np.eye(3)) == 0.
    return dict(ground_contact_equality=True, truncated_endpoint_only_upper_bound=True,
                no_downward_ray_unknown=True, slab_midpoints_checked=int(ok.sum()),
                gravity_direction=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    start = time.perf_counter(); cpu_start = time.process_time()
    args.output.mkdir(parents=True, exist_ok=True)
    outpath = args.output / 'frames.jsonl'
    if outpath.exists():
        raise FileExistsError(f'Preserve existing evidence; choose a new --output: {outpath}')
    manifest = json.loads((CACHE / 'cache_manifest.json').read_text())
    meta = {(k, r['i']): r for k, s in manifest['sessions'].items() for r in s['frames']}
    height = manifest['height']; K = AtomCamera().K_upright
    Rct = np.asarray(json.loads((X.OUT / 'camera_tof_extrinsic.json').read_text())['rotation'])
    verified = checks(); counts = Counter(); sessions = {}; seen = set()
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    with SOURCE.open(encoding='utf8') as source, outpath.open('w', encoding='utf8') as out:
        for line in source:
            if time.perf_counter() - start > 180:
                raise TimeoutError('Per-script 180-second ceiling; retain partial evidence')
            row = json.loads(line); key, i = row['key'], row['i']
            assert (key, i) not in seen; seen.add((key, i))
            m = meta[key, i]; result = []
            valid_pose = bool(m['pose_ok']) and np.isfinite([m['pitch'], m['roll']]).all()
            rotation = F.rot(m['pitch'], m['roll']) @ Rct.T if valid_pose else None
            sc = sessions.setdefault(key, Counter()); sc['frames'] += 1
            for j, candidate in enumerate(row['candidates']):
                c = candidate['visual_candidate']
                g = geometry(c, K, rotation, height) if valid_pose else dict(
                    geometry_status='UNKNOWN_POSE', distance_verified=False, safety_decision=None)
                result.append(dict(candidate_index=j, visual_candidate=c, **g))
                sc['candidates'] += 1; sc[g['geometry_status']] += 1
                if valid_pose:
                    for name, b in g['box_ray_geometry'].items():
                        sc[f'{name}_angular_20_possible'] += b['possible_ray_points'] >= 20
            out.write(json.dumps(dict(key=key, i=i, rgb_index=m['rgb_index'],
                                     pitch_deg=float(m['pitch']) if np.isfinite(m['pitch']) else None,
                                     roll_deg=float(m['roll']) if np.isfinite(m['roll']) else None,
                                     candidates=result), allow_nan=False) + '\n')
    for sc in sessions.values(): counts.update(sc)
    assert counts['frames'] == 1677 and counts['candidates'] == 18483
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before
    summary = dict(goal='Independent candidate metric observability without DAV2 affine extrapolation',
                   budget=dict(total_branch_cpu_limit_s=900, gpu=0, downloads=0),
                   evidence='Existing consumed Development; no metric truth or new protected data',
                   source=str(SOURCE.relative_to(ROOT)), source_sha256=before,
                   provisional_camera_height_m=height, height_source=manifest['height_source'],
                   height_verified_by_measurement=False, samples_per_candidate=SAMPLES,
                   assumptions=['Ground plane is horizontal and at the same level as under the camera.',
                                'Each sampled target point is at or above that plane.',
                                'Minimum per-point bound applies to the complete line only if its forward distance is constant.',
                                'Actual endpoint ground contact and camera height are not verified.'],
                   semantics=dict(conditional_ground_upper_z_over_height='One-sided upper bound under stated assumptions; lower bound remains zero.',
                                  provisional_upper_z_m='Upper bound times inherited ToF-fitted height, not measured range.',
                                  box_ray_geometry='Necessary angular possibility for sampled rays; no depth or target-presence decision.',
                                  elevation_sensitivity='Fixed +/-2 degree scenario, not estimated uncertainty or a confidence interval.'),
                   counts=counts, sessions=sessions, checks=verified,
                   wall_s=time.perf_counter() - start, process_cpu_s=time.process_time() - cpu_start)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(dict(counts=counts, checks=verified, wall_s=summary['wall_s'], process_cpu_s=summary['process_cpu_s'])))


if __name__ == '__main__':
    main()
