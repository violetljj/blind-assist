"""Evaluator-only geometric coverage on retained/replayed synthetic exposures.

No photons, CNH histograms, images, neural models, or protected data. Natural
motion is deterministic replay of the retained generator, not saved real motion.
All outputs of this module are evaluator truth; never pass them to online rules.
"""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
NATURAL = ROOT / 'artifacts.local/work/cnh-margin-confirm-20261002'
EDGE = np.tan(np.pi/8)
MAX_RADIAL_M = 2.1
OCCLUSION_TOLERANCE_M = .025


@lru_cache(maxsize=1)
def load_natural_manifest():
    rows = json.loads((NATURAL/'scene_manifest.json').read_text(encoding='utf8'))
    result = {(r['split'], int(r['unit']), int(r['config'])): r for r in rows}
    if len(result) != 5760 or len(result) != len(rows):
        raise ValueError('Natural manifest expected 5760 unique scenes')
    return result


def natural_poses(unit, config):
    """Return true_sensor, true_travel, estimated_sensor, each [16,4,4]."""
    from cnh_cvr_pilot import motion_metadata
    return motion_metadata(int(unit), int(config))


@lru_cache(maxsize=1)
def public_rays():
    import cnh_proposal_attribution_scenes as S
    return S.angular_rays(16)[0].reshape(-1, 3)


def tangent_ray_indices(local):
    """Nearest midpoint in the original128x128 tangent grid, zone-major ID."""
    tangent = np.divide(local[:, :2], local[:, 2:3],
                        out=np.zeros_like(local[:, :2]), where=local[:, 2:3] != 0)
    xy = np.clip(np.floor((tangent+EDGE)*128/(2*EDGE)).astype(np.int64), 0, 127)
    x, y = xy[:, 0], xy[:, 1]
    return ((y//16)*8+x//16)*256+(y%16)*16+x%16


def _sealed_distance(sealed):
    if isinstance(sealed, dict):
        for key in ('distance', 'ray_distance', 'radial_m'):
            if key in sealed:
                value = np.asarray(sealed[key], float)
                break
        else:
            raise ValueError('Sealed geometry lacks radial ray distance; IDs alone cannot establish occlusion')
    else:
        value = np.asarray(sealed, float)
    if value.shape == (128, 128):
        value = value.reshape(8, 16, 8, 16).transpose(0, 2, 1, 3).reshape(8, 8, 256)
    if value.shape not in ((8, 8, 256), (16384,)):
        raise ValueError('Single exposure distance must be[8,8,256],[128,128],or[16384]')
    return value.reshape(-1)


def observed_points(points, pose, boxes=None, sealed=None):
    """World points sampled in FOV/range with no first-hit surface in front.

    Select nearest original midpoint subray, then compare its first-hit radial
    range >= the point radial range minus2.5cm. No-hit infinity means the
    geometric ray reaches the point; it is not evidence of usable photon SNR.
    Only unique queried subrays are recast for natural box replay.
    """
    points, pose = np.asarray(points, float), np.asarray(pose, float)
    if points.ndim != 2 or points.shape[1] != 3 or pose.shape != (4, 4):
        raise ValueError('Expected world points[N,3] and pose[4,4]')
    if (boxes is None) == (sealed is None):
        raise ValueError('Supply exactly one of evaluator boxes or sealed ray distances')
    local = (points-pose[:3, 3]) @ pose[:3, :3]
    radius = np.linalg.norm(local, axis=1)
    inside = (np.isfinite(local).all(1) & (local[:, 2] > 0) &
              (radius <= MAX_RADIAL_M+1e-12) &
              (np.abs(local[:, 0]) <= EDGE*local[:, 2]+1e-12) &
              (np.abs(local[:, 1]) <= EDGE*local[:, 2]+1e-12))
    result = np.zeros(len(points), bool)
    if not inside.any():
        return result
    ray = tangent_ray_indices(local[inside])
    if sealed is not None:
        first = _sealed_distance(sealed)[ray]
    else:
        import cnh_proposal_attribution_scenes as S
        unique, reverse = np.unique(ray, return_inverse=True)
        hit = S.raycast_boxes(pose[:3, 3], public_rays()[unique]@pose[:3, :3].T, boxes)
        first = hit['distance'][reverse]
    result[inside] = first >= radius[inside]-OCCLUSION_TOLERANCE_M
    return result


def target_intrusion_points(box, half_width=.29):
    """Boundary intersections on target front face; evaluator world coordinates.

    Used for the controlled straight-world corridor. Empty if neither corridor
    boundary crosses target; do not turn missing points into a negative result.
    """
    lo, hi = np.asarray(box['lo'], float), np.asarray(box['hi'], float)
    xs = [x for x in (-half_width, half_width) if lo[0]-1e-12 <= x <= hi[0]+1e-12]
    return np.array([[x, (lo[1]+hi[1])/2, lo[2]] for x in xs], float).reshape(-1, 3)


def intrusion_point(box):
    points = target_intrusion_points(box)
    return points[0] if len(points) else None


def check_sealed_ids(sensor, branch_boxes, sealed):
    import cnh_proposal_attribution_scenes as S
    differences = 0
    for b, boxes in enumerate(branch_boxes):
        for f, pose in enumerate(sensor):
            ids = S.raycast_boxes(pose[:3,3],public_rays()@pose[:3,:3].T,boxes)['object_id']
            differences += int(np.count_nonzero(ids != sealed[b,f].reshape(-1)))
    if differences:
        raise ValueError('Evaluator geometry differs from original sealed IDs')
    return dict(status='PASS',subrays=int(sealed.size),object_id_differences=differences)


def engineering_check():
    import cnh_proposal_attribution_scenes as S
    pose = np.eye(4)
    points = np.array([[0, 0, 1.], [0, 0, 2.], [0, 0, 2.2], [.5, 0, 1.]])
    wall = dict(lo=[-.3, -.3, 1.5], hi=[.3, .3, 1.6], rho=.5)
    replay = observed_points(points, pose, boxes=[wall])
    np.testing.assert_array_equal(replay, [True, False, False, False])
    all_hits = S.raycast_boxes(pose[:3, 3], public_rays(), [wall])
    np.testing.assert_array_equal(replay, observed_points(points, pose, sealed=all_hits))
    empty = observed_points(points[:2], pose, boxes=[])
    assert empty.all()
    # Ray index mapping is exact for every original midpoint direction.
    ids = tangent_ray_indices(public_rays())
    np.testing.assert_array_equal(ids, np.arange(16384))
    # A sealed ID-only payload must not silently infer ray range.
    try:
        observed_points(points[:1], pose, sealed={'object_id': np.zeros(16384)})
    except ValueError:
        pass
    else:
        raise AssertionError('ID-only payload accepted')
    return dict(status='PASS', checks=['original16384midpoint ray index parity',
        'occluded/visible/radial-out/FOV-out points', 'selected replay equals full sealed distance',
        'no-hit ray reaches free-space point', 'ID-only sealed payload rejected'],
        photon_or_histogram_render_calls=0)
