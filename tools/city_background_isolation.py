"""Offline conservative descriptor/frustum audit; never grants source admission.

Actor package identity is an aggregate, not component/instance identity. Infinite
far distance deliberately retains skyline geometry. AABB plane rejection ignores
occlusion and can overestimate visibility. HLOD source membership remains unknown.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np

GLOBAL_LIGHT_CLASSES = {'/Script/Engine.SkyLight', '/Script/Engine.DirectionalLight'}


def native_class(row):
    value = row.get('native_class', '')
    match = re.search(r"'(/Script/[^']+)'", value)
    return match.group(1) if match else value


def frustum_candidates(bounds, camera, hfov=100., aspect=640/360):
    """Return a superset using AABB support against five infinite-frustum planes.

    Unreal camera: X forward at zero yaw, Y right, Z up; positive pitch looks up.
    No near/far clipping. Invalid bounds are retained, never silently rejected.
    """
    values = [camera[k] for k in ('x', 'y', 'z', 'yaw')] + [camera.get('pitch', 0), hfov, aspect]
    if not all(math.isfinite(x) for x in values) or not 0 < hfov < 180 or aspect <= 0:
        raise ValueError('Finite camera, valid FOV and aspect required')
    if camera.get('roll', 0) != 0:
        raise ValueError('Nonzero camera roll is unsupported')
    yaw, pitch = map(math.radians, (camera['yaw'], camera.get('pitch', 0)))
    forward = np.array([math.cos(pitch)*math.cos(yaw), math.cos(pitch)*math.sin(yaw), math.sin(pitch)])
    right = np.array([-math.sin(yaw), math.cos(yaw), 0.])
    up = np.cross(forward, right)
    th = math.tan(math.radians(hfov)/2)
    planes = np.array([forward, forward*th+right, forward*th-right,
                       forward*th/aspect+up, forward*th/aspect-up])
    bounds = np.asarray(bounds, dtype=float)
    valid = np.isfinite(bounds).all(axis=(1, 2)) & (bounds[:, 1] >= bounds[:, 0]).all(axis=1)
    center = (bounds[:, 0]+bounds[:, 1])/2 - np.array([camera[k] for k in ('x', 'y', 'z')])
    extent = (bounds[:, 1]-bounds[:, 0])/2
    # Touching a plane is retained. Tolerance also absorbs floating point error.
    result = np.ones(len(bounds), dtype=bool)
    for normal in planes:
        result &= (np.sum(center*normal, axis=1) + np.sum(extent*np.abs(normal), axis=1)) >= -1e-7
    return result | ~valid


def audit(descriptors, views, hfov=100., aspect=640/360):
    rows = descriptors['rows']
    packages = [r['actor_package'] for r in rows]
    if any(not p.startswith('/') for p in packages) or len(set(packages)) != len(packages):
        raise ValueError('Unique absolute actor package paths required')
    if not views or len({v['view_id'] for v in views}) != len(views):
        raise ValueError('Nonempty uniquely identified views required')
    excluded = [r for r in rows if native_class(r) in GLOBAL_LIGHT_CLASSES]
    rows = [r for r in rows if native_class(r) not in GLOBAL_LIGHT_CLASSES]
    bounds = np.array([[r['bounds_m']['min'], r['bounds_m']['max']] for r in rows], dtype=float)
    hlod = {r['actor_package'] for r in rows if native_class(r) == '/Script/Engine.WorldPartitionHLOD'}
    by_id = {r['actor_package']: r for r in rows}
    results, sets = [], {}
    for view in views:
        selected = [r['actor_package'] for r, keep in zip(rows, frustum_candidates(bounds, view['camera'], hfov, aspect)) if keep]
        sets[view['view_id']] = set(selected)
        results.append(dict(view, candidate_actor_packages=selected, candidate_count=len(selected),
                            unmapped_hlod_count=len(set(selected) & hlod)))
    overlaps = []
    for a, b in itertools.combinations(views, 2):
        if a['split'] == b['split']:
            continue
        common = sets[a['view_id']] & sets[b['view_id']]
        direct = common - hlod
        overlaps.append(dict(view_a=a['view_id'], view_b=b['view_id'], count=len(common),
            direct_actor_count=len(direct), shared_hlod_count=len(common & hlod),
            direct_building_label_count=sum(by_id[p].get('label', '').startswith('BLDG') for p in direct),
            example_actor_packages=sorted(direct)[:12]))
    return dict(schema='city-background-descriptor-frustum-v1', status='NOT_ADMITTED',
        visible_background_isolation='UNVERIFIED', map_asset=descriptors['map_asset'],
        backend='CPU', backend_reason='TASK_NOT_GPU_SUITABLE_METADATA',
        identity='map_asset + actor_package; aggregate actor identity only',
        geometry=dict(horizontal_fov_degrees=hfov, aspect=aspect, far_clip_m=None, occlusion=False),
        source_count=len(packages), unique_guid_repr_count=len({r.get('guid') for r in descriptors['rows']}),
        invalid_guid_repr_count=sum(str(r.get('guid', '')).startswith('<Struct') for r in descriptors['rows']),
        invalid_bounds_count=int((~(np.isfinite(bounds).all(axis=(1, 2)) & (bounds[:, 1]>=bounds[:, 0]).all(axis=1))).sum()),
        excluded_global_lights=[dict(actor_package=r['actor_package'], native_class=native_class(r), reason='Global illumination source, no physical building identity') for r in excluded],
        views=results, cross_split_overlaps=overlaps,
        limitations=['Potential overlap is not confirmed visible leakage; descriptor bounds and no occlusion overestimate.',
                     'Even zero overlap cannot admit: descriptor completeness, bounds coverage, HLOD source membership and component/instance mapping are unverified.',
                     'No distance, label, spatially-loaded flag, collision-only guess or region-box culling is applied.',
                     'Mirrors, reflections and indirect rendering dependencies are not represented by a primary-view frustum.'])


def candidate_views(candidates):
    """Hypothetical yaw sweep only; ground descriptor height is not a floor probe."""
    return [dict(view_id=f"{r['region_id']}_yaw{yaw}", region_id=r['region_id'], split=r['split'],
                 camera=dict(x=r['candidate_center_xy_m'][0], y=r['candidate_center_xy_m'][1],
                             z=r['ground_descriptor']['bounds_m']['max'][2]+1.6, yaw=yaw, pitch=0),
                 pose_authority='HYPOTHETICAL_CENTER_NOT_FLOOR_OR_OBSTRUCTION_VERIFIED')
            for r in candidates['regions'] for yaw in range(0, 360, 45)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--descriptors', type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--views', type=Path, help='JSON views with view_id, split, camera')
    group.add_argument('--candidates', type=Path, help='Hypothetical center yaw sweep')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    canonical = (Path(__file__).resolve().parents[1]/'artifacts.local').resolve()
    if args.output.exists() or not args.output.resolve().is_relative_to(canonical):
        raise ValueError('Fresh canonical artifacts.local output required')
    load = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
    views = load(args.views) if args.views else candidate_views(load(args.candidates))
    result = audit(load(args.descriptors), views)
    result['input_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (args.descriptors, args.views or args.candidates)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'source_count', 'invalid_guid_repr_count', 'unique_guid_repr_count')}))


if __name__ == '__main__':
    main()
