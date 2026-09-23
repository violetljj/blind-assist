"""Conservative pre-render native-clearance broadphase (world coordinates, metres).

Input: native_entities [{id, bounds_min_m, bounds_max_m, bounds_conservative,
 deformation_bounded}], clips [{id, trajectory_model: 'piecewise_linear_fixed_orientation',
 poses: [{x,y,z,pitch,yaw,roll}, ...]}], native_coverage_complete: bool.
Bounds must include all native surfaces and their motion/deformation over all clips.
This receipt never admits a layout for collection; other engineering gates still apply.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
from cnh_route_capture import basis

MARGIN_M = .15
CLIP_KINDS = {'centre', 'boundary', 'outside', 'removed'}


def _vector(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError('Expected a three-dimensional vector')
    out = [float(v) for v in value]
    if not all(math.isfinite(v) for v in out):
        raise ValueError('Coordinates must be finite')
    return out


def swept_query_bounds(start, end):
    """Six conservative swept AABBs. Translation is linear; orientation is fixed."""
    for pose in (start, end):
        _vector([pose[k] for k in ('x', 'y', 'z')])
        _vector([pose.get(k, 0.) for k in ('pitch', 'yaw', 'roll')])
    if any(start.get(k, 0.) != end.get(k, 0.) for k in ('pitch', 'yaw', 'roll')):
        raise ValueError('Changing orientation has no conservative endpoint-only sweep')
    forward, right, up = basis(start)
    result = []
    for center in (-.3, 0., .3):
        for region, yrange in (('HEAD', (-.2, .42)), ('BODY', (.42, .9))):
            points = []
            for pose in (start, end):
                origin = [pose[k] for k in ('x', 'y', 'z')]
                for x, y, z in itertools.product((center-.3, center+.3), yrange, (.3, 3.)):
                    points.append([origin[k]+x*right[k]-y*up[k]+z*forward[k] for k in range(3)])
            result.append(dict(region=region, center_x_m=center,
                               min_m=[min(p[k] for p in points)-MARGIN_M for k in range(3)],
                               max_m=[max(p[k] for p in points)+MARGIN_M for k in range(3)]))
    return result


def evaluate(manifest):
    """Fail closed on incomplete inputs; AABB overlap rejects, never proves collision."""
    entities = manifest.get('native_entities', [])
    clips = manifest.get('clips', [])
    reasons, conflicts, boxes = [], [], []
    if manifest.get('native_coverage_complete') is not True:
        reasons.append('NATIVE_COVERAGE_NOT_AUTHORITATIVE')
    if not entities:
        reasons.append('EMPTY_NATIVE_INVENTORY')
    ids = [e['id'] for e in entities]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate native entity IDs')
    validated = []
    for entity in entities:
        if entity.get('kind', 'native') != 'native':
            raise ValueError('Inventory must contain only native entities')
        low, high = _vector(entity['bounds_min_m']), _vector(entity['bounds_max_m'])
        if any(a > b for a, b in zip(low, high)):
            raise ValueError('Inverted native bounds')
        for flag in ('bounds_conservative', 'deformation_bounded'):
            if entity.get(flag) is not True:
                reasons.append(f'{entity["id"]}:{flag.upper()}_NOT_ESTABLISHED')
        validated.append((entity['id'], low, high))
    if len(clips) != 4 or {c['id'] for c in clips} != CLIP_KINDS:
        reasons.append('FOUR_COMPLETE_CLIPS_REQUIRED')
    for clip in clips:
        poses = clip.get('poses', [])
        if clip.get('trajectory_model') != 'piecewise_linear_fixed_orientation' or len(poses) < 2:
            reasons.append(f'{clip["id"]}:TRAJECTORY_COVERAGE_NOT_ESTABLISHED')
            continue
        for segment, (start, end) in enumerate(zip(poses, poses[1:])):
            try:
                swept = swept_query_bounds(start, end)
            except ValueError as exc:
                reasons.append(f'{clip["id"]}:SEGMENT_{segment}:{exc}')
                continue
            for box in swept:
                boxes.append(dict(clip_id=clip['id'], segment=segment, **box))
                for identifier, low, high in validated:
                    # Touching the expanded closed volume is conservatively rejected.
                    if all(high[k] >= box['min_m'][k] and low[k] <= box['max_m'][k] for k in range(3)):
                        conflicts.append(dict(entity_id=identifier, clip_id=clip['id'], segment=segment,
                                              region=box['region'], center_x_m=box['center_x_m']))
    return dict(schema='cnh_native_clearance_v1', margin_m=MARGIN_M,
                status='FAIL' if conflicts else ('NOT_RUN' if reasons else 'PASS'),
                scope='CONSERVATIVE_NATIVE_CLEARANCE_PREFILTER_ONLY', runtime_admission=False,
                geometry_status='CONFLICT' if conflicts else ('POSSIBLE_CLEAR' if boxes else 'NOT_RUN'),
                broadphase_clear=bool(boxes) and not conflicts,
                input_coverage_authoritative=not reasons,
                selected=bool(boxes) and not conflicts and not reasons,
                native_entity_count=len(entities), clip_count=len(clips),
                reasons=reasons, conflicts=conflicts, swept_query_aabbs=boxes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(json.loads(args.manifest.read_text(encoding='utf-8-sig')))
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
