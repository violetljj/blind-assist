"""Freeze two fresh same-site City1 Development candidates without rendering.

Old loaded-world bounds are used only to reject obvious collisions. Fresh UE
probe/coverage, render-depth parity and RGB checks remain authoritative.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from cnh_route_native_clearance import evaluate
from cnh_route_source_compare_adapter import validate_spec


SHIFTS_M = (2.0, 2.5)
SITE = 'city-consumed-engineering-site-1'
AUTHORITY = 'FRESH_CITY1_SAME_SITE_DEVELOPMENT'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source_spec, old_clearance):
    source_spec, old_clearance = Path(source_spec), Path(old_clearance)
    source = json.loads(source_spec.read_text(encoding='utf-8-sig'))
    clearance = json.loads(old_clearance.read_text(encoding='utf-8-sig'))
    if (source.get('native_geometry_policy') != 'CITY_NEARFIELD_VEHICLE_ZERO_SCALE'
            or source.get('map_asset') != '/Game/Map/Small_City_LVL'
            or len(source['layouts']) != 2 or source['layouts'][1]['layout_id'] != 'city-engineering-1'
            or clearance.get('loaded_world_coverage_complete') is not True
            or len(clearance.get('candidates', [])) != 2):
        raise ValueError('Expected the prior consumed City1 vehicle-mask source and inventory')
    parent = source['layouts'][1]
    old_poses = {(p['x'], p['y'], p['z'], p['pitch'], p['yaw'], p['roll'])
                 for row in source['layouts'] for clip in row['clips'] for p in clip['poses']}
    layouts, screens = [], []
    for index, shift in enumerate(SHIFTS_M):
        layout = copy.deepcopy(parent)
        layout['layout_id'] = f'city1-fresh-development-{index:02d}'
        layout['physical_site_id'] = SITE
        layout['camera']['x'] += shift
        for clip in layout['clips']:
            for pose in clip['poses']:
                pose['x'] += shift
                key = tuple(pose[k] for k in ('x','y','z','pitch','yaw','roll'))
                if key in old_poses:
                    raise ValueError('Fresh pose repeats an old consumed pose')
            for obj in clip['insertions']:
                obj['center_m'][0] += shift
        manifest = copy.deepcopy(clearance['candidates'][1]['manifest'])
        manifest['clips'] = layout['clips']
        screen = evaluate(manifest)
        screens.append(dict(layout_id=layout['layout_id'], shift_x_m=shift,
                            prior_loaded_world_screen_status=screen['status'],
                            prior_loaded_world_conflicts=len(screen['conflicts']),
                            authority='REJECTION_PREFILTER_ONLY_NOT_FRESH_UE_COVERAGE'))
        if screen['status'] != 'PASS':
            raise ValueError('Old loaded-world broadphase rejects fresh City1 pose: '+str(screens[-1]))
        layouts.append(layout)
    result = copy.deepcopy(source)
    result['layouts'] = layouts
    result['data_role'] = 'Development'
    if result.get('exposure_ev100') != 13.2:
        raise ValueError('Expected consumed City1 13.2 EV100 exposure source')
    result['exposure_ev100'] = 12.2
    result['city_derived_control'] = dict(
        authority=AUTHORITY, benchmark_eligible=False, new_layouts=True,
        physical_site_id=SITE, independent_site_count=1, radius_m=8.,
        source_spec_sha256=sha(source_spec), old_clearance_sha256=sha(old_clearance),
        source_selection='CITY1_ONLY_TRANSLATE_X_2_AND_2_5_METRES_NO_OUTCOME_SELECTION',
        geometry_thresholds='UNCHANGED_V1', required_global_render_toggles=False,
        rgb_exposure_decision='PRESELECT_12_2_EV_FROM_OLD_CITY1_DEVELOPMENT_P60_0_092_AT_13_2_EV',
        prior_loaded_world_screens=screens)
    # The adapter is required to accept this explicit same-site Development case.
    validate_spec(result)
    return result


def main():
    import argparse
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--source-spec', required=True, type=Path)
    p.add_argument('--old-clearance', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    spec = prepare(args.source_spec, args.old_clearance)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(layout_ids=[x['layout_id'] for x in spec['layouts']],
                          physical_site_id=SITE, output=str(args.output))))


if __name__ == '__main__':
    main()
