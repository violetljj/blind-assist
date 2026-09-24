"""Adjudicate the bounded fresh City1 Development acquisition without test access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


LAYOUTS = ('city1-fresh-development-00', 'city1-fresh-development-01')
CLIPS = ('centre', 'boundary', 'outside', 'removed')
SITE = 'city-consumed-engineering-site-1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def poses(spec):
    return {
        tuple(p[k] for k in ('x', 'y', 'z', 'pitch', 'yaw', 'roll'))
        for layout in spec['layouts'] for clip in layout['clips'] for p in clip['poses']
    }


def rgb_metrics(path):
    with Image.open(path) as image:
        if image.size != (640, 360):
            raise ValueError(f'RGB dimensions differ: {path}')
        pixels = image.convert('RGB').load()
        def levels(x0, x1, y0, y1):
            values = []
            for y in range(int(360*y0), int(360*y1), 4):
                for x in range(int(640*x0), int(640*x1), 4):
                    r, g, b = pixels[x, y]
                    values.append((.2126*r + .7152*g + .0722*b)/255.)
            return sorted(values)
        centre = levels(.15, .85, .30, .85)
        near = levels(.421875, .671875, .50, .9583333333)
    q = lambda values, fraction: values[int(fraction*(len(values)-1))]
    p10, p60, p90, p99 = (q(centre, fraction) for fraction in (.10, .60, .90, .99))
    n30, n60 = (q(near, fraction) for fraction in (.30, .60))
    return dict(roi_luma_p10=p10, roi_luma_p60=p60, roi_luma_p90=p90,
                roi_luma_p99=p99, roi_contrast_p90_minus_p10=p90-p10,
                nearfield_luma_p30=n30, nearfield_luma_p60=n60,
                passed=.12 <= p60 <= .85 and p90-p10 >= .08 and p99 <= .98
                       and n30 >= .10 and n60 >= .14)


def gate(capture, prior_spec, geometry):
    capture = Path(capture).resolve(strict=True)
    spec = read(capture/'source/spec.json')
    prior = read(prior_spec)
    fmt = read(capture/'format-receipt.json')
    raw = read(capture/'raw-manifest.json')
    selection = read(capture/'candidate-selection.json')
    clearance = read(capture/'clearance.json')
    intervention = read(capture/'native-geometry-intervention.json')
    source_receipt = read(capture/'source-receipt.json')
    source_integrity = read(capture/'source-integrity.json')
    map_integrity = read(capture/'map-integrity.json')
    process_release = read(capture/'process-release.json')
    geometry_result = read(geometry)
    control = spec.get('city_derived_control', {})
    rows = raw['frames']
    formatted = {row['id']: row for row in fmt['frames']}
    expected = {(layout, clip, pose) for layout in LAYOUTS for clip in CLIPS for pose in (0, 1)}
    observed = {(row['layout_id'], row['clip_id'], row['pose_index']) for row in rows}
    source_checks = dict(
        city1_only_map=spec.get('map_asset') == '/Game/Map/Small_City_LVL',
        development_only=spec.get('data_role') == fmt.get('data_role') == 'Development'
            and spec.get('benchmark_eligible') is False and fmt.get('benchmark_eligible') is False,
        same_site_explicit=[row['layout_id'] for row in spec['layouts']] == list(LAYOUTS)
            and all(row.get('physical_site_id') == SITE for row in spec['layouts'])
            and control.get('physical_site_id') == SITE
            and control.get('independent_site_count') == 1,
        prior_source_bound=control.get('source_spec_sha256') == digest(prior_spec)
            and prior.get('layouts', [{}])[-1].get('layout_id') == 'city-engineering-1',
        old_poses_excluded=not (poses(spec) & poses(prior)),
        fixed_candidate_selection=selection.get('selected_indices') == [0, 0]
            and selection.get('candidate_counts') == [1, 1],
        all_frames_exact=len(rows) == len(fmt.get('frames', [])) == 16 and observed == expected,
        transport_pass=fmt.get('status') == 'PASS_SEVEN_PASS_SOURCE_TRANSPORT_ONLY',
        fresh_loaded_world_screen=clearance.get('loaded_world_coverage_complete') is True
            and all(row.get('status') == 'PASS_LOADED_WORLD_ONLY' and not row.get('required_missing')
                    for row in clearance.get('candidates', []))
            and len(clearance.get('candidates', [])) == 2,
        vehicle_mask_applied=intervention.get('policy') == 'CITY_NEARFIELD_VEHICLE_ZERO_SCALE'
            and intervention.get('source_authority') == 'FRESH_CITY1_SAME_SITE_DEVELOPMENT'
            and intervention.get('near_vehicles_removed') is True
            and intervention.get('instance_removal_method') == 'ZERO_SCALE_KEEP_INDEX_AND_CUSTOM_DATA'
            and intervention.get('saved') is False,
        vehicle_instance_identity_preserved=len(intervention.get('components', [])) > 0
            and all(row.get('metadata_unchanged') is True
                    and row.get('far_transforms_unchanged') is True
                    for row in intervention.get('components', [])),
        map_unchanged=map_integrity.get('map_unchanged') is True,
        map_hash_bound=source_receipt.get('map_sha256_before') == spec.get('map_sha256')
            == map_integrity.get('map_sha256_after'),
        source_project_unchanged=source_integrity.get('project_unchanged') is True,
        ue_process_released=process_release.get('released') is True
            and not process_release.get('survivors'),
        fixed_ev100=spec.get('exposure_ev100') == 12.2
            and all(row.get('rgb_exposure', {}).get('initial_ev100') == 12.2
                    for row in fmt.get('frames', [])),
    )
    frame_metrics = []
    for row in rows:
        folder = (capture/row['folder']).resolve(strict=True)
        if not folder.is_relative_to(capture):
            raise ValueError('Frame path escapes capture')
        for side in ('left', 'right'):
            path = folder/f'{side}.png'
            item = rgb_metrics(path)
            item.update(frame_id=row['id'], layout_id=row['layout_id'], side=side,
                        image_sha256=digest(path),
                        transport_hash_matches=digest(path) == formatted[row['id']]['hashes'][f'{side}.png'])
            frame_metrics.append(item)
    rgb_pass = len(frame_metrics) == 32 and all(
        item['passed'] and item['transport_hash_matches'] for item in frame_metrics)
    g_layouts = geometry_result.get('layouts', [])
    geometry_pass = (geometry_result.get('status') == 'PASS'
                     and [row['layout_id'] for row in g_layouts] == list(LAYOUTS)
                     and all(row.get('tested_gates_status') == 'PASS' for row in g_layouts)
                     and geometry_result.get('capture_manifest_sha256') == digest(capture/'raw-manifest.json'))
    return dict(
        status='PASS_FRESH_CITY1_DEVELOPMENT_NUMERIC' if all(source_checks.values()) and rgb_pass and geometry_pass
            else 'FAIL_FRESH_CITY1_DEVELOPMENT_NUMERIC',
        data_role='Development', benchmark_eligible=False, physical_site_id=SITE,
        independent_physical_site_count=1, new_pose_layout_count=2, frame_count=16,
        source_gate=dict(status='PASS' if all(source_checks.values()) else 'FAIL', checks=source_checks,
                         scope='SAME_EXISTING_CITY1_PHYSICAL_SITE; loaded-world clearance only; no independent site or benchmark admission'),
        geometry_gate=dict(status='PASS' if geometry_pass else 'FAIL',
                           result_path=str(Path(geometry).resolve()),
                           result_sha256=digest(geometry), layouts=g_layouts),
        rgb_gate=dict(status='NUMERIC_PASS_VISUAL_REVIEW_REQUIRED' if rgb_pass else 'FAIL',
                      policy='EXISTING_CNH_ROUTE_SOURCE_LAUNCH_V1_ROI_EACH_EYE',
                      thresholds='centre p60 0.12..0.85, p90-p10 >=0.08, p99 <=0.98; near p30 >=0.10, p60 >=0.14',
                      checked_image_count=len(frame_metrics), failed_image_count=sum(not row['passed'] for row in frame_metrics),
                      visual_review_status='PENDING', frames=frame_metrics),
        capture=str(capture), prior_source_spec_sha256=digest(prior_spec),
        capture_spec_sha256=digest(capture/'source/spec.json'),
        limitation='Two new spatial pose layouts translated from prior City1; same physical site and same inserted asset families. No formal test, independent-site evidence, energy convergence, or label precision gate.')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--prior-spec', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = gate(args.capture, args.prior_spec, args.geometry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], source=result['source_gate']['status'],
                          geometry=result['geometry_gate']['status'], rgb=result['rgb_gate']['status'],
                          failed_images=result['rgb_gate']['failed_image_count'])))


if __name__ == '__main__':
    main()
