"""Prepare fixed-grid Development reference on original and extra cached sources.

No predictions are read and no calibration is selected. UNKNOWN is not FREE.
Preserve train/cal roles: training references are exposed only as a separately
marked diagnostic pool, never silently promoted to held-out calibration.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import hashlib
import io
import re
import shutil
import time
import zipfile

import cv2
import numpy as np
from PIL import Image

from rgb_body_query_fixed_grid import EDGES, PUBLIC, independent_xyz_labels, queries
from rgb_body_query_3rscan import color_coordinates, optical_z, parse_info, sensor_labels
from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_scene_diagnostic import write_csv


FREE = 'FREE_ON_SAMPLED_RAYS'
SOURCE = 'rgb-body-query-interval-distribution-dev-20261010/train-cal-sensor'
EXPECTED = {'train': 56, 'cal': 16}


def select_additional(repo):
    """Metadata/cache/frame-name selection only, before depth bytes or labels."""
    root = repo/'artifacts.local/datasets/3rscan'
    exclusions = [repo/'artifacts.local/work'/relative/'dataset_manifest.json' for relative in (
        'rgb-body-query-cross-session-dev-20261009/sensor',
        'rgb-body-query-transfer-dev-20261009/sensor')]
    used_ids = {value for path in exclusions for row in load(path)['rows']
                for value in (row['environment'], row['scan'])}
    groups, excluded = [], []
    for group in load(root/'3RScan.json'):
        if group['type'] != 'train':
            continue
        ids = [group['reference']] + [r['reference'] for r in group['scans']]
        cached = sorted(s for s in ids if (root/s/'sequence.zip').is_file())
        if not cached:
            continue
        if used_ids.intersection(ids):
            excluded.append(dict(environment=group['reference'], group_scan_ids=ids))
            continue
        scan = group['reference'] if group['reference'] in cached else cached[0]
        archive_path = root/scan/'sequence.zip'
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
        paired = sorted(int(match.group(1)) for name in names if
            (match := re.fullmatch(r'frame-(\d{6})\.color\.jpg', name)) and
            f'frame-{int(match.group(1)):06d}.depth.pgm' in names)
        indices = np.linspace(0, len(paired)-1, 16).round().astype(int)
        if len(set(indices)) != 16:
            raise ValueError('Need 16 unique paired frames; selected environment is not replaced')
        groups.append(dict(environment=group['reference'], scan=scan, official_split='train',
            split='additional_cal', group_sha256=hashlib.sha256(group['reference'].encode()).hexdigest(),
            group_scan_ids=ids, cached_scan_ids=cached, frames=[paired[i] for i in indices],
            paired_frame_count=len(paired), source_zip=str(archive_path),
            archive_size_bytes=archive_path.stat().st_size, archive_mtime_ns=archive_path.stat().st_mtime_ns))
    groups.sort(key=lambda row: row['group_sha256'])
    if len(groups) != 15 or len(excluded) != 20:
        raise ValueError('Expected cached35 = excluded20 + additional15; inventory changed')
    return dict(groups=groups, excluded_groups=excluded, metadata_sha256=sha(root/'3RScan.json'),
        exclusions=[dict(path=str(path), sha256=sha(path)) for path in exclusions],
        method='All unused cached official-train environments, whole-group exclusion incl rescans; reference-preferred scan; 16 uniform paired frames; all fixed27 boxes')


def additional_rows(selection, output, fixed, check_budget):
    """Extract existing official-train cache only; no model/prediction access."""
    rows = []
    dest = output/'additional-sensor'
    dest.mkdir(exist_ok=True)
    for group in selection['groups']:
        archive_path = Path(group['source_zip'])
        if (archive_path.stat().st_size, archive_path.stat().st_mtime_ns) != (
                group['archive_size_bytes'], group['archive_mtime_ns']):
            raise ValueError('Archive changed after metadata/frame selection')
        with zipfile.ZipFile(archive_path) as archive:
            info_bytes = archive.read('_info.txt')
            info = parse_info(info_bytes)
            mx, my = color_coordinates(info['depth_K'], info['color_K'], info['depth_shape'])
            ch, cw = info['color_shape']
            observed = (mx >= 0) & (mx <= cw-1) & (my >= 0) & (my <= ch-1)
            for frame in group['frames']:
                check_budget()
                rgb_bytes = archive.read(f'frame-{frame:06d}.color.jpg')
                depth_bytes = archive.read(f'frame-{frame:06d}.depth.pgm')
                rgb = Image.open(io.BytesIO(rgb_bytes))
                raw = cv2.imdecode(np.frombuffer(depth_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                if (rgb.height, rgb.width) != info['color_shape'] or raw.shape != info['depth_shape'] or raw.dtype != np.uint16:
                    raise ValueError('Source native image/depth/calibration dimensions disagree')
                depth = optical_z(raw, info['shift'])
                labels, states = [], []
                for query in fixed:
                    label, state = sensor_labels(depth, info['depth_K'], query, observed)
                    domain = label != 255
                    state.update(observed_reachable_rays=int((domain & observed).sum()))
                    labels.append(label); states.append(state)
                stem = f'{group["scan"]}_{frame:06d}'
                rgb_path, ref_path = dest/f'{stem}.jpg', dest/f'{stem}.npz'
                rgb_path.write_bytes(rgb_bytes)
                np.savez_compressed(ref_path, labels=np.stack(labels), depth=depth,
                    depth_K=info['depth_K'], color_K=info['color_K'], map_x=mx, map_y=my, observed=observed)
                rows.append(dict(environment=group['environment'], scan=group['scan'],
                    split='additional_cal', official_split='train', group_sha256=group['group_sha256'],
                    frame=frame, rgb_path=str(rgb_path), rgb_sha256=sha(rgb_path),
                    reference_path=str(ref_path), reference_sha256=sha(ref_path), source_zip=str(archive_path),
                    source_rgb_sha256=hashlib.sha256(rgb_bytes).hexdigest(),
                    source_depth_sha256=hashlib.sha256(depth_bytes).hexdigest(),
                    calibration_sha256=hashlib.sha256(info_bytes).hexdigest(),
                    color_K=info['color_K'].tolist(), depth_K=info['depth_K'].tolist(),
                    color_shape=list(info['color_shape']), depth_shape=list(info['depth_shape']),
                    depth_shift=info['shift'], queries=states))
            write(output/'source_progress.json', dict(completed_frames=len(rows),
                completed_environment=group['environment']))
    write(dest/'dataset_manifest.json', dict(rows=rows, queries=fixed, groups=selection['groups'],
        role='Additional official-train Development calibration reference, no old train/eval environment overlap',
        selection_sha256=sha(output/'plan.json')))
    return rows


def load(path):
    return json.loads(Path(path).read_text('utf-8-sig'))


def coverage(records):
    counts = Counter(r['reference_state'] for r in records)
    free = [r for r in records if r['reference_state'] == FREE]
    return dict(query_records=len(records),
                frames=len({(r['scan'], r['frame']) for r in records}),
                environments=len({r['environment'] for r in records}),
                positive_queries=counts['POSITIVE'], unknown_queries=counts['UNKNOWN'],
                strict_sampled_free_queries=len(free),
                strict_sampled_free_frames=len({(r['scan'], r['frame']) for r in free}),
                strict_sampled_free_environments=len({r['environment'] for r in free}),
                reachable_query_ray_units=sum(r['domain_pixels'] for r in records),
                positive_query_ray_units=sum(r['positive_pixels'] for r in records),
                free_query_ray_units=sum(r['free_ray_pixels'] for r in records),
                unknown_query_ray_units=sum(r['unknown_pixels'] for r in records),
                observed_reachable_query_ray_units=sum(r['observed_reachable_rays'] for r in records),
                zero_reachable_queries=sum(r['domain_pixels'] == 0 for r in records))


def run(repo, output, budget_s):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'plan.json').exists():
        raise FileExistsError('Preserve prior negative reference run')
    source = repo/'artifacts.local/work'/SOURCE
    manifest_path = source/'dataset_manifest.json'
    fixed = queries()
    selection = select_additional(repo)
    expected = dict(EXPECTED, additional_cal=240)
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                goal='Prepare original train56/cal16 and new cached official-train environment fixed27 reference pool',
                source_manifest_path=str(manifest_path), source_manifest_sha256=sha(manifest_path),
                expected_frames_by_role=expected, queries=fixed, axis_edges=EDGES,
                selection=selection,
                role_policy='Keep original train/cal identities; train pool is diagnostic, eval never becomes cal; new disjoint official-train groups are additional_cal Development',
                negative_policy='>=16 FREE rays and zero POSITIVE/UNKNOWN rays; missing/before-box/zero-domain remain UNKNOWN',
                budget_cpu_wall_s=budget_s, gpu_s=0, downloads=0, new_training=False,
                new_inference=False, source_sha256=sha(__file__),
                limits='Sampled first-return FREE is not volume clearance; frames/queries/rays correlated; no fresh confirmation')
    write(output/'plan.json', plan)
    complete, failure = False, None
    checks = Counter()
    records, rows, identities = [], [], []
    try:
        def check_budget():
            if time.perf_counter()-started >= budget_s:
                raise TimeoutError('Negative reference CPU wall budget reached')
        manifest = load(manifest_path)
        if manifest['queries'] != fixed:
            raise ValueError('Fixed grid identity changed')
        source_rows = manifest['rows'].copy()
        if dict(Counter(r['split'] for r in source_rows)) != EXPECTED:
            raise ValueError('Train/cal frame identities changed')
        groups = {role: {r['environment'] for r in source_rows if r['split'] == role}
                  for role in EXPECTED}
        if len(groups['train']) != 7 or len(groups['cal']) != 2 or groups['train'] & groups['cal']:
            raise ValueError('Train/cal environment roles changed')
        unique = {(r['scan'], r['frame']) for r in source_rows}
        if len(unique) != sum(EXPECTED.values()):
            raise ValueError('Duplicated frame identity')
        source_rows.extend(additional_rows(selection, output, fixed, check_budget))
        if dict(Counter(r['split'] for r in source_rows)) != expected:
            raise ValueError('Expanded source count mismatch')
        groups['additional_cal'] = {r['environment'] for r in source_rows if r['split'] == 'additional_cal'}
        (output/'references').mkdir(exist_ok=True)
        for row in source_rows:
            if time.perf_counter()-started >= budget_s:
                raise TimeoutError('Negative reference CPU wall budget reached')
            ref_path = Path(row['reference_path'])
            if sha(ref_path) != row['reference_sha256'] or sha(row['rgb_path']) != row['rgb_sha256']:
                raise ValueError('Cached reference/RGB identity changed')
            with np.load(ref_path, allow_pickle=False) as f:
                labels = f['labels'].copy()
                depth, observed, k = f['depth'], f['observed'], f['depth_K']
                if list(depth.shape) != row['depth_shape'] or not np.array_equal(k, row['depth_K']):
                    raise ValueError('Native reference geometry identity changed')
                if not np.array_equal(f['color_K'], row['color_K']):
                    raise ValueError('Color intrinsic identity changed')
                if labels.shape != (27, *depth.shape):
                    raise ValueError('Label array is not fixed27/native shape')
                for index, query in enumerate(fixed):
                    independent, domain = independent_xyz_labels(depth, k, query, observed)
                    label = labels[index]
                    if not np.array_equal(independent, label):
                        raise ValueError('Independent XYZ labels differ')
                    count = {key: int((label == value).sum()) for key, value in
                             [('positive_pixels', 1), ('free_ray_pixels', 0), ('unknown_pixels', 2)]}
                    state = ('POSITIVE' if count['positive_pixels'] >= 16 else
                             FREE if count['positive_pixels'] == 0 and count['unknown_pixels'] == 0
                             and count['free_ray_pixels'] >= 16 else 'UNKNOWN')
                    inherited = row['queries'][index]
                    reach = int(domain.sum())
                    if state != inherited['state'] or any(count[key] != inherited[key] for key in count):
                        raise ValueError('Saved state/count disagrees with strict contract')
                    if reach != inherited['domain_pixels'] or sum(count.values()) != reach:
                        raise ValueError('Reachable denominator mismatch')
                    observed_count = int((domain & observed).sum())
                    if observed_count != inherited['observed_reachable_rays']:
                        raise ValueError('Observed denominator mismatch')
                    known = count['positive_pixels'] + count['free_ray_pixels']
                    records.append(dict(split=row['split'], environment=row['environment'],
                        scan=row['scan'], frame=row['frame'], query=query['name'], query_index=index,
                        distance_band=f'{query["low"][2]:g}-{query["high"][2]:g}m',
                        reference_state=state, negative_eligible=state == FREE,
                        calibration_eligible=row['split'] in ('cal', 'additional_cal') and state == FREE,
                        train_diagnostic_only=row['split'] == 'train', domain_pixels=reach,
                        **count, observed_reachable_rays=observed_count,
                        known_reference_fraction=known/reach if reach else None,
                        frame_native_pixels=label.size, reference_path=str(output/'references'/ref_path.name),
                        reference_sha256=row['reference_sha256']))
                    checks['independent_xyz_queries'] += 1
                    checks['strict_state_count_checks'] += 1
                checks['native_geometry_frames'] += 1
            dest = output/'references'/ref_path.name
            shutil.copyfile(ref_path, dest)
            if sha(dest) != row['reference_sha256']:
                raise ValueError('Frozen copy differs')
            new = dict(row, reference_path=str(dest), source_reference_path=str(ref_path),
                       source_reference_sha256=row['reference_sha256'])
            rows.append(new)
            identities.append(dict(split=row['split'], environment=row['environment'],
                scan=row['scan'], frame=row['frame'], rgb_sha256=row['rgb_sha256'],
                source_reference_path=str(ref_path), source_reference_sha256=row['reference_sha256'],
                copied_reference_path=str(dest), copied_reference_sha256=sha(dest)))
        by_role = {role: coverage([r for r in records if r['split'] == role]) for role in expected}
        by_group = []
        for role in expected:
            for env in sorted(groups[role]):
                rr = [r for r in records if r['split'] == role and r['environment'] == env]
                by_group.append(dict(split=role, environment=env, **coverage(rr)))
        by_band = [dict(split=role, distance_band=band, **coverage([
            r for r in records if r['split'] == role and r['distance_band'] == band]))
            for role in expected for band in ('0.3-0.8m', '0.8-1.5m', '1.5-3m')]
        frozen_manifest = dict(manifest, status='FROZEN_EXISTING_REFERENCE_POOL_COMPLETE', rows=rows,
            frames=len(rows), groups=manifest['groups'] + selection['groups'],
            state_counts_by_split={role: dict(Counter(r['reference_state'] for r in records
                if r['split'] == role)) for role in expected},
            official_data_role='Original train56 diagnostic/cal16 plus 15 additional official-train Development calibration environments; no old eval reassigned',
            source_manifest_path=str(manifest_path), source_manifest_sha256=sha(manifest_path),
            plan_sha256=sha(output/'plan.json'), source_sha256=sha(__file__),
            preparation_policy=plan['selection'], role_policy=plan['role_policy'])
        write(output/'dataset_manifest.json', frozen_manifest)
        # Observation artifact remains evaluator-field-free. It is not a model run.
        write(output/'observations.json', dict(rows=[{key: row[key] for key in PUBLIC} for row in rows],
            queries=fixed, dataset_manifest_sha256=sha(output/'dataset_manifest.json')))
        write_csv(output/'query_reference_counts.csv', records)
        write_csv(output/'environment_coverage.csv', by_group)
        write_csv(output/'distance_band_coverage.csv', by_band)
        write(output/'negative_query_index.json', dict(
            role_policy=plan['role_policy'], queries=[r for r in records if r['negative_eligible']],
            calibration_queries=[r for r in records if r['calibration_eligible']],
            selection='Index strict labels after preparing ALL queries; predictions are never read'))
        result = dict(status='REFERENCE_PREPARATION_COMPLETE',
            total=coverage(records), by_role=by_role, by_environment=by_group, by_band=by_band,
            actual_new_negative_queries=by_role['additional_cal']['strict_sampled_free_queries'],
            new_frames=240, new_environments=15,
            negative_pool_note='Original train is diagnostic only; original cal and additional_cal eligible but remain different source pools',
            next_decision='Reference ready; no predictions scored or calibration thresholds selected. Assess actual environment/distance coverage before calibration.',
            limitations=plan['limits'], sources=identities, checks=dict(checks))
        if by_role['cal']['strict_sampled_free_queries'] != 2:
            raise ValueError('Original cal16 strict FREE count changed')
        write(output/'coverage.json', result)
        # Verify serialized identity/roles/negative index rather than only in-memory output.
        saved = load(output/'negative_query_index.json')
        if len(saved['queries']) != 4 + result['actual_new_negative_queries'] or len(saved['calibration_queries']) != 2 + result['actual_new_negative_queries']:
            raise ValueError('Saved strict negative index mismatch')
        if any(r['split'] not in ('cal', 'additional_cal') or r['reference_state'] != FREE for r in saved['calibration_queries']):
            raise ValueError('Saved negative role leakage')
        checks['saved_negative_rows_rechecked'] = len(saved['queries'])
        complete = True
        print(json.dumps(dict(status=result['status'], by_role=by_role, by_band=by_band)), flush=True)
    except Exception as error:
        failure = repr(error)
        write(output/'failure.json', dict(error=failure, completed_frames=len(rows), query_records=len(records)))
        raise
    finally:
        shutil.copyfile(__file__, output/'executed_negative_reference_prepare.py')
        write(output/'terminal.json', dict(status='COMPLETE' if complete else 'INCOMPLETE',
            failure=failure, elapsed_cpu_wall_s=time.perf_counter()-started, budget_cpu_wall_s=budget_s,
            gpu_s=0, downloads=0, inference_calls=0, new_training=False, checks=dict(checks),
            plan_sha256=sha(output/'plan.json'), source_sha256=sha(__file__),
            resources='No workers/services/processes retained; payload owner this run'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-s', type=float, default=600.)
    args = parser.parse_args()
    run(args.repo.resolve(), args.output.resolve(), args.budget_s)
