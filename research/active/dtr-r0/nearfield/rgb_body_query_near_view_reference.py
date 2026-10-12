"""Fixed unconsumed rescan sessions in 15 Development environments.

All cached non-consumed scans and eight uniform paired frames per scan are
frozen before reading depth. Sessions preserve environment identity; physical
viewpoint change is not verified. The stopped same-scan expansion stays stopped.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import re
import shutil
import time
import zipfile

import numpy as np

from rgb_body_query_fixed_grid import EDGES, PUBLIC, independent_xyz_labels, queries
from rgb_body_query_negative_reference_prepare import (
    FREE, additional_rows, coverage, load, select_additional)
from rgb_body_query_input_diagnostic import sha, write
from rgb_body_query_scene_diagnostic import write_csv

BANDS = ('0.3-0.8m', '0.8-1.5m', '1.5-3m')
PRIOR = 'rgb-body-query-negative-reference-dev-20261010'


def run(repo, output, budget_s):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'plan.json').exists():
        raise FileExistsError('Preserve previous viewpoint run; no implicit restart')
    complete, failure, checks = False, None, Counter()
    rows, records = [], []
    fixed = queries()
    prior = repo/'artifacts.local/work'/PRIOR
    old_selection = select_additional(repo)
    prior_manifest = load(prior/'dataset_manifest.json')
    consumed_scans = {r['scan'] for r in prior_manifest['rows']}
    for exclusion in old_selection['exclusions']:
        consumed_scans.update(r['scan'] for r in load(exclusion['path'])['rows'])
    excluded_ids = {v for g in old_selection['excluded_groups']
                    for v in g['group_scan_ids']}
    expected_envs = {g['environment'] for g in old_selection['groups']}
    groups = []
    for group in old_selection['groups']:
        if excluded_ids.intersection(group['group_scan_ids']):
            raise ValueError('Additional environment overlaps old train/cal/eval group')
        for scan in sorted(set(group['cached_scan_ids']) - consumed_scans):
            archive_path = repo/'artifacts.local/datasets/3rscan'/scan/'sequence.zip'
            with zipfile.ZipFile(archive_path) as archive:
                names = set(archive.namelist())
                paired = sorted(int(m.group(1)) for name in names if
                    (m := re.fullmatch(r'frame-(\d{6})\.color\.jpg', name)) and
                    f'frame-{int(m.group(1)):06d}.depth.pgm' in names)
                calibration_sha = hashlib.sha256(archive.read('_info.txt')).hexdigest()
            indices = np.linspace(0, len(paired)-1, 8).round().astype(int)
            if len(set(indices)) != 8:
                raise ValueError('Eight distinct uniform paired frames required; no replacement')
            groups.append(dict(group, scan=scan, frames=[paired[i] for i in indices],
                paired_frame_count=len(paired), source_zip=str(archive_path),
                archive_size_bytes=archive_path.stat().st_size,
                archive_mtime_ns=archive_path.stat().st_mtime_ns,
                calibration_sha256=calibration_sha, source_zip_sha256=sha(archive_path)))
    groups.sort(key=lambda g: (g['group_sha256'], g['scan']))
    if len(groups) != 16 or {g['environment'] for g in groups} != expected_envs:
        raise ValueError('Expected all 16 unseen rescans across the existing 15 environments')
    selection = dict(old_selection, groups=groups,
        method='All unconsumed cached rescans of the existing15 additional_cal environments; eight uniform paired native frames per scan; fixed27 boxes',
        consumed_scan_ids=sorted(consumed_scans), old_train_cal_eval_group_ids=sorted(excluded_ids))
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='ALL128_IDENTITIES_FROZEN_BEFORE_DEPTH_READS', selection=selection,
        goal='Test new viewpoints for cross-environment near strict sampled-FREE coverage',
        expected_frames=128, expected_scans=16, expected_environments=15,
        queries=fixed, axis_edges=EDGES, source_sha256=sha(__file__),
        prior_manifest_path=str(prior/'dataset_manifest.json'),
        prior_manifest_sha256=sha(prior/'dataset_manifest.json'),
        role_policy='Same15 additional_cal official train Development groups; all old20 environment groups excluded including rescans; no eval reassignment',
        strict_state_policy='FREE requires >=16 free rays and zero positive/unknown rays; zero domain, missing or occluded stays UNKNOWN',
        budget_cpu_wall_s=budget_s, inference_calls=0, gpu_s=0, downloads=0, new_training=False,
        stop='If near FREE remains in at most one environment, stop this viewpoint path without more same-view frames or inference',
        limitations='New rescan sessions do not create new environment identities or verify physical viewpoint change; native first-return rays and frames correlated; Development only, no whole-volume/body clearance')
    write(output/'plan.json', plan)
    print('PLAN_FROZEN: 16 rescans / 15 environments / 128 frames / 3456 queries', flush=True)
    try:
        def budget():
            if time.perf_counter()-started >= budget_s:
                raise TimeoutError('View-reference CPU wall budget reached; preserve partial run')
        rows = additional_rows(selection, output, fixed, budget)
        if len(rows) != 128 or len({(r['scan'], r['frame']) for r in rows}) != 128:
            raise ValueError('128-frame identity mismatch')
        for row in rows:
            budget()
            if sha(row['reference_path']) != row['reference_sha256'] or sha(row['rgb_path']) != row['rgb_sha256']:
                raise ValueError('Saved RGB/reference SHA mismatch')
            with np.load(row['reference_path'], allow_pickle=False) as data:
                labels, depth, observed, k = data['labels'], data['depth'], data['observed'], data['depth_K']
                if labels.shape != (27, *depth.shape) or list(depth.shape) != row['depth_shape']:
                    raise ValueError('Native fixed27 dimensions changed')
                if not np.array_equal(k, row['depth_K']) or not np.array_equal(data['color_K'], row['color_K']):
                    raise ValueError('Native K changed')
                for qi, query in enumerate(fixed):
                    budget()
                    independent, domain = independent_xyz_labels(depth, k, query, observed)
                    if not np.array_equal(independent, labels[qi]):
                        raise ValueError('Independent XYZ labels differ')
                    counts = {name: int((labels[qi] == value).sum()) for name, value in
                        [('positive_pixels', 1), ('free_ray_pixels', 0), ('unknown_pixels', 2)]}
                    state = ('POSITIVE' if counts['positive_pixels'] >= 16 else
                        FREE if counts['positive_pixels'] == 0 and counts['unknown_pixels'] == 0
                        and counts['free_ray_pixels'] >= 16 else 'UNKNOWN')
                    inherited = row['queries'][qi]
                    reach = int(domain.sum())
                    if state != inherited['state'] or any(counts[key] != inherited[key] for key in counts):
                        raise ValueError('Strict query state/count differs')
                    if reach != inherited['domain_pixels'] or sum(counts.values()) != reach:
                        raise ValueError('Query reachable denominator differs')
                    observed_count = int((domain & observed).sum())
                    if observed_count != inherited['observed_reachable_rays']:
                        raise ValueError('Observed ray denominator differs')
                    records.append(dict(split='additional_cal', official_split='train',
                        environment=row['environment'], scan=row['scan'], frame=row['frame'],
                        query=query['name'], query_index=qi,
                        distance_band=f'{query["low"][2]:g}-{query["high"][2]:g}m',
                        reference_state=state, negative_eligible=state == FREE,
                        calibration_eligible=state == FREE, domain_pixels=reach,
                        observed_reachable_rays=observed_count, **counts,
                        reference_path=row['reference_path'], reference_sha256=row['reference_sha256']))
                    checks['independent_xyz_queries'] += 1
                    checks['strict_state_count_checks'] += 1
                checks['native_identity_frames'] += 1
        envs = sorted(expected_envs)
        by_env = [dict(environment=env, **coverage([r for r in records if r['environment'] == env])) for env in envs]
        by_band = [dict(distance_band=band, **coverage([r for r in records if r['distance_band'] == band])) for band in BANDS]
        by_env_band = [dict(environment=env, distance_band=band,
            **coverage([r for r in records if r['environment'] == env and r['distance_band'] == band]))
            for env in envs for band in BANDS]
        manifest = dict(status='FIXED128_VIEW_REFERENCE_COMPLETE', rows=rows, frames=128,
            scans=16, environments=15, groups=groups, queries=fixed,
            plan_sha256=sha(output/'plan.json'), role_policy=plan['role_policy'])
        write(output/'dataset_manifest.json', manifest)
        write(output/'observations.json', dict(rows=[{k: r[k] for k in PUBLIC} for r in rows],
            queries=fixed, dataset_manifest_sha256=sha(output/'dataset_manifest.json')))
        write_csv(output/'query_reference_counts.csv', records)
        write_csv(output/'environment_coverage.csv', by_env)
        write_csv(output/'distance_band_coverage.csv', by_band)
        write_csv(output/'environment_distance_band_coverage.csv', by_env_band)
        negatives = [r for r in records if r['negative_eligible']]
        write(output/'negative_query_index.json', dict(queries=negatives, calibration_queries=negatives,
            role_policy=plan['role_policy'], selection='All frozen128 frames and all fixed27 queries retained'))
        nearenvs = sorted({r['environment'] for r in negatives if r['distance_band'] == BANDS[0]})
        result = dict(status='VIEW_REFERENCE_COMPLETE', total=coverage(records),
            by_environment=by_env, by_band=by_band, by_environment_band=by_env_band,
            near_free_environments=nearenvs, near_cross_environment_coverage_improved=len(nearenvs) > 1,
            next_decision=('Cross-environment near FREE found; eligible to inspect frozen inference costs' if len(nearenvs) > 1 else
                'Near FREE remains in at most one environment; stop viewpoint path without further same-view expansion or inference'),
            checks=dict(checks), limitations=plan['limitations'])
        write(output/'coverage.json', result)
        saved = load(output/'negative_query_index.json')
        obs = load(output/'observations.json')
        if len(saved['queries']) != len(negatives) or any(r['reference_state'] != FREE for r in saved['queries']):
            raise ValueError('Serialized negatives differ')
        if any(set(r) != set(PUBLIC) for r in obs['rows']):
            raise ValueError('Public observation evaluator-field leakage')
        checks['serialized_negative_rows'] = len(saved['queries'])
        checks['serialized_public_observation_rows'] = len(obs['rows'])
        complete = True
        print({b['distance_band']: (b['strict_sampled_free_queries'], b['strict_sampled_free_environments']) for b in by_band}, flush=True)
        print(result['next_decision'], flush=True)
    except Exception as error:
        failure = repr(error)
        write(output/'failure.json', dict(error=failure, completed_frames=len(rows), completed_queries=len(records)))
        raise
    finally:
        shutil.copyfile(__file__, output/'executed_near_view_reference.py')
        write(output/'terminal.json', dict(status='COMPLETE' if complete else 'INCOMPLETE', failure=failure,
            elapsed_cpu_wall_s=time.perf_counter()-started, budget_cpu_wall_s=budget_s,
            checks=dict(checks), gpu_s=0, inference_calls=0, downloads=0, new_training=False,
            source_sha256=sha(__file__), plan_sha256=sha(output/'plan.json'),
            resources='No processes/workers/services retained; durable payload owned by this run'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-s', type=float, default=600)
    args = parser.parse_args()
    run(args.repo.resolve(), args.output.resolve(), args.budget_s)
