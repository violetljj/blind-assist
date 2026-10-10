"""Expand the same 15 cached Development groups before reading labels.

Retain the original 16 uniform native paired frames, add 48 selected uniformly
from their complement, and retain every fixed27 query. No model is accessed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
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

PRIOR = 'rgb-body-query-negative-reference-dev-20261010'
BANDS = ('0.3-0.8m', '0.8-1.5m', '1.5-3m')


def run(repo, output, budget_s):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    if (output/'plan.json').exists():
        raise FileExistsError('Preserve prior expansion run')
    prior = repo/'artifacts.local/work'/PRIOR
    fixed = queries()
    selection = select_additional(repo)
    expanded, newgroups = [], []
    for group in selection['groups']:
        with zipfile.ZipFile(group['source_zip']) as archive:
            names = set(archive.namelist())
        paired = sorted(int(m.group(1)) for name in names if
            (m := re.fullmatch(r'frame-(\d{6})\.color\.jpg', name)) and
            f'frame-{int(m.group(1)):06d}.depth.pgm' in names)
        uniform64 = [paired[i] for i in np.linspace(0, len(paired)-1, 64).round().astype(int)]
        old = set(group['frames'])
        remaining = [f for f in paired if f not in old]
        fresh = [remaining[i] for i in np.linspace(0, len(remaining)-1, 48).round().astype(int)]
        if len(set(fresh)) != 48 or old.intersection(fresh):
            raise ValueError('Need 48 unique remaining paired frames; no source replacement')
        frames = sorted(old | set(fresh))
        expanded.append(dict(group, frames=frames, original_frames=group['frames'],
            newly_selected_frames=fresh, direct_uniform64_contains_old16=old.issubset(uniform64),
            direct_uniform64_frames=uniform64))
        newgroups.append(dict(group, frames=fresh))
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='SOURCE_SELECTION_FROZEN_BEFORE_NEW_DEPTH_AND_LABELS',
        goal='Test whether uniform extra temporal coverage provides cross-environment near FREE',
        baseline='15 environments x16 frames; near FREE6 in one environment',
        decision='If near FREE still occurs in one environment, stop uniform frame expansion as an ineffective gap remedy',
        selection=dict(selection, groups=expanded),
        selection_method='Same original16 plus uniform48 from remaining paired frame names; same scan, all15 environments, all27 queries',
        expected_frames=960, new_frames=720, reused_frames=240, queries=fixed, axis_edges=EDGES,
        prior_manifest=str(prior/'dataset_manifest.json'),
        prior_manifest_sha256=sha(prior/'dataset_manifest.json'),
        role_policy='Additional_cal official train Development only; original20 groups excluded including rescans; no eval reassignment',
        query_reference_policy='FREE>=16 free rays, no positive or unknown rays; UNKNOWN remains UNKNOWN; closed native optical-Z boxes',
        budget_cpu_wall_s=budget_s, new_training=False, inference_calls=0, gpu_s=0, downloads=0,
        source_sha256=sha(__file__),
        limitations='Correlated native first-return rays and frames, same camera family, Development only; FREE is not whole-volume clearance')
    write(output/'plan.json', plan)
    complete, failure = False, None
    checks = Counter()
    rows, records = [], []
    try:
        def budget():
            if time.perf_counter()-started >= budget_s:
                raise TimeoutError('600s CPU wall expansion budget reached; preserve partial source, no restart')
        oldmanifest = load(prior/'dataset_manifest.json')
        if oldmanifest['queries'] != fixed:
            raise ValueError('Original fixed grid changed')
        oldrows = [r for r in oldmanifest['rows'] if r['split'] == 'additional_cal']
        expected_old = {(g['scan'], f) for g in selection['groups'] for f in g['frames']}
        if {(r['scan'], r['frame']) for r in oldrows} != expected_old or len(oldrows) != 240:
            raise ValueError('Old16 source identities changed')
        freshrows = additional_rows(dict(selection, groups=newgroups), output, fixed, budget)
        rows = sorted(oldrows + freshrows, key=lambda r: (r['environment'], r['frame']))
        if len(rows) != 960 or len({(r['scan'], r['frame']) for r in rows}) != 960:
            raise ValueError('960-frame uniqueness mismatch')
        for row in rows:
            budget()
            if sha(row['reference_path']) != row['reference_sha256'] or sha(row['rgb_path']) != row['rgb_sha256']:
                raise ValueError('Reference or RGB identity changed')
            with np.load(row['reference_path'], allow_pickle=False) as f:
                labels, depth, observed, k = f['labels'], f['depth'], f['observed'], f['depth_K']
                if labels.shape != (27, *depth.shape) or list(depth.shape) != row['depth_shape']:
                    raise ValueError('Fixed27 native geometry differs')
                if not np.array_equal(k, row['depth_K']) or not np.array_equal(f['color_K'], row['color_K']):
                    raise ValueError('Native intrinsics changed')
                for qi, query in enumerate(fixed):
                    budget()
                    independent, domain = independent_xyz_labels(depth, k, query, observed)
                    label = labels[qi]
                    if not np.array_equal(independent, label):
                        raise ValueError('Independent XYZ label mismatch')
                    counts = {key: int((label == value).sum()) for key, value in
                        [('positive_pixels', 1), ('free_ray_pixels', 0), ('unknown_pixels', 2)]}
                    state = ('POSITIVE' if counts['positive_pixels'] >= 16 else
                        FREE if counts['positive_pixels'] == 0 and counts['unknown_pixels'] == 0
                        and counts['free_ray_pixels'] >= 16 else 'UNKNOWN')
                    inherited = row['queries'][qi]
                    reach = int(domain.sum())
                    observed_count = int((domain & observed).sum())
                    if state != inherited['state'] or any(counts[key] != inherited[key] for key in counts):
                        raise ValueError('Strict state/count mismatch')
                    if reach != inherited['domain_pixels'] or sum(counts.values()) != reach or observed_count != inherited['observed_reachable_rays']:
                        raise ValueError('Reachable/observed ray denominator mismatch')
                    records.append(dict(split='additional_cal', environment=row['environment'], scan=row['scan'],
                        frame=row['frame'], reused_original16=(row['scan'], row['frame']) in expected_old,
                        query=query['name'], query_index=qi,
                        distance_band=f'{query["low"][2]:g}-{query["high"][2]:g}m', reference_state=state,
                        negative_eligible=state == FREE, calibration_eligible=state == FREE,
                        domain_pixels=reach, **counts, observed_reachable_rays=observed_count,
                        reference_path=row['reference_path'], reference_sha256=row['reference_sha256']))
                    checks['independent_xyz_queries'] += 1
                    checks['strict_state_count_checks'] += 1
                checks['native_identity_frames'] += 1
        envs = sorted({r['environment'] for r in rows})
        by_env = [dict(environment=env, **coverage([r for r in records if r['environment'] == env])) for env in envs]
        by_band = [dict(distance_band=band, **coverage([r for r in records if r['distance_band'] == band])) for band in BANDS]
        newrecords = [r for r in records if not r['reused_original16']]
        new_by_band = [dict(distance_band=band, **coverage([r for r in newrecords if r['distance_band'] == band])) for band in BANDS]
        manifest = dict(status='FIXED960_REFERENCE_COMPLETE', rows=rows, frames=960, groups=expanded,
            queries=fixed, state_counts=dict(Counter(r['reference_state'] for r in records)),
            plan_sha256=sha(output/'plan.json'), role_policy=plan['role_policy'],
            reuse_policy='Original240 RGB/reference paths retained and SHA checked; new720 native raw RGB/depth refs stored here')
        write(output/'dataset_manifest.json', manifest)
        write(output/'observations.json', dict(rows=[{k: r[k] for k in PUBLIC} for r in rows],
            queries=fixed, dataset_manifest_sha256=sha(output/'dataset_manifest.json')))
        write_csv(output/'query_reference_counts.csv', records)
        write_csv(output/'environment_coverage.csv', by_env)
        write_csv(output/'distance_band_coverage.csv', by_band)
        write_csv(output/'new_distance_band_coverage.csv', new_by_band)
        negatives = [r for r in records if r['negative_eligible']]
        write(output/'negative_query_index.json', dict(queries=negatives, calibration_queries=negatives,
            role_policy=plan['role_policy'], selection='Labels indexed only after ALL frozen source queries'))
        near = [r for r in records if r['distance_band'] == BANDS[0] and r['negative_eligible']]
        nearenvs = sorted({r['environment'] for r in near})
        result = dict(status='REFERENCE_EXPANSION_COMPLETE', total=coverage(records), new=coverage(newrecords),
            by_environment=by_env, by_band=by_band, new_by_band=new_by_band,
            near_free_environments=nearenvs,
            direct_uniform64_contains_old16_environments=sum(g['direct_uniform64_contains_old16'] for g in expanded),
            near_cross_environment_coverage_improved=len(nearenvs)>1,
            next_decision=('Near FREE remains concentrated in one environment; stop uniform temporal frame expansion' if len(nearenvs)<=1 else
                'Near FREE now spans multiple environments; inspect actual discrete/calibration coverage before use'),
            checks=dict(checks), limitations=plan['limitations'])
        write(output/'coverage.json', result)
        saved = load(output/'negative_query_index.json')
        if len(saved['queries']) != len(negatives) or any(r['reference_state'] != FREE for r in saved['queries']):
            raise ValueError('Saved negative index differs')
        obs = load(output/'observations.json')
        if any(set(r) != set(PUBLIC) for r in obs['rows']):
            raise ValueError('Public observation truth leakage')
        checks['serialized_negative_rows'] = len(saved['queries'])
        checks['serialized_public_observation_rows'] = len(obs['rows'])
        complete = True
        print(result['next_decision'], flush=True)
        print({b['distance_band']: (b['strict_sampled_free_queries'], b['strict_sampled_free_environments']) for b in by_band}, flush=True)
    except Exception as error:
        failure = repr(error)
        write(output/'failure.json', dict(error=failure, completed_frames=len(rows), query_records=len(records)))
        raise
    finally:
        shutil.copyfile(__file__, output/'executed_negative_reference_expand.py')
        write(output/'terminal.json', dict(status='COMPLETE' if complete else 'INCOMPLETE', failure=failure,
            elapsed_cpu_wall_s=time.perf_counter()-started, budget_cpu_wall_s=budget_s,
            checks=dict(checks), gpu_s=0, downloads=0, inference_calls=0, new_training=False,
            source_sha256=sha(__file__), plan_sha256=sha(output/'plan.json'),
            resources='No processes/workers/services retained; durable payload owned by this run'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-s', type=float, default=600.)
    args = parser.parse_args()
    run(args.repo.resolve(), args.output.resolve(), args.budget_s)
