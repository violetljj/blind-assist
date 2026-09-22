"""Evaluator-only native support opportunity on a fixed saved LOCAL transfer.

No fitting, model invocation, cutoff change, capture, interpolated depth, or
physical ownership claim. All-zone contributor availability is not causal HGB
attribution. Bounds compatibility is not renderer object identity.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

from ba_camera_corridor import sample_indices, sample_native
from ba_camera_corridor_metrics import evaluate_rows
from inherit_spatial_model import FOCAL, QUERIES, RAW_SIZE, canonical_tof, query_bands
from local_transfer_metrics import comparison
from query_occupancy_data import camera_bounds, observation_tokens, read, sha, write
from tof_fov45_core import boxes45, simulate

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from tools.research_backend import BackendCandidate, DeviceObservation, select_backend

ARMS = ('A_current', 'local', 'privileged_winner', 'privileged_either')
BANDS = ('definite', 'possible_only', 'outside')
COMPATIBILITY = ('target_only', 'background_only', 'both', 'unassigned')
PARTITIONS = ('WINNER_OBSERVED', 'OTHER_CENTRE_ONLY_OBSERVED',
              'NATIVE_VISIBLE_ONLY', 'NO_CENTRE_OBSERVED_OR_VISIBLE')
AXES = ('appearance', 'layer', 'layout_relation', 'base_group_id')
BOUND_TOLERANCE_M = .02
NATIVE_W, NATIVE_H = 640, 360
RGB_W, RGB_H = 320, 180


def require_governed(repo: Path, result: Path):
    artifact = (repo/'artifacts.local').resolve()
    resolved_result = result.resolve()
    if not resolved_result.is_relative_to(artifact) or resolved_result.parent == artifact:
        raise ValueError('Canonical task artifact output required')
    journal_name = os.environ.get('BLINDASSIST_ASSET_RUN_JOURNAL')
    if not journal_name:
        raise ValueError('Use active governed research-ue RunSpec')
    journal_path = Path(journal_name).resolve()
    if not journal_path.is_relative_to(artifact) or not journal_path.is_file():
        raise ValueError('Governed journal must exist inside artifact tree')
    journal = read(journal_path)
    if journal.get('schema') != 'blindassist-asset-run-journal-v1' or journal.get('state') != 'running':
        raise ValueError('Governed journal is not actively running')
    outputs = []
    for output in journal.get('outputs', []):
        path = Path(output['path'])
        path = (path if path.is_absolute() else artifact/path).resolve()
        if path == resolved_result:
            outputs.append(output['alias'])
    if not outputs:
        raise ValueError('Active governed journal does not own exact result')
    command = journal.get('command', [])
    if command.count('--result') != 1 or command.index('--result')+1 >= len(command):
        raise ValueError('Governed command must bind one result')
    if Path(command[command.index('--result')+1]).resolve() != resolved_result:
        raise ValueError('Governed command owns another result')
    out = resolved_result.parent
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Preserve nonempty diagnostic output')
    out.mkdir(parents=True, exist_ok=True)
    return dict(journal_path=str(journal_path), run_id=journal.get('id'), output_aliases=outputs), journal


def native_points(depth):
    """One optical-Z point per native pixel center; never average or resize Z."""
    depth = np.asarray(depth)
    if depth.shape != (NATIVE_H, NATIVE_W) or depth.dtype.kind != 'f':
        raise ValueError('Expected native 360x640 optical-Z floating depth')
    valid = np.isfinite(depth) & (depth > 0)
    z = np.where(valid, depth, 0).astype(np.float64)
    ys, xs = np.indices(depth.shape)
    focal = 2*FOCAL
    points = np.stack(((xs+.5-320)/focal*z, (ys+.5-180)/focal*z, z), axis=-1)
    return points.reshape(-1, 3), valid.ravel()


def points_in_query(points, valid, query):
    xl, xh, yl, yh = query
    return (valid & (points[:, 2] >= .3) & (points[:, 2] <= 3)
            & (points[:, 0] >= xl) & (points[:, 0] <= xh)
            & (points[:, 1] >= yl) & (points[:, 1] <= yh))


def compatibility_codes(points, valid, target_bounds, background_bounds):
    """Disjoint compatibility only: -1 invalid, 0 target, 1 background, 2 both, 3 neither."""
    def matches(bounds):
        mask = np.zeros(len(points), bool)
        for lo, hi in bounds:
            mask |= ((points >= np.asarray(lo)-BOUND_TOLERANCE_M)
                     & (points <= np.asarray(hi)+BOUND_TOLERANCE_M)).all(1)
        return mask & valid
    target, background = matches(target_bounds), matches(background_bounds)
    result = np.full(len(points), -1, np.int8)
    result[valid] = 3
    result[target & ~background] = 0
    result[background & ~target] = 1
    result[target & background] = 2
    return result


def compatibility_counts(codes, mask=None):
    selected = codes if mask is None else codes[mask]
    return {name: int(np.count_nonzero(selected == i)) for i, name in enumerate(COMPATIBILITY)}


def rgb_subpixel_indices(rgb_flat):
    """Each RGB pixel owns four separate native cells in row-major 2x2 order."""
    rgb_flat = np.asarray(rgb_flat, np.int64)
    y, x = np.divmod(rgb_flat, RGB_W)
    anchor = 2*y*NATIVE_W+2*x
    return anchor[:, None]+np.array([0, 1, NATIVE_W, NATIVE_W+1])


def local_band_membership(tof, queries=QUERIES):
    """Exact old LOCAL public membership; invalid zones never enter a band."""
    z, valid, boxes, low, high = canonical_tof(tof)
    zone_map = np.full((RGB_H, RGB_W), -1, np.int16)
    for zone, box in enumerate(boxes):
        y0, x0, y1, x1 = box*[180, 320, 180, 320]
        ys = np.arange(max(0, int(np.ceil(y0-.5))), min(180, int(np.ceil(y1-.5))))
        xs = np.arange(max(0, int(np.ceil(x0-.5))), min(320, int(np.ceil(x1-.5))))
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        if np.any(zone_map[yy, xx] != -1):
            raise ValueError('Overlapping original ToF footprints')
        zone_map[yy, xx] = zone
    footprint = zone_map >= 0
    eligible = footprint & valid[np.maximum(zone_map, 0)]
    yy, xx = np.nonzero(eligible)
    zones = zone_map[yy, xx]
    ax, ay = (xx+.5-160)/FOCAL, (yy+.5-90)/FOCAL
    codes = np.full((len(queries), RGB_H, RGB_W), -2, np.int8)
    codes[:, footprint] = -1
    fractions = np.zeros((len(queries), 3), np.float64)
    for qi, query in enumerate(queries):
        masks = query_bands(ax, ay, low[zones], high[zones], query)
        if not np.all(sum(m.astype(np.int8) for m in masks) == 1):
            raise ValueError('Public bands are not a disjoint complete partition')
        for bi, mask in enumerate(masks):
            codes[qi, yy[mask], xx[mask]] = bi
            fractions[qi, bi] = int(mask.sum())/max(1, int(footprint.sum()))
    return dict(codes=codes, zone_map=zone_map, zone_valid=valid, low=low, high=high,
                fractions=fractions, footprint_pixels=int(footprint.sum()),
                invalid_zone_pixels=int((footprint & ~eligible).sum()), values=z)


def band_native_counts(rgb_indices, zone_indices, points, valid, inside, compatibility, low, high):
    subpixels = rgb_subpixel_indices(rgb_indices)
    native_indices = subpixels.ravel()
    zones = np.repeat(np.asarray(zone_indices), 4)
    v = valid[native_indices]
    q = inside[native_indices]
    agreement = v & (points[native_indices, 2] >= low[zones]) & (points[native_indices, 2] <= high[zones])
    comp = compatibility[native_indices]
    return dict(rgb_pixels=len(rgb_indices), subpixels_total=len(native_indices),
        valid_native=int(v.sum()), invalid_native=int((~v).sum()),
        native_query_inside=int(q.sum()), native_query_outside_observed=int((v & ~q).sum()),
        regional_range_agreement=int(agreement.sum()), regional_range_disagreement=int((v & ~agreement).sum()),
        query_inside_and_range_agreement=int((q & agreement).sum()),
        compatibility_counts=compatibility_counts(comp),
        native_query_inside_compatibility=compatibility_counts(comp, q),
        regional_range_agreement_compatibility=compatibility_counts(comp, agreement))


def contributor_lineage(traces):
    offsets, sampled, weights = [0], [], []
    for zone, trace in enumerate(traces):
        if trace['zone_id'] != zone:
            raise ValueError('Trace zone order changed')
        selected = np.asarray(trace['pixel_indices'], np.int32)
        weight = np.asarray(trace['weights'], np.float64)
        if len(selected) != len(weight) or (not trace['observed'] and len(selected)):
            raise ValueError('Unobserved returns cannot supply contributors')
        sampled.append(selected)
        weights.append(weight)
        offsets.append(offsets[-1]+len(selected))
    sampled = np.concatenate(sampled)
    weights = np.concatenate(weights)
    ys, xs = sample_indices()
    y, x = np.divmod(sampled, 256)
    native = (ys[y]*NATIVE_W+xs[x]).astype(np.int32)
    if len(np.unique(sampled)) != len(sampled):
        raise ValueError('Contributors overlap across disjoint zones')
    return dict(zone_offsets=np.asarray(offsets, np.int32), sampled_indices=sampled,
                native_indices=native, weights=weights)


def support_partition(query_records, winner):
    other = 4 if winner == 1 else 1
    if query_records[winner]['observed_contributor_points'] > 0:
        return PARTITIONS[0]
    if query_records[other]['observed_contributor_points'] > 0:
        return PARTITIONS[1]
    if query_records[1]['native_visible_points'] or query_records[4]['native_visible_points']:
        return PARTITIONS[2]
    return PARTITIONS[3]


def analyze_frame(depth, tof, sensor_identity, rendered_objects, camera, target_name,
                  saved_features, saved_row):
    values, traces = simulate(sample_native(depth), sensor_identity, boxes45())
    reproduced = observation_tokens(values, boxes45())
    parity = bool(np.array_equal(reproduced, tof))
    line = contributor_lineage(traces)
    points, valid = native_points(depth)
    target_objects = [o for o in rendered_objects if o['name'] == target_name]
    other_objects = [o for o in rendered_objects if o['name'] != target_name]
    if len(target_objects) != 1:
        raise ValueError('Need exactly one declared target bounds object')
    comp = compatibility_codes(points, valid, camera_bounds(target_objects, camera), camera_bounds(other_objects, camera))
    membership = local_band_membership(tof)
    contributor_indices = line['native_indices']
    zone_ids = np.repeat(np.arange(64), np.diff(line['zone_offsets']))
    query_records, fraction_mismatches, visible_mismatches = [], 0, 0
    for qi, query in enumerate(QUERIES):
        inside = points_in_query(points, valid, query)
        observed_inside = inside[contributor_indices]
        observed_count = int(observed_inside.sum())
        visible_count = int(inside.sum())
        visible_mismatches += int(observed_count > 0 and visible_count == 0)
        bands = []
        for bi, name in enumerate(BANDS):
            indices = np.flatnonzero(membership['codes'][qi].ravel() == bi)
            zones = membership['zone_map'].ravel()[indices]
            record = band_native_counts(indices, zones, points, valid, inside, comp, membership['low'], membership['high'])
            expected = float(saved_features[qi, RAW_SIZE+15*bi+14])
            actual = float(membership['fractions'][qi, bi])
            error = abs(expected-actual)
            matches = error <= 1e-6
            fraction_mismatches += int(not matches)
            record.update(band_index=bi, name=name, saved_pixel_fraction=expected,
                reconstructed_pixel_fraction=actual, pixel_fraction_matches=matches,
                pixel_fraction_absolute_error=error)
            bands.append(record)
        query_records.append(dict(query_index=qi, query_bounds=query.tolist(),
            query_truth=saved_row['query_truth'][qi], query_label_valid=saved_row['query_valid'][qi],
            observed_contributor_points=observed_count, native_visible_points=visible_count,
            observed_contributor_weight=float(line['weights'][observed_inside].sum()),
            observed_contributing_zones=np.unique(zone_ids[observed_inside]).tolist(),
            observed_source_compatibility=compatibility_counts(comp[contributor_indices], observed_inside),
            visible_source_compatibility=compatibility_counts(comp, inside),
            observed_support_state=('SUPPORTED' if observed_count else 'OTHER_OBSERVED_RETURNS_ONLY'
                if len(contributor_indices) else 'NO_OBSERVED_RETURNS'),
            bands=bands))
    winner = 1 if saved_row['query_probabilities']['local'][1] >= saved_row['query_probabilities']['local'][4] else 4
    lineage = dict(**line, band_codes=membership['codes'], rgb_zone_indices=membership['zone_map'],
        zone_valid=membership['zone_valid'], interval_low=membership['low'], interval_high=membership['high'],
        simulated_values=values)
    zones = [dict(zone_id=t['zone_id'], observed=t['observed'], reason=t['reason'],
        winner_bin=t['winner_bin'], distance_m=t['distance_m'],
        contributor_count=int(line['zone_offsets'][i+1]-line['zone_offsets'][i])) for i, t in enumerate(traces)]
    detail = dict(index=saved_row['index'], id=saved_row['id'],
        **{key: saved_row[key] for key in ('clip_id', 'frame_in_clip', 'time_s', 'type_id',
                                         'appearance_pair_id', *AXES)},
        truth=saved_row['truth'], winner_query=winner,
        local_centre_scores=[saved_row['query_probabilities']['local'][q] for q in (1, 4)],
        local_standalone_alert=saved_row['predictions']['local_standalone']['alert'],
        support_partition=support_partition(query_records, winner), queries=query_records,
        public_tof_exact_parity=parity, band_fraction_mismatches=fraction_mismatches,
        contributor_visible_mismatches=visible_mismatches,
        native_valid_pixels=int(valid.sum()), native_invalid_pixels=int((~valid).sum()),
        observed_contributor_points_total=len(contributor_indices),
        all_observed_source_compatibility=compatibility_counts(comp[contributor_indices]),
        footprint_rgb_pixels=membership['footprint_pixels'], invalid_zone_rgb_pixels=membership['invalid_zone_pixels'],
        zone_observations=zones)
    return detail, lineage


def cohort_tags(row):
    a, local = (row['predictions'][arm]['alert'] for arm in ('A_current', 'local'))
    return (['local_rescue'] if row['truth'] is True and local and not a else
            ['local_added_fp'] if row['truth'] is False and local and not a else
            ['local_miss'] if row['truth'] is True and not local else [])


def diagnostic_readout_rows(saved_rows, diagnostics):
    rows = []
    for old, detail in zip(saved_rows, diagnostics):
        if old['id'] != detail['id']:
            raise ValueError('Diagnostic/readout frame order mismatch')
        row = copy.deepcopy(old)
        a = old['predictions']['A_current']
        local = old['predictions']['local_standalone']['alert']
        queries = detail['queries']
        witness = dict(privileged_winner=queries[detail['winner_query']]['observed_contributor_points'] > 0,
            privileged_either=any(queries[q]['observed_contributor_points'] > 0 for q in (1, 4)))
        for arm, supported in witness.items():
            alert = bool(a['alert'] or (local and supported))
            row['predictions'][arm] = dict(alert=alert, unknown=bool(a['unknown']),
                                          ambiguous=bool(alert and a['unknown']))
        rows.append(row)
    return rows


def true_event_gaps(rows, metrics):
    clips = {}
    for row in rows:
        clips.setdefault(row['clip_id'], {})[row['frame_in_clip']] = row
    result = {}
    for arm in ARMS:
        records = []
        for event in metrics['arms'][arm]['events']:
            event_rows = [clips[event['clip_id']][i] for i in range(event['start_frame'], event['end_frame']+1)]
            hits = [i for i, r in enumerate(event_rows) if r['predictions'][arm]['alert']]
            first, last = (hits[0], hits[-1]) if hits else (len(event_rows), -1)
            gaps, i = [], first+1
            while i < last:
                if event_rows[i]['predictions'][arm]['alert']:
                    i += 1
                    continue
                begin = i
                while i < last and not event_rows[i]['predictions'][arm]['alert']:
                    i += 1
                gaps.append(dict(start_frame=event_rows[begin]['frame_in_clip'],
                    end_frame=event_rows[i-1]['frame_in_clip'], frames=i-begin, sampled_duration_s=(i-begin)*.2))
            records.append(dict(clip_id=event['clip_id'], start_frame=event['start_frame'],
                end_frame=event['end_frame'], truth_frames=len(event_rows), alerted_true_frames=len(hits),
                alert_coverage=len(hits)/len(event_rows), sampled_alert_coverage_s=len(hits)*.2,
                unalerted_true_ids=[r['id'] for r in event_rows if not r['predictions'][arm]['alert']],
                leading_silence_frames=first, trailing_silence_frames=len(event_rows)-last-1 if hits else 0,
                internal_gap_segments=len(gaps), internal_gap_frames=sum(g['frames'] for g in gaps),
                max_internal_gap_frames=max((g['frames'] for g in gaps), default=0), internal_gaps=gaps))
        result[arm] = dict(events=records, total_alerted_true_frames=sum(r['alerted_true_frames'] for r in records),
            total_internal_gap_frames=sum(r['internal_gap_frames'] for r in records),
            total_internal_gap_segments=sum(r['internal_gap_segments'] for r in records),
            convention='Internal gaps lie between first and last alert within one true event; all silence remains in unalerted_true_ids')
    return result


def summarize_diagnostics(details):
    parts = Counter(d['support_partition'] for d in details)
    queries = []
    for qi in range(6):
        records = [d['queries'][qi] for d in details]
        bands = []
        for bi, name in enumerate(BANDS):
            sub = [q['bands'][bi] for q in records]
            keys = ('rgb_pixels', 'subpixels_total', 'valid_native', 'invalid_native', 'native_query_inside',
                'native_query_outside_observed', 'regional_range_agreement', 'regional_range_disagreement',
                'query_inside_and_range_agreement')
            b = dict(band_index=bi, name=name, **{key: sum(s[key] for s in sub) for key in keys})
            for key in ('compatibility_counts', 'native_query_inside_compatibility', 'regional_range_agreement_compatibility'):
                b[key] = {c: sum(s[key][c] for s in sub) for c in COMPATIBILITY}
            bands.append(b)
        queries.append(dict(query_index=qi, frame_query_denominator=len(records),
            label_invalid=sum(not q['query_label_valid'] for q in records),
            observed_supported_frames=sum(q['observed_contributor_points'] > 0 for q in records),
            observed_unsupported_frames=sum(q['observed_contributor_points'] == 0 for q in records),
            visible_frames=sum(q['native_visible_points'] > 0 for q in records),
            observed_contributor_points=sum(q['observed_contributor_points'] for q in records),
            native_visible_points=sum(q['native_visible_points'] for q in records),
            full_extent_positive_without_observed_support=sum(q['query_label_valid'] and q['query_truth'] and not q['observed_contributor_points'] for q in records),
            full_extent_negative_with_observed_support=sum(q['query_label_valid'] and not q['query_truth'] and q['observed_contributor_points'] > 0 for q in records),
            visible_without_observed_support=sum(q['native_visible_points'] > 0 and not q['observed_contributor_points'] for q in records),
            observed_source_compatibility={c: sum(q['observed_source_compatibility'][c] for q in records) for c in COMPATIBILITY},
            bands=bands))
    return dict(frames=len(details), frame_ids=[d['id'] for d in details],
        support_partition_counts={name: parts[name] for name in PARTITIONS}, queries=queries,
        public_tof_mismatch_frames=sum(not d['public_tof_exact_parity'] for d in details),
        band_fraction_mismatches=sum(d['band_fraction_mismatches'] for d in details),
        contributor_visible_mismatches=sum(d['contributor_visible_mismatches'] for d in details))


def make_report(saved_rows, diagnostics):
    rows = diagnostic_readout_rows(saved_rows, diagnostics)
    metrics = evaluate_rows(rows, arms=ARMS)
    comps = {arm+'_vs_'+ref: comparison(rows, metrics, arm, ref)
        for arm in ('privileged_winner', 'privileged_either') for ref in ('A_current', 'local')}
    strata = {key: {value: evaluate_rows([r for r in rows if r[key] == value], arms=ARMS)
        for value in sorted({r[key] for r in rows})} for key in AXES}
    cohorts = {'all': diagnostics}
    for name, tag in [('local_rescues', 'local_rescue'), ('local_added_fp', 'local_added_fp'), ('local_misses', 'local_miss')]:
        cohorts[name] = [d for d in diagnostics if tag in d['cohort_tags']]
    cohort_summaries = {name: summarize_diagnostics(value) for name, value in cohorts.items()}
    summaries = {key: {value: summarize_diagnostics([d for d in diagnostics if d[key] == value])
        for value in sorted({d[key] for d in diagnostics})} for key in AXES}
    by_id = {r['id']: r for r in rows}
    rescued = cohort_summaries['local_rescues']['frame_ids']
    added_fp = cohort_summaries['local_added_fp']['frame_ids']
    opportunity = {}
    reconstruction_ok = not any(cohort_summaries['all'][k] for k in ('public_tof_mismatch_frames',
        'band_fraction_mismatches', 'contributor_visible_mismatches'))
    invalid_query_labels = sum(not value for row in saved_rows for value in row['query_valid'])
    reconstruction_ok = reconstruction_ok and invalid_query_labels == 0
    scope_ok = len(rows) == 576 and len(rescued) == 36 and len(added_fp) == 2 and len(cohorts['local_misses']) == 28
    for arm in ('privileged_winner', 'privileged_either'):
        kept = [name for name in rescued if by_id[name]['predictions'][arm]['alert']]
        removed = [name for name in added_fp if not by_id[name]['predictions'][arm]['alert']]
        vs_a = comps[arm+'_vs_A_current']
        a_preserved = not vs_a['lost_true_frames'] and not any(e['lost'] or
            (e['delay_s'] is not None and e['delay_s'] > 1e-9) for e in vs_a['event_differences'])
        evaluable = reconstruction_ok and scope_ok and all(r['truth'] is not None for r in rows)
        passed = len(kept) >= 32 and len(removed) == 2 and a_preserved
        opportunity[arm] = dict(status='NOT_EVALUABLE' if not evaluable else 'PASS' if passed else 'FAIL',
            label='PRIVILEGED_OPPORTUNITY_NOT_DEPLOYABLE', rescues_before=len(rescued), rescues_retained=len(kept),
            rescue_ids_retained=kept, rescue_ids_lost=[name for name in rescued if name not in kept],
            added_FP_before=len(added_fp), added_FP_removed=len(removed), added_FP_ids_removed=removed,
            A_preserved=a_preserved, A_preservation_by_OR_construction=True,
            role='PRIMARY_INFORMATION_OPPORTUNITY' if arm == 'privileged_winner' else 'DESCRIPTIVE_OTHER_CENTRE_CONTROL')
    report = dict(status='PASS', stage_status='PASS',
        hypothesis_status=opportunity['privileged_winner']['status'],
        label='PRIVILEGED_OPPORTUNITY_NOT_DEPLOYABLE', metrics=metrics, strata=strata, comparisons=comps,
        event_gaps=true_event_gaps(rows, metrics), cohort_summaries=cohort_summaries,
        diagnostic_strata=summaries, opportunity=opportunity,
        reconstruction_admissible=reconstruction_ok, expected_consumed_cohort_counts_match=scope_ok,
        denominators=dict(frames=len(rows), queries=len(rows)*6, bands=len(rows)*6*3,
                          invalid_query_labels=invalid_query_labels),
        boundaries=['Observed winning-bin samples are evaluator-only lineage of the original simulator, not hardware ownership.',
            'All-zone support availability is not causal attribution of an HGB decision to a zone or pixel.',
            'Rendered-bounds +/-0.02m establishes compatibility only; neither material nor object identity is inferred.',
            'Native witness removal of false positives is largely guaranteed geometrically; rescue retention and timing costs carry the opportunity question.',
            'No contributor or no visible point never certifies free space; invalid native values and invalid public zones remain explicit.',
            'Neither privileged readout is deployable or a promotion; original A/LOCAL models, cutoffs and historical roles remain unchanged.'])
    return rows, report


def run(repo: Path, result: Path):
    repo, result = Path(repo).resolve(), Path(result)
    governance, journal = require_governed(repo, result)
    out, begun = result.resolve().parent, time.perf_counter()
    root = repo/'artifacts.local/evidence/ba-local-transfer-20260922'
    capture, prepared, predictions, evaluated = [root.with_name(root.name+'-'+s)
        for s in ('capture', 'prepared', 'predictions', 'evaluated')]
    paths = dict(spec=root/'plan/spec.json', geometry=capture/'evaluator/geometry.json',
        materialization=prepared/'materialization.json', labels=prepared/'labels/evaluation.npz',
        rgb=prepared/'observations/rgb.npy', tof=prepared/'observations/tof.npy',
        identities=prepared/'observations/identities.json', feature=predictions/'features.npz',
        probability=predictions/'probabilities.npz', prediction_seal=predictions/'prediction-seal.json',
        prediction_freeze=predictions/'freeze.json', selection=predictions/'selection.json',
        baseline=predictions/'baseline.json', prediction_identities=predictions/'identities.json',
        evaluation_result=evaluated/'result.json', saved_metrics=evaluated/'metrics.json',
        saved_rows=evaluated/'frame-results.json',
        protocol=repo/'research/active/dtr-r0/nearfield/LOCAL_SUPPORT_PROTOCOL_20260922.md')
    spec, geometry, manifest = read(paths['spec']), read(paths['geometry']), read(paths['materialization'])
    saved_rows, seal, pred_freeze, old_result = [read(paths[k]) for k in
        ('saved_rows', 'prediction_seal', 'prediction_freeze', 'evaluation_result')]
    if len(spec['cases']) != len(geometry) or len(geometry) != len(saved_rows) or len(saved_rows) != 576:
        raise ValueError('Exactly all 576 original transfer frames required')
    if sha(paths['spec']) != manifest['source_spec_sha256']:
        raise ValueError('Source plan identity changed')
    for name, digest in seal['hashes'].items():
        if sha(predictions/name) != digest:
            raise ValueError('Original prediction seal changed: '+name)
        paths['sealed_prediction/'+name] = predictions/name
    for name, key in [('saved_rows', 'frame_results_sha256'), ('saved_metrics', 'metrics_sha256'),
                       ('prediction_seal', 'prediction_seal_sha256'), ('labels', 'evaluator_labels_sha256')]:
        if sha(paths[name]) != old_result[key]:
            raise ValueError('Original evaluation seal changed: '+name)
    for name in ('rgb', 'tof', 'identities'):
        key = 'observations/'+paths[name].name
        if sha(paths[name]) != manifest['hashes'][key] or sha(paths[name]) != pred_freeze['input_hashes'][key]:
            raise ValueError('Original public input changed: '+name)
    if sha(paths['materialization']) != pred_freeze['materialization_sha256']:
        raise ValueError('Materialization receipt changed')
    for i, (case, geo, saved) in enumerate(zip(spec['cases'], geometry, saved_rows)):
        if case['name'] != geo['id'] or geo['id'] != saved['id'] or saved['index'] != i or geo['sample_index'] != i:
            raise ValueError('Native/case/saved frame identity mismatch')
        if case['camera'] != geo['declared_camera'] or any(case['camera'][a] != 0 for a in ('pitch', 'yaw', 'roll')):
            raise ValueError('Unexpected camera pose contract')
        native = capture/'evaluator'/geo['native_path']
        if not native.resolve().is_relative_to((capture/'evaluator').resolve()):
            raise ValueError('Native file escaped evaluator root')
        if sha(native) != geo['native_sha256']:
            raise ValueError('Native payload hash changed')
        paths[f'native/{i:04}'] = native
    sources = [HERE/name for name in ('local_support_diagnostic.py', 'test_local_support_diagnostic.py',
        'inherit_spatial_model.py', 'ba_camera_corridor.py', 'tof_fov45_core.py',
        'query_occupancy_data.py', 'ba_camera_corridor_metrics.py', 'local_transfer_metrics.py')]
    sources.append(repo/'tools/research_backend.py')
    source_hashes = {str(p.relative_to(repo)): sha(p) for p in sources}
    for name, digest in source_hashes.items():
        if name in pred_freeze['source_hashes'] and pred_freeze['source_hashes'][name] != digest:
            raise ValueError('Original feature/metric code changed after prediction sealing: '+name)
        dest = out/'source-snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo/name, dest)
    file_hashes = {str(p.relative_to(repo)).replace('\\', '/'): sha(p) for p in paths.values()}
    input_seal = dict(schema='local-support-input-seal-v1',
        roots={k: str(v.resolve()) for k, v in dict(transfer=root, capture=capture, prepared=prepared,
            predictions=predictions, evaluated=evaluated).items()},
        input_paths={name: str(p.resolve()) for name, p in paths.items()}, input_file_hashes=file_hashes,
        source_hashes=source_hashes, governance=governance,
        fits=0, model_inference_calls=0, cutoff_changes=0, capture_calls=0,
        evaluator_only=True, bound_compatibility_tolerance_m=BOUND_TOLERANCE_M,
        source_noise_identity_prefix='local-transfer/', query_boxes=QUERIES.tolist())
    write(out/'input-seal.json', input_seal)
    write(out/'governed-journal-snapshot.json', journal)
    labels = dict(np.load(paths['labels'], allow_pickle=False))
    rgb, tof = [np.load(paths[k], mmap_mode='r', allow_pickle=False) for k in ('rgb', 'tof')]
    features = np.load(paths['feature'], allow_pickle=False)['local']
    probabilities = np.load(paths['probability'], allow_pickle=False)['local']
    if rgb.shape != (576, 3, 180, 320) or rgb.dtype != np.uint8 or tof.shape != (576, 64, 6) or features.shape != (576, 6, 961):
        raise ValueError('Original input tensor schema changed')
    if not np.array_equal(labels['indices'], np.arange(576)):
        raise ValueError('Saved evaluator index mismatch')
    for i, row in enumerate(saved_rows):
        if row['query_truth'] != (labels['classes'][i] < 6).tolist() or row['query_valid'] != labels['valid'][i].astype(bool).tolist():
            raise ValueError('Saved query truth changed')
        if row['query_probabilities']['local'] != probabilities[i].tolist():
            raise ValueError('Saved LOCAL query probabilities changed')
        score = max(probabilities[i, [1, 4]])
        if row['predictions']['local_standalone']['alert'] != bool(score >= pred_freeze['thresholds']['local']):
            raise ValueError('Saved cutoff readout changed')
    select_backend('scalar-scoring', cpu=BackendCandidate('native-support-numpy-cpu', 'cpu',
        lambda: local_band_membership(tof[0]),
        lambda _: DeviceObservation('cpu', platform.processor() or 'host CPU', 'numpy '+np.__version__)),
        cpu_reason='TASK_NOT_GPU_SUITABLE', record_path=out/'backend.json',
        capabilities=dict(reason='Fixed evaluator-only point membership, index lineage and scalar metric recount',
            python_executable=sys.executable, model_inference=False, fitting=False))
    (out/'lineage').mkdir()
    diagnostics, elapsed = [], []
    for i, (case, geo, row) in enumerate(zip(spec['cases'], geometry, saved_rows)):
        start = time.perf_counter()
        depth = np.load(paths[f'native/{i:04}'], allow_pickle=False)
        detail, lineage = analyze_frame(depth, tof[i], 'local-transfer/'+case['sensor_noise_key'],
            geo['objects'], case['camera'], case['target_name'], features[i], row)
        detail['cohort_tags'] = cohort_tags(row)
        detail['native_input_sha256'] = geo['native_sha256']
        file = out/'lineage'/f'frame-{i:04}.npz'
        np.savez_compressed(file, **lineage)
        detail['lineage_path'] = file.relative_to(out).as_posix()
        detail['lineage_sha256'] = sha(file)
        diagnostics.append(detail)
        elapsed.append(time.perf_counter()-start)
        if (i+1) % 96 == 0:
            print(json.dumps(dict(stage='native_support_diagnostic', frames=i+1, seconds=time.perf_counter()-begun)), flush=True)
    rows, measured = make_report(saved_rows, diagnostics)
    write(out/'frame-diagnostics.json', diagnostics)
    write(out/'readout-rows.json', rows)
    write(out/'metrics.json', measured)
    for relative, digest in file_hashes.items():
        if sha(repo/relative) != digest:
            raise ValueError('Input mutated during diagnostic: '+relative)
    for relative, digest in source_hashes.items():
        if sha(repo/relative) != digest:
            raise ValueError('Source mutated during diagnostic: '+relative)
    costs = dict(frames=576, per_frame_seconds_p50=float(np.median(elapsed)),
        per_frame_seconds_p95=float(np.quantile(elapsed, .95)), total_elapsed_s=time.perf_counter()-begun,
        fits=0, model_inference_calls=0, cutoff_changes=0, capture_calls=0,
        scope='Host evaluator-only CPU cost including native loading, support counts and lineage serialization; not endpoint latency')
    write(out/'costs.json', costs)
    hashes = {p.relative_to(out).as_posix(): sha(p) for p in out.rglob('*') if p.is_file()}
    write(out/'output-seal.json', dict(status='PASS', input_seal_sha256=sha(out/'input-seal.json'), files=hashes))
    all_summary = measured['cohort_summaries']['all']
    answer = dict(status='PASS', stage_status='PASS', hypothesis_status=measured['hypothesis_status'],
        label='PRIVILEGED_OPPORTUNITY_NOT_DEPLOYABLE', opportunity=measured['opportunity'],
        denominators=measured['denominators'], public_tof_exact_parity_frames=576-all_summary['public_tof_mismatch_frames'],
        band_fraction_checks=576*6*3, band_fraction_mismatches=all_summary['band_fraction_mismatches'],
        contributor_visible_mismatches=all_summary['contributor_visible_mismatches'],
        support_partition_counts={k: v['support_partition_counts'] for k, v in measured['cohort_summaries'].items()},
        metrics_sha256=sha(out/'metrics.json'), frame_diagnostics_sha256=sha(out/'frame-diagnostics.json'),
        readout_rows_sha256=sha(out/'readout-rows.json'), output_seal_sha256=sha(out/'output-seal.json'),
        input_seal_sha256=sha(out/'input-seal.json'), costs=costs,
        resource_state='CPU command completes; no capture, model process, worker or paid allocation created',
        scope='Consumed controlled transfer diagnostic; no baseline or algorithm promotion')
    write(result, answer)
    print(json.dumps(answer), flush=True)
    return answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.repo, arguments.result)
