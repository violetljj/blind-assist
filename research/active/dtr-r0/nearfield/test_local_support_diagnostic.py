"""Synthetic-only checks of native cells, support availability and fixed readouts."""
from __future__ import annotations

import copy
import unittest

import numpy as np

from inherit_spatial_model import extract
from local_support_diagnostic import (ARMS, BANDS, COMPATIBILITY, PARTITIONS,
    analyze_frame, band_native_counts, cohort_tags, compatibility_codes,
    compatibility_counts, contributor_lineage, diagnostic_readout_rows,
    local_band_membership, make_report, native_points, points_in_query,
    rgb_subpixel_indices, support_partition)
from ba_camera_corridor import sample_indices, sample_native
from query_occupancy_data import observation_tokens
from tof_fov45_core import boxes45, simulate


def synthetic_saved_rows():
    rows = []
    for group in range(8):
        layer = 'BODY' if group < 4 else 'HEAD'
        qi = 1 if layer == 'BODY' else 4
        for appearance in ('base', 'changed'):
            for relation in ('INSIDE', 'BOUNDARY', 'OUTSIDE'):
                clip = f'g{group}_{appearance}_{relation}'
                for frame in range(12):
                    truth = relation != 'OUTSIDE' and 2 <= frame <= 9
                    a = truth and frame >= 4
                    probabilities = [0.1]*6
                    if a:
                        probabilities[qi] = .9
                    row = dict(index=len(rows), id=f'{clip}_{frame:02}', clip_id=clip,
                        frame_in_clip=frame, time_s=.2*frame, type_id=f'family{group//2}', layer=layer,
                        appearance=appearance, layout_relation=relation, base_group_id=f'g{group}',
                        appearance_pair_id=f'g{group}_{relation}_{frame:02}', truth=truth,
                        boundary=relation == 'BOUNDARY', query_truth=[truth and q == qi for q in range(6)],
                        query_valid=[True]*6, query_probabilities=dict(local=probabilities),
                        predictions={name: dict(alert=bool(a), unknown=True, ambiguous=bool(a))
                            for name in ('A_current', 'local', 'local_standalone')})
                    rows.append(row)
    rescues = [r for r in rows if r['truth'] and not r['predictions']['A_current']['alert']][:36]
    false = [r for r in rows if r['layout_relation'] == 'OUTSIDE' and r['base_group_id'] == 'g0'
             and r['appearance'] == 'base' and r['frame_in_clip'] in (3, 9)]
    for row in rescues+false:
        qi = 1 if row['layer'] == 'BODY' else 4
        row['query_probabilities']['local'][qi] = .9
        for name in ('local', 'local_standalone'):
            row['predictions'][name].update(alert=True, ambiguous=True)
    return rows


def empty_band(index):
    keys = ('rgb_pixels', 'subpixels_total', 'valid_native', 'invalid_native',
        'native_query_inside', 'native_query_outside_observed', 'regional_range_agreement',
        'regional_range_disagreement', 'query_inside_and_range_agreement')
    return dict(band_index=index, name=BANDS[index], **dict.fromkeys(keys, 0),
        **{key: dict.fromkeys(COMPATIBILITY, 0) for key in ('compatibility_counts',
            'native_query_inside_compatibility', 'regional_range_agreement_compatibility')})


def synthetic_details(rows):
    details = []
    for row in rows:
        queries = [dict(query_index=q, query_truth=row['query_truth'][q], query_label_valid=True,
            observed_contributor_points=int(row['query_truth'][q]), native_visible_points=int(row['query_truth'][q]),
            observed_source_compatibility=dict.fromkeys(COMPATIBILITY, 0), bands=[empty_band(b) for b in range(3)])
            for q in range(6)]
        scores = row['query_probabilities']['local']
        winner = 1 if scores[1] >= scores[4] else 4
        detail = dict(**{key: row[key] for key in ('id', 'index', 'appearance', 'layer', 'layout_relation', 'base_group_id')},
            winner_query=winner, queries=queries, public_tof_exact_parity=True, band_fraction_mismatches=0,
            contributor_visible_mismatches=0, cohort_tags=cohort_tags(row))
        detail['support_partition'] = support_partition(queries, winner)
        details.append(detail)
    return details


class NativeSupportDiagnosticTests(unittest.TestCase):
    def test_exact_native_subcells_preserve_partial_query_occupancy(self):
        depth = np.full((360, 640), np.nan, np.float32)
        depth[180:182, 320:322] = 1
        points, valid = native_points(depth)
        indices = rgb_subpixel_indices([90*320+160])[0]
        self.assertEqual(indices.tolist(), [180*640+320, 180*640+321, 181*640+320, 181*640+321])
        inside = points_in_query(points, valid, [0, .003, 0, .9])
        self.assertEqual(inside[indices].tolist(), [True, False, True, False])
        comp = np.full(360*640, -1, np.int8)
        comp[valid] = 3
        counts = band_native_counts(np.array([90*320+160]), np.array([0]), points, valid, inside,
            comp, np.array([.9]), np.array([1.1]))
        self.assertEqual(counts['subpixels_total'], 4)
        self.assertEqual(counts['native_query_inside'], 2)
        self.assertEqual(counts['regional_range_agreement'], 4)
        self.assertEqual(counts['query_inside_and_range_agreement'], 2)

    def test_partial_range_agreement_and_invalid_subpixel_not_averaged(self):
        depth = np.full((360, 640), np.nan, np.float32)
        depth[180:182, 320:322] = [[1, 3.1], [np.nan, 1.5]]
        points, valid = native_points(depth)
        inside = points_in_query(points, valid, [-.3, .3, -.2, .9])
        comp = compatibility_codes(points, valid,
            [(np.array([-1, -1, .9]), np.array([1, 1, 1.1]))],
            [(np.array([-1, -1, 1.4]), np.array([1, 1, 3.2]))])
        counts = band_native_counts(np.array([90*320+160]), np.array([0]), points, valid, inside,
            comp, np.array([.9]), np.array([1.1]))
        self.assertEqual(counts['valid_native'], 3)
        self.assertEqual(counts['invalid_native'], 1)
        self.assertEqual(counts['native_query_inside'], 2)
        self.assertEqual(counts['regional_range_agreement'], 1)
        self.assertEqual(counts['compatibility_counts'], dict(target_only=1, background_only=2, both=0, unassigned=0))

    def test_source_compatibility_and_query_membership_are_different(self):
        points = np.array([[0, 0, 1], [1, 0, 1], [0, 0, 4], [2, 0, 1], [0, 0, 2]], float)
        valid = np.array([True, True, True, True, False])
        codes = compatibility_codes(points, valid,
            [(np.array([.9, -.1, .9]), np.array([2.1, .1, 1.1]))],
            [(np.array([-.1, -.1, 3.9]), np.array([.1, .1, 4.1])),
             (np.array([1.9, -.1, .9]), np.array([2.1, .1, 1.1]))])
        self.assertEqual(codes.tolist(), [3, 0, 1, 2, -1])
        inside = points_in_query(points, valid, [-.3, .3, -.2, .42])
        self.assertEqual(inside.tolist(), [True, False, False, False, False])
        self.assertEqual(compatibility_counts(codes)['both'], 1)

    def test_original_band_fractions_and_invalid_zone_denominator(self):
        values = np.full(64, 1.8, np.float32)
        values[0] = np.nan
        tof = observation_tokens(values, boxes45())
        rgb = np.zeros((3, 180, 320), np.uint8)
        original = extract(rgb, tof)[:, 1]
        member = local_band_membership(tof)
        expected = original[:, [924, 939, 954]]
        np.testing.assert_allclose(member['fractions'], expected, rtol=0, atol=1e-6)
        self.assertGreater(member['invalid_zone_pixels'], 0)
        for q in range(6):
            self.assertEqual(int((member['codes'][q] == -1).sum()), member['invalid_zone_pixels'])
            self.assertEqual(int((member['codes'][q] >= 0).sum()), member['footprint_pixels']-member['invalid_zone_pixels'])

    def test_simulator_lineage_native_mapping_and_query_counts(self):
        depth = np.full((360, 640), 1.8, np.float32)
        identity = 'local-transfer/synthetic-only'
        values, traces = simulate(sample_native(depth), identity, boxes45())
        lineage = contributor_lineage(traces)
        ys, xs = sample_indices()
        sampled = lineage['sampled_indices']
        np.testing.assert_array_equal(lineage['native_indices'], ys[sampled//256]*640+xs[sampled % 256])
        self.assertEqual(lineage['zone_offsets'][-1], len(lineage['weights']))
        for i, trace in enumerate(traces):
            if not trace['observed']:
                self.assertEqual(lineage['zone_offsets'][i], lineage['zone_offsets'][i+1])
        tof = observation_tokens(values, boxes45())
        features = extract(np.zeros((3, 180, 320), np.uint8), tof)[:, 1]
        saved = synthetic_saved_rows()[2]
        saved['query_probabilities']['local'][1] = saved['query_probabilities']['local'][4] = .9
        objects = [dict(name='target', render_bounds_center_m=[1.8, 0, 0], render_bounds_extent_m=[.01, 3, 3]),
            dict(name='background', render_bounds_center_m=[4.5, 0, 0], render_bounds_extent_m=[.01, 10, 10])]
        detail, stored = analyze_frame(depth, tof, identity, objects, dict(x=0, y=0, z=0), 'target', features, saved)
        self.assertTrue(detail['public_tof_exact_parity'])
        self.assertEqual(detail['band_fraction_mismatches'], 0)
        self.assertEqual(detail['winner_query'], 1)
        self.assertEqual(len(detail['queries']), 6)
        self.assertEqual(sum(len(q['bands']) for q in detail['queries']), 18)
        for query in detail['queries']:
            self.assertGreater(query['native_visible_points'], 0)
            self.assertGreater(query['observed_contributor_points'], 0)
            self.assertEqual(query['observed_source_compatibility']['target_only'], query['observed_contributor_points'])
        np.testing.assert_array_equal(stored['native_indices'], lineage['native_indices'])

    def test_four_way_support_and_unobserved_are_not_free(self):
        queries = [dict(observed_contributor_points=0, native_visible_points=0) for _ in range(6)]
        self.assertEqual(support_partition(queries, 1), PARTITIONS[3])
        queries[4]['native_visible_points'] = 1
        self.assertEqual(support_partition(queries, 1), PARTITIONS[2])
        queries[4]['observed_contributor_points'] = 1
        self.assertEqual(support_partition(queries, 1), PARTITIONS[1])
        queries[1]['observed_contributor_points'] = 1
        self.assertEqual(support_partition(queries, 1), PARTITIONS[0])
        values, traces = simulate(np.full((192, 256), np.nan, np.float32), 'missing', boxes45())
        self.assertFalse(np.isfinite(values).any())
        self.assertEqual(len(contributor_lineage(traces)['native_indices']), 0)
        self.assertTrue(all(t['reason'] == 'INSUFFICIENT_HITS' for t in traces))

    def test_fixed_gate_retention_and_visible_timing_costs(self):
        saved = synthetic_saved_rows()
        details = synthetic_details(saved)
        rescues = [i for i, d in enumerate(details) if 'local_rescue' in d['cohort_tags']]
        self.assertEqual(len(rescues), 36)
        for position in (0, 3, 4, 7):
            d = details[rescues[position]]
            d['queries'][d['winner_query']]['observed_contributor_points'] = 0
            d['support_partition'] = support_partition(d['queries'], d['winner_query'])
        rows, report = make_report(saved, details)
        primary = report['opportunity']['privileged_winner']
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(primary['status'], 'PASS')
        self.assertEqual(primary['rescues_retained'], 32)
        self.assertEqual(primary['added_FP_removed'], 2)
        self.assertTrue(primary['A_preserved'])
        self.assertEqual(len(report['comparisons']['privileged_winner_vs_local']['lost_true_frames']), 4)
        self.assertTrue(any(e['delay_s'] and e['delay_s'] > 0 for e in report['comparisons']['privileged_winner_vs_local']['event_differences']))
        self.assertGreater(report['event_gaps']['privileged_winner']['total_internal_gap_frames'], 0)
        self.assertTrue(all(r['predictions']['privileged_winner']['unknown'] for r in rows))
        d = details[rescues[8]]
        d['queries'][d['winner_query']]['observed_contributor_points'] = 0
        d['support_partition'] = support_partition(d['queries'], d['winner_query'])
        _, failed = make_report(saved, details)
        self.assertEqual(failed['status'], 'PASS')
        self.assertEqual(failed['hypothesis_status'], 'FAIL')
        self.assertEqual(failed['opportunity']['privileged_winner']['rescues_retained'], 31)

    def test_other_centre_control_and_invalid_query_coverage(self):
        saved = synthetic_saved_rows()
        details = synthetic_details(saved)
        chosen = next(i for i, d in enumerate(details) if 'local_rescue' in d['cohort_tags'])
        d = details[chosen]
        winner = d['winner_query']
        other = 4 if winner == 1 else 1
        d['queries'][winner]['observed_contributor_points'] = 0
        d['queries'][other]['observed_contributor_points'] = 1
        d['support_partition'] = support_partition(d['queries'], winner)
        rows = diagnostic_readout_rows(saved, details)
        self.assertFalse(rows[chosen]['predictions']['privileged_winner']['alert'])
        self.assertTrue(rows[chosen]['predictions']['privileged_either']['alert'])
        self.assertEqual(d['support_partition'], PARTITIONS[1])
        saved[0]['query_valid'][0] = False
        _, report = make_report(saved, details)
        self.assertEqual(report['denominators']['frames'], 576)
        self.assertEqual(report['denominators']['invalid_query_labels'], 1)
        self.assertEqual(report['hypothesis_status'], 'NOT_EVALUABLE')


if __name__ == '__main__':
    unittest.main()
