"""Small synthetic contract checks; backend reason TASK_NOT_GPU_SUITABLE."""
from __future__ import annotations

import unittest

import numpy as np

from contact_boundary_data import (
    boundary_metrics, camera_boxes, choose_threshold, contact_labels, counts,
    critical_values, queries,
)


def world_slab_oracle(geometry, query):
    """Independent world-coordinate prism intersection, no label helper calls."""
    camera = geometry['declared_camera']
    hits, travel = [], []
    for width, horizon, layer in np.asarray(query, np.float64):
        low_y, high_y = ((.42, .9), (-.2, .42))[int(layer)]
        query_low = (camera['x'] + .3, camera['y'] - width / 2,
                     camera['z'] - high_y)
        query_high = (camera['x'] + horizon, camera['y'] + width / 2,
                      camera['z'] - low_y)
        first = np.inf
        for obj in geometry['objects']:
            center = obj['render_bounds_center_m']
            extent = obj['render_bounds_extent_m']
            lower = [max(center[a] - extent[a], query_low[a]) for a in range(3)]
            upper = [min(center[a] + extent[a], query_high[a]) for a in range(3)]
            if all(lower[a] <= upper[a] for a in range(3)):
                first = min(first, lower[0] - camera['x'] - .3)
        hits.append(np.isfinite(first))
        travel.append(first)
    return np.array(hits), np.array(travel)


def source_geometry(camera, objects):
    return dict(declared_camera=dict(zip(('x', 'y', 'z'), camera),
                                    pitch=0., yaw=0., roll=0.),
                actual_camera_location_m=list(camera), objects=objects)


def box(low, high):
    return np.array(low, np.float64), np.array(high, np.float64)


class ContactGeometryTests(unittest.TestCase):
    def test_random_world_slab_labels_and_first_hit(self):
        rng = np.random.default_rng(202609231)
        for _ in range(30):
            camera = rng.uniform(-5, 5, 3)
            objects = [dict(render_bounds_center_m=(camera + rng.uniform(
                [0., -1., -1.5], [4., 1., .5])).tolist(),
                render_bounds_extent_m=rng.uniform(.02, .6, 3).tolist())
                for _ in range(5)]
            geometry = source_geometry(camera, objects)
            query = np.column_stack((rng.uniform(.2, 1.2, 80),
                                     rng.uniform(.3, 3., 80), rng.integers(0, 2, 80)))
            expected_hit, expected_travel = world_slab_oracle(geometry, query)
            hit, travel = contact_labels(camera_boxes(geometry), query)
            np.testing.assert_array_equal(hit, expected_hit)
            np.testing.assert_allclose(travel, expected_travel, rtol=0, atol=3e-15)

    def test_closed_lateral_horizon_height_and_near_boundaries(self):
        # One box just touches the right edge, horizon and shared BODY/HEAD edge.
        boxes = [box([.25, .42, 1.], [.5, .42, 1.2])]
        query = np.array([[.5, 1., 0], [.5, 1., 1],
                          [.5 - 1e-7, 1., 0], [.5, 1. - 1e-7, 0]])
        hit, distance = contact_labels(boxes, query)
        np.testing.assert_array_equal(hit, [True, True, False, False])
        np.testing.assert_allclose(distance[:2], [.7, .7])
        self.assertTrue(np.isinf(distance[2:]).all())
        near = [box([-.1, .5, .1], [.1, .6, .3])]
        hit, distance = contact_labels(near, [[.6, .3, 0]])
        self.assertTrue(hit[0])
        self.assertEqual(distance[0], 0.)

    def test_full_extent_all_objects_and_nearest_not_list_order(self):
        # First object is farther. Near object's centre lies outside the query,
        # but its extent intersects; target-only or centre tests give wrong labels.
        boxes = [box([-.2, .5, 2.], [.2, .7, 2.3]),
                 box([.2, .5, .8], [.9, .7, 1.2]),
                 box([-.1, -.1, 1.4], [.1, .1, 1.8])]
        query = [[.6, 3., 0], [.6, 3., 1], [.2, .6, 0]]
        for order in (boxes, boxes[::-1]):
            hit, travel = contact_labels(order, query)
            np.testing.assert_array_equal(hit, [True, True, False])
            np.testing.assert_allclose(travel[:2], [.5, 1.1])
            self.assertTrue(np.isinf(travel[2]))

    def test_exact_critical_values_and_censoring(self):
        boxes = [box([.4, .5, 1.2], [.6, .7, 1.4]),
                 box([-.1, -.1, 3.3], [.1, .1, 3.5])]
        np.testing.assert_array_equal(critical_values(boxes, 'width'), [.8, np.inf])
        np.testing.assert_array_equal(critical_values(boxes, 'horizon'), [np.inf, 3.3])
        crossing = [box([-.1, .5, .1], [.1, .7, .4])]
        np.testing.assert_array_equal(critical_values(crossing, 'width'), [0., np.inf])
        np.testing.assert_array_equal(critical_values(crossing, 'horizon'), [.3, np.inf])

    def test_query_and_pose_domain_rejected(self):
        for query in ([[0., 1., 0]], [[1.3, 1., 0]], [[.6, .2, 0]],
                      [[.6, 3.1, 0]], [[.6, 1., .5]], [[np.nan, 1., 0]]):
            with self.subTest(query=query), self.assertRaises(ValueError):
                contact_labels([], query)
        geometry = source_geometry([0., 0., 0.], [])
        geometry['declared_camera']['yaw'] = .01
        with self.assertRaises(ValueError):
            camera_boxes(geometry)
        geometry['declared_camera']['yaw'] = 0.
        geometry['actual_camera_location_m'][0] = .002
        with self.assertRaises(ValueError):
            camera_boxes(geometry)


class QuerySplitTests(unittest.TestCase):
    def test_seen_and_novel_query_categories(self):
        grids = queries()
        expected = dict(seen=72, width=54, horizon=64, both=48, extrapolation=32,
                        width_curve=102, horizon_curve=110)
        for name, count in expected.items():
            self.assertEqual(grids[name].shape, (count, 3))
            self.assertEqual(len(set(map(tuple, grids[name]))), count)
            contact_labels([], grids[name])
        for left in ('seen', 'width', 'horizon', 'both', 'extrapolation'):
            for right in ('seen', 'width', 'horizon', 'both', 'extrapolation'):
                if left != right:
                    self.assertFalse(set(map(tuple, grids[left])) & set(map(tuple, grids[right])))
        seen_widths = set(grids['seen'][:, 0])
        seen_horizons = set(grids['seen'][:, 1])
        self.assertFalse(seen_widths & set(grids['both'][:, 0]))
        self.assertFalse(seen_horizons & set(grids['both'][:, 1]))
        self.assertEqual(seen_horizons, set(grids['width'][:, 1]))
        self.assertEqual(seen_widths, set(grids['horizon'][:, 0]))
        self.assertLess(grids['extrapolation'][:, 0].min(), min(seen_widths))
        self.assertGreater(grids['extrapolation'][:, 0].max(), max(seen_widths))


class ThresholdTests(unittest.TestCase):
    def test_tied_scores_are_atomic_and_negative_budget_enforced(self):
        scores = np.array([.9, .8, .8] + [.1] * 18)
        truth = np.array([True, True, False] + [False] * 18)
        threshold = choose_threshold(scores, truth)
        result = counts(scores, truth, threshold)
        self.assertEqual(threshold, .9)  # One FP / 19 negatives exceeds 5%.
        self.assertEqual((result['TP'], result['FP']), (1, 0))
        self.assertLessEqual(result['FPR'], .05)
        scores = np.array([.9, .8, .8] + [.1] * 19)
        truth = np.array([True, True, False] + [False] * 19)
        result = counts(scores, truth, choose_threshold(scores, truth))
        self.assertEqual((result['TP'], result['FP']), (2, 1))
        self.assertEqual(result['FPR'], .05)

    def test_zero_recall_tie_prefers_all_negative_over_false_positive(self):
        scores = np.array([.9] + [.1] * 20)
        truth = np.array([False] + [True] + [False] * 19)
        threshold = choose_threshold(scores, truth)
        result = counts(scores, truth, threshold)
        self.assertEqual((result['TP'], result['FP']), (0, 0))

    def test_all_negative_threshold_survives_float32_scores(self):
        scores = np.full(21, .5, np.float32)
        truth = np.array([True] + [False] * 20)
        threshold = choose_threshold(scores, truth)
        result = counts(scores, truth, threshold)
        self.assertEqual((result['TP'], result['FP']), (0, 0))
        self.assertLessEqual(result['FPR'], .05)

    def test_matches_exhaustive_atomic_feasible_candidates(self):
        rng = np.random.default_rng(17)
        for _ in range(40):
            scores = rng.integers(0, 12, 150).astype(np.float64) / 12
            truth = np.r_[np.ones(30, bool), np.zeros(120, bool)]
            rng.shuffle(truth)
            candidates = np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]
            outcomes = [counts(scores, truth, threshold) for threshold in candidates]
            feasible = [r for r in outcomes if r['FPR'] <= .05]
            expected = max((r['TP'], -r['FP']) for r in feasible)
            actual = counts(scores, truth, choose_threshold(scores, truth))
            self.assertEqual((actual['TP'], -actual['FP']), expected)


class BoundaryMetricTests(unittest.TestCase):
    def test_all_negative_cutoff_preserved_for_float32_boundary_curves(self):
        probability = np.full((1, 2, 51), .5, np.float32)
        threshold = float(np.nextafter(np.float64(.5), np.inf))
        result = boundary_metrics(probability, threshold, np.array([[.5, .8]]), 'width')
        self.assertEqual(result['predicted_boundary_coverage'], 0.)
        self.assertEqual(result['joint_within_5cm'], 0.)

    def test_missing_predictions_remain_in_joint_denominator(self):
        probability = np.zeros((1, 2, 51))
        probability[0, 0, 15:] = 1.  # Exact width .50; second boundary missing.
        result = boundary_metrics(probability, .5, np.array([[.5, .8]]), 'width')
        self.assertEqual(result['interior_true_boundaries'], 2)
        self.assertEqual(result['predicted_boundary_coverage'], .5)
        self.assertAlmostEqual(result['conditional_MAE_m'], 0.)
        self.assertEqual(result['within_5cm'], 1)
        self.assertEqual(result['joint_within_5cm'], .5)

    def test_left_right_censoring_and_reversals_are_reported(self):
        probability = np.zeros((2, 2, 51))
        probability[0, 0, :] = 1.
        probability[0, 1, 10:] = 1.  # Wrong crossing on right-censored truth.
        probability[1, 0, 20:30] = 1.
        probability[1, 0, 35:] = 1.  # First boundary .60 but one reversal.
        truth = np.array([[.1, np.inf], [.6, 1.5]])
        result = boundary_metrics(probability, .5, truth, 'width')
        self.assertEqual(result['left_censored_truth'], 1)
        self.assertEqual(result['right_censored_truth'], 2)
        self.assertEqual(result['interior_true_boundaries'], 1)
        self.assertEqual(result['wrong_crossings_on_right_censored'], 1)
        self.assertEqual(result['binary_reversals'], 1)
        self.assertEqual(result['probability_monotonic_violations'], 1)

    def test_horizon_all_missing_and_no_finite_boundary(self):
        probability = np.zeros((1, 2, 55))
        result = boundary_metrics(probability, .5, np.array([[1., 2.]]), 'horizon')
        self.assertEqual(result['predicted_boundary_coverage'], 0.)
        self.assertIsNone(result['conditional_MAE_m'])
        self.assertEqual(result['joint_within_5cm'], 0.)
        result = boundary_metrics(probability, .5, np.array([[.3, np.inf]]), 'horizon')
        self.assertEqual(result['interior_true_boundaries'], 0)
        self.assertIsNone(result['predicted_boundary_coverage'])
        self.assertIsNone(result['joint_within_5cm'])


if __name__ == '__main__':
    unittest.main()
