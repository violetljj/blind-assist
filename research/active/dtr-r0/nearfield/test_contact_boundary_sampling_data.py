"""Synthetic CPU sampler checks: TASK_NOT_GPU_SUITABLE; no captured data."""
from __future__ import annotations

import json
import unittest

import numpy as np

from contact_boundary_sampling_data import critical_slice, sample_queries


def world_oracle(world_boxes, camera, query):
    result = []
    for width, horizon, layer in np.asarray(query, np.float64):
        yl, yh = ((.42, .9), (-.2, .42))[int(layer)]
        low = np.array([camera[0]+.3, camera[1]-width/2, camera[2]-yh])
        high = np.array([camera[0]+horizon, camera[1]+width/2, camera[2]-yl])
        result.append(any(np.all(np.maximum(lo, low) <= np.minimum(hi, high))
                          for lo, hi in world_boxes))
    return np.array(result)


def to_camera_boxes(world_boxes, camera):
    # Convert only in the test fixture; oracle above remains in world axes.
    result = []
    for lo, hi in world_boxes:
        result.append((np.array([lo[1]-camera[1], camera[2]-hi[2], lo[0]-camera[0]]),
                       np.array([hi[1]-camera[1], camera[2]-lo[2], hi[0]-camera[0]])))
    return result


class SamplingTests(unittest.TestCase):
    def test_world_slice_boundaries_and_all_object_minimum(self):
        camera = np.array([8., -2., 1.82])
        # World X forward, Y lateral, Z up. Include nearer/farther and both sides.
        world = [(np.array([9.2, -1.7, 1.12]), np.array([9.4, -1.4, 1.32])),
                 (np.array([8.9, -2.6, 1.12]), np.array([9.1, -2.2, 1.32])),
                 (np.array([8.4, -2.1, 1.8]), np.array([8.6, -1.9, 1.9]))]
        boxes = to_camera_boxes(world, camera)
        self.assertAlmostEqual(critical_slice(boxes, 0, 'width', 1.), .4)
        self.assertAlmostEqual(critical_slice(boxes, 0, 'width', 1.5), .4)
        self.assertAlmostEqual(critical_slice(boxes, 0, 'horizon', .5), .9)
        self.assertTrue(np.isinf(critical_slice(boxes, 0, 'horizon', .3)))
        self.assertEqual(critical_slice(boxes, 1, 'width', 1.), 0.)
        for layer in (0, 1):
            for kind, fixed_values in (('width', [.7, 1., 1.6, 2.8]),
                                       ('horizon', [.4, .7, 1.])):
                for fixed in fixed_values:
                    boundary = critical_slice(boxes, layer, kind, fixed)
                    if not np.isfinite(boundary) or boundary <= 0:
                        continue
                    axis = 0 if kind == 'width' else 1
                    q = np.array([[fixed, fixed, layer], [fixed, fixed, layer]])
                    q[:, axis] = [boundary-1e-5, boundary+1e-5]
                    np.testing.assert_array_equal(world_oracle(world, camera, q), [False, True])

    def test_boundary_pairs_opposite_labels_and_strict_domain(self):
        boxes = [(np.array([.3, .5, 1.2]), np.array([.5, .7, 1.4])),
                 (np.array([-.5, -.1, 1.8]), np.array([-.3, .1, 2.]))]
        for seed in range(6):
            q, labels, receipt = sample_queries(boxes, seed)
            self.assertEqual(q.shape, (72, 3))
            self.assertEqual(q.dtype, np.float32)
            self.assertEqual(labels.shape, (72,))
            self.assertEqual(labels.dtype, np.bool_)
            self.assertEqual(np.count_nonzero(q[:, 2] == 0), 36)
            self.assertEqual(np.count_nonzero(q[:, 2] == 1), 36)
            self.assertEqual(receipt['boundary_queries'] + receipt['fallback_queries'], 36)
            self.assertGreater(receipt['boundary_pairs'], 0)
            self.assertEqual(sum(r['kind'] == 'width' for r in receipt['pairs']), 9)
            self.assertEqual(sum(r['kind'] == 'horizon' for r in receipt['pairs']), 9)
            self.assertLessEqual(receipt['attempted_slices'], 144)
            for item in receipt['pairs']:
                self.assertGreaterEqual(item['attempts'], 1)
                self.assertLessEqual(item['attempts'], 8)
                if item['accepted']:
                    pair = q[item['rows']]
                    np.testing.assert_array_equal(labels[item['rows']], [False, True])
                    self.assertTrue((pair[:, :2] > [.36, .6]).all())
                    self.assertTrue((pair[:, :2] < [1.08, 3.]).all())
                    axis = 0 if item['kind'] == 'width' else 1
                    np.testing.assert_allclose(pair[:, axis],
                        [item['critical_value']-item['offset_m'], item['critical_value']+item['offset_m']], atol=1e-7)
                    self.assertEqual(pair[0, 1-axis], pair[1, 1-axis])

    def test_no_hit_fallback_is_uniform_bounded_and_truthful(self):
        q, labels, receipt = sample_queries([], 202609231)
        self.assertEqual(receipt['boundary_pairs'], 0)
        self.assertEqual(receipt['fallback_queries'], 36)
        self.assertEqual(receipt['attempted_slices'], 144)
        self.assertFalse(labels.any())
        self.assertTrue((q[:, :2] >= [.36, .6]).all())
        self.assertTrue((q[:, :2] <= [1.08, 3.]).all())
        self.assertEqual(len(np.unique(q[:, :2], axis=0)), 72)
        self.assertTrue(all(r['critical_value'] is None and r['fixed_value'] is None
                            and r['fallback_reason'] for r in receipt['pairs']))
        self.assertEqual(receipt['per_layer']['0']['fallback_pairs'], 9)
        self.assertEqual(receipt['per_layer']['1']['fallback_pairs'], 9)

    def test_outside_admissible_boundary_falls_back_without_clipping(self):
        # Both layers permanently occupied inside the sampled domain; boundary
        # width 0 and horizon .3 must not be moved into the training domain.
        boxes = [(np.array([-.1, -.2, .1]), np.array([.1, .9, .5]))]
        _, labels, receipt = sample_queries(boxes, 9)
        self.assertTrue(labels.all())
        self.assertEqual(receipt['boundary_pairs'], 0)
        self.assertEqual(receipt['fallback_queries'], 36)

    def test_determinism_separate_metadata_and_seed_variation(self):
        boxes = [(np.array([.3, .5, 1.2]), np.array([.5, .7, 1.4]))]
        q1, y1, r1 = sample_queries(boxes, 70)
        q2, y2, r2 = sample_queries(boxes, 70)
        np.testing.assert_array_equal(q1, q2)
        np.testing.assert_array_equal(y1, y2)
        self.assertEqual(r1, r2)
        self.assertNotEqual(q1.tolist(), sample_queries(boxes, 71)[0].tolist())
        json.dumps(r1, allow_nan=False)
        # The model-facing object is exactly numeric w/h/layer, with no appended
        # critical value, type, material, ID, truth or receipt fields.
        self.assertEqual(q1.shape[1], 3)
        self.assertIsNone(q1.dtype.names)
        self.assertIn('NOT_MODEL_FEATURES', r1['authority'])

    def test_random_world_geometry_labels_and_slice_transitions(self):
        rng = np.random.default_rng(2409)
        for index in range(20):
            camera = rng.uniform(-3., 3., 3)
            world = []
            for _ in range(4):
                center = camera + rng.uniform([.5, -.8, -.8], [3.5, .8, .2])
                extent = rng.uniform(.02, .15, 3)
                world.append((center-extent, center+extent))
            q, labels, receipt = sample_queries(to_camera_boxes(world, camera), index)
            np.testing.assert_array_equal(labels, world_oracle(world, camera, q))
            for pair in receipt['pairs']:
                if pair['accepted']:
                    np.testing.assert_array_equal(world_oracle(world, camera, q[pair['rows']]), [False, True])

    def test_bad_geometry_and_seed_rejected(self):
        with self.assertRaises(ValueError):
            sample_queries([([1, 0, 0], [0, 1, 1])], 0)
        with self.assertRaises(ValueError):
            sample_queries([([np.nan, 0, 0], [1, 1, 1])], 0)
        with self.assertRaises(ValueError):
            sample_queries([], -1)
        with self.assertRaises(ValueError):
            critical_slice([], 2, 'width', 1.)


if __name__ == '__main__':
    unittest.main()
