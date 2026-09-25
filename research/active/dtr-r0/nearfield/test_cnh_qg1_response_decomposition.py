"""Synthetic-only checks before frozen Development response evaluation."""
import unittest

import numpy as np

from cnh_route_sensor import angular_rays, synthesize_response, derive_readout, H3, RAW_BINS
from cnh_qg1_response_decomposition import (
    COMPONENTS, PARAMETERS, aggregate, exchange, frame_stages, raw_stage, checked_identity,
    metric_bundle, scale_diagnostic,
)


class ResponseDecompositionTest(unittest.TestCase):
    def geometry(self):
        _, weights = angular_rays(4)
        radial = np.linspace(.03, 6., weights.size).reshape(weights.shape)
        radial[0, 0, 0] = np.nan
        radial[0, 1, 0] = -1
        return radial, weights

    def test_full_endpoint_exactly_matches_actual_simulator(self):
        radial, weights = self.geometry()
        for seed in (1, 2**63 + 17):
            actual = synthesize_response(radial, .5, 1., weights, seed=seed)
            recreated = raw_stage(radial, weights, COMPONENTS, seed)
            np.testing.assert_array_equal(recreated, actual['histogram'])
            np.testing.assert_array_equal(aggregate(recreated), derive_readout(actual, H3)['histogram'])
            np.testing.assert_array_equal(aggregate(recreated).astype(np.float32),
                                          derive_readout(actual, H3)['histogram'].astype(np.float32))

    def test_neighbor_energy_and_nonwrapping(self):
        raw = np.zeros((8, 8, RAW_BINS))
        raw[0, 0, 7] = 100
        result = exchange(raw, .02)
        self.assertAlmostEqual(float(result.sum()), 100)
        self.assertEqual(result[7, 0, 7], 0)
        self.assertEqual(result[0, 7, 7], 0)
        self.assertEqual(result[0, 0, 7], 99)
        self.assertEqual(result[1, 0, 7], .5)

    def test_area_keeps_invalid_and_outside_ray_denominator(self):
        _, weights = angular_rays(2)
        radial = np.full_like(weights, 1.)
        radial[..., 0] = np.nan
        radial[..., 1] = 6.
        raw = raw_stage(radial, weights, (), 1)
        expected = PARAMETERS.signal_counts * weights[..., 2:].sum(-1) / weights.sum(-1)
        np.testing.assert_allclose(raw.sum(-1), expected, rtol=1e-14)

    def test_stage_membership_and_paired_noise(self):
        radial, weights = self.geometry()
        stages = frame_stages(radial, weights, 123)
        self.assertEqual(len(stages), 11)
        np.testing.assert_array_equal(stages['RETURN_LAW'], stages['ONLY_RETURN_LAW'])
        np.testing.assert_array_equal(stages['SHOT_BACKGROUND'], aggregate(raw_stage(radial, weights, COMPONENTS, 123)))
        self.assertTrue(np.any(stages['SHOT_BACKGROUND'] < 0))

    def test_identity_rejects_seed_or_hash_drift(self):
        from cnh_street_e2e_materialize import frame_identity
        manifest = dict(source_manifest_sha256='a' * 64)
        row = dict(id='frame-001', camera_sha256='b' * 64, depth_sha256='c' * 64, depth_valid_sha256='d' * 64)
        key, seed = frame_identity(manifest['source_manifest_sha256'], row, row['camera_sha256'], row['depth_sha256'])
        row.update(frame_key=key, seed=seed)
        frame = dict(frame_id=row['id'], frame_key=key, original_files={
            'camera.json': dict(sha256=row['camera_sha256']),
            'depth_left.exr': dict(sha256=row['depth_sha256']),
            'depth_left_valid.npy': dict(sha256=row['depth_valid_sha256'])})
        self.assertEqual(checked_identity(manifest, row, frame), seed)
        row['seed'] += 1
        with self.assertRaises(ValueError):
            checked_identity(manifest, row, frame)

    def test_scale_diagnostic_preserves_scores_and_reports_rounding(self):
        # Adjacent float64 values can merge under a non-power-of-two multiplier.
        values = np.arange(48, dtype=np.float64) * np.spacing(1.5) + 1.5
        scores = values.reshape(8, 6)
        labels = (np.arange(48).reshape(8, 6) % 3 == 0).astype(np.int8)
        train = np.arange(8) < 4
        dev = ~train
        layouts = np.where(train, 'train', 'dev')
        original = scores.copy()
        baseline = metric_bundle(labels, scores, train, dev, layouts)
        report = scale_diagnostic(labels, scores, 7., train, dev, layouts, baseline)
        self.assertIn('numerical_invariance_verified', report)
        self.assertEqual(set(report['splits']['train']['per_query']), set(map(str, range(6))))
        self.assertGreater(report['splits']['train']['pooled']['adjacent_original_order_ties_changed'], 0)
        self.assertFalse(report['numerical_invariance_verified'])
        binary_scale = scale_diagnostic(labels, scores, .5, train, dev, layouts, baseline)
        self.assertTrue(binary_scale['numerical_invariance_verified'])
        np.testing.assert_array_equal(scores, original)


if __name__ == '__main__':
    unittest.main()
