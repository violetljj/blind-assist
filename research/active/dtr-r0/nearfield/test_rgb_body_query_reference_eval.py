"""Focused reference-label checks; no model, dataset or deployment assertions."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rgb_body_query_geometry import camera_queries
from rgb_body_query_reference_eval import (
    IGNORE, UNKNOWN, NEGATIVE, POSITIVE, MIN_SUPPORT,
    confusion, evaluate, interval_labels, ray_interval, ratios, sha,
)


class ReferenceEvaluationTests(unittest.TestCase):
    def test_signed_and_zero_rays_have_physical_closed_intervals(self):
        rx = np.array([[-1., -.5, 0., .5, 1.]])
        q = dict(name='left', low=[-2., -1., 1.], high=[-1., 1., 4.])
        low, high, reachable = ray_interval(rx, np.zeros_like(rx), q)
        np.testing.assert_allclose(low[0, :2], [1., 2.])
        np.testing.assert_allclose(high[0, :2], [2., 4.])
        np.testing.assert_array_equal(reachable, [[True, True, False, False, False]])

    def test_shared_depth_boundary_is_positive_in_both_closed_boxes(self):
        z = np.full((1, MIN_SUPPORT), 3.)
        target = np.ones(z.shape, bool)
        zero = np.zeros_like(z)
        for query in (camera_queries()[0], camera_queries()[3]):
            label, record = interval_labels(z, z, target, zero, zero, query)
            self.assertTrue(np.all(label == POSITIVE))
            self.assertEqual(record['state'], 'POSITIVE')

    def test_missing_disagreement_and_interval_crossing_remain_unknown(self):
        q = dict(name='small', low=[-.1, -.1, 1.], high=[.1, .1, 1.2])
        cres = np.array([[1.1, 1.1, 1.1, 3., 1.1]])
        zed = np.array([[np.nan, 2., 1.3, 3.1, 1.1]])
        target = np.array([[True, True, True, True, False]])
        label, record = interval_labels(cres, zed, target, np.zeros_like(cres), np.zeros_like(cres), q)
        np.testing.assert_array_equal(label, [[UNKNOWN, UNKNOWN, UNKNOWN, NEGATIVE, IGNORE]])
        self.assertEqual(record['state'], 'UNKNOWN')

    def test_presence_threshold_and_empty_domain_are_distinct(self):
        z = np.full((1, MIN_SUPPORT), 1.)
        zero = np.zeros_like(z)
        q = camera_queries()[0]
        target = np.ones(z.shape, bool)
        target[0, -1] = False
        _, below = interval_labels(z, z, target, zero, zero, q)
        self.assertEqual(below['state'], 'UNKNOWN')
        target[:] = True
        _, positive = interval_labels(z, z, target, zero, zero, q)
        self.assertEqual(positive['state'], 'POSITIVE')
        _, empty = interval_labels(z, z, np.zeros_like(target), zero, zero, q)
        self.assertEqual(empty['state'], 'NEGATIVE_VISIBLE_TARGET_SET')
        self.assertTrue(empty['vacuous_negative'])
        self.assertEqual(empty['domain_pixels'], 0)
        outside = np.full(z.shape, 4.)
        label, negative = interval_labels(outside, outside, target, zero, zero, q)
        self.assertTrue(np.all(label == NEGATIVE))
        self.assertEqual(negative['state'], 'NEGATIVE_VISIBLE_TARGET_SET')
        self.assertFalse(negative['vacuous_negative'])
        self.assertEqual(negative['domain_pixels'], MIN_SUPPORT)

    def test_unknown_and_ignore_never_become_false_positives_or_true_negatives(self):
        labels = np.array([[POSITIVE, POSITIVE, NEGATIVE, NEGATIVE, UNKNOWN, IGNORE]], np.uint8)
        predicted = np.array([[True, False, True, False, True, False]])
        counts = confusion(labels, predicted)
        self.assertEqual(counts, dict(tp=1, fn=1, fp=1, tn=1))
        self.assertEqual(ratios(dict(tp=0, fn=0, fp=0, tn=0)), dict(iou=None, recall=None, fpr=None))

    def test_evaluate_excludes_unknown_and_vacuous_queries_and_names_oracle(self):
        # Retain tiny fixture evidence under ignored canonical artifacts.
        repo = Path(__file__).resolve().parents[4]
        base = repo/'artifacts.local/work/rgb-body-query-eval-dev-20261009/audit'
        base.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix='reference-fixture-', dir=base))
        k = np.array([[1000., 0., 7.5], [0., 1000., 0.], [0., 0., 1.]])
        target = np.zeros((2, 16), bool)
        target[0] = True
        labels = np.full((6, 2, 16), IGNORE, np.uint8)
        labels[0, 0] = NEGATIVE  # Nonvacuous target-domain negative.
        labels[3, 0] = UNKNOWN  # Predictor enters far box; reference unknown.
        ref_file = folder/'reference.npz'
        np.savez(ref_file, labels=labels, target=target, K=k)
        states = [dict(name=q['name'], state='UNKNOWN', vacuous_negative=False) for q in camera_queries()]
        states[0]['state'] = 'NEGATIVE_VISIBLE_TARGET_SET'
        states[1].update(state='NEGATIVE_VISIBLE_TARGET_SET', vacuous_negative=True)
        ref = dict(status='COMPLETE', rows=[dict(frame=1, path=str(ref_file), sha256=sha(ref_file), rgb_sha256='fixture-rgb', annotation_type='HUMAN_ANNOTATED', queries=states)])
        (folder/'reference_manifest.json').write_text(json.dumps(ref), 'utf-8')
        pred_file = folder/'prediction.npz'
        # Prediction only reaches near box on NON-target second row.
        np.savez(pred_file, depth=np.array([[4.]*16, [1.]*16]), valid_mask=np.ones((2, 16), bool))
        pred = dict(status='COMPLETE', rows=[dict(frame=1, arm='fixture', depth_path=str(pred_file), depth_sha256=sha(pred_file), rgb_sha256='fixture-rgb', public_K=k.tolist())])
        pred_manifest = folder/'predictions.json'
        pred_manifest.write_text(json.dumps(pred), 'utf-8')
        output = folder/'evaluation.json'
        with contextlib.redirect_stdout(io.StringIO()):
            evaluate(folder, [pred_manifest], output)
        result = json.loads(output.read_text('utf-8'))
        summary = result['summaries']['fixture']
        self.assertEqual(summary['pixel_confusion'], dict(tp=0, fn=0, fp=0, tn=16))
        self.assertEqual(summary['scoreable_query_frames'], 1)
        self.assertEqual(summary['excluded_vacuous_query_negatives'], 1)
        self.assertEqual(summary['oracle_region_query_confusion'], dict(tp=0, fn=0, fp=0, tn=1))
        self.assertEqual(result['matched_comparison']['frames'], [1])
        self.assertEqual(result['matched_comparison']['summaries']['fixture'], summary)
        near = next(r for r in result['rows'] if r['query'] == 'center_near')
        self.assertEqual(near['observation_support_pixels'], 16)
        self.assertEqual(near['oracle_region_support_pixels'], 0)
        self.assertFalse(near['oracle_region_prediction'])
        far = next(r for r in result['rows'] if r['query'] == 'center_far')
        self.assertTrue(far['oracle_region_prediction'])
        self.assertEqual(far['reference_state'], 'UNKNOWN')
        self.assertEqual(far['labeled_reference_pixels'], 0)


if __name__ == '__main__':
    unittest.main()
