"""Focused checks for natural-target visibility and UNKNOWN boundaries."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import evaluate_city_native_route as route


class RouteEvaluationTest(unittest.TestCase):
    def test_native_pooling_positive_then_unknown(self):
        mask = np.zeros((1, 2, 360, 640), np.int8)
        mask[0, 0, 0, 0] = -1
        mask[0, 0, 0, 1] = 1
        mask[0, 1, 0, 0] = -1
        result = route.pooled(mask)
        self.assertEqual(result[0, 0, 0, 0], 1)
        self.assertEqual(result[0, 1, 0, 0], -1)
        self.assertEqual(result[0, 1, 1, 1], 0)

    def test_absent_labels_unknown(self):
        near, support, info = route.labels(None, [0, 1])
        self.assertTrue((near == -1).all())
        self.assertTrue((support == -1).all())
        self.assertEqual(info['status'], 'UNKNOWN')

    def target(self, status='EVALUABLE'):
        masks = np.full((2, 2, 18, 32), -1, np.int8)
        masks[0, 0, 4, 4] = 1
        pred = {}
        config = {}
        for name in route.METHODS:
            pred[name + '_near'] = np.zeros((2, 2))
            pred[name + '_support'] = np.zeros((2, 2, 18, 32))
            config[name] = dict(near_thresholds=[dict(value=.5), dict(value=.5)])
        data = dict(targets=[dict(target_id='tree', category='pole', status=status,
            authority='independent collision and isolated native-depth validation',
            frame_indices=[0, 1], masks='mask.npy')])
        with patch.object(route, 'read', return_value=data), \
             patch.object(route, 'within', return_value=Path(__file__)), \
             patch.object(route, 'sha', return_value='test'), \
             patch.object(route.np, 'load', return_value=masks):
            return route.target_metrics(Path(__file__), [0, 1], pred, config)[0]

    def test_unknown_frame_not_a_false_negative(self):
        result = self.target()
        body = result['methods']['g10']['BODY']
        self.assertEqual(body['alert_FN'], 1)
        self.assertEqual(body['support_overlap_misses'], 1)
        self.assertEqual(body['unknown_or_no_positive_frames'], 1)
        self.assertIsNone(body['false_alerts'])
        self.assertEqual(result['methods']['g10']['HEAD']['alert_FN'], 0)

    def test_unreliable_target_not_scored(self):
        result = self.target('UNKNOWN')
        self.assertEqual(result['evaluation'], 'UNKNOWN')
        self.assertEqual(result['methods'], {})


if __name__ == '__main__':
    unittest.main()
