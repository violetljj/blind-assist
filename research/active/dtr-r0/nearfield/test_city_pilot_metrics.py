"""Hand-calculated fixed-threshold City metric checks."""
import json
import unittest

import numpy as np

from city_pilot_metrics import evaluate


class CityPilotMetricsTests(unittest.TestCase):
    def inputs(self, n):
        return (np.zeros((n,2)), np.zeros((n,2,18,32)),
                np.zeros((n,2),dtype=np.int8), np.zeros((n,2,18,32),dtype=np.int8))

    def test_inclusive_confusion_and_balanced_error(self):
        p, s, y, m = self.inputs(4)
        y[:] = [[1,1],[1,0],[0,1],[0,0]]
        p[:] = [[.5,.5],[.49,.5],[.5,.49],[.49,.49]]
        r = evaluate(p,s,y,m)
        for head in r['heads'].values():
            near = head['near']
            self.assertEqual([near[k] for k in ('TP','FP','FN','TN')],[1,1,1,1])
            for key in ('recall','precision','specificity','balanced_accuracy'):
                self.assertEqual(near[key],.5)
        self.assertEqual(r['near_balanced_error'],.5)
        json.dumps(r,allow_nan=False)

    def test_unknown_near_is_not_positive_or_negative(self):
        p,s,y,m = self.inputs(2)
        y[:] = [[-1,-1],[0,0]]
        p[0] = 1
        r = evaluate(p,s,y,m)
        for head in r['heads'].values():
            self.assertEqual(head['near']['FP'],0)
            self.assertEqual(head['near']['unknown_fraction'],.5)
            self.assertIsNone(head['near']['recall'])
            self.assertIsNone(head['near']['precision'])
            self.assertIsNone(head['near']['balanced_accuracy'])
        self.assertIsNone(r['near_balanced_error'])

    def test_iou_ignores_unknown_predictions_but_peak_does_not_move(self):
        p,s,y,m = self.inputs(2)
        y[:,0] = 1
        m[:,0] = -1
        m[:,0,0,:3] = [1,1,0]
        s[0,0,0,:4] = [.5,.1,.5,.99]  # IoU 1/3, peak on UNKNOWN.
        s[1,0,0,:4] = [.8,.6,.1,.2]   # IoU 1, positive peak.
        r = evaluate(p,s,y,m)['heads']['BODY']['support']
        self.assertAlmostEqual(r['positive_frame_mean_iou'],2/3)
        self.assertEqual(r['peak_hits'],1)
        self.assertEqual(r['peak_unknown_misses'],1)
        self.assertEqual(r['peak_known_misses'],0)
        self.assertEqual(r['peak_hit_rate'],.5)
        self.assertAlmostEqual(r['unknown_fraction'],573/576)

    def test_negative_false_positive_frame_and_pixel_denominators(self):
        p,s,y,m = self.inputs(3)
        m[:,0] = -1
        m[0,0,0,:2] = 0
        m[1,0,0,:2] = 0
        s[0,0,0,0] = .5
        s[1,0,0,2] = 1  # Unknown false alarm excluded.
        s[2,0] = 1      # Entirely unknown frame excluded.
        r = evaluate(p,s,y,m)['heads']['BODY']['support']
        self.assertEqual(r['negative_evaluable_frames'],2)
        self.assertEqual(r['negative_false_positive_mask_frames'],1)
        self.assertEqual(r['negative_false_positive_mask_rate'],.5)
        self.assertEqual(r['negative_false_positive_pixels'],1)
        self.assertEqual(r['negative_false_positive_pixel_rate'],.25)

    def test_no_known_support_is_not_perfect_iou(self):
        p,s,y,m = self.inputs(1)
        y[:] = 1
        m[:] = -1
        r = evaluate(p,s,y,m)['heads']['BODY']['support']
        self.assertEqual(r['positive_not_evaluable_frames'],1)
        self.assertIsNone(r['positive_frame_mean_iou'])
        self.assertIsNone(r['peak_hit_rate'])

    def test_group_joint_requires_all_frames_and_both_known_heads(self):
        p,s,y,m = self.inputs(4)
        p[1,1] = .5
        y[3,0] = -1
        r = evaluate(p,s,y,m,['a','a','b','c'])['group_joint']
        self.assertEqual(r['groups'],3)
        self.assertEqual(r['evaluable_groups'],2)
        self.assertEqual(r['unknown_groups'],1)
        self.assertEqual(r['all_correct_groups'],1)
        self.assertEqual(r['accuracy'],.5)

    def test_empty_is_json_safe_without_zero_denominator_substitution(self):
        r = evaluate(*self.inputs(0),group_ids=[])
        self.assertIsNone(r['near_balanced_error'])
        self.assertIsNone(r['group_joint']['accuracy'])
        json.dumps(r,allow_nan=False)

    def test_bad_inputs_rejected(self):
        for field,value in ((0,float('nan')),(1,1.1),(2,2),(3,-2)):
            args = list(self.inputs(1))
            args[field].flat[0] = value
            with self.subTest(field=field),self.assertRaises(ValueError):
                evaluate(*args)
        p,s,y,m = self.inputs(1)
        with self.assertRaises(ValueError):
            evaluate(p,s[:,:,:17],y,m)
        with self.assertRaises(ValueError):
            evaluate(p,s,y,m,[])


if __name__ == '__main__':
    unittest.main(verbosity=2)
