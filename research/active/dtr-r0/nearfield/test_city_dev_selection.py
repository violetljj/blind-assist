import json
import unittest

import numpy as np

from city_dev_selection import apply_thresholds, select_checkpoint, select_threshold


class CityDevSelectionTests(unittest.TestCase):
    def test_inclusive_ties_and_ten_percent_boundary(self):
        scores = [.8,.5]+[.5]+[.1]*9
        labels = [1,1]+[0]*10
        result = select_threshold(scores,labels,min_count=1)
        self.assertEqual(result['threshold'],.5)
        self.assertEqual([result[k] for k in ('TP','FP','FN','TN')],[2,1,0,9])
        self.assertEqual(result['FPR'],.1)

    def test_recall_tie_prefers_lower_fpr(self):
        result = select_threshold([.9,.8,.7]+[.1]*9,[1]+[0]*11,min_count=1)
        self.assertEqual(result['threshold'],.9)
        self.assertEqual(result['FP'],0)

    def test_all_negative_and_higher_threshold_tie(self):
        result = select_threshold([1.,1.],[1,0],min_count=1)
        self.assertEqual(result['threshold'],np.nextafter(1.,np.inf))
        self.assertEqual([result[k] for k in ('TP','FP','FN','TN')],[0,0,1,1])
        json.dumps(result,allow_nan=False)

    def test_unknown_excluded_and_coverage_explicit(self):
        result = select_threshold([.8,.1,1.],[1,0,-1],min_count=1)
        self.assertEqual(result['threshold'],.8)
        self.assertEqual(result['TP'],1)
        self.assertEqual(result['unknown_count'],1)
        self.assertEqual(result['known_fraction'],2/3)

    def test_float32_application_preserves_sentinel_and_repeats(self):
        scores = np.array([.7,.7],dtype=np.float32)
        result = select_threshold(scores,[1,0],min_count=1)
        threshold = result['threshold']
        self.assertGreater(threshold,float(scores.max()))
        self.assertFalse(apply_thresholds(scores,threshold).any())
        self.assertTrue(apply_thresholds(scores,float(scores.max())).all())
        pair = np.stack([scores,np.ones_like(scores)],axis=1)
        thresholds = [threshold,np.nextafter(1.,np.inf)]
        self.assertFalse(apply_thresholds(pair,thresholds).any())
        self.assertEqual(select_threshold(scores,[1,0],min_count=1),result)
        np.testing.assert_array_equal(apply_thresholds(pair,thresholds),
                                      apply_thresholds(pair.copy(),thresholds))

    def test_default_minimum_and_missing_classes_fail(self):
        for labels in ([1]*47+[0]*48,[1]*48+[0]*47,[1]*48,[-1]*100,[]):
            with self.subTest(labels=len(labels)),self.assertRaises(ValueError):
                select_threshold(np.zeros(len(labels)),labels)
        result = select_threshold([.8]*48+[.1]*48,[1]*48+[0]*48)
        self.assertEqual(result['positive_count'],48)
        self.assertEqual(result['recall'],1)

    def test_invalid_inputs_fail(self):
        for scores,labels in (([float('nan'),.1],[1,0]),([1.1,.1],[1,0]),
                              ([.8,.1],[2,0]),([[.8,.1]],[[1,0]])):
            with self.assertRaises(ValueError):
                select_threshold(scores,labels,min_count=1)
        for minimum in (0,-1,True,1.5):
            with self.assertRaises(ValueError):
                select_threshold([.8,.1],[1,0],min_count=minimum)

    def test_checkpoint_is_whole_model_and_fixed_tie_order(self):
        labels = np.array([[1,1],[1,1],[0,0],[0,0]])
        perfect = np.array([[.9,.9],[.8,.8],[.1,.1],[.1,.1]])
        a = perfect.copy(); a[1,1] = .05  # min recall .5, macro .75
        b = perfect.copy(); b[1,0] = .05  # same; choose A on full tie
        c = perfect.copy(); c[:2,:] = .05
        result = select_checkpoint(dict(A=a,B=b,C=c),labels,min_count=1)
        self.assertEqual(result['selected_arm'],'A')
        self.assertEqual(result['arms']['A']['minimum_head_recall'],.5)
        self.assertEqual(result['selected_thresholds']['HEAD'],.9)
        # Perfect C beats both; do not create a hypothetical perfect A-body/B-head model.
        result = select_checkpoint(dict(A=a,B=b,C=perfect),labels,min_count=1)
        self.assertEqual(result['selected_arm'],'C')
        json.dumps(result,allow_nan=False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
