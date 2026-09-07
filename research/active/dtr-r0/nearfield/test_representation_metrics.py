"""Focused synthetic tests for G12 recall-first operating-point selection."""
import copy
import unittest
import numpy as np
from representation_evaluate import recall_thresholds

class RepresentationMetricsTest(unittest.TestCase):
    def setUp(self):
        self.key=('repvit','17','normal')
        self.samples=[dict(sample_id=str(i),split='val') for i in range(30)]
        self.targets={str(i):[1,1,-1,-1] if i<20 else [0,0,-1,-1] for i in range(30)}
        self.variants={str(i):'bar_only' if i<5 else 'both' for i in range(30)}
        self.rows={str(i):np.array([.8,.8,-1,-1]) if i<20 else np.array([.5,.5,-1,-1]) for i in range(30)}
        self.rows['0'][:2]=.1

    def select(self):
        return recall_thresholds({self.key:self.rows},self.samples,self.targets,self.variants)[self.key]

    def test_overall_recall_constraint_and_inclusive_tied_maximum(self):
        chosen=self.select()[0]
        self.assertEqual(chosen['value'],.8)
        self.assertEqual(chosen['constraints']['overall_positive']['required_TP'],19)
        self.assertEqual(chosen['constraints']['overall_positive']['actual_TP'],19)
        self.assertEqual(chosen['constraints']['overall_positive']['actual_recall'],.95)
        self.assertEqual(chosen['actual_FP'],0)
        above=np.nextafter(chosen['value'],float('inf'))
        self.assertLess(sum(self.rows[str(i)][0]>=above for i in range(20)),19)

    def test_head_bar_only_recall_prevents_subgroup_sacrifice(self):
        head=self.select()[1]
        self.assertEqual(head['value'],.1)
        self.assertEqual(head['constraints']['bar_only_positive']['positives'],5)
        self.assertEqual(head['constraints']['bar_only_positive']['required_TP'],5)
        self.assertEqual(head['constraints']['bar_only_positive']['actual_recall'],1.)
        self.assertEqual(head['constraints']['overall_positive']['actual_recall'],1.)
        self.assertEqual(head['actual_FP'],10)
        self.assertEqual(head['actual_FPR'],1.)

    def test_all_alarm_and_unit_threshold_boundaries_are_reported(self):
        for rows in self.rows.values():
            rows[:2]=0.
        for head in self.select()[:2]:
            self.assertEqual(head['value'],0.)
            self.assertEqual(head['actual_FP'],10)
            self.assertEqual(head['actual_FPR'],1.)
            self.assertEqual(head['constraints']['overall_positive']['actual_recall'],1.)
        for rows in self.rows.values():
            rows[:2]=1.
        for head in self.select()[:2]:
            self.assertEqual(head['value'],1.)
            self.assertEqual(head['actual_FP'],10)

    def test_unknown_missing_positive_and_absent_bar_coverage_rejected(self):
        original=copy.deepcopy(self.rows)
        self.rows['25'][0]=-1
        with self.assertRaisesRegex(ValueError,'prediction coverage'):
            self.select()
        self.rows=original;self.targets['25'][1]=-1
        with self.assertRaisesRegex(ValueError,'truth coverage'):
            self.select()
        self.targets['25'][1]=0
        self.variants={sid:'both' for sid in self.variants}
        with self.assertRaisesRegex(ValueError,'bar_only_positive'):
            self.select()
        for value in self.targets.values():
            value[0]=0
        with self.assertRaisesRegex(ValueError,'overall_positive'):
            self.select()

    def test_training_and_test_values_never_enter_selection(self):
        expected=self.select()
        # These IDs are deliberately absent from score/label/variant mappings.
        self.samples.extend([dict(sample_id='tr',split='train'),dict(sample_id='te',split='test')])
        self.assertEqual(expected,self.select())
        self.rows.update(tr=np.array([1.,1.,-1,-1]),te=np.array([0.,0.,-1,-1]))
        self.targets.update(tr=[0,0,-1,-1],te=[1,1,-1,-1])
        self.assertEqual(expected,self.select())

if __name__=='__main__':
    unittest.main()
