"""Only decision-changing invariants for the paired diagnostic."""
import unittest
import numpy as np
from head_paired_evaluate import assess, ranking


class PairedEvaluationTest(unittest.TestCase):
    def inputs(self):
        rows, y, logits, maps, support = [], [], [], [], []
        for role in ('train', 'eval'):
            for relation in ('CLEAR', 'HEAD_ONLY'):
                for appearance in ('A', 'L', 'M'):
                    positive = relation == 'HEAD_ONLY'
                    rows.append(dict(sample_index=len(rows), group_id=role+'-unit', source_partition=role,
                        family='fixture', relation=relation, appearance=appearance,
                        native_mask_sha256=relation, native_rgb_sha256=relation+appearance))
                    y.append([0, int(positive)])
                    logits.append([-4, 4 if positive else -4])
                    s = np.zeros((2,18,32), np.int8)
                    s[:,0,0] = -1
                    if positive: s[1,1,1] = 1
                    support.append(s)
                    maps.append(np.where(s == 1, .9, .1))
        z = np.array(logits, float)
        return 1/(1+np.exp(-z)), z, np.array(maps), np.array(y), np.array(support), rows, [.5,.5]

    def test_reversal_is_damage_not_relation_success(self):
        inputs = list(self.inputs())
        for i, row in enumerate(inputs[5]):
            if row['appearance'] == 'L':
                inputs[0][i,1] = 1-inputs[0][i,1]
                inputs[1][i,1] *= -1
        result = assess(*inputs)
        self.assertEqual(result['factors']['L']['damaged_units'], 2)
        self.assertTrue(result['appearance_failure_signal'])
        self.assertEqual(result['factors']['L']['positive_relation_B'], 0)
        self.assertEqual(result['factors']['M']['damaged_units'], 0)

    def test_constant_negative_is_not_success(self):
        inputs = list(self.inputs())
        inputs[0][:,1] = .1
        inputs[1][:,1] = -4
        result = assess(*inputs)
        self.assertFalse(result['appearance_failure_signal'])
        self.assertEqual(result['factors']['L']['four_HEAD_correct'], 0)
        self.assertEqual(result['factors']['L']['positive_relation_B'], 0)

    def test_unknown_or_changed_native_mask_never_admits_pair(self):
        inputs = list(self.inputs())
        for i, row in enumerate(inputs[5]):
            if row['appearance'] == 'L':
                row['native_mask_sha256'] = 'different'
                inputs[2][i,1,1,1] = .1
        result = assess(*inputs)
        self.assertEqual(result['factors']['L']['evaluable_units'], 0)
        self.assertEqual(result['factors']['L']['damaged_units'], 0)
        # Unknown maxima are recorded separately from known target hits.
        inputs[2][:,1,0,0] = 1.
        result = assess(*inputs)
        self.assertEqual(result['partitions']['train']['A']['HEAD']['peak_unknown'], 1)
        self.assertEqual(result['partitions']['train']['A']['HEAD']['peak_hits'], 0)

    def test_rank_ties_and_missing_class(self):
        self.assertEqual(ranking(np.array([.5,.5]),np.array([0,1])),dict(auc=.5,ap=.5))
        self.assertEqual(ranking(np.array([.2,.8]),np.array([0,1])),dict(auc=1.,ap=1.))
        self.assertEqual(ranking(np.array([.5]),np.array([0])),dict(auc=None,ap=None))


if __name__ == '__main__':
    unittest.main()
