import json
import unittest
import numpy as np
from mz77_sequence_metrics import analyze


class SequenceMetricsTests(unittest.TestCase):
    def setUp(self):
        self.a = dict(clip=np.array(['test']*40), index=np.arange(40), time_s=np.arange(40)/10,
                      truth=np.zeros((40,4),bool), known=np.ones((40,4),bool),
                      target_truth=np.zeros((40,4),bool), front_distance_m=5-np.arange(40)/10)
        self.p = {c:{'DIVERSE':np.full((40,4), -1.)} for c in ('CLEAN','CENTER_GAP')}

    def result(self):
        result = analyze(self.a,self.p)
        json.dumps(result, allow_nan=False)
        return result['clips']['test']

    def metric(self, q='BODY_NEAR', condition='CLEAN', scope='full_scene'):
        return self.result()['conditions'][condition]['methods']['DIVERSE'][scope][q]

    def test_missed_episode_and_boundary_censoring(self):
        self.a['truth'][:5,0] = True
        self.a['truth'][35:,0] = True
        m = self.metric()
        self.assertEqual(m['confusion'],dict(TP=0,FN=10,FP=0,TN=30))
        self.assertEqual(m['missed_episodes'],2)
        self.assertTrue(m['episodes'][0]['left_censored'])
        self.assertTrue(m['episodes'][1]['right_censored'])
        self.assertIsNone(m['episodes'][0]['nominal_delay_s'])
        self.assertEqual(m['longest_silence_positive_samples'],5)

    def test_preexisting_false_alert_is_not_new_onset(self):
        self.a['truth'][10:15,0] = True
        self.p['CLEAN']['DIVERSE'][8:13,0] = 0
        m = self.metric(); ep=m['episodes'][0]
        self.assertEqual(m['confusion']['FP'],2)
        self.assertEqual(ep['onset_status'],'PREEXISTING_ALERT')
        self.assertEqual(ep['first_true_hit']['index'],10)
        self.assertAlmostEqual(ep['first_true_hit']['front_distance_m'],4)
        self.assertEqual(m['alert_on_transitions'],1)
        self.assertEqual(m['alert_off_transitions'],1)

    def test_union_prevents_wrong_range_missing_warning(self):
        self.a['truth'][5:10,1] = True
        self.p['CLEAN']['DIVERSE'][5:10,0] = 1
        self.assertEqual(self.metric('BODY_FAR')['confusion']['FN'],5)
        self.assertEqual(self.metric('BODY_NEAR')['confusion']['FP'],5)
        self.assertEqual(self.metric('BODY_ANY')['confusion']['TP'],5)
        target = self.metric('BODY_ANY',scope='intended_target')
        self.assertEqual(target['positive_samples'],0)
        self.assertEqual(target['activations_without_target_support'],5)
        self.assertNotIn('confusion',target)

    def test_unknowns_excluded_and_union_known_if_positive_member(self):
        self.a['known'][5:8,0] = False
        self.a['truth'][5:8,1] = True
        self.p['CLEAN']['DIVERSE'][5:8,0] = 1
        m = self.metric()
        self.assertEqual(m['unknown_alert_samples'],3)
        self.assertEqual(sum(m['confusion'].values()),37)
        self.assertEqual(self.metric('BODY_ANY')['confusion']['TP'],3)
        self.a['truth'][5:8,1] = False
        self.assertEqual(self.metric('BODY_ANY')['unknown_samples'],3)

    def test_gap_dropout_and_delayed_recovery(self):
        self.a['truth'][15:,2] = True
        for c in self.p:
            self.p[c]['DIVERSE'][15:,2] = 1
        self.p['CENTER_GAP']['DIVERSE'][20:25,2] = -1
        w = self.result()['gap_comparisons']['DIVERSE'][0]
        q = w['queries']['HEAD_NEAR']
        self.assertEqual(q['lost_true_indices'],[20,21,22])
        self.assertEqual(w['head_near_miss_indices'],[20,21,22])
        self.assertEqual(q['recovery']['first_recovered_index'],25)
        self.assertEqual(q['recovery']['samples_after_packet_return'],2)

    def test_persistent_silence_is_not_recovery_and_later_episode_excluded(self):
        self.a['truth'][15:,0] = True
        self.p['CLEAN']['DIVERSE'][15:,0] = 1
        q = self.result()['gap_comparisons']['DIVERSE'][0]['queries']['BODY_NEAR']
        self.assertEqual(q['recovery']['status'],'NOT_EVALUABLE_NO_PRIOR_WARNING')
        self.p['CENTER_GAP']['DIVERSE'][15:20,0] = 1
        q = self.result()['gap_comparisons']['DIVERSE'][0]['queries']['BODY_NEAR']
        self.assertEqual(q['recovery']['status'],'RIGHT_CENSORED')
        self.a['truth'][24:27,0] = False
        self.p['CENTER_GAP']['DIVERSE'][28:,0] = 1
        q = self.result()['gap_comparisons']['DIVERSE'][0]['queries']['BODY_NEAR']
        self.assertEqual(q['recovery']['status'],'POSITIVE_EPISODE_ENDED')
        self.assertIsNone(q['recovery']['first_recovered_index'])

    def test_noop_packet_gap_not_evaluable(self):
        self.a['packets'] = {c:dict(valid=np.zeros((40,64,2),bool),ranges=np.zeros((40,64,2))) for c in self.p}
        r = self.result()
        self.assertEqual(r['gap_pressure'][0]['status'],'NOT_EVALUABLE_NO_VALID_RETURN_REMOVED')
        self.assertEqual(r['gap_comparisons']['DIVERSE'][0]['queries']['BODY_ANY']['lost_true_indices'],[])

    def test_confusion_matches_independent_scalar_loop(self):
        rng=np.random.default_rng(7)
        self.a['known'] = rng.random((40,4))>.2
        self.a['truth'] = (rng.random((40,4))>.5) & self.a['known']
        self.p['CLEAN']['DIVERSE'] = rng.normal(size=(40,4))
        expected = dict(TP=0,FP=0,FN=0,TN=0)
        for i in range(40):
            if self.a['known'][i,0]:
                positive=self.a['truth'][i,0]; alert=self.p['CLEAN']['DIVERSE'][i,0]>=0
                expected[('TP' if positive else 'FP') if alert else ('FN' if positive else 'TN')] += 1
        self.assertEqual(self.metric()['confusion'],expected)


if __name__ == '__main__':
    unittest.main()
