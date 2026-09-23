import unittest

from stereo_adapt_audit_20260923 import grouped_pairs,negative_query_far_support


def rows(identifier, layout_predictions, truth=1., cohort='new_eval', episode='ep'):
    return [dict(cohort=cohort, panel='new_eval' if cohort=='new_eval' else 'mz101',
                 id=identifier, part='HEAD', width=width, family='thin', episode=episode,
                 arm=arm+suffix, truth=truth, pred=prediction)
            for width in (.24,.36,.48)
            for arm,prediction in zip(('baseline','ordinary','balanced'),layout_predictions)
            for suffix in ('','_union')]


class AuditGroupingTests(unittest.TestCase):
    def test_far_support_restricted_to_default_width_negatives(self):
        truth=[dict(cohort='new_eval',panel='new_eval',id='a',truth=dict(distance={
            'BODY':[None,1.,None],'HEAD':[1.,None,1.]}))]
        pixels=[dict(cohort='new_eval',panel='new_eval',id='a',part=part,arm=arm,
                     far_native_estimated_query=value)
                for part,arm,value in [('BODY','baseline',100),('HEAD','baseline',7),
                                       ('HEAD','ordinary',0),('HEAD','balanced',2)]]
        result=negative_query_far_support(truth,pixels)['rows']
        self.assertEqual(len(result),3)
        self.assertTrue(all(r['part']=='HEAD' and r['negative_queries']==1 for r in result))
        by_arm={r['arm']:r for r in result}
        self.assertEqual(by_arm['baseline']['far_native_estimated_query_pixels'],7)
        self.assertEqual(by_arm['ordinary']['negative_queries_with_far_support'],0)
        self.assertEqual(by_arm['balanced']['negative_queries_with_far_support'],1)

    def test_correlated_widths_remain_two_layout_groups(self):
        data=rows('a',(1.2,1.,1.)) + rows('b',(1.,None,1.))
        result=grouped_pairs(data,{'a':'layout_a','b':'layout_b'})
        family=result['baseline_to_ordinary']['families']['new_eval/thin']
        self.assertEqual(family,dict(groups=2,groups_net_improved=1,
            groups_net_regressed=1,groups_net_tied=0,corrected_5cm=3,regressed_5cm=3))
        self.assertEqual(result['ordinary_to_balanced']['groups'][1]['recovered_missing'],3)

    def test_historical_frames_group_by_episode(self):
        data=rows('a',(1.2,1.,1.),cohort='historical')+rows('b',(1.2,1.,1.),cohort='historical')
        result=grouped_pairs(data,{})['baseline_to_ordinary']
        self.assertEqual(len(result['groups']),1)
        self.assertEqual(result['groups'][0]['queries'],6)

    def test_no_contact_improvement_is_not_positive_accuracy(self):
        result=grouped_pairs(rows('a',(1.,None,None),truth=None),{'a':'layout_a'})
        group=result['baseline_to_ordinary']['groups'][0]
        self.assertEqual(group['removed_false_contact'],3)
        self.assertEqual(group['corrected_5cm'],0)

    def test_missing_arm_rejected(self):
        with self.assertRaises(KeyError):
            grouped_pairs(rows('a',(1.,1.,1.))[:-1],{'a':'layout_a'})


if __name__=='__main__':unittest.main()
