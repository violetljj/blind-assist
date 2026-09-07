"""Synthetic G9-A pairing, frozen-threshold and attribution denominator checks."""
import copy
import unittest
import numpy as np
from grounding_evaluate import paired_metrics, support_metrics, frozen_thresholds, group_metadata

class GroundingMetricsTest(unittest.TestCase):
    def setUp(self):
        self.variants=('both','bar_only','box_only','neither')
        self.rows={'both':[.9,.9,.1,.1],'bar_only':[.1,.9,.1,.1],
                   'box_only':[.9,.1,.1,.1],'neither':[.1,.1,.1,.1]}
        self.truth={'both':[1,1,-1,-1],'bar_only':[0,1,-1,-1],
                    'box_only':[1,0,-1,-1],'neither':[0,0,-1,-1]}
        self.thresholds=[{'value':.5} for _ in range(4)]
        self.groups={'g004':dict(counterfactual_geometry_verified=True,
                     members={v:{'sample_id':v} for v in self.variants})}

    def test_correct_removals_require_correct_both_baseline(self):
        metrics=paired_metrics(self.rows,self.truth,self.thresholds,self.groups)
        bar=metrics['removal']['bar_removal']
        self.assertEqual(bar['both_initially_correct'],1)
        self.assertEqual(bar['joint_success'],1)
        self.assertEqual(bar['conditional_joint_success_given_both_correct'],1)
        self.assertEqual(metrics['all_four_correct']['BODY_HEAD_joint']['correct'],1)
        self.rows['both'][0]=.1
        failed=paired_metrics(self.rows,self.truth,self.thresholds,self.groups)
        self.assertEqual(failed['removal']['bar_removal']['target_flip_correct'],1)
        self.assertEqual(failed['removal']['bar_removal']['retained_head_correct'],0)
        self.assertEqual(failed['removal']['bar_removal']['joint_success'],0)
        self.assertIsNone(failed['removal']['bar_removal']['conditional_joint_success_given_both_correct'])

    def test_unknown_truth_and_predictions_never_become_correct_clear(self):
        self.truth['box_only'][1]=-1
        metrics=paired_metrics(self.rows,self.truth,self.thresholds,self.groups)
        self.assertEqual(metrics['removal']['bar_removal']['truth_UNKNOWN_groups'],1)
        self.assertEqual(metrics['removal']['bar_removal']['truth_eligible_groups'],0)
        self.truth['box_only'][1]=0
        self.rows['box_only'][1]=-1
        metrics=paired_metrics(self.rows,self.truth,self.thresholds,self.groups)
        self.assertEqual(metrics['removal']['bar_removal']['after_prediction_UNKNOWN'],1)
        self.assertEqual(metrics['removal']['bar_removal']['joint_success'],0)
        self.assertEqual(metrics['all_four_correct']['HEAD']['prediction_UNKNOWN_groups'],1)
        self.assertEqual(metrics['all_four_correct']['HEAD']['correct'],0)

    def test_query_coverage_and_evidence_fraction_are_distinct(self):
        samples=[{'sample_id':'s'}]; targets={'s':[1,0,-1,-1]}
        truth=np.zeros((2,18,32),bool); truth[0,0,:2]=True
        maps=np.zeros((2,18,32),float); maps[0,0,0]=1.;maps[0,1,:9]=1.
        objects={'s__box':truth[0], 's__bar':np.zeros((18,32),bool)}
        result=support_metrics(samples,targets,{'s':truth},objects,{'s':maps})
        row=result['BODY']['samples_detail'][0]
        self.assertEqual(row['GT_query_coverage_at_0_5'],.5)
        self.assertEqual(row['evidence_fraction_in_GT_query'],.1)
        self.assertEqual(row['evidence_fraction_off_GT_query'],.9)
        self.assertAlmostEqual(row['IoU_at_0_5'],1/11)
        self.assertTrue(row['pointing_hit'])
        empty=support_metrics(samples,targets,{'s':truth},objects,{'s':np.zeros_like(maps)})
        self.assertIsNone(empty['BODY']['samples_detail'][0]['evidence_fraction_in_GT_query'])
        self.assertIsNone(empty['BODY']['samples_detail'][0]['pointing_hit'])

    def test_frozen_threshold_requires_exact_source_key(self):
        source={'evaluations':{'ordinary_video/17/normal':{'thresholds':self.thresholds}}}
        self.assertEqual(frozen_thresholds(source,'ordinary_video/17/normal'),self.thresholds)
        with self.assertRaises(KeyError):
            frozen_thresholds(source,'ordinary_video/29/normal')

    def test_geometry_mismatch_excludes_pairing_and_preflight_skipped(self):
        objects={'bar':{'name':'bar','size':[1,1,1]},'box':{'name':'box','size':[2,2,2]}}
        named={'both':['bar','box'],'bar_only':['bar'],'box_only':['box'],'neither':[]}
        samples=[];clips=[];cases=[]
        for i,v in enumerate(self.variants):
            samples.append(dict(sample_id=v,clip_id=v,group_id='g004',split='test',frame_indices=[i]))
            clips.append(dict(clip_id=v,group_id='g004',variant=v,objects=[objects[n] for n in named[v]]))
            cases.append(dict(camera={'x':0},wearer={'x':0}))
        dataset={'samples':samples};spec={'clips':clips,'cases':cases}
        _,groups,_=group_metadata(dataset,spec)
        self.assertTrue(groups['g004']['counterfactual_geometry_verified'])
        cases[2]['camera']['x']=1
        _,groups,_=group_metadata(dataset,spec)
        self.assertFalse(groups['g004']['counterfactual_geometry_verified'])
        result=paired_metrics(self.rows,self.truth,self.thresholds,groups)
        self.assertEqual(result['removal']['bar_removal']['geometry_NOT_EVALUABLE'],1)
        extra=copy.deepcopy(samples[0]);extra.update(sample_id='preflight',clip_id='preflight',group_id='g000')
        samples.append(extra)
        clips.append(dict(clip_id='preflight',group_id='g000',variant='both',preflight=True))
        active,_,excluded=group_metadata(dataset,spec)
        self.assertEqual(len(active),4)
        self.assertEqual(excluded,['preflight'])

if __name__=='__main__':
    unittest.main()
