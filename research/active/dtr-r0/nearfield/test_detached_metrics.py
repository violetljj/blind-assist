"""G11 synthetic boundary, per-head calibration and UNKNOWN checks."""
import copy
import unittest
import numpy as np
from detached_evaluate import (select_mask_thresholds,calibrated_mask_metrics,fresh_scopes,
    primary_thresholds,secondary_thresholds,baseline_reproduction,SEEDS)

class DetachedMetricsTest(unittest.TestCase):
    def setUp(self):
        self.key=('detached_gate','17','normal')
        self.samples=[dict(sample_id='tr',split='train'),dict(sample_id='v',split='val'),dict(sample_id='t',split='test')]
        self.truth=np.zeros((2,18,32),bool);self.truth[:,0,0]=True
        self.prob=np.zeros((2,18,32),float);self.prob[:,0,0]=.9
        self.prob[0,0,1]=.1;self.prob[1,0,1]=.6

    def test_per_head_mask_selection_reads_only_validation_positives(self):
        result=select_mask_thresholds({self.key:{'v':self.prob}},self.samples,{'v':[1,1,-1,-1]},{'v':self.truth})
        self.assertEqual(result[self.key][0]['value'],.15)
        self.assertEqual(result[self.key][1]['value'],.65)
        self.assertEqual(result[self.key][0]['mean_validation_IoU'],1.)
        # Train/test entries can change arbitrarily; neither influences selection.
        altered=np.full_like(self.prob,1.)
        again=select_mask_thresholds({self.key:{'v':self.prob,'tr':altered,'t':altered}},self.samples,
            {'v':[1,1,-1,-1],'tr':[0,0,-1,-1],'t':[0,0,-1,-1]}, {'v':self.truth,'tr':~self.truth,'t':~self.truth})
        self.assertEqual(result,again)

    def test_ties_smallest_threshold_and_unknown_positive_support_excluded(self):
        zero=np.zeros_like(self.prob)
        selected=select_mask_thresholds({self.key:{'v':zero}},self.samples,{'v':[1,-1,-1,-1]},{'v':self.truth})[self.key]
        self.assertEqual(selected[0]['value'],.05)
        self.assertEqual(selected[0]['mean_validation_IoU'],0.)
        self.assertIsNone(selected[1]['value'])
        no_support=select_mask_thresholds({self.key:{'v':zero}},self.samples,{'v':[1,1,-1,-1]},{'v':np.zeros_like(self.truth)})[self.key]
        self.assertEqual(no_support[0]['positive_support_NOT_EVALUABLE'],1)
        self.assertIsNone(no_support[0]['value'])

    def test_negative_activation_coverage_and_unknown_separate(self):
        selected=[{'value':.5},{'value':.5}]
        samples=[{'sample_id':'n'},{'sample_id':'u'},{'sample_id':'p'}]
        mask=np.zeros_like(self.prob);mask[0,0,0]=.8
        result=calibrated_mask_metrics({sid:mask for sid in ('n','u','p')},samples,
            {'n':[0,0,-1,-1],'u':[-1,-1,-1,-1],'p':[1,1,-1,-1]},
            {sid:self.truth for sid in ('n','u','p')},selected)
        self.assertEqual(result['BODY']['known_negative_samples'],1)
        self.assertEqual(result['BODY']['negative_activated_samples'],1)
        self.assertEqual(result['BODY']['negative_activation_sample_rate'],1.)
        self.assertAlmostEqual(result['BODY']['negative_activation_pixel_fraction']['mean'],1/576)
        self.assertEqual(result['BODY']['truth_UNKNOWN'],1)
        self.assertEqual(result['BODY']['positive_mean_IoU']['mean'],1.)
        self.assertEqual(result['HEAD']['negative_activated_samples'],0)
        self.assertIsNone(result['HEAD']['positive_pointing']['mean'])

    def test_scalar_thresholds_keep_frozen_primary_and_reference_val_only(self):
        keys=[(arm,seed,'normal') for arm in ('original_gate','detached_gate') for seed in SEEDS]
        source={'evaluations':{f'ordinary_video/{seed}/normal':{'thresholds':[{'value':.3+i*.1} for _ in range(4)]}
                               for i,seed in enumerate(SEEDS)}}
        frozen=primary_thresholds(source,keys)
        for seed in SEEDS:
            self.assertEqual(frozen['original_gate',seed,'normal'],frozen['detached_gate',seed,'normal'])
        chosen=secondary_thresholds({self.key:{'v':np.array([.1,.2,-1,-1])}},self.samples,{'v':[0,0,-1,-1]})
        self.assertGreater(chosen[self.key][0]['value'],.1)
        self.assertGreater(chosen[self.key][1]['value'],.2)
        self.assertEqual(chosen[self.key][0]['negatives'],1)

    def test_fresh_all_test_quartets_and_strata_need_no_training_partition(self):
        samples=[];clips=[];cases=[]
        objects={'bar':{'name':'bar'},'box':{'name':'box'}}
        for gid,stratum in (('g5000','narrow'),('g5001','diverse')):
            for variant in ('both','bar_only','box_only','neither'):
                sid=gid+'_'+variant;idx=len(cases)
                named=[objects[n] for n in objects if variant=='both' or variant==n+'_only']
                samples.append(dict(sample_id=sid,clip_id=sid,group_id=gid,split='test',frame_indices=[idx]))
                clips.append(dict(clip_id=sid,group_id=gid,variant=variant,stratum=stratum,objects=named))
                cases.append(dict(camera={'x':0},wearer={'x':0},objects=named))
        scopes=fresh_scopes({'samples':samples},{'clips':clips,'cases':cases})
        self.assertEqual(len(scopes['all_TEST'][0]),8)
        self.assertEqual(len(scopes['narrow'][1]),1)
        self.assertEqual(len(scopes['diverse'][1]),1)
        altered=copy.deepcopy(samples);altered[0]['split']='val'
        with self.assertRaises(ValueError):
            fresh_scopes({'samples':altered},{'clips':clips,'cases':cases})

if __name__=='__main__':
    unittest.main()
