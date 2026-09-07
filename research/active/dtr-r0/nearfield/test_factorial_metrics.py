"""Five focused checks for factorial data evaluation boundaries."""
import copy
import unittest
import numpy as np
from factorial_evaluate import (ARMS,SEEDS,partitions,primary_thresholds,
    secondary_thresholds,score_policy,reproduction_comparison)

class FactorialMetricsTest(unittest.TestCase):
    def setUp(self):
        self.samples=[];self.clips=[];self.cases=[];self.targets={};self.rows={}
        variants={'both':(1,1),'bar_only':(0,1),'box_only':(1,0),'neither':(0,0)}
        objects={'bar':{'name':'bar','center':[2,0,1.8]},'box':{'name':'box','center':[2,0,.6]}}
        for gid,split in (('g1000','train'),('g1040','val'),('g1052','test')):
            for variant,near in variants.items():
                cid=gid+'_'+variant;sid=cid+'_t0';index=len(self.cases)
                self.samples.append(dict(sample_id=sid,clip_id=cid,group_id=gid,split=split,frame_indices=[index]))
                named=[objects[n] for n in ('bar','box') if variant=='both' or variant==n+'_only']
                self.clips.append(dict(clip_id=cid,group_id=gid,split=split,variant=variant,objects=named))
                self.cases.append(dict(camera={'x':1},wearer={'x':1},objects=named))
                self.targets[sid]=list(near)+[-1,-1]
                self.rows[sid]=np.array([.9 if t else .1 for t in near]+[-1,-1])
        self.dataset={'samples':self.samples};self.spec={'clips':self.clips,'cases':self.cases}
        self.key=('factorial_data','17','normal')

    def test_every_arm_uses_matching_ordinary_frozen_seed(self):
        source={'evaluations':{}}
        for i,seed in enumerate(SEEDS):
            source['evaluations'][f'ordinary_video/{seed}/normal']={'thresholds':[{'value':.1*(i+1)} for _ in range(4)]}
            source['evaluations'][f'factorial_data/{seed}/normal']={'thresholds':[{'value':.99} for _ in range(4)]}
        keys=[(arm,seed,'normal') for arm in ARMS for seed in SEEDS]
        selected=primary_thresholds(source,keys)
        for arm,seed,mode in keys:
            self.assertEqual(selected[arm,seed,mode],source['evaluations'][f'ordinary_video/{seed}/normal']['thresholds'])
        selected['factorial_data','17','normal'][0]['value']=.7
        self.assertEqual(source['evaluations']['ordinary_video/17/normal']['thresholds'][0]['value'],.1)

    def test_secondary_sees_only_common_validation_quartet(self):
        val_ids={s['sample_id'] for s in self.samples if s['split']=='val'}
        only_val_rows={sid:self.rows[sid] for sid in val_ids}
        only_val_truth={sid:self.targets[sid] for sid in val_ids}
        selected=secondary_thresholds({self.key:only_val_rows},self.samples,only_val_truth)
        self.assertGreater(selected[self.key][0]['value'],.1)
        self.assertEqual(selected[self.key][0]['actual_fp'],0)
        self.assertEqual(selected[self.key][0]['negatives'],2)
        self.assertEqual(selected[self.key][2]['status'],'NOT_EVALUABLE')
        changed=copy.deepcopy(self.targets)
        for sample in self.samples:
            if sample['split']!='val':
                self.rows[sample['sample_id']]=np.array([1.,1.,-1,-1])
                changed[sample['sample_id']]=[0,0,-1,-1]
        self.assertEqual(selected,secondary_thresholds({self.key:self.rows},self.samples,changed))

    def test_complete_test_group_only_is_scored_with_single_frame(self):
        split,test,groups=partitions(self.dataset,self.spec)
        self.assertEqual({s['group_id'] for s in test},{'g1052'})
        self.assertEqual(len(test),4)
        self.assertEqual(set(groups),{'g1052'})
        self.assertEqual([len(split[k]) for k in ('train','val','test')],[4,4,4])
        ev=score_policy({self.key:self.rows},{self.key:[{'value':.5} for _ in range(4)]},test,self.targets,groups)['/'.join(self.key)]
        self.assertEqual(ev['cells']['BODYnear']['total'],4)
        self.assertEqual(ev['cells']['BODYnear']['TP'],2)
        self.assertEqual(ev['cells']['BODYapproaching']['UNKNOWN'],4)
        self.assertEqual(ev['counterfactual']['all_four_correct']['BODY_HEAD_joint']['correct'],1)
        self.assertEqual(ev['counterfactual']['removal']['bar_removal']['joint_success'],1)

    def test_group_split_leak_and_incomplete_quartet_rejected(self):
        altered=copy.deepcopy(self.dataset)
        altered['samples'][0]['group_id']='g1052'
        with self.assertRaises(ValueError):
            partitions(altered,self.spec)
        altered=copy.deepcopy(self.dataset);altered['samples'].pop()
        with self.assertRaises(ValueError):
            partitions(altered,self.spec)

    def test_regression_numeric_agreement_reports_threshold_flip(self):
        samples=[{'sample_id':'s'}]
        fresh={('frozen_ordinary',seed,'normal'):{'s':[.90001,.49999,-1,-1]} for seed in SEEDS}
        old={('ordinary_video',seed,'normal'):{'s':[.9,.5,.1,.1]} for seed in SEEDS}
        selected={key:[{'value':.5} for _ in range(4)] for key in fresh}
        result=reproduction_comparison(fresh,old,selected,samples)
        for row in result['comparisons'].values():
            self.assertTrue(row['all_known_scores_within_tolerance'])
            self.assertEqual(row['near_alert_mismatches'],1)
            self.assertEqual(row['known_near_cells'],2)

if __name__=='__main__':
    unittest.main()
