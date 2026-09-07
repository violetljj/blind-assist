"""Synthetic G10 tests; no capture, learned weights or outcome files opened."""
import copy
import unittest
import numpy as np
from diversity_evaluate import (ARMS,SEEDS,evaluation_scopes,normalize_for_scope,
    primary_thresholds,secondary_thresholds,factorial_contrasts)

class DiversityMetricsTest(unittest.TestCase):
    def setUp(self):
        self.samples=[];self.clips=[];self.cases=[];self.targets={};self.rows={}
        variants={'both':(1,1),'bar_only':(0,1),'box_only':(1,0),'neither':(0,0)}
        objects={'bar':{'name':'bar','center':[2,0,1.8]},'box':{'name':'box','center':[2,0,.6]}}
        settings=(('tr','train','diverse'),('vn','val','narrow'),('vd','val','diverse'),
                  ('tn','test','narrow'),('td','test','diverse'))
        for gid,split,stratum in settings:
            for variant,near in variants.items():
                cid=gid+'_'+variant;sid=cid+'_t0';index=len(self.cases)
                self.samples.append(dict(sample_id=sid,clip_id=cid,group_id=gid,split=split,frame_indices=[index]))
                named=[objects[n] for n in ('bar','box') if variant=='both' or variant==n+'_only']
                self.clips.append(dict(clip_id=cid,group_id=gid,split=split,stratum=stratum,variant=variant,objects=named))
                self.cases.append(dict(camera={'x':1},wearer={'x':1},objects=named))
                self.targets[sid]=list(near)+[-1,-1]
                self.rows[sid]=np.array([.9 if t else .1 for t in near]+[-1,-1])
        self.dataset={'samples':self.samples};self.spec={'clips':self.clips,'cases':self.cases}

    def test_test_scopes_keep_whole_groups_and_exclude_train_val(self):
        splits,scopes=evaluation_scopes(self.dataset,self.spec)
        self.assertEqual(set(scopes),{'all_TEST','narrow','diverse'})
        self.assertEqual([len(scopes[k][0]) for k in scopes],[8,4,4])
        self.assertEqual(set(scopes['narrow'][1]),{'tn'})
        self.assertEqual(set(scopes['diverse'][1]),{'td'})
        self.assertTrue(all(s['split']=='test' for s in scopes['all_TEST'][0]))
        _,regression=evaluation_scopes(self.dataset,self.spec,with_strata=False)
        self.assertEqual(set(regression),{'all_TEST'})
        self.assertEqual(len(regression['all_TEST'][0]),8)

    def test_split_and_stratum_crossing_rejected(self):
        dataset=copy.deepcopy(self.dataset);dataset['samples'][0]['group_id']='td'
        with self.assertRaises(ValueError):
            evaluation_scopes(dataset,self.spec)
        spec=copy.deepcopy(self.spec);spec['clips'][-1]['stratum']='narrow'
        with self.assertRaises(ValueError):
            evaluation_scopes(self.dataset,spec)
        dataset=copy.deepcopy(self.dataset);dataset['samples'].pop()
        with self.assertRaises(ValueError):
            evaluation_scopes(dataset,self.spec)

    def test_primary_shared_frozen_seed_secondary_validation_only(self):
        source={'evaluations':{f'ordinary_video/{seed}/normal':{'thresholds':[{'value':.2+i*.1} for _ in range(4)]}
                                for i,seed in enumerate(SEEDS)}}
        keys=[(arm,seed,'normal') for arm in ARMS for seed in SEEDS]
        frozen=primary_thresholds(source,keys)
        for arm,seed,mode in keys:
            self.assertEqual(frozen[arm,seed,mode],source['evaluations'][f'ordinary_video/{seed}/normal']['thresholds'])
        val_ids={s['sample_id'] for s in self.samples if s['split']=='val'}
        selected=secondary_thresholds({keys[0]:{sid:self.rows[sid] for sid in val_ids}},self.samples,
                                      {sid:self.targets[sid] for sid in val_ids})
        self.assertEqual(selected[keys[0]][0]['negatives'],4)
        self.assertGreater(selected[keys[0]][0]['value'],.1)
        self.assertEqual(selected[keys[0]][0]['actual_fp'],0)
        changed=copy.deepcopy(self.targets)
        for s in self.samples:
            if s['split']!='val':
                self.rows[s['sample_id']]=np.array([1.,1.,-1,-1]);changed[s['sample_id']]=[0,0,-1,-1]
        self.assertEqual(selected,secondary_thresholds({keys[0]:self.rows},self.samples,changed))

    def test_regression_accepts_exact_test_only_and_rejects_partial(self):
        test=[s for s in self.samples if s['split']=='test']
        ids=[s['sample_id'] for s in test]
        def payload(sample_ids):
            return dict(sample_ids=sample_ids,unknown_score=-1,
                arms={arm:dict(seeds={seed:dict(normal=[[.5,.5,-1,-1] for _ in sample_ids])
                    for seed in SEEDS if seed!='ensemble'}) for arm in ARMS})
        predictions=normalize_for_scope(payload(ids),self.samples,test_only=test)
        self.assertEqual(len(predictions),16)
        self.assertEqual(set(next(iter(predictions.values()))),set(ids))
        with self.assertRaises(ValueError):
            normalize_for_scope(payload(ids[:-1]),self.samples,test_only=test)

    def test_paired_2x2_difference_in_differences_counts_and_rates(self):
        correct={'existing_plain':0,'existing_region':1,'expanded_plain':2,'expanded_region':4}
        evaluations={}
        for arm,n in correct.items():
            for seed in SEEDS:
                cell=dict(TP=n,FP=4-n,FN=4-n,TN=n,UNKNOWN=0,all_opportunity_recall=n/4,actual_FPR=(4-n)/4)
                ev=dict(cells={'BODYnear':cell,'HEADnear':cell},counterfactual={'all_four_correct':{
                    'BODY_HEAD_joint':dict(correct=n,truth_evaluable_groups=4,all_four_correct_rate=n/4)}})
                evaluations[f'{arm}/{seed}/normal']={'all_TEST':ev}
        contrasts=factorial_contrasts(evaluations)['by_seed']['ensemble']['all_TEST']
        self.assertEqual(contrasts['paired_contrasts']['data_at_plain']['differences']['joint_correct_count'],2)
        self.assertEqual(contrasts['paired_contrasts']['head_at_existing']['differences']['joint_correct_count'],1)
        self.assertEqual(contrasts['interaction']['differences']['joint_correct_count'],1)
        self.assertEqual(contrasts['interaction']['differences']['joint_correct_rate'],.25)

if __name__=='__main__':
    unittest.main()
