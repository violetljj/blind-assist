"""Synthetic checks only; never opens experiment labels."""
import copy
import unittest
import numpy as np
from evaluate_whisker import thresholds, evaluate, normalize_predictions, HEADS, sample_metadata, matched_near_retention

class WhiskerMetricsTest(unittest.TestCase):
    def setUp(self):
        self.samples = [dict(sample_id=k, clip_id=k, group_id=k, split=split)
                        for k,split in [('tr','train'),('v','val'),('stop','test'),('static','test'),('rotation','test')]]
        self.targets = {s['sample_id']: [0]*4 for s in self.samples}
        self.rows = {s['sample_id']: np.array([.2]*4) for s in self.samples}
        self.metadata = {k: {'motion': k} for k in ('stop','static','rotation')}

    def test_no_test_label_or_score_leakage(self):
        val_only = {'v': [0]*4}
        frozen = thresholds(self.rows,self.samples,val_only)
        self.rows['stop'][:] = 1
        self.targets['stop'] = [1]*4
        self.assertEqual(frozen,thresholds(self.rows,self.samples,val_only))
        self.assertGreater(frozen[0]['value'], .2)
        self.assertLess(frozen[0]['value'], .20000001)

    def test_exact_five_percent_and_ties(self):
        samples = [dict(sample_id=str(i), split='val') for i in range(20)]
        targets = {str(i): [0]*4 for i in range(20)}
        rows = {str(i): [i/20]*4 for i in range(20)}
        selected = thresholds(rows,samples,targets)
        self.assertEqual(selected[0]['actual_fp'],1)
        self.assertAlmostEqual(selected[0]['fpr'],.05)
        rows['19'] = [.9]*4
        selected = thresholds(rows,samples,targets)
        self.assertEqual(selected[0]['actual_fp'],0)

    def test_unknown_never_becomes_negative(self):
        self.targets['stop'] = [-1]*4
        self.rows['stop'][:] = 1
        ev = evaluate(self.rows,self.samples,self.targets,thresholds(self.rows,self.samples,self.targets),self.metadata)
        c = ev['cells']['BODYnear']
        self.assertEqual(c['UNKNOWN'],1)
        self.assertEqual(c['FP'],0)
        self.assertEqual(c['TN'],2)
        self.assertEqual(c['coverage'],2/3)

    def test_stopped_near_is_distinct_from_approaching(self):
        self.targets['stop'] = [1,1,0,0]
        self.targets['static'] = [1,0,0,0]
        self.rows['stop'] = np.array([.9,.9,.1,.1])
        self.rows['static'] = np.array([.9,.1,.1,.1])
        self.rows['rotation'] = np.array([.1,.1,.9,.1])
        ev = evaluate(self.rows,self.samples,self.targets,thresholds(self.rows,self.samples,self.targets),self.metadata)
        self.assertEqual(ev['persisted_near_stop_static']['BODY']['recovered'],2)
        self.assertEqual(ev['cells']['BODYnear']['TP'],2)
        self.assertEqual(ev['cells']['BODYapproaching']['FP'],1)
        self.assertEqual(ev['rotation_false_approach']['BODYapproaching']['FP'],1)
        self.assertEqual(ev['false_alert_clips'],['rotation'])

    def test_actual_support_localization_and_missing_support(self):
        self.targets['stop'] = [1,1,0,0]
        truth = np.zeros((2,18,32), np.uint8); truth[0,3,4] = 1
        maps = truth.astype(float)
        ev = evaluate(self.rows,self.samples,self.targets,thresholds(self.rows,self.samples,self.targets),self.metadata,{'stop':truth},{'stop':maps})
        self.assertEqual(ev['localization']['BODY']['mean_IoU'],1)
        self.assertEqual(ev['localization']['BODY']['pointing_hit_rate'],1)
        self.assertEqual(ev['localization']['HEAD']['NOT_EVALUABLE'],1)
        self.assertIsNone(ev['localization']['HEAD']['mean_IoU'])

    def test_ensemble_recomputed_and_repeat_parent_threshold(self):
        ids = [s['sample_id'] for s in self.samples]
        payload = dict(target_order=list(HEADS),sample_ids=ids,arms={'video':{'seeds':{
            '17': {'normal': [[.1]*4]*5,'repeated_history':[[.9]*4]*5},
            '29': {'normal': [[.3]*4]*5,'repeated_history':[[.9]*4]*5}},
            'ensemble': {'normal': [[1]*4]*5}}})
        result = normalize_predictions(payload,self.samples)
        self.assertAlmostEqual(result['video','ensemble','normal']['v'][0],.2)
        parent = thresholds(result['video','ensemble','normal'],self.samples,self.targets)
        repeated = evaluate(result['video','ensemble','repeated_history'],self.samples,self.targets,parent,self.metadata)
        self.assertEqual(repeated['thresholds'],parent)
        self.assertEqual(repeated['cells']['BODYnear']['FP'],3)

    def test_matched_geometry_stop_drop_and_pose_verification(self):
        samples, clips, cases = [], [], []
        for i, role in enumerate(('approach', 'stop', 'static')):
            cid = 'g001_v0_'+role
            samples.append(dict(sample_id=role,clip_id=cid,group_id='g001',split='test',frame_indices=[i]))
            clips.append(dict(clip_id=cid,group_id='g001',split='test',motion=role,control='near'))
            cases.append(dict(phase=role,frame_in_clip=2 if role=='approach' else 5,
                              camera={'x':1.,'z':1.7},wearer={'x':1.},objects=[{'center':[2,0,1]}]))
        dataset, spec = {'samples':samples}, {'clips':clips,'cases':cases}
        metadata = sample_metadata(dataset,spec)
        rows = {'approach': [.9]*4, 'stop':[.1]*4, 'static':[.8]*4}
        targets = {role:[1,1,0,0] for role in rows}
        frozen = [dict(value=.5) for _ in HEADS]
        result = matched_near_retention(rows,samples,targets,frozen,metadata)
        cell = result['parts']['BODY']
        self.assertEqual(result['verified_geometry_groups'],1)
        self.assertEqual(cell['approach_detected'],1)
        self.assertEqual(cell['stop_kept_given_approach_detected'],0)
        self.assertEqual(cell['stop_dropped_given_approach_detected'],1)
        self.assertEqual(cell['static_kept_given_approach_detected'],1)
        self.assertEqual(cell['all_three_detected'],0)
        self.assertAlmostEqual(cell['stop_minus_approach_probability_delta']['mean'],-.8)
        cases[1]['camera']['x'] = 1.01
        unmatched = matched_near_retention(rows,samples,targets,frozen,sample_metadata(dataset,spec))
        self.assertEqual(unmatched['verified_geometry_groups'],0)
        self.assertEqual(unmatched['NOT_EVALUABLE_geometry'],1)
        self.assertEqual(unmatched['parts']['BODY']['positive_triplets'],0)

if __name__ == '__main__':
    unittest.main()
