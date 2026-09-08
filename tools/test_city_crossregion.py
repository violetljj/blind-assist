import copy
import json
from pathlib import Path
import unittest
from city_crossregion import compile_design,compile_route,causal_flip,interventions

P=json.loads((Path(__file__).resolve().parents[1]/'experiments/city-field/crossregion-v1.json').read_text())


def route():return dict(route_id='r0',region_id='g0',split='train',scene_type='sidewalk',status='ADMITTED',floor_z_m=0.,camera=dict(x=0.,y=0.,z=1.7,yaw=0.,pitch=-5.,roll=0.))
def records(always=None):
    truths={'NONE':[0,0],'BODY_ONLY':[1,0],'HEAD_ONLY':[0,1],'BOTH':[1,1]}
    return [dict(region_id='g0',pair_id='p0',split='test',condition=c,eligible=True,context_hash='fixed-context',truth=t,
                 prediction=t.copy() if always is None else [always,always],probability=[float(x) for x in t]) for c,t in truths.items()]


class CrossregionTests(unittest.TestCase):
    def test_carriers_are_identical_across_all_conditions(self):
        variants=interventions((1,2,3),37,'r')
        common=[[o for o in objects if not o.get('target_part')] for objects in variants.values()]
        self.assertTrue(all(x==common[0] for x in common))
        self.assertEqual([sum(o.get('target_part',False) for o in os) for os in variants.values()],[0,1,1,2])
        head=next(o for o in variants['HEAD_ONLY'] if o.get('target_part'))
        self.assertEqual(head['size_m'][1],1.56)

    def test_four_poses_sixteen_views_and_exact_pair_camera(self):
        rows=compile_route(route(),P['camera_perturbations']);self.assertEqual(len(rows),16)
        for i in range(0,16,4):self.assertTrue(all(c['camera']==rows[i]['camera'] for c in rows[i:i+4]))

    def test_full_design_requires_background_admission(self):
        with self.assertRaisesRegex(ValueError,'isolation'):compile_design(P,dict(routes=[route()]),{})

    def test_budget_336_and_visible_instance_leak_rejected(self):
        source=dict(status='ADMITTED',visible_background_isolation='PASS',map_asset='/Game/Test',regions=[],routes=[])
        for i,split in enumerate(P['region_splits']):
            source['regions'].append(dict(region_id=f'g{i}',split=split,bounds_xy_m=[i*100,0,i*100+50,50],visible_background_instance_ids=[f'building{i}']))
            for j,scene in enumerate(P['scene_types']):
                r=route();r.update(route_id=f'g{i}r{j}',region_id=f'g{i}',split=split,scene_type=scene);r['camera'].update(x=i*100+5,y=5);source['routes'].append(r)
        template=dict(map_asset='/Game/Test')
        self.assertEqual(len(compile_design(P,source,template)['cases']),336)
        source['regions'][-1]['visible_background_instance_ids']=['building0']
        with self.assertRaisesRegex(ValueError,'leak'):compile_design(P,source,template)

    def test_perfect_pairs_and_constant_shortcut(self):
        for h in ('BODY','HEAD'):
            self.assertEqual(causal_flip(records())['scores'][h]['correct_pairs'],2)
            self.assertEqual(causal_flip(records(1))['scores'][h]['accuracy'],0)
            self.assertEqual(causal_flip(records(0))['scores'][h]['accuracy'],0)

    def test_missing_unknown_and_truth_conflict_exclude_whole_quartet(self):
        for rows in (records()[:-1],records(),records()):
            if len(rows)==4:
                if rows[0]['truth']==[0,0]:rows[0]['eligible']=False
            result=causal_flip(rows);self.assertEqual(result['excluded_quartets'],1)
            self.assertIsNone(result['scores']['HEAD']['accuracy'])
        rows=records();rows[0]['truth']=[0,1]
        self.assertEqual(causal_flip(rows)['excluded_quartets'],1)

    def test_changed_context_cannot_score(self):
        rows=records();rows[0]['context_hash']='other-camera'
        with self.assertRaisesRegex(ValueError,'context mismatch'):causal_flip(rows)

    def test_split_metrics_cannot_mix_training_with_test(self):
        rows=records();other=copy.deepcopy(rows);other[0]['split']='train'
        with self.assertRaisesRegex(ValueError,'one split'):causal_flip(other)


if __name__=='__main__':unittest.main()
