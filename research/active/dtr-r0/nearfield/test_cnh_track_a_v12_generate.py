import unittest
import copy
import numpy as np
from cnh_track_a_v12_generate import (purpose_seed,schedule,object_templates,
    plan_centres,boundary_centres,candidate_check,distinct_contributors)
from cnh_track_a_geometry import signed_margin
from cnh_track_a_generate import BOXES


def synthetic_path():
    poses=np.repeat(np.eye(4)[None],12,axis=0)
    poses[:,1,3]=-1.6
    poses[:,2,3]=(np.arange(12)-7)*.1
    return dict(world_from_Q=poses.tolist(),angles=np.zeros((12,3)).tolist(),speed=.5,seed=123)


class V12GeneratorTest(unittest.TestCase):
    def test_seed_purpose_candidate_isolation(self):
        # Deliberately outside pilot unit/config domain; no cohort generation.
        keys=[purpose_seed(100,100,p,c) for p in ('trajectory','size_placement','material','noise') for c in (0,1)]
        self.assertEqual(len(set(keys)),8)
        self.assertEqual(purpose_seed(100,100,'trajectory'),purpose_seed(100,100,'trajectory'))
        with self.assertRaises(ValueError):purpose_seed(100,100,'unknown')

    def test_frozen_pattern_cardinality(self):
        even,odd=schedule(0),schedule(1)
        self.assertEqual(len(even),32)
        self.assertEqual(len(set(even[:28])),14)
        self.assertEqual(len(set(even[:28])|set(odd[:28])),21)
        self.assertEqual(even[:14],even[14:28])

    def test_milp_immutable_path_and_multi_main(self):
        path=synthetic_path();before=copy.deepcopy(path)
        objects=object_templates('train',(2,4),np.random.default_rng(123),np.random.default_rng(456))
        centres,receipt=plan_centres(objects,path,1.6,True)
        self.assertIsNotNone(centres,receipt)
        self.assertEqual(path,before)
        self.assertGreaterEqual(len(receipt['planned_main']),6)
        self.assertGreaterEqual(len(receipt['planned_target']),2)
        poses=np.array(path['world_from_Q'])
        for t in receipt['planned_main']:
            contributors=[[] for _ in range(6)]
            for obj,center in zip(objects,centres):
                local=(obj['triangles']+center-poses[t,:3,3])@poses[t,:3,:3]
                m=np.array([signed_margin(local,*box) for box in BOXES])
                self.assertTrue(np.all(np.abs(m)>=.05-1e-8))
                for q in np.flatnonzero(m>=0):contributors[q].append(obj['id'])
            self.assertTrue(distinct_contributors(contributors))

    def test_boundary_actual_template_maximum(self):
        path=synthetic_path();anchor=np.array(path['world_from_Q'])[7]
        for split in ('train','calib','audit'):
            for ci,pattern in ((28,(4,0)),(29,(0,2)),(30,(3,0)),(31,(0,3))):
                objects=object_templates(split,pattern,np.random.default_rng(123),np.random.default_rng(456))
                centres,plan=boundary_centres(objects,path,ci,np.random.default_rng(789))
                obj=objects[0];local=obj['triangles']+(centres[0]-anchor[:3,3])
                actual=signed_margin(local,*BOXES[plan['query']])
                self.assertAlmostEqual(actual,plan['target_margin'],places=8)

    def test_nonconsecutive_actual_main_permitted(self):
        main=np.array([False,False,False,True,True,False,True,True,False,True,True,False])
        y=np.zeros((12,6),int);y[main]=[1,0,1,0,0,1]
        contrib=[[[1],[],[1],[],[],[2]] for _ in range(12)]
        config=dict(main=main.tolist(),labels=y.tolist(),target_labels=[1,0,1,0,0,1],contributors=contrib)
        good,why=candidate_check(config,7)
        self.assertTrue(good,why)


if __name__=='__main__':unittest.main()
