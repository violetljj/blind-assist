import unittest
import random
from collections import Counter
import mz96_ue_decision_capture as m


class CaptureSpecTest(unittest.TestCase):
    def test_split_frozen_and_scene_disjoint(self):
        a=m.source();self.assertEqual(a,m.source())
        self.assertEqual(Counter(s['split'] for s in a['scenes']),dict(train=80,validation=16,test=32))
        self.assertEqual(len({s['episode_id'] for s in a['scenes']}),128)
        self.assertEqual(len({repr(s['objects']) for s in a['scenes'] if s['objects']}),112)

    def test_future_and_receding_truth(self):
        self.assertEqual(m.hazard(0,3.5,0,-.7),(True,False))
        self.assertEqual(m.hazard(1.5,2.,0,.5),(False,False))
        self.assertTrue(m.intersection(1.,3.,-.7,0.))
        self.assertFalse(m.intersection(1.,3.,.7,0.))

    def test_no_behind_or_out_of_horizon_truth(self):
        self.assertEqual(m.hazard(0,-1,0,0),(False,False))
        self.assertEqual(m.hazard(0,3.7,0,-1),(False,False))
        self.assertFalse(m.intersection(0,.2,0,-1))

    def test_ghost_cohort_independent_and_explicit(self):
        scenes=m.source(96013)['scenes'];ghosts=[s for s in scenes if s['radar_ghost']]
        self.assertEqual(len(ghosts),32)
        self.assertGreater(len({s['family'] for s in ghosts}),1)
        self.assertTrue(all(s['radar_ghost']['authority']=='GENERATED_RADAR_GHOST_NOT_ENGINE_GEOMETRY' for s in ghosts))

    def test_ghost_kinematics_and_shared_measurement(self):
        self.assertEqual(m.ghost_return(dict(x=0,z=3),.7,1.,0.),(2.3,0.,-.7))
        self.assertIsNone(m.ghost_return(dict(x=0,z=-3),.7,1.,0.))
        real=[m.measure_radar(2.,10.,-.6,10.,random.Random(i)) for i in range(100)]
        ghost=[m.measure_radar(2.,10.,-.6,10.,random.Random(i)) for i in range(100)]
        self.assertEqual(real,ghost)
        self.assertTrue(any(v is None for v in ghost))


if __name__=='__main__':unittest.main()
