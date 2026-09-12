import unittest
import numpy as np
import mz99_angle_information_eval as e


def box(x=0.,z=1.,vx=0.,vz=-1.):
    return dict(x=x,z=z,vx=vx,vz=vz,extent_m=[.1,.1,.5])


class GeometryTests(unittest.TestCase):
    def test_current_far_route_is_not_imminent_contact(self):
        b=box(z=3.,vz=-.7)
        self.assertTrue(e.route_current(b));self.assertIsNone(e.contact_interval(b))

    def test_continuous_entry_and_receding(self):
        self.assertAlmostEqual(e.contact_interval(box())[0],.7)
        self.assertIsNone(e.contact_interval(box(vz=1.)))
        self.assertAlmostEqual(e.contact_interval(box(x=.7,z=0.,vx=-1.,vz=0.))[0],.3)

    def test_footprint_not_only_center(self):
        self.assertIsNotNone(e.contact_interval(box(x=.39,z=0.,vz=0.),0.))
        self.assertIsNone(e.contact_interval(box(x=.41,z=0.,vz=0.),0.))

    def test_session_gap_and_episode_reset(self):
        p=np.array([1,0,0,0,1,0,0,0,0,1,1],bool);ids=np.array(['a']*10+['b'])
        active,starts=e.sessions(p,ids)
        np.testing.assert_array_equal(np.flatnonzero(starts),[0,9,10])
        self.assertFalse(active[8]);self.assertTrue(active[3])
        for n in range(1,len(p)):np.testing.assert_array_equal(e.sessions(p[:n],ids[:n])[0],active[:n])

    def test_censor_and_early_carried_warning(self):
        ids=np.array(['a']*40);times=np.arange(40)*.1
        geometry=[dict(native_bounds=[box(z=2.3),box(z=.5),box(z=6.)])]*40
        events=e.contact_events(ids,times,geometry)
        self.assertEqual([v['status'] for v in events],['evaluable','left_censored','right_censored'])
        pred=np.ones(40,bool);m=e.timing_metrics(pred,ids,times,events)
        self.assertEqual(m['timely'],1);self.assertEqual(m['timely_carried_session'],1)

    def test_carried_session_then_restart_is_repeat(self):
        ids=np.array(['a']*7);pred=np.array([1,0,0,0,0,1,0],bool)
        truth=np.array([0,1,1,1,1,1,1],bool)
        self.assertEqual(e.session_metrics(pred,ids,truth)['repeated_starts_within_truth_interval'],1)

if __name__=='__main__':unittest.main()
