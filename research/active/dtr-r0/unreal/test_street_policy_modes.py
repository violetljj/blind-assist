import unittest
from street_live_policy import MotionPolicy


class ModesTest(unittest.TestCase):
    def call(self,mode,risk=False,front=None,valid=1):
        c={'clearance_m':{str(k):12.0 for k in (0.,-.72,.72,-.52,.52)},
           'front_obstacle_m':front,'valid_fraction':valid}
        return MotionPolicy(mode).command(t=1,x=0,y=0,goal_x=8,dtr_risk=risk,corridors=c)

    def test_dtr_only_has_no_depth_control(self):
        self.assertEqual(self.call('DTR_ONLY',front=.2,valid=0)['action'],'WALK')
        self.assertEqual(self.call('DTR_ONLY',risk=True,front=.2)['action'],'WAIT_DYNAMIC')

    def test_depth_only_has_no_dtr_control(self):
        a=self.call('DEPTH_ONLY',risk=True)
        self.assertEqual(a['action'],'WALK')
        self.assertFalse(a['dtr_route_risk'])
        self.assertEqual(self.call('DEPTH_ONLY',front=.2)['action'],'BRAKE_IMMINENT')

    def test_joint_retains_both_channels(self):
        self.assertEqual(self.call('JOINT',risk=True)['action'],'WAIT_DYNAMIC')
        self.assertEqual(self.call('JOINT',front=.2)['action'],'BRAKE_IMMINENT')
        self.assertEqual(self.call('JOINT',valid=0)['action'],'WAIT_UNKNOWN_DEPTH')

    def test_disabled_channel_not_reported_at_goal(self):
        p=MotionPolicy('DEPTH_ONLY')
        c={'clearance_m':{'0.0':12},'front_obstacle_m':None,'valid_fraction':1}
        r=p.command(t=10,x=8,y=0,goal_x=8,dtr_risk=True,corridors=c)
        self.assertEqual(r['action'],'ARRIVED')
        self.assertFalse(r['dtr_route_risk'])

if __name__=='__main__':unittest.main()
