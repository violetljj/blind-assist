import unittest
from visual_geometry import assess_bounds

class VisualGeometryTests(unittest.TestCase):
    def test_offset_limb_envelope_cannot_pass_torso_proxy(self):
        s={'shape':'disc','radius_m':.28,'base_m':0.,'height_m':1.75}
        a=assess_bounds(s,[0,-.0623,.875],[.611,.189,.875],[0,0])
        self.assertFalse(a['passed'])
        s['radius_m']=.70
        self.assertTrue(assess_bounds(s,[0,-.0623,.875],[.611,.189,.875],[0,0])['passed'])
    def test_box_size_and_placement_both_checked(self):
        s={'shape':'box','half_extents_m':[.2,.3],'base_m':0.,'height_m':.12}
        self.assertTrue(assess_bounds(s,[1,2,.06],[.2,.3,.06],[1,2])['passed'])
        self.assertFalse(assess_bounds(s,[1.02,2,.06],[.2,.3,.06],[1,2])['passed'])
        self.assertFalse(assess_bounds(s,[1,2,.08],[.2,.3,.06],[1,2])['passed'])
if __name__=='__main__':unittest.main()
