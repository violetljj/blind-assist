import unittest
from audit_city_target_uncertainty import explain_ray


class RayExplanationTest(unittest.TestCase):
    def ray(self,depth,component='other',instance=0):
        return dict(status='UNKNOWN',isolated_axial_m=2.,collision_axial_m=depth,
                    component_path=component,instance_index=instance)

    def test_near_occluder_is_not_target_identity(self):
        self.assertEqual(explain_ray(self.ray(1.99),'target',None),
                         'OTHER_COMPONENT_NEARER_WITHIN_RENDER_TOLERANCE')

    def test_no_hit_and_farther_other_are_distinct(self):
        self.assertEqual(explain_ray({'status':'UNKNOWN'},'target',None),'NO_COLLISION_HIT')
        self.assertEqual(explain_ray(self.ray(2.07),'target',None),'OTHER_COMPONENT_AT_OR_BEHIND_TARGET')

    def test_wrong_instance_does_not_match(self):
        self.assertEqual(explain_ray(self.ray(2.07,'target',1),'target',0),'OTHER_COMPONENT_AT_OR_BEHIND_TARGET')
        self.assertEqual(explain_ray(self.ray(2.07,'target',0),'target',0),'TARGET_COLLISION_DEPTH_DISAGREEMENT')

    def test_passed_evidence_keeps_status(self):
        self.assertEqual(explain_ray({'status':'OCCLUDED'},'target',None),'OCCLUDED')


if __name__=='__main__':unittest.main()
