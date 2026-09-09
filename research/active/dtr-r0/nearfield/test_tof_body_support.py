"""Ideal cone containment and positive-only evidence regression."""
import math
import unittest
from tof_body_support import support,add_positive_support
from contact_retina_spec import BODY_BOXES


class ConeSupportTest(unittest.TestCase):
    def test_invalid_does_not_clear(self):
        result=support(None,False,15,.05)
        self.assertEqual(result['status'],'UNKNOWN')
        self.assertEqual(add_positive_support([False,True,True,False],result),[False,True,True,False])

    def test_narrow_cone_supports_near_without_target_identity(self):
        result=support(1.,True,15,.05)
        self.assertEqual(result['supported'],['HEAD_NEAR'])
        self.assertEqual(add_positive_support([False,False,False,True],result),[False,False,True,True])

    def test_wide_or_far_return_cannot_be_assigned_to_head(self):
        self.assertEqual(support(1.,True,27,.05)['supported'],[])
        self.assertEqual(support(2.6,True,15,.05)['supported'],[])

    def test_radial_shell_bounds_contain_sampled_directions(self):
        # Independent explicit ray construction, including edges, center and corners.
        for diagonal in (15,27):
            slope=math.tan(math.radians(diagonal/2))/math.sqrt(2)
            for distance in (.3,.8,1.,1.5,2.6,4.):
                result=support(distance,True,diagonal,.05);lo,hi=result['bounds']
                for radius in (distance-.05,distance,distance+.05):
                    for u in (-slope,-slope/2,0,slope/2,slope):
                        for v in (-slope,-slope/2,0,slope/2,slope):
                            n=math.sqrt(1+u*u+v*v);point=(radius/n,radius*u/n,1.7+radius*v/n)
                            for j in range(3):self.assertTrue(lo[j]-1e-12<=point[j]<=hi[j]+1e-12)
                            if 'HEAD_NEAR' in result['supported']:
                                bodylo,bodyhi=BODY_BOXES[1]
                                self.assertTrue(bodyhi[0]-1e-12<=point[0]<=bodyhi[0]+1.5+1e-12)
                                self.assertTrue(bodylo[1]-1e-12<=point[1]<=bodyhi[1]+1e-12)
                                self.assertTrue(bodylo[2]-1e-12<=point[2]<=bodyhi[2]+1e-12)


if __name__=='__main__':unittest.main()
