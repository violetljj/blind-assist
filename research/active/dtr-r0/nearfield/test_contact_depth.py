import unittest
import numpy as np
import torch
from predict_contact_depth import contact_scores


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA geometry implementation')
class DepthContactTest(unittest.TestCase):
    def test_world_wall_remains_body_contact_when_head_turns(self):
        for yaw in (0., 12.):
            yy,xx=np.mgrid[:360,:640]
            f=320/np.tan(np.deg2rad(50))
            a=np.deg2rad(yaw)
            ray_x=np.cos(a)-np.sin(a)*(xx-319.5)/f
            depth=np.asarray(3./ray_x,dtype=np.float32)
            frame=dict(camera_transform=dict(x=0.,y=0.,z=1.82,pitch=0.,yaw=yaw,roll=0.),
                       wearer_transform=dict(x=0.,y=0.,z=.12,pitch=0.,yaw=0.,roll=0.),speed_m_s=1.)
            scores,_,detail=contact_scores(depth,frame,100.)
            self.assertEqual(scores,[[0,0,1],[0,0,1]])
            self.assertAlmostEqual(detail['contact_s'][0],2.82,places=4)
            self.assertAlmostEqual(detail['contact_s'][1],2.87,places=4)


if __name__=='__main__':unittest.main()
