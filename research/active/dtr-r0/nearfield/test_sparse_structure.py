"""Coordinate and observability tests; no synthetic score tuning."""
import unittest
import numpy as np
import torch
import cv2
from sparse_structure import pose_matrix, project, match


class GeometryTest(unittest.TestCase):
    def test_projection_translation_and_rotation(self):
        pose=dict(x=0.,y=0.,z=1.7,pitch=0.,yaw=0.)
        rotation, _ = pose_matrix(pose)
        np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-7)
        points=torch.tensor([[49.5,49.5]])
        inv=torch.tensor([.5,.25])
        uv,z=project(points,inv,{**pose,"y":-.2},pose,100,100,90,"cpu")
        np.testing.assert_allclose(uv.numpy()[0,:,0],[54.5,52.0],atol=1e-5)
        np.testing.assert_allclose(z.numpy(),[[2.,4.]])
        uv,_=project(points,inv,{**pose,"yaw":5},pose,100,100,90,"cpu")
        np.testing.assert_allclose(uv.numpy()[:,0],uv.numpy()[:,1],atol=1e-5)

    def test_pitch_projection_world_up(self):
        rotation,_=pose_matrix(dict(x=0,y=0,z=0,pitch=-10,yaw=0))
        self.assertLess(rotation[2,2],0)
        self.assertLess(rotation[2,1],0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA matcher required")
    def test_textured_plane_recovery_and_zero_baseline_unknown(self):
        image=np.random.default_rng(7).integers(0,256,(128,128,3),dtype=np.uint8)
        focal=64/np.tan(np.deg2rad(50))
        offsets=[-.2,-.1,0.]
        images=[cv2.warpAffine(image,np.array([[1,0,-offset*focal/2],[0,1,0]],np.float32),
            (128,128),borderMode=cv2.BORDER_REFLECT) for offset in offsets]
        poses=[dict(x=0,y=offset,z=1.7,pitch=0,yaw=0) for offset in offsets]
        result=match(images,poses)
        recovered=np.asarray(result["depth_m"])[np.asarray(result["accepted"],bool)]
        self.assertGreaterEqual(len(recovered),3)
        self.assertLess(abs(float(np.median(recovered))-2),.15)
        no_baseline=match([image]*3,[poses[-1]]*3)
        self.assertFalse(any(no_baseline["accepted"]))


if __name__ == "__main__":
    unittest.main()
