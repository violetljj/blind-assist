import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import unittest
import numpy as np
import torch

from cnh_qg1_model import QueryGeometryModel, depth_features
from cnh_qg1_train import select32
from cnh_learning_diagnostic import select32 as original_select32


class QG1ModelTest(unittest.TestCase):
    def model(self):
        return QueryGeometryModel(np.array([[20.,0,15.5],[0,20.,11.5],[0,0,1]]), (32,24))

    def test_roi_centres_shape_and_feature_locality(self):
        model=self.model()
        model.roi_bounds[:]=torch.tensor([4.,4.,10.,10.])
        features=torch.zeros(1,24,12,16)
        features[...,2:6,2:6]=3
        result=model.roi_pool(features)
        self.assertEqual(tuple(result.shape),(1,6,24,8,8))
        self.assertTrue(torch.all(result[...,:4,:4]==3))
        self.assertTrue(torch.all(result[...,4:,:]==0))
        altered=features.clone()
        altered[...,:2,:]=999
        altered[...,6:,:]=999
        altered[...,:, :2]=999
        altered[...,:,6:]=999
        torch.testing.assert_close(model.roi_pool(altered),result,rtol=0,atol=0)

    def test_empty_roi_does_not_sample_origin(self):
        model=self.model()
        model.roi_bounds.zero_()
        self.assertTrue(torch.all(model.roi_pool(torch.ones(1,24,12,16))==0))

    def test_depth_channels_finite_and_geometric(self):
        depth=torch.tensor([[[2.,float('nan'),0.,100.,-1.]]])
        rays=torch.tensor([[[.5,.5,.5,.5,.5]], [[-.25,-.25,-.25,-.25,-.25]]])
        features=depth_features(depth,rays)
        self.assertEqual(tuple(features.shape),(1,5,1,5))
        self.assertTrue(torch.isfinite(features).all())
        torch.testing.assert_close(features[0,:,0,0],torch.tensor([.1,-.05,.2,1.,1/3]))
        self.assertTrue(torch.all(features[0,:,:,1:]==0))

    def test_frozen_32_selection_is_same_label_blind_function(self):
        self.assertIs(select32,original_select32)
        rows=[dict(layout_id=name,frame_key=f'{name}:{i}') for name in ('A','B','C','dev') for i in range(160)]
        data=dict(rows=rows,train=np.array([r['layout_id']!='dev' for r in rows]))
        expected=np.concatenate([base+np.linspace(0,159,n).round().astype(int)
                                 for base,n in ((0,11),(160,11),(320,10))])
        np.testing.assert_array_equal(select32(data),expected)
        self.assertEqual(len(set(select32(data))),32)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_deterministic_cuda_backward_all_arms(self):
        previous=torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            for arm in ('full_depth','tof_sim','tof_rgb'):
                torch.manual_seed(17)
                model=self.model().cuda()
                image=torch.randn(2,5,24,32,device='cuda')
                histogram=torch.randn(2,64,16,device='cuda')
                ambient=torch.ones(2,64,device='cuda')
                scalar=torch.ones_like(ambient)
                valid=torch.ones_like(ambient,dtype=torch.bool)
                logits=model(image,histogram,ambient,scalar,valid,arm=arm)
                self.assertEqual(tuple(logits.shape),(2,6))
                logits.square().mean().backward()
                modules=[model.head]
                if arm!='tof_sim':modules.append(model.image_encoder)
                if arm!='full_depth':modules.append(model.tof_project)
                for module in modules:
                    gradients=[p.grad for p in module.parameters() if p.grad is not None]
                    self.assertTrue(gradients)
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    self.assertGreater(sum(float(g.abs().sum()) for g in gradients),0)
        finally:
            torch.use_deterministic_algorithms(previous)


if __name__=='__main__':
    unittest.main()
