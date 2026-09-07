"""Only the three requested synthetic CPU detach invariants; no source fitting."""
import unittest
import torch
from detached_model import DetachedModel
from factorial_train import configure_trainable
from whisker_model import IMAGE_SIZE,masked_bce,support_bce


class DetachedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_near_gradient_detached_from_support_but_shared_features_remain(self):
        torch.manual_seed(17)
        model=DetachedModel(True); configure_trainable(model)
        near,_=model(torch.rand(2,3,*IMAGE_SIZE))
        masked_bce(near,torch.tensor([[1.,0.],[0.,1.]])).backward()
        self.assertIsNone(model.support.weight.grad)
        self.assertIsNone(model.support.bias.grad)
        self.assertGreater(float(model.spatial[0].weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.near[2].weight.grad.abs().sum()),0.)

    def test_support_loss_updates_support(self):
        torch.manual_seed(29)
        model=DetachedModel(True); configure_trainable(model)
        before=model.support.weight.detach().clone()
        _,support=model(torch.rand(2,3,*IMAGE_SIZE))
        targets=torch.zeros_like(support); targets[:,:,4,8]=1.
        loss=support_bce(support,targets,torch.ones(2,2))
        optimizer=torch.optim.SGD(model.support.parameters(),lr=.01)
        loss.backward(); optimizer.step()
        self.assertGreater(float(model.support.weight.grad.abs().sum()),0.)
        self.assertFalse(torch.equal(before,model.support.weight))

    def test_inference_and_parameters_identical(self):
        torch.manual_seed(43)
        original=DetachedModel(False); detached=DetachedModel(True)
        detached.load_state_dict(original.state_dict(),strict=True)
        self.assertEqual(set(original.state_dict()),set(detached.state_dict()))
        self.assertEqual(sum(p.numel() for p in original.parameters()),sum(p.numel() for p in detached.parameters()))
        rgb=torch.rand(2,3,*IMAGE_SIZE)
        with torch.inference_mode():
            left=original(rgb); right=detached(rgb)
        for a,b in zip(left,right): torch.testing.assert_close(a,b,rtol=0.,atol=0.)


if __name__=='__main__': unittest.main()
