"""Focused G13 initialization and attached support/deep-content path checks."""
from pathlib import Path
import unittest
import torch
from representation_model import RepresentationModel
from decoupled_model import DecoupledModel
from whisker_model import IMAGE_SIZE,masked_bce

PRETRAINED=Path(__file__).resolve().parents[4]/'artifacts.local/nearfield/representation-20260908/pretrained'


class DecoupledTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_exact_C_parameter_initialization(self):
        torch.manual_seed(17); original=RepresentationModel(PRETRAINED,True)
        torch.manual_seed(17); decoupled=DecoupledModel(PRETRAINED)
        self.assertEqual(set(original.state_dict()),set(decoupled.state_dict()))
        self.assertEqual(sum(p.numel() for p in original.parameters()),sum(p.numel() for p in decoupled.parameters()))
        for key,value in original.state_dict().items(): torch.testing.assert_close(value,decoupled.state_dict()[key],rtol=0.,atol=0.)

    def test_near_gradient_through_detail_support_but_contents_deep_only(self):
        torch.manual_seed(29); model=DecoupledModel(PRETRAINED).eval()
        rgb=torch.rand(2,3,*IMAGE_SIZE)
        near,_=model(rgb)
        masked_bce(near,torch.tensor([[1.,0.],[0.,1.]])).backward()
        self.assertGreater(float(model.detail[1].weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.support.weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.deep_projection.weight.grad.abs().sum()),0.)
        # Hold the predicted gate constant: changing E can no longer change
        # the near logits, proving E is not part of pooled feature contents.
        with torch.no_grad():
            model.support.weight.zero_(); model.support.bias.zero_()
            before,_=model(rgb)
            model.detail[1].bias.add_(2.)
            after,_=model(rgb)
        torch.testing.assert_close(before,after,rtol=0.,atol=0.)


if __name__=='__main__': unittest.main()
