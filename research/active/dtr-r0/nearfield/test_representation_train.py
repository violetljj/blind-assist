"""Focused official-backbone interface, paired initialization and gradients."""
from pathlib import Path
import unittest
import torch
from representation_model import RepresentationModel
from whisker_model import IMAGE_SIZE,masked_bce,support_bce

PRETRAINED=Path(__file__).resolve().parents[4]/'artifacts.local/nearfield/representation-20260908/pretrained'


class RepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_strict_official_load_and_common_initialization(self):
        torch.manual_seed(17); base=RepresentationModel(PRETRAINED,False)
        torch.manual_seed(17); detail=RepresentationModel(PRETRAINED,True)
        common=base.state_dict(); other=detail.state_dict()
        for key,value in common.items(): torch.testing.assert_close(value,other[key],rtol=0.,atol=0.)
        self.assertTrue(all(not key.startswith('classifier') for key in common))
        self.assertGreater(sum(p.numel() for p in detail.parameters()),sum(p.numel() for p in base.parameters()))
        self.assertEqual(base.deep_channels,384)
        self.assertEqual(base.shallow_channels,48)

    def test_current_rgb_outputs_and_end_to_end_head_gradient(self):
        torch.manual_seed(29); model=RepresentationModel(PRETRAINED,True)
        near,support=model(torch.rand(2,3,*IMAGE_SIZE))
        self.assertEqual(tuple(near.shape),(2,2)); self.assertEqual(tuple(support.shape),(2,2,18,32))
        target=torch.tensor([[1.,0.],[0.,1.]])
        loss=masked_bce(near,target)+.25*support_bce(support,torch.zeros_like(support),target)
        loss.backward()
        for module in (model.backbone[0],model.deep_projection,model.detail,model.near,model.support):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),0.)


if __name__=='__main__': unittest.main()
