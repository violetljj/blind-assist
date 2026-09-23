"""Matched regional carrier mathematical and permutation contracts."""
import copy
import unittest
import torch
from torch.nn import functional as F
from regional_contact_model import RegionContactModel


class RegionalContactTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(813);torch.set_num_threads(2)
        self.model=RegionContactModel().double()
        self.x=torch.randn(2,64,46,dtype=torch.float64)
        self.query=torch.tensor([[.6,.3,0],[.6,1.5,0],[.6,3.,0],[.8,2.,1]],dtype=torch.float64)

    def test_horizon_independence_and_cdf(self):
        q,mu,s=self.model.components(self.x,self.query)
        for value in (q,mu,s):torch.testing.assert_close(value[:,:3],value[:,:1].expand(-1,3))
        p=self.model(self.x,self.query).sigmoid()
        expected=q*torch.sigmoid((self.query[:,1]-mu)/s)/torch.sigmoid((3-mu)/s)
        torch.testing.assert_close(p,expected);torch.testing.assert_close(p[:,2],q[:,2])
        self.assertTrue(bool((p[:,:3].diff(dim=1)>=0).all()))

    def test_shared_per_image_gradient_parity(self):
        other=copy.deepcopy(self.model);query=self.query[None].expand(2,-1,-1).clone()
        a=self.model(self.x,self.query);b=other(self.x,query);torch.testing.assert_close(a,b)
        y=torch.tensor([[0.,0.,1.,1.],[0.,1.,1.,0.]],dtype=torch.float64)
        F.binary_cross_entropy_with_logits(a,y).backward();F.binary_cross_entropy_with_logits(b,y).backward()
        for left,right in zip(self.model.parameters(),other.parameters()):
            self.assertTrue(bool(torch.isfinite(left.grad).all()));torch.testing.assert_close(left.grad,right.grad)

    def test_distinct_per_image_queries(self):
        query=self.query[None].repeat(2,1,1);query[1,:,0]=1.
        expected=torch.cat([self.model(self.x[i:i+1],query[i]) for i in range(2)])
        torch.testing.assert_close(self.model(self.x,query),expected)

    def test_full_token_permutation_invariance(self):
        order=torch.randperm(64)
        torch.testing.assert_close(self.model(self.x[:,order],self.query),self.model(self.x,self.query),atol=1e-12,rtol=1e-12)

    def test_rgb_only_permutation_changes_output(self):
        changed=self.x.clone();changed[:,:,:40]=self.x[:,torch.arange(64).roll(17),:40]
        self.assertGreater(float((self.model(changed,self.query)-self.model(self.x,self.query)).detach().abs().max()),1e-8)

    def test_missing_flag_tokens_not_filtered(self):
        x=self.x.clone();x[:,:,40]=0.;x[:,:,41]=0.
        changed=x.clone();changed[:,:,:40]+=1.
        self.assertTrue(bool(torch.isfinite(self.model(x,self.query)).all()))
        self.assertGreater(float((self.model(x,self.query)-self.model(changed,self.query)).detach().abs().max()),1e-8)

    def test_continuous_crossing_consistency(self):
        threshold=.3;z=self.model.continuous_crossing(self.x,self.query,threshold)
        queries=self.query[None].repeat(2,1,1);queries[:,:,1]=z
        p=self.model(self.x,queries).sigmoid();interior=(z>.3)&(z<3.)
        self.assertTrue(bool(interior.any()));torch.testing.assert_close(p[interior],torch.full_like(p[interior],threshold))
        self.assertTrue(bool(torch.isinf(self.model.continuous_crossing(self.x,self.query,1.000001)).all()))

    def test_invalid_input_and_parameter_count(self):
        self.assertEqual(self.model.parameter_counts()['effective'],20163)
        self.assertEqual(self.model.parameter_counts()['effective'],sum(p.numel() for p in self.model.parameters()))
        for x in (self.x[:,:63],self.x[:,:,1:],self.x.long(),torch.full_like(self.x,float('nan'))):
            with self.assertRaises(ValueError):self.model(x,self.query)
        for q in (torch.zeros(3),torch.tensor([[.6,.1,0.]]),torch.zeros(3,4,3)):
            with self.assertRaises(ValueError):self.model(self.x,q)


if __name__=='__main__':unittest.main()
