import unittest
import torch
from mz24_decisive_loss import decisive_terms,availability_objective


class DecisiveLossTests(unittest.TestCase):
    def test_negative_maximum_receives_pressure_and_witness_is_protected(self):
        a=torch.tensor([[[3.,1.,2.]]],requires_grad=True)
        known=torch.tensor([[[False,False,True]]])
        eligible=torch.ones(1,1,1,3,1,dtype=torch.bool)
        query=known[:,:,None,:,None];teacher=torch.tensor([100.,10.,-3.]).reshape_as(eligible)
        n,p,counts=decisive_terms(a,known,eligible,query,teacher);(n+p).backward()
        self.assertGreater(a.grad[0,0,0],0);self.assertEqual(float(a.grad[0,0,1]),0)
        self.assertLess(a.grad[0,0,2],0);self.assertEqual(counts,dict(negative_bags=1,positive_bags=1))

    def test_empty_bags_do_not_create_targets(self):
        a=torch.randn(2,1,3,requires_grad=True);known=torch.zeros_like(a,dtype=torch.bool)
        eligible=torch.zeros(2,1,1,3,4,dtype=torch.bool);query=torch.ones_like(eligible)
        teacher=torch.zeros_like(eligible,dtype=torch.float32)
        n,p,c=decisive_terms(a,known,eligible,query,teacher);(n+p).backward()
        self.assertEqual(float(n+p),0);self.assertTrue((a.grad==0).all());self.assertEqual(sum(c.values()),0)

    def test_dense_availability_and_mask_shapes_have_finite_gradients(self):
        a=torch.randn(2,2,3,requires_grad=True);known=torch.rand(2,2,3)>.5
        eligible=torch.rand(2,2,2,3,4)>.5
        query=eligible&known[:,:,None,:,None];teacher=torch.randn(2,2,2,3,4)
        loss,parts=availability_objective(a,known,eligible,query,teacher);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(a.grad).all())
        self.assertAlmostEqual(float(loss),parts['dense']+.25*(parts['negative']+parts['positive']),places=5)


if __name__=='__main__':unittest.main()
