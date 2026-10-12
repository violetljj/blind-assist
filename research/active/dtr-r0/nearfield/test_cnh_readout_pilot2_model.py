"""Small CPU engineering checks; no generated scenes or model fitting."""
import unittest
import torch

from cnh_readout_pilot2_model import T2, parameter_count


class ModelChecks(unittest.TestCase):
    def test_background_context_padding_and_gradient(self):
        torch.set_num_threads(2);torch.manual_seed(123)
        net=T2()
        z=torch.randn(1,8,8,8,16)
        transforms=torch.eye(4,dtype=torch.float64)[None,None].repeat(1,8,1,1)
        length=torch.tensor([4]);ambient=torch.ones(1,8,8,8)
        # Zero query overlap must retain observed background context.
        weights=torch.zeros(1,8,2,8,8,16)
        scores=net(z,transforms,length,ambient,weights)
        self.assertEqual(tuple(scores.shape),(1,2))
        self.assertTrue(bool(torch.isfinite(scores).all()))
        noisy=z.clone();noisy[:,:4]=1000
        noisy_a=ambient.clone();noisy_a[:,:4]=1000
        noisy_t=transforms.clone();noisy_t[:,:4,:3,3]=100
        self.assertTrue(torch.equal(scores,net(noisy,noisy_t,length,noisy_a,weights)))
        changed=z.clone();changed[:,-1]-=20
        self.assertFalse(torch.equal(scores,net(changed,transforms,length,ambient,weights)))
        scores.square().sum().backward()
        grads=[p.grad for p in net.parameters() if p.grad is not None]
        self.assertTrue(grads and all(bool(torch.isfinite(g).all()) for g in grads))
        self.assertGreater(float(net.encoder[0].weight.grad.abs().sum()),0)
        self.assertEqual(parameter_count(),38161)


if __name__=='__main__':unittest.main()
