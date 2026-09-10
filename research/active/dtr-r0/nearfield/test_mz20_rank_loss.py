"""Gradient and label-boundary tests for cross-bag BODY witness ranking."""
import unittest
import torch
from mz20_rank_loss import body_rank_query_loss
from test_mz18_body_loss import readout, original_query_loss


class RankLossTests(unittest.TestCase):
    def loss(self, f, t, k, e, y):
        log, sup = readout(f, e)
        return body_rank_query_loss(f, t, k, e, y, log, sup)

    def test_cross_frame_hard_negative_rewards_true_witness(self):
        f = torch.zeros((2, 1, 1, 3, 4), requires_grad=True)
        t = torch.zeros_like(f, dtype=torch.bool)
        t[0, ..., 0, :2] = True
        e = torch.ones_like(t); e[..., 2:] = False
        k = torch.ones_like(t)
        with torch.no_grad():
            f[0, ..., 0, :2] = 2.
            f[1, ..., 1, :2] = 4.
        y = torch.tensor([[True, True, False, False], [False, False, False, False]])
        g = torch.autograd.grad(self.loss(f, t, k, e, y), f)[0]
        self.assertTrue((g[0, ..., 0, :2] < 0).all())
        self.assertTrue((g[1, ..., 1, :2] > 0).all())

    def test_unknown_and_ineligible_do_not_supply_false_pairs(self):
        f = torch.tensor([1., 2., 3., 20., 30.]).reshape(1,1,1,5,1).expand(-1,-1,-1,-1,4).clone().requires_grad_()
        t = torch.zeros_like(f, dtype=torch.bool); t[..., :2, :2] = True
        k = torch.ones_like(t); k[..., 3, :2] = False
        e = torch.ones_like(t); e[..., 4, :2] = False; e[..., 2:] = False
        y = torch.ones((1,4), dtype=torch.bool)
        g = torch.autograd.grad(self.loss(f,t,k,e,y),f,retain_graph=True)[0]
        self.assertTrue((g[...,1,:2] < 0).all())
        self.assertTrue((g[...,2,:2] > 0).all())
        self.assertTrue((g[...,3:,:2] == 0).all())
        empty = self.loss(f,t,torch.zeros_like(k),e,y)
        self.assertEqual(empty.item(),0.)
        self.assertTrue((torch.autograd.grad(empty,f)[0] == 0).all())

    def test_head_query_gradient_exact_including_ties(self):
        torch.manual_seed(20)
        f=torch.randn(3,1,1,5,4,requires_grad=True)
        t=torch.zeros_like(f,dtype=torch.bool); t[...,0,:]=True
        e=torch.ones_like(t); e[1,...,3]=False
        k=torch.ones_like(t); k[0,...,1,2:]=False
        y=torch.tensor([[True,False,True,True],[False,True,False,False],[True,True,True,False]])
        with torch.no_grad(): f[0,...,:2,2:]=5.
        old=torch.autograd.grad(original_query_loss(f,t,e,y),f,retain_graph=True)[0]
        new=torch.autograd.grad(self.loss(f,t,k,e,y),f)[0]
        self.assertTrue(torch.equal(old[...,2:],new[...,2:]))

    def test_negative_bag_retains_task_supervision_without_local_relabel(self):
        f=torch.zeros(2,1,1,2,4,requires_grad=True)
        t=torch.zeros_like(f,dtype=torch.bool);t[0,...,0,0]=True
        k=torch.zeros_like(t);k[0,...,0,0]=True
        e=torch.zeros_like(t);e[...,0]=True
        y=torch.zeros(2,4,dtype=torch.bool);y[0,0]=True
        with torch.no_grad():f[1,...,1,0]=4.
        before=k.clone()
        g=torch.autograd.grad(self.loss(f,t,k,e,y),f)[0]
        self.assertGreater(g[1,0,0,1,0].item(),0.)
        self.assertLess(g[0,0,0,0,0].item(),0.)
        self.assertTrue(torch.equal(k,before))


if __name__ == '__main__': unittest.main()
