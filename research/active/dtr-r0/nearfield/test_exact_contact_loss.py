"""Independent probability arithmetic and gradient checks for interval supervision."""
import unittest

import torch

from exact_contact_loss import interval_nll


class ExactContactLossTests(unittest.TestCase):
    def test_direct_cdf_arithmetic_all_censoring_cases(self):
        q=torch.tensor([.8,.7,.6,.9],dtype=torch.float64)
        mu=torch.tensor([1.2,.8,2.,2.7],dtype=torch.float64)
        scale=torch.tensor([.3,.4,.5,.2],dtype=torch.float64)
        targets=torch.tensor([1.4,.3,float('inf'),3.],dtype=torch.float64)
        probabilities=[]
        for i,z in enumerate(targets):
            def cdf(h):
                return q[i]*torch.sigmoid((h-mu[i])/scale[i])/torch.sigmoid((3-mu[i])/scale[i])
            if torch.isinf(z): p=1-q[i]
            elif z==.3: p=cdf(.325)
            else: p=cdf(min(float(z)+.025,3.))-cdf(max(float(z)-.025,.3))
            probabilities.append(p)
        expected=-torch.stack(probabilities).log().mean()
        torch.testing.assert_close(interval_nll(q,mu,scale,targets),expected)

    def test_all_censored_has_no_position_or_scale_gradient(self):
        q=torch.full((2,3),.7,dtype=torch.float64,requires_grad=True)
        mu=torch.full((2,3),1.5,dtype=torch.float64,requires_grad=True)
        scale=torch.full((2,3),.01,dtype=torch.float64,requires_grad=True)
        loss=interval_nll(q,mu,scale,torch.full((2,3),float('inf'),dtype=torch.float64))
        self.assertEqual(loss.ndim,0)
        torch.testing.assert_close(loss,-torch.log(torch.tensor(.3,dtype=torch.float64)))
        loss.backward()
        torch.testing.assert_close(mu.grad,torch.zeros_like(mu))
        torch.testing.assert_close(scale.grad,torch.zeros_like(scale))
        self.assertTrue(bool((q.grad>0).all()))

    def test_far_interval_small_scale_has_finite_nonzero_gradient(self):
        q=torch.tensor([.8],requires_grad=True)
        mu=torch.tensor([2.9],requires_grad=True)
        scale=torch.tensor([.010001],requires_grad=True)
        loss=interval_nll(q,mu,scale,torch.tensor([.5]))
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        for value in (q,mu,scale):
            self.assertTrue(bool(torch.isfinite(value.grad).all()))
            self.assertTrue(bool((value.grad.abs()>0).all()))
        self.assertGreater(float(mu.grad),0.)

    def test_large_scale_and_mixed_censoring_finite_gradient(self):
        q=torch.tensor([.4,.6,.8],dtype=torch.float64,requires_grad=True)
        mu=torch.tensor([.4,1.5,2.8],dtype=torch.float64,requires_grad=True)
        scale=torch.full((3,),1e6,dtype=torch.float64,requires_grad=True)
        loss=interval_nll(q,mu,scale,torch.tensor([.3,3.,float('inf')],dtype=torch.float64))
        self.assertTrue(bool(torch.isfinite(loss)));loss.backward()
        for value in (q,mu,scale): self.assertTrue(bool(torch.isfinite(value.grad).all()))
        self.assertEqual(float(mu.grad[-1]),0.)

    def test_nearer_location_reduces_interior_nll(self):
        q=torch.tensor([.8],dtype=torch.float64);scale=torch.tensor([.1],dtype=torch.float64)
        target=torch.tensor([1.5],dtype=torch.float64)
        good=interval_nll(q,torch.tensor([1.5],dtype=torch.float64),scale,target)
        bad=interval_nll(q,torch.tensor([2.5],dtype=torch.float64),scale,target)
        self.assertLess(float(good),float(bad))

    def test_exact_three_contact_distinct_from_right_censoring(self):
        q=torch.tensor([.8],dtype=torch.float64,requires_grad=True)
        mu=torch.tensor([2.5],dtype=torch.float64,requires_grad=True)
        scale=torch.tensor([.3],dtype=torch.float64,requires_grad=True)
        interval_nll(q,mu,scale,torch.tensor([3.],dtype=torch.float64)).backward()
        self.assertLess(float(q.grad),0.)  # observed contact raises q
        self.assertNotEqual(float(mu.grad),0.)


if __name__=='__main__': unittest.main()
