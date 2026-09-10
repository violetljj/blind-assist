import unittest
import torch
import numpy as np
from mz15_shared_support import matched_pair
from mz15_evaluator import align_stress_labels


class SupportTests(unittest.TestCase):
    def test_shared_field_and_matched_initialization(self):
        torch.manual_seed(115)
        a, b = matched_pair()
        x = torch.randn(2, 64, 18, 32)
        r, v = torch.full((2, 64, 2), 2.), torch.ones((2, 64, 2), dtype=torch.bool)
        o, p = a.inspect(x, r, v), b.inspect(x, r, v)
        torch.testing.assert_close(o['logits'], p['logits'])
        for q in range(4):
            torch.testing.assert_close(o['candidate_logits'][..., q], o['field'][..., 0])
        o['field'].square().mean().backward()
        self.assertGreater(float(a.local[0].weight.grad.abs().sum()), 0)

    def test_invalid_and_visual_correspondence(self):
        torch.manual_seed(116)
        a, _ = matched_pair()
        x = torch.randn(2, 64, 18, 32)
        r, v = torch.full((2, 64, 2), 2.), torch.ones((2, 64, 2), dtype=torch.bool)
        correct, wrong = a.inspect(x, r, v), a.inspect(x, r, v, True)
        self.assertTrue(torch.equal(correct['eligible'], wrong['eligible']))
        self.assertGreater(float((correct['field']-wrong['field']).detach().abs().max()), 0)
        invalid = a.inspect(x, r.fill_(float('nan')), v.fill_(False))
        self.assertFalse(bool(invalid['observation_valid'].any()))
        self.assertFalse(bool(invalid['support'].any()))
        self.assertTrue(torch.isfinite(invalid['logits']).all())
        self.assertTrue((invalid['logits'] == -20).all())

    def test_stress_echo_relocation_keeps_actual_source_identity(self):
        src=np.zeros((1,2,2,49),dtype=bool);q=np.zeros((1,2,2,49,4),dtype=bool)
        src[0,0,0,3]=True;src[0,0,1,17]=True;q[0,0,1,17,3]=True
        cr=np.array([[[1.,3.],[1.,2.]]]);cv=np.ones_like(cr,dtype=bool)
        sr=cr.copy();sr[0,0]=[3.,0.];sv=cv.copy();sv[0,0]=[True,False]
        s,t,m=align_stress_labels(src,q,cr,cv,sr,sv)
        self.assertTrue(m[0,0]);self.assertFalse(m[0,1])
        self.assertTrue(s[0,0,0,17]);self.assertTrue(t[0,0,0,17,3])
        self.assertFalse(s[0,0,0,3]);self.assertFalse(s[0,0,1].any())
        self.assertTrue(src[0,0,0,3])


if __name__ == '__main__':
    unittest.main()
