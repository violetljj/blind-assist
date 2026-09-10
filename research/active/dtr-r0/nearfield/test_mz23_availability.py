import unittest
import torch
from mz23_availability import AngularAvailability,restrict_candidates


class AvailabilityTests(unittest.TestCase):
    def test_restriction_is_monotone_and_keeps_missing_explicit(self):
        f=torch.tensor([1.,3.]).reshape(1,1,1,2,1).expand(-1,-1,-1,-1,4)
        original=dict(candidate_logits=f,eligible=torch.ones_like(f,dtype=torch.bool))
        out=restrict_candidates(original,torch.tensor([[[1.,-1.]]]))
        self.assertTrue((out['logits']==1).all());self.assertTrue(out['support'].all())
        out=restrict_candidates(original,torch.full((1,1,2),-1.))
        self.assertFalse(out['support'].any());self.assertTrue((out['logits']==-20).all())
        self.assertTrue(original['eligible'].all())

    def test_invalid_packets_finite_and_no_query_or_native_input(self):
        m=AngularAvailability(torch.zeros(64,49,2))
        x=torch.randn(2,64,28,28);r=torch.full((2,64,2),float('nan'));v=torch.zeros_like(r,dtype=torch.bool)
        out=m(x,r,v);self.assertEqual(out.shape,(2,64,49));self.assertTrue(torch.isfinite(out).all())


if __name__=='__main__':unittest.main()
