import unittest
import torch
from mz22_evidence_arbiter import EvidenceArbiter,observable_features
from mz16_detail_readout import make_model
from mz15_shared_support import LocalSupportReadout


class ArbiterTests(unittest.TestCase):
    def test_initial_identity_and_unsupported_identity_after_update(self):
        model=EvidenceArbiter();x=torch.randn(8,4,18);b=torch.randn(8,4);s=torch.rand(8,4)>.5
        self.assertTrue(torch.equal(model(x,b,s),b))
        with torch.no_grad():
            for h in model.heads:h[-1].bias.fill_(3.)
        self.assertTrue(torch.equal(model(x,b,s)[~s],b[~s]))

    def test_features_finite_and_truth_free_for_empty_packets(self):
        torch.manual_seed(22)
        m=make_model(LocalSupportReadout(shared=False).state_dict()).eval()
        x=torch.randn(2,64,28,28);r=torch.ones(2,64,2);v=torch.zeros_like(r,dtype=torch.bool)
        with torch.no_grad():
            o=m.inspect(x,r,v);f=observable_features(o,torch.zeros(2,4),r,v,m.rays)
        self.assertEqual(tuple(f.shape),(2,4,18));self.assertTrue(torch.isfinite(f).all())
        self.assertFalse(o['support'].any())


if __name__=='__main__':unittest.main()
