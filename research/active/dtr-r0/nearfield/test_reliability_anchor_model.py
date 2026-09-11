"""Focused synthetic CPU checks, no actual checkpoints/data/model experiment."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path('E:/linnan/linnan/research/active/dtr-r0/nearfield')))
import torch
from test_mz56_global_anchor_model import reference
from mz56_global_anchor_model import AnchorQuery,loss_for as original_loss
from reliability_anchor_model import ReliabilityAnchorQuery,loss_for


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def models(self):
        geometry=reference();torch.manual_seed(151)
        old=AnchorQuery(geometry.state_dict(),'GLOBAL_ANCHOR')
        # Exercise existing nonzero trained-like global columns.
        with torch.no_grad():old.head[0].weight[:,41:].fill_(.04)
        a=ReliabilityAnchorQuery(geometry.state_dict(),old.state_dict(),'BASE')
        b=ReliabilityAnchorQuery(geometry.state_dict(),old.state_dict(),'RELIABILITY')
        return old,a,b
    def test_initialization_geometry_and_logits(self):
        old,a,b=self.models();self.assertIs(loss_for,original_loss)
        for model in (a,b):
            self.assertEqual(sum(p.numel() for p in model.parameters()),11116)
            torch.testing.assert_close(model.head[0].weight[:,:49],old.head[0].weight,atol=0,rtol=0)
            self.assertFalse(model.head[0].weight[:,49:].any())
            for name,value in old.named_buffers():torch.testing.assert_close(dict(model.named_buffers())[name],value,atol=0,rtol=0)
        for (ka,va),(kb,vb) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(ka,kb);torch.testing.assert_close(va,vb,atol=0,rtol=0)
        x=torch.randn(2,64,45,80);r=torch.linspace(.1,4,256).reshape(2,64,2);v=torch.ones_like(r,dtype=torch.bool);v[1]=False
        expected=old.inspect(x,r,v)
        for model in (a,b):
            actual=model.inspect(x,r,v)
            torch.testing.assert_close(actual['field'],expected['field'],atol=2e-5,rtol=1e-6)
            for key in ('OPEN','CROP'):
                torch.testing.assert_close(actual[key]['raw'],expected[key]['raw'],atol=2e-5,rtol=1e-6)
                self.assertTrue(torch.equal(actual[key]['support'],expected[key]['support']))
            self.assertFalse(actual['geometry_proxies'][1].any())
    def test_known_scalar_neighbors_and_missing(self):
        _,_,m=self.models();r=torch.zeros(1,64,2);v=torch.zeros_like(r,dtype=torch.bool)
        # Center27: up19=1,left26=2,right28=3,down35=4. Lower median2, lower MAD1.
        for z,d in [(27,2.5),(19,1),(26,2),(28,3),(35,4)]:r[0,z,0]=d;v[0,z,0]=True
        g,a=m.neighbor_geometry(r,v)
        torch.testing.assert_close(g[0,27],torch.tensor([.125,.25,1.]),atol=0,rtol=0)
        self.assertTrue(a[0,27]);self.assertFalse(g[0,0].any())
        reversed_r=r.flip(-1);reversed_v=v.flip(-1)
        torch.testing.assert_close(m.neighbor_geometry(reversed_r,reversed_v)[0],g,atol=0,rtol=0)
        bad=r.clone();bad[~v]=torch.nan
        torch.testing.assert_close(m.neighbor_geometry(bad,v)[0],g,atol=0,rtol=0)
        v[:]=False;v[0,0,0]=True;r[:]=0;r[0,0,0]=1
        g,a=m.neighbor_geometry(r,v);self.assertFalse(g.any());self.assertFalse(a.any())
    def test_gradient_and_local_only_new_feature_effect(self):
        _,a,b=self.models();x=torch.randn(1,64,45,80);r=torch.ones(1,64,2);v=torch.ones_like(r,dtype=torch.bool)
        for m in (a,b):
            m.zero_grad();out=m.inspect(x,r,v);out['field'].sum().backward()
            gradient=m.head[0].weight.grad[:,49:]
            self.assertEqual(bool(gradient.abs().sum()>0),m.augmentation_arm=='RELIABILITY')
        with torch.no_grad():
            # Both receive identical weights; only supplied geometry differs.
            a.head[0].weight[:,49:].fill_(.2);b.head[0].weight[:,49:].fill_(.2)
            a.head[0].bias.fill_(2);b.head[0].bias.fill_(2)
            a.head[2].weight.fill_(.2);b.head[2].weight.fill_(.2)
        af=a.inspect(x,r,v)['field'];bf=b.inspect(x,r,v)['field'];coverage=a.sensor_coverage
        torch.testing.assert_close(af[:,~coverage],bf[:,~coverage],atol=0,rtol=0)
        self.assertTrue(bool((af[:,coverage]-bf[:,coverage]).abs().max()>0))

if __name__=='__main__':unittest.main()
