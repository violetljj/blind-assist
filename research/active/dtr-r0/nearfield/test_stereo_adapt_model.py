"""Small CPU invariants; full-resolution CUDA feasibility lives in canary receipt."""
import copy
import unittest
import torch
from stereo_adapt_model import checkpoint_update_block,freeze_scope,state_hashes,selected_name


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.BatchNorm1d(3))
        self.update_block=torch.nn.Linear(3,3)
        self.spx_2_gru=torch.nn.Linear(3,3)
        self.spx_gru=torch.nn.Linear(3,1)
    def forward(self,x):
        return self.spx_gru(self.spx_2_gru(self.update_block(self.feature(x))))


class MutableBlock(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(3,3)
    def forward(self,net,inp,corr,disp,att):
        net[0]=torch.tanh(self.linear(net[0])+inp[0]+corr+disp+att[0])
        return net,net[0]*.25,net[0].mean(1,keepdim=True)


class AdaptationTests(unittest.TestCase):
    def test_exact_parameter_scope_and_eval_buffers(self):
        model=TinyModel();freeze_scope(model);before=state_hashes(model)
        self.assertFalse(any(m.training for m in model.modules()))
        optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
        model(torch.ones(4,3)).sum().backward();optimizer.step()
        after=state_hashes(model)
        self.assertEqual(before['frozen'],after['frozen'])
        self.assertEqual(before['buffers'],after['buffers'])
        self.assertNotEqual(before['selected'],after['selected'])
        self.assertTrue(all(p.requires_grad==selected_name(name) for name,p in model.named_parameters()))

    def test_module_prefix_is_exact(self):
        self.assertTrue(selected_name('update_block.mask.weight'))
        self.assertFalse(selected_name('update_block_backup.weight'))
        self.assertFalse(selected_name('feature.update_block.weight'))

    def test_checkpoint_mutated_list_matches_full_recurrence_gradient(self):
        torch.manual_seed(19)
        normal=MutableBlock();checked=copy.deepcopy(normal);checkpoint_update_block(checked)
        x=torch.randn(2,3)
        def run(block):
            initial=x.clone().requires_grad_();net=[initial]
            frozen=[torch.ones_like(x)*.1]
            loss=0
            for _ in range(4):
                net,mask,delta=block(net,frozen,frozen[0],frozen[0],frozen)
                loss=loss+mask.square().mean()+delta.square().mean()
            loss.backward()
            return loss.detach(),initial.grad,[p.grad.clone() for p in block.parameters()]
        a,b=run(normal),run(checked)
        torch.testing.assert_close(a[0],b[0]);torch.testing.assert_close(a[1],b[1])
        for x,y in zip(a[2],b[2]):torch.testing.assert_close(x,y)


if __name__=='__main__':unittest.main()
