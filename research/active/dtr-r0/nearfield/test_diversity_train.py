"""Synthetic CPU checks; temporary artifacts remain under artifacts.local."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from diversity_model import DiversityModel
from diversity_train import load_inputs,make_pools,predict,balanced_indices,configure_trainable,VARIANTS,EXPECTED
from whisker_model import IMAGE_SIZE,masked_bce

ARTIFACTS=Path(__file__).resolve().parents[4]/'artifacts.local/nearfield'


class DiversityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def fixture(self,root,prefix='g'):
        (root/'model').mkdir(); (root/'training').mkdir()
        (root/'verification.json').write_text(json.dumps({'status':'PASS'}))
        frames=[]; samples=[]; labels={}; variants={}; support={}
        for group,split in enumerate(('train','val','test')):
            for variant in VARIANTS:
                index=len(frames); clip=f'{prefix}{group}_{variant}'; sid=clip+'_t0'
                frames.append(dict(sample_index=index,rgb_path=f'{index}.png',time_s=0.,clip_id=clip,frame_in_clip=0))
                samples.append(dict(sample_id=sid,clip_id=clip,group_id=f'{prefix}{group}',split=split,frame_indices=[index]))
                if split!='test': labels[sid]=EXPECTED[variant]+[-1,-1]; support[sid]=np.zeros((2,18,32),np.uint8)
                if split=='train': variants[sid]=variant
        dataset=dict(calibration=dict(width=640,height=360,horizontal_fov_degrees=100),frames=frames,samples=samples)
        (root/'model/dataset.json').write_text(json.dumps(dataset))
        (root/'training/labels.json').write_text(json.dumps({'targets':labels}))
        (root/'training/partition.json').write_text(json.dumps({'variants':variants}))
        np.savez(root/'training/support.npz',**support)
        return dataset,labels,variants

    def test_region_same_parameters_and_unnormalized_per_head_gate(self):
        torch.manual_seed(17); plain=DiversityModel(False); region=DiversityModel(True)
        region.load_state_dict(plain.state_dict(),strict=True)
        self.assertEqual(sum(p.numel() for p in plain.parameters()),sum(p.numel() for p in region.parameters()))
        self.assertEqual(set(plain.state_dict()),set(region.state_dict()))
        rgb=torch.rand(2,3,*IMAGE_SIZE)
        with torch.no_grad():
            region.support.weight.zero_(); region.support.bias.zero_()
            original=plain(rgb)[0]; gated=region(rgb)[0]; bias=region.near[2].bias
            torch.testing.assert_close(gated,.5*(original-bias)+bias,atol=2e-6,rtol=1e-5)
            region.support.bias.copy_(torch.tensor([80.,-80.]))
            gated=region(rgb)[0]
            torch.testing.assert_close(gated[:,0],original[:,0],atol=2e-6,rtol=1e-5)
            torch.testing.assert_close(gated[:,1],bias[1].expand(2),atol=1e-7,rtol=0.)

    def test_near_loss_backpropagates_through_region_support_not_frozen_modules(self):
        torch.manual_seed(29); model=DiversityModel(True); configure_trainable(model)
        near,_=model(torch.rand(2,3,*IMAGE_SIZE))
        masked_bce(near,torch.tensor([[1.,0.],[0.,1.]])).backward()
        self.assertGreater(float(model.support.weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.spatial[0].weight.grad.abs().sum()),0.)
        for name,param in model.named_parameters():
            if name.split('.')[0] not in {'spatial','near','support'}:
                self.assertFalse(param.requires_grad); self.assertIsNone(param.grad)

    def test_only_train_pools_and_equal_variant_priors(self):
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as a,tempfile.TemporaryDirectory(dir=ARTIFACTS) as b:
            old,_,ov=self.fixture(Path(a),'old'); new,_,nv=self.fixture(Path(b),'new')
            load_inputs(Path(a),{'train':1,'val':1,'test':1})
            for expanded,count in ((False,4),(True,8)):
                pools=make_pools(old['samples'],ov,new['samples'],nv,expanded)
                self.assertEqual(sum(map(len,pools.values())),count)
                combined=old['samples']+new['samples']
                self.assertTrue(all(combined[i]['split']=='train' for values in pools.values() for i in values))
                selected=balanced_indices(pools,np.random.default_rng(17))
                for variant,values in pools.items(): self.assertEqual(sum(i in values for i in selected),8)

    def test_loader_prohibits_test_targets_and_extra_metadata(self):
        with tempfile.TemporaryDirectory(dir=ARTIFACTS) as directory:
            root=Path(directory); dataset,labels,variants=self.fixture(root)
            test_id=next(s['sample_id'] for s in dataset['samples'] if s['split']=='test')
            labels[test_id]=[0,0,-1,-1]; (root/'training/labels.json').write_text(json.dumps({'targets':labels}))
            with self.assertRaisesRegex(ValueError,'test labels prohibited'): load_inputs(root,{'train':1,'val':1,'test':1})
            del labels[test_id]; (root/'training/labels.json').write_text(json.dumps({'targets':labels}))
            dataset['frames'][0]['speed_m_s']=0; (root/'model/dataset.json').write_text(json.dumps(dataset))
            with self.assertRaisesRegex(ValueError,'Privileged'): load_inputs(root,{'train':1,'val':1,'test':1})

    def test_prediction_maps_only_test_and_closing_unknown(self):
        model=DiversityModel(True)
        scores,maps=predict(model,torch.rand(3,3,*IMAGE_SIZE),[{'split':s} for s in ('train','val','test')])
        self.assertEqual(scores.shape,(3,4)); self.assertEqual(maps.shape,(1,2,18,32))
        np.testing.assert_array_equal(scores[:,2:],-np.ones((3,2)))


if __name__=='__main__': unittest.main()
