"""Focused sampling, frozen-weight and protected-input checks; synthetic only."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from factorial_train import (VARIANTS,EXPECTED,balanced_indices,variant_pools,
    load_inputs,configure_trainable,spatial_only,predict)
from whisker_model import WhiskerModel, IMAGE_SIZE, masked_bce, support_bce


TEST_TMP=Path(__file__).resolve().parents[4]/'artifacts.local/nearfield/factorial-test-tmp'


class FactorialTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEST_TMP.mkdir(parents=True,exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        TEST_TMP.rmdir()
    def fixture(self,root):
        (root/'model').mkdir(); (root/'training').mkdir()
        (root/'verification.json').write_text(json.dumps({'status':'PASS'}))
        samples=[]; frames=[]; labels={}; variants={}; support={}
        for group in range(64):
            split='train' if group<40 else 'val' if group<52 else 'test'
            for variant in VARIANTS:
                index=len(frames); clip=f'g{1000+group}_{variant}'; sid=clip+'_t0'
                samples.append(dict(sample_id=sid,clip_id=clip,group_id=f'g{1000+group}',split=split,frame_indices=[index]))
                frames.append(dict(sample_index=index,rgb_path=f'{index}.png',time_s=0.,clip_id=clip,frame_in_clip=0))
                if split!='test':
                    labels[sid]=EXPECTED[variant]+[-1,-1]; support[sid]=np.zeros((2,18,32),np.uint8)
                if split=='train': variants[sid]=variant
        dataset=dict(calibration=dict(width=640,height=360,horizontal_fov_degrees=100),samples=samples,frames=frames)
        (root/'model/dataset.json').write_text(json.dumps(dataset))
        (root/'training/labels.json').write_text(json.dumps({'targets':labels}))
        (root/'training/partition.json').write_text(json.dumps({'variants':variants}))
        np.savez(root/'training/support.npz',**support)
        return dataset,labels,variants

    def test_variant_balance_equal_priors_and_groups(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            root=Path(directory); dataset,labels,variants=self.fixture(root)
            load_inputs(root)
            samples=dataset['samples']
            for arm,count in (('single_data',80),('factorial_data',160)):
                pools=variant_pools(samples,variants,arm)
                self.assertEqual(sum(map(len,pools.values())),count)
                self.assertEqual(len({samples[i]['group_id'] for values in pools.values() for i in values}),40)
                first=balanced_indices(pools,np.random.default_rng(17))
                np.testing.assert_array_equal(first,balanced_indices(pools,np.random.default_rng(17)))
                target=np.asarray([labels[samples[i]['sample_id']][:2] for i in first])
                np.testing.assert_array_equal(target.sum(0),[16,16])
                for variant in pools:
                    self.assertEqual(sum(variants[samples[i]['sample_id']]==variant for i in first),32//len(pools))

    def test_protected_labels_variant_map_and_pose_rejected(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as directory:
            root=Path(directory); dataset,labels,variants=self.fixture(root)
            test_id=next(s['sample_id'] for s in dataset['samples'] if s['split']=='test')
            labels[test_id]=[0,0,-1,-1]
            (root/'training/labels.json').write_text(json.dumps({'targets':labels}))
            with self.assertRaisesRegex(ValueError,'test labels prohibited'): load_inputs(root)
            del labels[test_id]; (root/'training/labels.json').write_text(json.dumps({'targets':labels}))
            variants[test_id]='neither'; (root/'training/partition.json').write_text(json.dumps({'variants':variants}))
            with self.assertRaisesRegex(ValueError,'exactly TRAIN'): load_inputs(root)
            del variants[test_id]; (root/'training/partition.json').write_text(json.dumps({'variants':variants}))
            dataset['frames'][0]['speed_m_s']=0.
            (root/'model/dataset.json').write_text(json.dumps(dataset))
            with self.assertRaisesRegex(ValueError,'Privileged'): load_inputs(root)

    def test_spatial_only_preserves_frozen_weights_and_sentinel(self):
        torch.set_num_threads(2); torch.manual_seed(17)
        model=WhiskerModel('ordinary_video'); params=configure_trainable(model)
        frozen={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        rgb=torch.rand(2,3,*IMAGE_SIZE)
        near,support=spatial_only(model,rgb)
        target=torch.tensor([[1.,0.],[0.,1.]])
        optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-4)
        loss=masked_bce(near,target)+.25*support_bce(support,torch.zeros_like(support),target)
        loss.backward(); optimizer.step()
        for name,p in model.named_parameters():
            if name in frozen:
                self.assertIsNone(p.grad); torch.testing.assert_close(p,frozen[name],rtol=0,atol=0)
        scores,maps=predict(model,rgb,[1])
        self.assertEqual(scores.shape,(2,4)); self.assertEqual(maps.shape,(1,2,18,32))
        np.testing.assert_array_equal(scores[:,2:],-np.ones((2,2)))
        self.assertTrue(((scores[:,:2]>=0)&(scores[:,:2]<=1)).all())


if __name__=='__main__': unittest.main()
