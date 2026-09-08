"""Focused City cache boundaries and CUDA support-supervision regression."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from city_data import CityRGBDataset, CitySupervisedDataset, pixel_support_bce, pool_support


class CityDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        artifacts = Path(__file__).resolve().parents[4] / 'artifacts.local'
        evidence = artifacts / 'tests/city-data'
        evidence.mkdir(parents=True, exist_ok=True)
        # Retain small, hash-bound synthetic fixtures as verification evidence.
        cls.evidence = Path(tempfile.mkdtemp(prefix='focused-', dir=evidence)).resolve()
        print('City test fixtures:', cls.evidence, flush=True)

    def cache(self, *, rgb_shape=(2,144,256,3), support_shape=(2,2,18,32),
              support_value=0, near_value=0., role='train'):
        root = Path(tempfile.mkdtemp(prefix='cache-', dir=self.evidence))
        def record(name, array):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, array, allow_pickle=False)
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            return dict(path=name, sha256=digest)
        rgb = np.full(rgb_shape, 128, dtype=np.uint8)
        rgb_record = record('model/rgb.npy', rgb)
        near = record('supervision/near.npy', np.full((2,2),near_value,dtype=np.float32))
        support = np.full(support_shape,support_value,dtype=np.int8)
        if support_value == 0:
            support.flat[:3] = [-1,0,1]
        support_record = record('supervision/support.npy',support)
        (root/'supervision'/f'{role}.json').write_text(json.dumps(dict(
            sample_indices=[7,11],near=near,support=support_record)),encoding='utf-8')
        (root/'manifest.json').write_text(json.dumps(dict(schema='city-training-cache-v1',
            status='PASS',partitions={role:dict(sample_indices=[7,11],rgb=rgb_record)})),encoding='utf-8')
        return root

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for pooling regression')
    def test_pool_positive_unknown_negative_and_thin_trace(self):
        native = torch.zeros((1,2,360,640),dtype=torch.int8,device='cuda')
        native[0,0,0,0] = -1
        native[0,0,0,1] = 1  # Positive wins even when the same cell is unknown.
        native[0,0,0,20] = -1
        native[0,1,359,639] = 1  # One-pixel edge support must survive pooling.
        pooled = pool_support(native)
        expected = torch.zeros((1,2,18,32),dtype=torch.int8,device='cuda')
        expected[0,0,0,0] = 1
        expected[0,0,0,1] = -1
        expected[0,1,17,31] = 1
        self.assertTrue(torch.equal(pooled,expected))
        self.assertTrue(torch.equal(pool_support(torch.full_like(native,-1)),torch.full_like(expected,-1)))
        self.assertTrue(torch.equal(pool_support(torch.zeros_like(native)),torch.zeros_like(expected)))

    def test_pool_rejects_invalid_shape_and_values(self):
        for shape in ((2,360,640),(1,1,360,640),(1,2,359,640)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                pool_support(torch.zeros(shape,dtype=torch.int8))
        for value in (2.,-2.,float('nan')):
            native = torch.zeros((1,2,360,640))
            native[0,0,0,0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                pool_support(native)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for gradient regression')
    def test_support_gradient_excludes_unknown_and_balances_known(self):
        logits = torch.zeros((1,1,1,4),device='cuda',requires_grad=True)
        targets = torch.tensor([[[[-1,1,0,0]]]],device='cuda')
        loss = pixel_support_bce(logits,targets)
        self.assertAlmostEqual(loss.item(),np.log(2),places=6)
        loss.backward()
        expected = torch.tensor([[[[0.,-.25,.125,.125]]]],device='cuda')
        torch.testing.assert_close(logits.grad,expected,rtol=0,atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for gradient regression')
    def test_all_unknown_has_zero_loss_and_gradient(self):
        logits = torch.tensor([[[[-20.,0.,20.]]]],device='cuda',requires_grad=True)
        loss = pixel_support_bce(logits,torch.full_like(logits,-1))
        self.assertEqual(loss.item(),0.)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad,torch.zeros_like(logits)))

    def test_loss_rejects_bad_target_or_shape(self):
        for target in (torch.tensor([2.]),torch.tensor([float('nan')]),torch.zeros(2)):
            with self.subTest(target=target), self.assertRaises(ValueError):
                pixel_support_bce(torch.zeros(1),target)

    def test_supervised_refuses_test_before_opening_any_path(self):
        with patch.object(Path,'open',side_effect=AssertionError('Unexpected file access')):
            with self.assertRaisesRegex(ValueError,'TEST'):
                CitySupervisedDataset(self.evidence/'absent','test')

    def test_test_rgb_never_reads_supervision_or_evaluator(self):
        root = self.cache(role='test')
        (root/'evaluator').mkdir()
        (root/'evaluator/test.json').write_text('INVALID TEST LABELS',encoding='utf-8')
        original_open = Path.open
        accessed = []
        def guarded_open(path,*args,**kwargs):
            relative = path.resolve().relative_to(root)
            accessed.append(relative.as_posix())
            if relative.parts[0] in ('supervision','evaluator'):
                raise AssertionError('RGB dataset accessed privileged labels')
            return original_open(path,*args,**kwargs)
        with patch.object(Path,'open',guarded_open):
            dataset = CityRGBDataset(root,'test')
            item = dataset[1]
        self.assertEqual(set(item),{'rgb','sample_index'})
        self.assertEqual(item['sample_index'],11)
        self.assertEqual(tuple(item['rgb'].shape),(3,144,256))
        torch.testing.assert_close(item['rgb'],torch.full_like(item['rgb'],128/255))
        self.assertTrue(accessed)

    def test_supervised_valid_trivalued_cache(self):
        dataset = CitySupervisedDataset(self.cache())
        self.assertEqual(len(dataset),2)
        item = dataset[0]
        self.assertEqual(tuple(item['near'].shape),(2,))
        self.assertEqual(tuple(item['support'].shape),(2,18,32))
        self.assertEqual(item['support'].flatten()[:3].tolist(),[-1,0,1])

    def test_supervised_rejects_redirected_test_labels_before_read(self):
        for key in ('near','support'):
            for prefix in ('evaluator','supervision/../evaluator'):
                with self.subTest(key=key,prefix=prefix):
                    root = self.cache()
                    labels_path = root/'supervision/train.json'
                    labels = json.loads(labels_path.read_text())
                    source = root/labels[key]['path']
                    protected = root/'evaluator'/f'{key}.npy'
                    protected.parent.mkdir()
                    protected.write_bytes(source.read_bytes())  # The original SHA remains valid.
                    labels[key]['path'] = f'{prefix}/{key}.npy'
                    labels_path.write_text(json.dumps(labels),encoding='utf-8')
                    original_open = Path.open
                    def guarded_open(path,*args,**kwargs):
                        if path.resolve() == protected:
                            raise AssertionError('Opened evaluator labels before boundary rejection')
                        return original_open(path,*args,**kwargs)
                    with patch.object(Path,'open',guarded_open):
                        with self.assertRaisesRegex(ValueError,'boundary'):
                            CitySupervisedDataset(root)

    def test_rgb_rejects_label_subtree_redirect_before_read(self):
        for folder in ('evaluator','supervision'):
            with self.subTest(folder=folder):
                root = self.cache(role='test')
                manifest_path = root/'manifest.json'
                manifest = json.loads(manifest_path.read_text())
                record = manifest['partitions']['test']['rgb']
                protected = root/folder/'rgb-shaped-label.npy'
                protected.parent.mkdir(exist_ok=True)
                protected.write_bytes((root/record['path']).read_bytes())
                record['path'] = f'{folder}/rgb-shaped-label.npy'
                manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
                original_open = Path.open
                def guarded_open(path,*args,**kwargs):
                    if path.resolve() == protected:
                        raise AssertionError('Opened privileged payload before boundary rejection')
                    return original_open(path,*args,**kwargs)
                with patch.object(Path,'open',guarded_open):
                    with self.assertRaisesRegex(ValueError,'boundary'):
                        CityRGBDataset(root,'test')

    def test_supervised_rejects_invalid_values_and_shape(self):
        for kwargs in (dict(support_value=2),dict(near_value=float('nan')),
                       dict(near_value=2),dict(support_shape=(2,2,17,32))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CitySupervisedDataset(self.cache(**kwargs))

    def test_rgb_shape_and_actual_file_hash_are_enforced(self):
        with self.assertRaises(ValueError):
            CityRGBDataset(self.cache(rgb_shape=(2,143,256,3)),'train')
        root = self.cache()
        with (root/'model/rgb.npy').open('ab') as stream:
            stream.write(b'changed-after-manifest')
        with self.assertRaisesRegex(ValueError,'hash'):
            CityRGBDataset(root,'train')


if __name__ == '__main__':
    unittest.main(verbosity=2)
