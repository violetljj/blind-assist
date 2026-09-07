"""Bounded geometry and causal closing tests; no capture/model inference."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from predict_whisker_depth import frame_geometry, temporal_scores, infer_images, batch_preflight
from evaluate_whisker import confusion, thresholds

class FakeMetric:
    """CPU-only mock of the inspected cached metric DAv2 method contract."""
    def __init__(self, batch_error=None):
        self.batch_error = batch_error
        self.preprocess_sizes = []

    def image2tensor(self, image, input_size):
        self.preprocess_sizes.append(input_size)
        value = torch.from_numpy(image.astype(np.float32)).permute(2,0,1)[None]/255
        value = torch.nn.functional.interpolate(value,(8,12),mode='bilinear',align_corners=True)
        return value, image.shape[:2]

    def forward(self, tensor):
        if len(tensor)>1 and self.batch_error=='oom':
            raise torch.cuda.OutOfMemoryError('Synthetic CPU mock OOM')
        depth = tensor.sum(dim=1)*2
        return depth+.01 if len(tensor)>1 and self.batch_error=='drift' else depth

    def infer_image(self, image, input_size):
        tensor,size = self.image2tensor(image,input_size)
        return torch.nn.functional.interpolate(self.forward(tensor)[:,None],size,
            mode='bilinear',align_corners=True)[0,0].numpy()

class WhiskerDepthTest(unittest.TestCase):
    def test_batch_adapter_preserves_preprocessing_and_resize(self):
        images = [np.random.default_rng(i).integers(0,256,(9,15,3),dtype=np.uint8) for i in range(5)]
        model = FakeMetric()
        reference = infer_images(model,images,1)
        candidate = infer_images(model,images,4)
        self.assertEqual(model.preprocess_sizes,[518]*10)
        for old,new in zip(reference,candidate):
            self.assertEqual(new.shape,(9,15))
            self.assertEqual(new.dtype,np.float32)
            np.testing.assert_array_equal(old,new)

    def test_preflight_rejects_batch_drift_without_gpu(self):
        images = [np.full((9,15,3),i,np.uint8) for i in range(8)]
        geometry = ({'near':[1,0],'ground':{'status':'ACCEPTED'}}, None)
        with patch('predict_whisker_depth.torch.cuda.synchronize'), patch('predict_whisker_depth.frame_geometry',return_value=geometry):
            result = batch_preflight(FakeMetric('drift'),images,90,6144)
        self.assertEqual(result['selected_batch'],1)
        self.assertEqual(result['fallback_reason'],'BATCH4_AGREEMENT_FAILED')
        self.assertEqual(result['frame_attempts'],37)
        self.assertEqual(result['forward_call_attempts'],22)
        self.assertEqual(result['probe_frames'],8)

    def test_preflight_specific_oom_falls_back_without_gpu(self):
        images = [np.zeros((9,15,3),np.uint8) for _ in range(8)]
        geometry = ({'near':[1,0],'ground':{'status':'ACCEPTED'}}, None)
        with patch('predict_whisker_depth.torch.cuda.synchronize'), patch('predict_whisker_depth.torch.cuda.empty_cache') as release, patch('predict_whisker_depth.frame_geometry',return_value=geometry):
            result = batch_preflight(FakeMetric('oom'),images,90,6144)
        self.assertEqual(result['selected_batch'],1)
        self.assertEqual(result['fallback_reason'],'BATCH4_CUDA_OUT_OF_MEMORY')
        release.assert_called_once()

    def test_supported_near_wall_without_ground_still_survives(self):
        detail,maps = frame_geometry(np.full((90,160),2.,np.float32),90,device='cpu')
        self.assertEqual(detail['near'],[1,1])
        self.assertEqual(maps.shape,(2,18,32))
        self.assertTrue((maps.sum(axis=(1,2))>0).all())
        self.assertEqual(detail['ground']['status'],'UNKNOWN')

    def test_invalid_depth_is_unknown_not_clear(self):
        detail,maps = frame_geometry(np.full((90,160),np.nan,np.float32),90,device='cpu')
        self.assertEqual(detail['near'],[-1,-1])
        self.assertEqual(detail['nearest_m'],[None,None])
        self.assertFalse(maps.any())

    def test_stopped_near_independent_of_approach(self):
        history = [dict(near=[1,1],nearest_m=[2.,2.]) for _ in range(3)]
        scores,speed = temporal_scores(history,[0.,.2,.4])
        self.assertEqual(scores,[1,1,0,0])
        self.assertTrue(np.allclose(speed,0))
        history[0]['nearest_m']=[2.2,None]
        scores,speed = temporal_scores(history,[0.,.2,.4])
        self.assertEqual(scores,[1,1,1,-1])
        self.assertGreater(speed[0],.1)
        self.assertIsNone(speed[1])

    def test_unknown_baseline_approach_is_not_negative(self):
        samples = [dict(sample_id='v',split='val'),dict(sample_id='t',split='test')]
        rows = {'v':np.zeros(4),'t':np.array([1,1,-1,-1])}
        targets = {'v':[0]*4,'t':[1,1,1,0]}
        cells = confusion(rows,samples[1:],targets,thresholds(rows,samples,targets))
        self.assertEqual(cells['BODYapproaching']['FN'],0)
        self.assertEqual(cells['BODYapproaching']['unknown_positive'],1)
        self.assertEqual(cells['BODYapproaching']['all_opportunity_recall'],0)
        self.assertEqual(cells['HEADapproaching']['TN'],0)
        self.assertEqual(cells['HEADapproaching']['unknown_negative'],1)

if __name__=='__main__':
    unittest.main()
