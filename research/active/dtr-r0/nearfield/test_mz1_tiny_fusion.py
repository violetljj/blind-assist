"""Artificial forward-only isolation checks; no optimizer or scientific training."""
import unittest
import numpy as np
import torch
from mz1_tiny_fusion import TinyFusion, ARMS, schedule, state_sha


class TinyFusionTests(unittest.TestCase):
    def test_initialization_shape_and_isolation(self):
        if not torch.cuda.is_available():
            self.skipTest('CUDA unavailable')
        torch.manual_seed(53)
        initial=TinyFusion().state_dict()
        visual=torch.randn(5,772,device='cuda');tof=torch.randn(5,256,device='cuda')
        models={}
        for arm in ARMS:
            model=TinyFusion(arm);model.load_state_dict(initial)
            self.assertEqual(state_sha(model.state_dict()),state_sha(initial))
            models[arm]=model.cuda().eval()
        with torch.inference_mode():
            for model in models.values():self.assertEqual(model(visual,tof).shape,(5,4))
            torch.testing.assert_close(models['RGB_ONLY'](visual,tof),models['RGB_ONLY'](visual,tof+100),rtol=0,atol=0)
            torch.testing.assert_close(models['TOF_ONLY'](visual,tof),models['TOF_ONLY'](visual-100,tof),rtol=0,atol=0)
            torch.testing.assert_close(models['RGB_ONLY'](visual,tof),models['FUSION'](visual,torch.zeros_like(tof)),rtol=0,atol=0)
            torch.testing.assert_close(models['TOF_ONLY'](visual,tof),models['FUSION'](torch.zeros_like(visual),tof),rtol=0,atol=0)

    def test_schedule_roles_and_repeatability(self):
        roles=np.array(['EVAL_ONLY','TRAIN_ONLY','DEV_ONLY','TRAIN_ONLY','TRAIN_ONLY'])
        a,b=schedule(roles),schedule(roles)
        self.assertEqual(a.shape,(300,128))
        np.testing.assert_array_equal(a,b)
        self.assertTrue(np.all(roles[a]=='TRAIN_ONLY'))
        self.assertEqual(set(a.ravel()),{1,3,4})


if __name__=='__main__':unittest.main()
