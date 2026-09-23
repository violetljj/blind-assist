"""CPU-only public/target admission tests; never creates an adapter or GPU work."""
import copy
import unittest
from unittest.mock import patch

from stereo_adapt_run_20260923 import admitted_examples


def manifest(stage):
    count=192 if stage=='train' else 96
    rows=[]
    for i in range(count):
        r=dict(panel='adapt',id=f'{stage}_{i:03d}',layout_id=f'{stage}_layout_{i//4}',split=stage)
        for key in ('left','right','ranges','valid'):r[key]='unused';r[key+'_sha256']='testhash'
        if stage=='train':r['target_depth']='unused-target'
        rows.append(r)
    return dict(frames=rows,authority='SUPERVISED_TRAIN_EXAMPLES_ONLY' if stage=='train' else 'PUBLIC_RGB_TOF_CALIBRATION_ONLY_NO_TARGETS',
                target_role='training_ground_truth_NOT_SENSOR_INPUT')


class RunnerContractTests(unittest.TestCase):
    @patch('stereo_adapt_run_20260923.sha',return_value='testhash')
    def test_train_targets_and_eval_public_admitted(self,_):
        self.assertEqual(len(admitted_examples(manifest('train'),'train')),192)
        self.assertEqual(len(admitted_examples(manifest('eval'),'infer')),96)

    @patch('stereo_adapt_run_20260923.sha',return_value='testhash')
    def test_eval_rejects_target_or_geometry_before_hash_read(self,_):
        for key in ('target_depth','native_depth','truth','camera','native_bounds','objects'):
            m=manifest('eval');m['frames'][0][key]='forbidden'
            with self.assertRaises(AssertionError):admitted_examples(m,'infer')

    @patch('stereo_adapt_run_20260923.sha',return_value='testhash')
    def test_train_rejects_eval_partition_and_unsupervised_authority(self,_):
        m=manifest('train');m['frames'][0]['split']='eval'
        with self.assertRaises(AssertionError):admitted_examples(m,'train')
        m=manifest('train');m['authority']='PUBLIC_RGB_TOF_CALIBRATION_ONLY_NO_TARGETS'
        with self.assertRaises(AssertionError):admitted_examples(m,'train')

    @patch('stereo_adapt_run_20260923.sha',return_value='testhash')
    def test_duplicate_ids_rejected(self,_):
        m=manifest('eval');m['frames'][1]=copy.deepcopy(m['frames'][0])
        with self.assertRaises(AssertionError):admitted_examples(m,'infer')


if __name__=='__main__':unittest.main()
