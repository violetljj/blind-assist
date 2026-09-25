import unittest
import numpy as np
import torch
from cnh_learning_diagnostic import select32
from cnh_rgb_visible_depth_ceiling import depth_batch


class LearningDiagnosticTest(unittest.TestCase):
    def test_selection_never_uses_dev_or_labels(self):
        rows = [{'layout_id': name} for name in ('trainC','dev','trainA','trainB') for _ in range(160)]
        train = np.array([r['layout_id'] != 'dev' for r in rows])
        data = dict(rows=rows,train=train)
        ids=select32(data)
        self.assertEqual(len(set(ids)),32)
        self.assertTrue(train[ids].all())
        self.assertEqual([sum(rows[i]['layout_id']==n for i in ids) for n in ('trainA','trainB','trainC')],[11,11,10])

    def test_actual_depth_transform_validity_and_units(self):
        depth=np.ones((1,360,640),dtype=np.float32)
        depth[0,0,:5]=[0,np.nan,np.inf,-1,100]
        value=depth_batch(depth,np.array([0]),torch.device('cpu')).numpy()
        self.assertTrue(np.isfinite(value).all())
        np.testing.assert_array_equal(value[0,:2,0,:5],0)
        self.assertAlmostEqual(float(value[0,0,1,1]),float(np.log(2)/np.log(101)),places=7)
        self.assertEqual(value[0,1,1,1],1)
        np.testing.assert_array_equal(value[0,2],0)


if __name__=='__main__':
    unittest.main()
