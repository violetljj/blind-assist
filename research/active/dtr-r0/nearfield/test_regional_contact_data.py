import unittest
import numpy as np
from regional_contact_data import derangements,paired_inputs


class RegionalDataTest(unittest.TestCase):
    def test_derangement_is_public_reproducible_and_perframe(self):
        x=np.random.default_rng(19).normal(size=(4,4864)).astype(np.float32);x[3]=x[0]
        a=derangements(x);b=derangements(x.copy())
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a[0],a[3])
        self.assertTrue(np.all(a!=np.arange(64)));self.assertFalse(np.array_equal(a[0],a[1]))
        np.testing.assert_array_equal(np.sort(a,axis=1),np.tile(np.arange(64),(4,1)))

    def test_only_visual_pairing_changes_and_normalization_train_only(self):
        rng=np.random.default_rng(20);rgb=rng.normal(size=(3,64,40)).astype(np.float32);sensor=rng.normal(size=(3,64,6)).astype(np.float32)
        p=np.stack([np.roll(np.arange(64),i+1) for i in range(3)])
        a,b,n=paired_inputs(rgb,sensor,[0,1],p)
        np.testing.assert_array_equal(a[...,40:],sensor);np.testing.assert_array_equal(b[...,40:],sensor)
        np.testing.assert_array_equal(b[...,:40],a[np.arange(3)[:,None],p,:40])
        rgb[2]+=1000;_,_,other=paired_inputs(rgb,sensor,[0,1],p)
        np.testing.assert_array_equal(n['mean'],other['mean']);np.testing.assert_array_equal(n['std'],other['std'])


if __name__=='__main__':unittest.main()
