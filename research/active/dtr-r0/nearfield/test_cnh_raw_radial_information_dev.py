"""Focused input isolation and row alignment tests; no fitted experiment."""
import unittest
import numpy as np
import cnh_raw_radial_information_dev as M


class Inputs(unittest.TestCase):
    def setUp(self):
        self.pair=np.zeros((2,13,2),np.float32)
        self.data={k:np.full((2,13,n),i+1,np.float32) for i,(k,n) in enumerate(M.DIMS.items())}

    def test_nested_exact_feature_schema_and_label_isolation(self):
        self.data['contact']=np.ones(2,bool)
        a=M.assemble_features(self.pair,self.data)
        self.assertEqual([a[k].shape[-1] for k in M.ARMS],[138,162,546])
        np.testing.assert_array_equal(a['current_support'][...,:138],a['current'])
        np.testing.assert_array_equal(a['temporal'][...,:162],a['current_support'])
        self.data['contact'][:]=False
        b=M.assemble_features(self.pair,self.data)
        for k in M.ARMS:np.testing.assert_array_equal(a[k],b[k])

    def test_missing_temporal_retained_but_inf_and_current_nan_rejected(self):
        self.data['temporal_delta'][0,0,0]=np.nan
        self.assertTrue(np.isnan(M.assemble_features(self.pair,self.data)['temporal'][0,0,162]))
        self.data['temporal_delta'][0,0,0]=np.inf
        with self.assertRaises(ValueError):M.assemble_features(self.pair,self.data)
        self.data['temporal_delta'][0,0,0]=0
        self.data['current'][0,0,0]=np.nan
        with self.assertRaises(ValueError):M.assemble_features(self.pair,self.data)

    def test_alignment_rejects_reordered_hb_tags(self):
        data=dict(unit=np.array([1,1]),config=np.array([4,4]),anchor_id=np.array(['a','a']),tag=np.array(['H','B']))
        M.check_alignment(data,data['unit'],data['config'],data['anchor_id'],data['tag'])
        with self.assertRaises(AssertionError):M.check_alignment(data,data['unit'],data['config'],data['anchor_id'],data['tag'][::-1])


if __name__=='__main__':unittest.main()
