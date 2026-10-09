"""Narrow CPU fixtures for split, deletion, label and public-input invariants."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import cnh_counterfactual_data_dev as D


class CounterfactualDataFixture(unittest.TestCase):
    def test_split_marginals_and_shared_physical_draws(self):
        rows=D.scene_spec();audit=D.validate_spec(rows)
        self.assertEqual(audit['train_joint_cells']['counterfactual'],768)
        self.assertLess(audit['train_joint_cells']['ordinary'],768)
        worlds={}
        for row in rows['cf_train']:
            worlds[(row['physical_key'],row['occurrence_draw_id'])]=D.sampling_seed(row,0)
        shared=0
        for row in rows['ordinary_train']:
            key=(row['physical_key'],row['occurrence_draw_id'])
            if key in worlds:
                self.assertEqual(D.sampling_seed(row,0),worlds[key]);shared+=1
        self.assertGreater(shared,384)
        train={r['target_id'] for r in rows['cf_train']}
        cal={r['target_id'] for r in rows['cal']}
        validation={r['target_id'] for r in rows['validation']}
        self.assertFalse(train&cal or train&validation or cal&validation)

    def test_all_object_surface_labels_against_frozen_clip(self):
        S,_,_,B=D.frozen_imports();sensor,_=B.poses(-10.)
        rows=D.scene_spec()['cal']
        fixtures=[r for r in rows if r['background_id']==8]
        fixtures += [dict(boxes=[D.box([-.80,-.50,.20],[.80,1.50,4.],.5)])]
        for row in fixtures:
            for f in (3,13,15):
                shift=sensor[f,:3,3]
                tri=np.concatenate([S.box_mesh(b['lo'],b['hi']) for b in row['boxes']])-shift
                expected=[]
                for yl,yh in ((-.20,.42),(.42,.90)):
                    lower=np.array([-.30+D.EPS,yl+D.EPS,.30+D.EPS]);upper=np.array([.30-D.EPS,yh-D.EPS,3.-D.EPS])
                    wide_lo=lower.copy();wide_hi=upper.copy();wide_lo[0]=-.4;wide_hi[0]=.4
                    expected.append('contact' if len(S.clip_triangles(tri,lower,upper)) else
                                    'pass' if len(S.clip_triangles(tri,wide_lo,wide_hi)) else 'clear')
                self.assertEqual(D.category_boxes(row['boxes'],shift),expected)
        # Background contact is neither hidden nor forced clear by target removal.
        physical=[D.box([.1,.50,.65],[.2,.70,.8],.5)]
        self.assertEqual(D.category_boxes(physical,np.zeros(3)),['clear','contact'])

    def test_target_removal_is_new_unoccluded_render(self):
        _,R,_,B=D.frozen_imports();sensor,_=B.poses(-10.)
        bg=D.backgrounds()[0]['boxes'];target=D.target('vertical',0,1,'in1cm',0,.65)
        poses=sensor[[13]]
        absent=R.expected(dict(poses=poses,boxes=bg))
        opaque_zero=R.expected(dict(poses=poses,boxes=[dict(target,rho=0.),*bg]))
        self.assertGreater(float(np.max(abs(absent['expectation']-opaque_zero['expectation']))),0.)
        self.assertEqual(D.physical_key(bg),D.physical_key(list(bg)))
        self.assertNotEqual(D.physical_key(bg),D.physical_key([target,*bg]))

    def test_native_padding_transform_and_training_masks(self):
        _,_,_,B=D.frozen_imports();sensor,query=B.poses(-10.)
        root=D.OUT/'tmp';root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='data-fixture-',dir=root) as temp:
            folder=Path(temp)/'cal';folder.mkdir()
            hist=np.zeros((1,2,16,8,8,16),np.int32)
            ambient=np.full((16,8,8),30.,float)
            categories=np.broadcast_to(np.array([['pass','clear']]),(1,13,2)).copy()
            receipt=D.materialize(hist,ambient,sensor,query,categories,folder,lambda:None)
            h=np.load(folder/'histories.npy');t=np.load(folder/'transforms.npy');le=np.load(folder/'length.npy')
            self.assertEqual(h.shape,(26,8,8,8,16));self.assertEqual(h.dtype,np.float16)
            self.assertTrue((h[0,:4]==0).all());self.assertEqual(int(le[0]),4)
            np.testing.assert_allclose(t[0,-1],query[3],atol=1e-12)
            np.testing.assert_array_equal(np.load(folder/'labels.npy'),np.zeros((26,2)))
            mask=np.load(folder/'mask.npy');self.assertTrue((mask[:,0]==0).all());self.assertTrue((mask[:,1]==1).all())
            self.assertAlmostEqual(receipt['normalized_weight_mass'],26,places=5)
            yaw=np.load(folder/'transforms_yaw3.npy')
            self.assertFalse(np.array_equal(yaw[0,-1],t[0,-1]))
            np.testing.assert_array_equal(yaw[0,:4],t[0,:4])


if __name__=='__main__':unittest.main()
