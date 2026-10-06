"""Independent geometric control scenes for target visibility semantics."""
import unittest
import numpy as np
import cnh_mode2_seam_dev as D


class GeometryControls(unittest.TestCase):
    def test_target_front_visible_back_hidden_and_occluder_blocks(self):
        box={'lo':[-.2,-.2,1.],'hi':[.2,.2,1.2]}
        points=np.array([[0.,0.,1.],[0.,0.,1.2]])
        np.testing.assert_array_equal(D.first_hit_visible(points,[box],np.zeros(3)),[True,False])
        blocker={'lo':[-.1,-.1,.5],'hi':[.1,.1,.6]}
        self.assertFalse(D.first_hit_visible(points,[box,blocker],np.zeros(3)).any())

    def test_forward_inside_both_splayed_views_not_on_edge(self):
        sensor=np.eye(4);sensor[:3,:3]=D.R.rotation(-10,'x')
        p=np.array([[0.,0.,1.]])
        v=D.visibility(p,np.ones(1),sensor)
        self.assertEqual(v['L_fraction'],1.);self.assertEqual(v['R_fraction'],1.)
        self.assertEqual(v['split_fraction'],0.)
        self.assertGreater(v['best_horizontal_margin_deg'],7.)

    def test_true_target_query_clip_excludes_unrelated_far_surface(self):
        box={'lo':[-.1,-.1,1.],'hi':[.1,.1,1.2]};far={'lo':[1.,-.1,1.],'hi':[1.2,.1,1.2]}
        sensor=np.eye(4);sensor[:3,:3]=D.R.rotation(-10,'x')
        G=D.geometry_module()
        self.assertEqual(D.target_geometry([box],np.eye(4),sensor,[0],G)['status'],'EVALUABLE')
        self.assertEqual(D.target_geometry([far],np.eye(4),sensor,[0],G)['status'],'NO_DESIGNATED_TARGET_SURFACE_IN_CONTACT_QUERIES')


if __name__=='__main__':unittest.main()
