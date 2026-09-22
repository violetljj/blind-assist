"""Synthetic constraint and support tests, without scientific input access."""
import unittest
import numpy as np

from boundary_error_math import (label_bracket, curve_crossing,
    disjoint_crossing, point_boundary, support_span)
from contact_boundary_data import queries, contact_labels


class BoundaryErrorMathTest(unittest.TestCase):
    def test_diagnostic_receipt_numpy_scalars_and_missing_values(self):
        import json
        from run_boundary_error import nullable
        row=dict(count=np.int64(3),flag=np.bool_(True),missing=float('inf'))
        self.assertEqual(json.loads(json.dumps(nullable(row),allow_nan=False)),
                         dict(count=3,flag=True,missing=None))

    def test_full_grid_same_axis_and_unseen_tail(self):
        q = queries()['seen']
        box = [(np.array([.56,.5,1.63]), np.array([.7,.8,2.]))]
        labels = contact_labels(box,q)[0]
        lo,hi = label_bracket(q,labels,'width',0)
        self.assertAlmostEqual(lo,1.08,places=6)
        self.assertTrue(np.isinf(hi))  # cannot infer outside1.2
        self.assertTrue(contact_labels(box,np.array([[1.2,3.,0]]))[0][0])

    def test_cross_axis_no_information_beyond_cartesian_slice(self):
        q = queries()['seen']
        boxes = [(np.array([.22,.5,1.63]),np.array([.4,.8,2.])),
                 (np.array([.45,.5,.8]),np.array([.6,.8,1.]))]
        y=contact_labels(boxes,q)[0]
        np.testing.assert_allclose(label_bracket(q,y,'width',0),[.36,.6],atol=1e-6)
        np.testing.assert_allclose(label_bracket(q,y,'horizon',0),[1.5,1.8],atol=1e-6)

    def test_all_positive_and_layer_isolation(self):
        q=queries()['seen'];y=q[:,2]==0
        np.testing.assert_allclose(label_bracket(q,y,'width',0),[0,.36],atol=1e-6)
        self.assertTrue(np.isinf(label_bracket(q,y,'width',1)[1]))

    def test_crossing_interval_accounts_for_sample_spacing(self):
        c=curve_crossing([.2,.22,.24],[.1,.2,.9],.5)
        self.assertFalse(disjoint_crossing((.225,.23),c))
        self.assertTrue(disjoint_crossing((.24,.3),c))
        self.assertFalse(disjoint_crossing((.2399999,.3),c))
        self.assertEqual(curve_crossing([.2,.22],[.9,.9],.5)[3],'left')
        self.assertEqual(curve_crossing([.2,.22],[.1,.1],.5)[3],'right')

    def test_negative_ray_and_vertical_clipping(self):
        np.testing.assert_allclose(support_span((-.2,-.1),(.2,.3),(1,3),'width',0),[.28,1.2])
        # Exact ray reaches BODY bottom at Z=2.1; x=.2Z limits corridor to1.5.
        self.assertIsNone(support_span((.2,.2),(.2,.2),(1,3),'horizon',0))
        np.testing.assert_allclose(support_span((0,0),(.2,.2),(1,3),'width',0),[0,0])

    def test_native_silence_and_layer_boundary(self):
        p=np.array([[.25,.42,1.6],[.1,.8,4.],[.2,-.3,1.]])
        self.assertEqual(point_boundary(p,[1,1,1],'width',0),(0.5,0))
        self.assertEqual(point_boundary(p,[1,1,1],'horizon',1),(1.6,0))
        self.assertTrue(np.isinf(point_boundary(p,[0,0,0],'width',0)[0]))
        pts=np.array([[.1,.5,1.],[.305,.5,1.]])
        self.assertEqual(point_boundary(pts,[1,1],'width',0),(.2,0))
        self.assertEqual(point_boundary(pts,[1,1],'width',0,.61),(.61,1))

    def test_truth_curve_resolution_is_not_training_knot_resolution(self):
        for axis in (np.linspace(.2,1.2,51),np.linspace(.3,3,55)):
            for t in np.linspace(axis[0]+.001,axis[-1]-.001,27):
                pred=curve_crossing(axis,axis>=t,.5)[2]
                self.assertLessEqual(abs(pred-t),.050001)


if __name__=='__main__':
    unittest.main()
