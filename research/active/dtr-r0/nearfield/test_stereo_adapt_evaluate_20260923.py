import unittest
import numpy as np
import stereo_adapt_evaluate_20260923 as e


class EvaluationTests(unittest.TestCase):
    def empty(self):return np.full((360,640),np.nan,dtype=np.float32)

    def test_missing_fails_fixed_camera_head_denominator(self):
        native=self.empty();native[180,320]=1.
        rows,far=e.pixel_stats(self.empty(),self.empty(),native)
        self.assertEqual(rows['HEAD']['native_query_pixels'],1)
        self.assertEqual(rows['HEAD']['missing'],1)
        self.assertEqual(rows['HEAD']['correct_5cm'],0)
        self.assertEqual(rows['BODY']['native_query_pixels'],0)

    def test_far_background_estimated_near_counted_separately(self):
        native=self.empty();pred=self.empty();native[180,320]=8.;pred[180,320]=1.
        rows,far=e.pixel_stats(pred,pred,native)
        self.assertEqual(rows['HEAD']['native_query_pixels'],0)
        self.assertEqual(rows['HEAD']['far_native_estimated_query'],1)
        self.assertEqual(far['far_native_estimated_near'],1)

    def test_unfiltered_above4_preserved_in_distribution(self):
        native=self.empty();raw=self.empty();native[180,320]=1.;raw[180,320]=7.
        rows,_=e.pixel_stats(self.empty(),raw,native)
        self.assertEqual(rows['HEAD']['raw_depth']['above4m'],1)
        self.assertEqual(rows['HEAD']['raw_depth']['finite_positive'],1)
        self.assertEqual(rows['HEAD']['predicted_valid'],0)

    def test_depth_accuracy_requires_same_query_membership(self):
        native=self.empty();pred=self.empty();native[180,400]=1.;pred[180,400]=1.04
        rows,_=e.pixel_stats(pred,pred,native)
        self.assertEqual(rows['HEAD']['abs_z_5cm'],1)
        self.assertEqual(rows['HEAD']['correct_5cm'],0)

    def test_hidden_scene_contact_remains_missing_failure(self):
        r=dict(truth=1.5,pred=None,native_visible=None)
        result=e.c.boundary_stats([r])
        self.assertEqual(result['truth_contact'],1)
        self.assertEqual(result['missing_contact'],1)
        self.assertEqual(result['hit_5cm'],0.)


if __name__=='__main__':unittest.main()
