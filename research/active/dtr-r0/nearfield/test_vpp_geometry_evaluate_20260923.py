"""Synthetic paired-accounting tests, no historical cohort execution."""
import unittest
import numpy as np
import vpp_geometry_evaluate_20260923 as v


class PairedGeometryTests(unittest.TestCase):
    def setUp(self):
        self.pose=dict(yaw=0.,pitch=0.,roll=0.,camera_in_body_m=[0.,0.,1.6])

    def empty(self):
        return np.full((360,640),np.nan,dtype=np.float32)

    def test_correction_regression_partition_includes_missing(self):
        native,a,b=self.empty(),self.empty(),self.empty()
        for x in range(320,324):native[180,x]=1.
        a[180,320]=1.;b[180,320]=1.
        b[180,321]=1.
        a[180,322]=1.
        row=v.paired_geometry(a,b,native,self.pose)[1]
        self.assertEqual(row,dict(native_query_pixels=4,both_correct=1,corrected=1,
            regressed=1,both_incorrect=1,unguided_correct=2,guided_correct=2))

    def test_axial_accuracy_without_query_membership_does_not_count(self):
        native,a,b=self.empty(),self.empty(),self.empty()
        native[180,400]=1.
        # Native side displacement .175 is within HEAD; +4cm shifts it outside.
        b[180,400]=1.04
        row=v.paired_geometry(a,b,native,self.pose)[1]
        self.assertEqual(row['native_query_pixels'],1)
        self.assertEqual(row['guided_correct'],0)

    def test_stamp_distances_partition_fixed_denominator(self):
        native,a,b=self.empty(),self.empty(),self.empty()
        for x in (320,321,325,340):native[180,x]=b[180,x]=1.
        rows=v.paired_geometry(a,b,native,self.pose,[dict(u=320,v=180)])[1]
        self.assertEqual(rows['stamp_0to1px_corrected'],2)
        self.assertEqual(rows['near_2to16px_corrected'],1)
        self.assertEqual(rows['beyond16px_corrected'],1)
        self.assertEqual(rows['zero_whole_frame_corrected'],0)

    def test_no_head_direct_hint_is_not_zero_frame(self):
        # A BODY seed, 0.2m below HEAD box: nearby HEAD, but not direct HEAD.
        hints=dict(total_frame_hints=1,seeds=[dict(xyz_camera=[1.,0.,-.4])])
        counts=v.public_hint_counts(hints,self.pose)
        head=v.hint_slices(counts,'HEAD')
        self.assertTrue(head['no_direct_hint'])
        self.assertTrue(head['nearby_hint'])
        self.assertFalse(head['zero_whole_frame'])
        self.assertEqual(counts['query_hints']['BODY']['direct'],1)

    def test_zero_frame_no_spurious_nearby_hint(self):
        counts=v.public_hint_counts(dict(total_frame_hints=0,seeds=[]),self.pose)
        self.assertEqual(v.hint_slices(counts,'HEAD'),dict(no_direct_hint=True,nearby_hint=False,zero_whole_frame=True))

    def test_task_transition_directions(self):
        a=np.array([True,True,False,False])
        b=np.array([False,False,True,True])
        gt=np.array([True,False,True,False])
        self.assertEqual(v.transitions(a,b,gt),dict(retained_tp=0,lost_tp=1,new_tp=1,
            removed_fp=1,added_fp=1,retained_fp=0))


if __name__=='__main__':unittest.main()
