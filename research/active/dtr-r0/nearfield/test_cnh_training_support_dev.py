"""Scoped tests for matched augmentation sampling and paired structure metrics."""
import unittest
import numpy as np
import cnh_training_support_dev as M


class TrainingSupport(unittest.TestCase):
    def setUp(self):
        self.contact=np.array([1,0,1,0,0,0],bool)
        self.control=~self.contact
        self.deadline=np.array([3,12,5,12,12,12])
        self.train=np.array([1,1,1,1,0,0],bool)
        self.source=np.array([0,1,4,5]);self.units=np.array([1,1,3,3])

    def schedule(self):
        return M.augmented_training_rows(self.contact,self.control,self.deadline,self.train,self.source,self.units,[1,2])

    def test_added_rows_train_only_and_joint_equal_control_weights(self):
        selected,ri,ti,y,w=self.schedule()
        np.testing.assert_array_equal(selected,[0,1])
        self.assertFalse(np.isin(ri,[4,5]).any())
        self.assertEqual(int(y.sum()),3)
        self.assertAlmostEqual(w[y==1].sum(),3)
        self.assertAlmostEqual(w[y==0].sum(),3)
        np.testing.assert_array_equal(ri[y==1],[0,2,6])
        np.testing.assert_array_equal(ti[y==1],[3,5,3])
        for t,expected in ((3,2/3),(5,1/3)):
            mask=(y==0)&(ti==t)
            np.testing.assert_array_equal(np.sort(ri[mask]),[1,3,7])
            np.testing.assert_allclose(w[mask],expected,rtol=0,atol=1e-12)

    def test_heldout_truth_and_control_deadline_do_not_change_schedule(self):
        before=self.schedule()
        self.contact[4]=True;self.control[4]=False;self.deadline[4]=1
        self.deadline[self.control]=-999
        after=self.schedule()
        for a,b in zip(before,after):np.testing.assert_array_equal(a,b)

    def test_source_unit_role_mismatch_rejected(self):
        with self.assertRaises(AssertionError):
            M.augmented_training_rows(self.contact,self.control,self.deadline,self.train,self.source,self.units,[3])

    def test_paired_failure_counts_keep_noncontact_out(self):
        contact=np.repeat([True,True,False],3)
        timely=np.array([1,0,0,0,0,1,1,1,0],bool)
        result=M.paired_structure_counts(timely,contact,np.tile(['H','B','HB'],3))
        self.assertEqual(result,dict(contact_anchors=2,single_any_timely=1,hb_timely=1,
            single_any_timely_hb_miss=1,hb_timely_both_single_miss=1))


if __name__=='__main__':unittest.main()
