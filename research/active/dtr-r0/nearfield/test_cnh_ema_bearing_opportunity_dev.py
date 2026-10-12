import unittest
import numpy as np
import cnh_ema_bearing_opportunity_dev as O


class BearingContract(unittest.TestCase):
    def test_head_relative_sign_boundaries_and_missing(self):
        np.testing.assert_array_equal(O.quantize(-np.array([20.,10.,0.,-10.,-20.,np.nan,181.])), [0,1,1,1,2,-1,2])

    def test_first_keeps_startup_deadline_and_no_later_repair(self):
        alarm=np.zeros((3,13),bool); alarm[0,[0,2,3]]=True; alarm[1,4]=True; alarm[2,2]=True
        first=O.first_notice(alarm,np.array([True,True,False]),np.array([2,3,12]))
        np.testing.assert_array_equal(first,[2,-1,-1])
        labels=np.ones((3,13),int);labels[0,2]=-1
        truth=np.zeros((3,13,3),bool);truth[:,:,1]=True
        rec,_=O.evaluate(labels,first,truth,np.array([True,True,False]))
        self.assertEqual((rec['events'],rec['timely'],rec['first_clean'],rec['first_abstain']),(2,1,0,1))

    def test_opportunity_multi_unavailable_and_empty_truth(self):
        labels=np.ones((5,13),int); labels[2:4]=-1
        truth=np.zeros((5,13,3),bool);truth[0,:,0]=True;truth[1,:,[0,2]]=True;truth[2,:,0]=True;truth[3,:,[0,2]]=True
        rec,_=O.evaluate(labels,np.full(5,2),truth,np.ones(5,bool))
        self.assertEqual((rec['first_wrong'],rec['first_abstain'],rec['truth_empty']),(3,2,1))
        self.assertEqual(rec['opportunity'],dict(a_wrong_unique=1,b_wrong_multiple=1,c_unavailable_supported=2,c_unavailable_unique=1,clean_oracle_headroom=4,unique_oracle_headroom=2))

    def test_clean_is_support_not_unique_object_localization(self):
        labels=np.ones((2,13),int);truth=np.ones((2,13,3),bool);truth[1,:,0]=False;truth[1,:,2]=False
        rec,_=O.evaluate(labels,np.array([2,2]),truth,np.ones(2,bool))
        self.assertEqual((rec['first_clean'],rec['first_unique'],rec['truth_multiple']),(2,1,1))


if __name__=='__main__':unittest.main()
