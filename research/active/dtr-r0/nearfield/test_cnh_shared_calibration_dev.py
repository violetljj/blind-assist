"""Focused common-control threshold contract; synthetic scores, no fitting."""
import unittest
import numpy as np
import cnh_shared_calibration_dev as M


class SharedThresholds(unittest.TestCase):
    def scores(self):
        data={}
        for fold in range(3):
            for method in M.METHODS:
                x=np.full((2,13),np.nan);x[0]=1000
                x[0,2:12]=np.array([.1]*5+[.9]*5)+(0 if method=='original_center' else fold)
                data[f'fold{fold}/{method}']=x
        return data

    def test_whole_ties_same_panel_and_ignore_outside_control_window(self):
        scores=self.scores();eligible=np.array([True,False])
        ts,records=M.shared_thresholds(scores,eligible,caps=(.25,.5))
        for fold in range(3):
            for method in M.METHODS:
                values=scores[f'fold{fold}/{method}'][0,2:12]
                self.assertEqual(int((values>=ts[fold,'0.250',method]).sum()),0)
                self.assertEqual(int((values>=ts[fold,'0.500',method]).sum()),5)
        self.assertEqual(len({ts[f,'0.500','original_center'] for f in range(3)}),1)
        self.assertTrue(all(r['calibration_controls']==1 and r['calibration_intervals']==10 for r in records))

    def test_changed_max_score_or_empty_panel_rejected(self):
        scores=self.scores();scores['fold2/original_center'][0,2]=.2
        with self.assertRaises(AssertionError):M.shared_thresholds(scores,np.array([True,False]))
        with self.assertRaises(ValueError):M.shared_thresholds(self.scores(),np.array([False,False]))


if __name__=='__main__':unittest.main()
