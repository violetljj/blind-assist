"""Guard vote weighting in a mixed cell and keep both endpoint and visibility evidence."""
import unittest
from unittest.mock import patch

import numpy as np

import holoset_soft_history_20261119 as soft


class WeightedHistoryTest(unittest.TestCase):
    def test_mixed_cell_before_log(self):
        # Three historical endpoints share a cell, two are marked. Current has the same rays.
        pose = np.eye(4)
        pose[:3,:3] = [[1,0,0],[0,0,1],[0,-1,0]]
        pose[2,3] = 1.
        frame = dict(pose=pose,center=np.zeros(2),depth=np.ones((1,3)),
                     points=np.array([[0.,1.02,1.02]]*3))
        frames = [frame,dict(frame)]
        baseline = soft.d8.original.features(frames,0.,soft.d8.V)
        used = np.nonzero(baseline[1])
        self.assertEqual(len(used[0]),1)
        def capture(_,features):
            np.testing.assert_array_equal(features[[0,2,3,4,5,6,7]],baseline[[0,2,3,4,5,6,7]])
            return features[1][used].item()
        with patch.object(soft.d8,"score",capture):
            values,count = soft.weighted_scores(frames,[np.array([[True,False,True]])],0.,None)
        self.assertEqual(count,2)
        for weight in soft.WEIGHTS:
            self.assertEqual(values[f"{weight:.2f}"],float(np.float16(np.log1p(1+2*weight))))


if __name__=="__main__":
    unittest.main()
