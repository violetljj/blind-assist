import unittest
import numpy as np
import torch
from gate_spatial_diagnostic import transform_gate,shuffle_indices
from detached_spec import specification


class GateSpatialTests(unittest.TestCase):
    def test_mean_and_value_preservation(self):
        gate=torch.arange(2*2*18*32,dtype=torch.float32).reshape(2,2,18,32)/2304
        ids=shuffle_indices(['a','b'],101)
        np.testing.assert_array_equal(ids,shuffle_indices(['a','b'],101))
        constant=transform_gate(gate,'constant_mean');shuffle=transform_gate(gate,'shuffle_101',torch.from_numpy(ids))
        torch.testing.assert_close(constant.mean((-2,-1)),gate.mean((-2,-1)))
        torch.testing.assert_close(shuffle.flatten(2).sort(-1).values,gate.flatten(2).sort(-1).values)
        self.assertFalse(torch.equal(shuffle,gate))

    def test_new_test_allocation(self):
        spec=specification();self.assertEqual(len(spec['samples']),128)
        self.assertTrue(all(s['split']=='test' for s in spec['samples']))
        for value in ('narrow','diverse'):
            self.assertEqual(len({c['group_id'] for c in spec['clips'] if c['stratum']==value}),16)
        self.assertEqual({s['group_id'] for s in spec['samples']},{f'g{i}' for i in range(5000,5032)})


if __name__=='__main__':unittest.main()
