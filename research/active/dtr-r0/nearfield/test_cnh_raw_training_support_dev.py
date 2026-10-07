"""Only new raw/source alignment contract; existing B weighting tests unchanged."""
import unittest
import numpy as np
import cnh_raw_training_support_dev as M


class CurrentAlignment(unittest.TestCase):
    def test_same_pair_current_dtype_order_and_hb_identity(self):
        pair=np.arange(2*13*2,dtype=np.float64).reshape(2,13,2)
        current=np.arange(2*13*136,dtype=np.float32).reshape(2,13,136)
        payload=dict(current=current,unit=np.array([4,4]),config=np.array([2,2]),anchor_id=np.array(['a','a']),tag=np.array(['H','HB']))
        x=M.assemble_current(pair,payload,payload['unit'],payload['config'],payload['anchor_id'],payload['tag'])
        self.assertEqual(x.dtype,np.float64)
        np.testing.assert_array_equal(x,np.concatenate((pair,current),axis=-1))
        with self.assertRaises(AssertionError):
            M.assemble_current(pair,payload,payload['unit'],payload['config'],payload['anchor_id'],payload['tag'][::-1])
        with self.assertRaises(AssertionError):
            M.assemble_current(pair,payload,payload['unit'],payload['config'],np.array(['b','b']),payload['tag'])


if __name__=='__main__':unittest.main()
