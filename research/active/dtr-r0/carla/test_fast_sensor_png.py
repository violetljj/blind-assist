import io
import unittest
import numpy as np
from PIL import Image
from fast_sensor_png import encode_bgra


class EncodingTest(unittest.TestCase):
    def test_independent_decoder_exact_channels_and_alpha(self):
        data=np.random.default_rng(412).integers(0,256,(33,47,4),dtype=np.uint8)
        decoded=np.array(Image.open(io.BytesIO(encode_bgra(data.tobytes(),47,33))).convert('RGBA'))
        for source,target in ((0,2),(1,1),(2,0),(3,3)):
            np.testing.assert_array_equal(data[:,:,source],decoded[:,:,target])

    def test_wrong_payload_length_rejected(self):
        with self.assertRaises(ValueError):encode_bgra(b'123',2,2)


if __name__=='__main__':unittest.main()
