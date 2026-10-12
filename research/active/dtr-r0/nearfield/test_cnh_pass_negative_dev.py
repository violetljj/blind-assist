"""CPU-only weak-pass supervision and continuation schedule checks."""
import unittest

import numpy as np

from cnh_pass_negative_dev import continuation_order, weak_supervision
from cnh_counterfactual_train_dev import slot_schedule


class PassNegativeTests(unittest.TestCase):
    def test_pass_only_labels_masks_and_raw_far_weight(self):
        labels = np.array([[1., 0.], [0., 1.], [0., 0.]], np.float32)
        mask = np.array([[1., 0.], [0., 1.], [1., 1.]], np.float32)
        far = np.array([1., 2., 1.])
        raw = mask*far[:, None]
        original = (raw*np.float32(3/raw.sum())).astype(np.float32)
        passes = mask == 0
        y, m, w, r = weak_supervision(labels, mask, original, passes, far, float(raw.sum()))
        np.testing.assert_array_equal(y, labels)
        np.testing.assert_array_equal(m, np.ones_like(mask))
        expected_raw = raw.copy()
        expected_raw[passes] = (.25*np.broadcast_to(far[:, None], raw.shape))[passes]
        expected = (expected_raw*np.float32(3/expected_raw.sum())).astype(np.float32)
        np.testing.assert_array_equal(w, expected)
        assert abs(w.sum(dtype=np.float64)-3) < 1e-6
        assert r['added_pass_raw_mass'] == .75
        assert r['original_valid_weight_ratio_max'] < 1.
        np.testing.assert_array_equal(mask, np.array([[1., 0.], [0., 1.], [1., 1.]], np.float32))
        np.testing.assert_array_equal(original, (raw*np.float32(3/raw.sum())).astype(np.float32))

    def test_common_continuation_order(self):
        for seed in (2026100955, 2026100956, 2026100957):
            for offset in (0, 11):
                actual = continuation_order(seed, offset)
                np.testing.assert_array_equal(actual, slot_schedule(seed, 32+offset, 39936, 19968)[0])
                assert len(actual) == 59904
                assert 12*(len(actual)//256) == 2808


if __name__ == '__main__':
    unittest.main()
