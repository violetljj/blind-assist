#!/usr/bin/env python3

import unittest

import numpy as np

from scripts.research.hftf.rewrite_depthart_qairt_static_shape import constant_mod, is_noop_expand


class DepthartQairtStaticShapeTest(unittest.TestCase):
    def test_expand_all_ones_is_noop(self) -> None:
        self.assertTrue(is_noop_expand((1, 256, 112, 112), np.array([1, 1, 1, 1])))
        self.assertTrue(is_noop_expand((1, 8, 128), np.array([1, 1, 1])))

    def test_expand_rejects_changed_shape(self) -> None:
        self.assertFalse(is_noop_expand((1, 8, 128), np.array([2, 8, 128])))

    def test_constant_mod(self) -> None:
        np.testing.assert_array_equal(constant_mod(np.array([2]), np.array([3])), np.array([2]))
        with self.assertRaises(ValueError):
            constant_mod(np.array([2]), np.array([0]))


if __name__ == "__main__":
    unittest.main()
