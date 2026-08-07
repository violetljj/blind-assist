#!/usr/bin/env python3

import unittest

from onnx import helper

from scripts.research.hftf.deployment.depthart.rewrite_depthart_qairt_hygiene import (
    attributes,
    clean_node,
)


class DepthartQairtHygieneTest(unittest.TestCase):
    def test_removes_only_default_inference_attributes(self) -> None:
        bn = helper.make_node("BatchNormalization", ["x", "s", "b", "m", "v"], ["y"], training_mode=0)
        reshape = helper.make_node("Reshape", ["x", "shape"], ["y"], allowzero=0)
        pool = helper.make_node(
            "AveragePool", ["x"], ["y"], ceil_mode=0, count_include_pad=1,
            kernel_shape=[2, 2], pads=[0, 0, 0, 0], strides=[2, 2],
        )
        self.assertEqual(clean_node(bn), ["training_mode"])
        self.assertEqual(clean_node(reshape), ["allowzero"])
        self.assertEqual(clean_node(pool), ["ceil_mode", "count_include_pad"])
        self.assertNotIn("training_mode", attributes(bn))
        self.assertNotIn("allowzero", attributes(reshape))
        self.assertNotIn("ceil_mode", attributes(pool))
        self.assertNotIn("count_include_pad", attributes(pool))

    def test_rejects_semantic_changes(self) -> None:
        bn = helper.make_node("BatchNormalization", ["x", "s", "b", "m", "v"], ["y"], training_mode=1)
        reshape = helper.make_node("Reshape", ["x", "shape"], ["y"], allowzero=1)
        with self.assertRaises(ValueError):
            clean_node(bn)
        with self.assertRaises(ValueError):
            clean_node(reshape)

    def test_keeps_count_include_pad_with_real_padding(self) -> None:
        pool = helper.make_node(
            "AveragePool", ["x"], ["y"], count_include_pad=1,
            kernel_shape=[2, 2], pads=[1, 1, 1, 1], strides=[2, 2],
        )
        self.assertEqual(clean_node(pool), [])
        self.assertEqual(attributes(pool)["count_include_pad"], 1)


if __name__ == "__main__":
    unittest.main()
