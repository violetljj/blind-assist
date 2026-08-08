from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from scripts.research.hftf.deployment.depthart.bisect_depthart_pre_scan_parity import (
    build_plan,
    compare,
)


class DepthArtPreScanParityBisectTest(unittest.TestCase):
    def test_plan_stops_before_first_selective_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            image = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 2])
            output = helper.make_tensor_value_info("scan_out", TensorProto.FLOAT, [1, 2])
            bias = helper.make_tensor("bias", TensorProto.FLOAT, [1], [1.0])
            nodes = [
                helper.make_node("Add", ["image", "bias"], ["added"], name="add"),
                helper.make_node("Relu", ["added"], ["relu"], name="relu"),
                helper.make_node(
                    "SelectiveScan",
                    ["relu"],
                    ["scan_out"],
                    name="scan",
                    domain="com.depthart",
                ),
            ]
            graph = helper.make_graph(nodes, "test", [image], [output], [bias])
            model = helper.make_model(
                graph,
                opset_imports=[
                    helper.make_opsetid("", 17),
                    helper.make_opsetid("com.depthart", 1),
                ],
            )
            onnx.save(model, path)
            plan = build_plan(path)
            self.assertEqual(plan["prefix_dependency_nodes"], 2)
            self.assertEqual(
                [item["op_type"] for item in plan["checkpoints"]], ["Add", "Relu"]
            )
            self.assertEqual(plan["first_selective_scan"]["first_input"], "relu")

    def test_compare_uses_frozen_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = root / "left.raw"
            exact = root / "exact.raw"
            changed = root / "changed.raw"
            value = np.asarray([1.0, 2.0], dtype=np.float32)
            value.tofile(left)
            value.tofile(exact)
            (value + np.asarray([0.0, 0.01], dtype=np.float32)).tofile(changed)
            self.assertTrue(compare(left, exact)["allclose"])
            self.assertFalse(compare(left, changed)["allclose"])


if __name__ == "__main__":
    unittest.main()
