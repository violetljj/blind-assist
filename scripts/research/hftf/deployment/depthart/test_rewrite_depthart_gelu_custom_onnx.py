import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from scripts.research.hftf.deployment.depthart.rewrite_depthart_gelu_custom_onnx import rewrite_model


def _constant(name: str, output: str, value: float) -> onnx.NodeProto:
    return helper.make_node(
        "Constant", [], [output], name=name,
        value=numpy_helper.from_array(np.asarray(value, dtype=np.float32)),
    )


def _fixture(divisor: float = float(np.sqrt(2.0))) -> onnx.ModelProto:
    nodes = [
        _constant("sqrt2", "sqrt2_out", divisor),
        _constant("one", "one_out", 1.0),
        _constant("half", "half_out", 0.5),
        helper.make_node("Div", ["x", "sqrt2_out"], ["div_out"], name="gelu_div"),
        helper.make_node("Erf", ["div_out"], ["erf_out"], name="gelu_erf"),
        helper.make_node("Add", ["erf_out", "one_out"], ["add_out"], name="gelu_add"),
        helper.make_node("Mul", ["x", "add_out"], ["mul_out"], name="gelu_mul"),
        helper.make_node("Mul", ["mul_out", "half_out"], ["y"], name="gelu_final"),
    ]
    graph = helper.make_graph(
        nodes, "gelu", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class RewriteDepthArtGeluTest(unittest.TestCase):
    def test_rewrites_exact_five_node_family(self) -> None:
        rewritten, records = rewrite_model(_fixture())
        gelu = [node for node in rewritten.graph.node if node.op_type == "DepthArtGelu"]
        self.assertEqual(len(gelu), 1)
        self.assertEqual(gelu[0].domain, "com.depthart")
        self.assertEqual(list(gelu[0].input), ["x"])
        self.assertEqual(list(gelu[0].output), ["y"])
        self.assertEqual(len(records), 1)
        self.assertFalse(any(node.op_type in {"Div", "Erf", "Add", "Mul"} for node in rewritten.graph.node))

    def test_rejects_non_exact_divisor(self) -> None:
        with self.assertRaisesRegex(ValueError, "no exact erf-GELU"):
            rewrite_model(_fixture(1.5))


if __name__ == "__main__":
    unittest.main()
