import unittest

import onnx
from onnx import TensorProto, helper

from .rewrite_depthart_first_patch_conv_custom_onnx import TARGET_NAME, rewrite_model


class RewriteDepthArtFirstPatchConvCustomOnnxTest(unittest.TestCase):
    def test_rewrites_only_frozen_first_patch_conv(self) -> None:
        node = helper.make_node(
            "Conv", ["x", "w"], ["y"], name=TARGET_NAME,
            dilations=[1, 1], group=1, kernel_shape=[3, 3],
            pads=[1, 1, 1, 1], strides=[2, 2],
        )
        model = helper.make_model(
            helper.make_graph(
                [node], "g",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 448, 448]),
                 helper.make_tensor_value_info("w", TensorProto.FLOAT, [24, 3, 3, 3])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 24, 224, 224])],
            ),
            opset_imports=[helper.make_opsetid("", 17)],
        )
        rewritten, record = rewrite_model(model)
        result = rewritten.graph.node[0]
        self.assertEqual((result.domain, result.op_type), ("com.depthart", "DepthArtPatchConv2d"))
        self.assertEqual(len(result.attribute), 0)
        self.assertEqual(record["index"], 0)

    def test_rejects_contract_change(self) -> None:
        node = helper.make_node(
            "Conv", ["x", "w"], ["y"], name=TARGET_NAME,
            dilations=[1, 1], group=1, kernel_shape=[3, 3],
            pads=[0, 0, 0, 0], strides=[2, 2],
        )
        model = helper.make_model(helper.make_graph([node], "g", [], []))
        with self.assertRaisesRegex(ValueError, "unsupported frozen Conv contract"):
            rewrite_model(model)


if __name__ == "__main__":
    unittest.main()
