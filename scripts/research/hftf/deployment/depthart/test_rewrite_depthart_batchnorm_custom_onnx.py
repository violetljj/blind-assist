import unittest

from onnx import TensorProto, helper

from .rewrite_depthart_batchnorm_custom_onnx import rewrite_model


class RewriteDepthArtBatchNormCustomOnnxTest(unittest.TestCase):
    def test_rewrites_frozen_inference_contract(self) -> None:
        node = helper.make_node(
            "BatchNormalization", ["x", "s", "b", "m", "v"], ["y"],
            name="bn", epsilon=1e-3, momentum=0.9,
        )
        model = helper.make_model(helper.make_graph(
            [node], "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8, 8])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])],
        ))
        rewritten, records = rewrite_model(model)
        result = rewritten.graph.node[0]
        self.assertEqual((result.domain, result.op_type), ("com.depthart", "DepthArtBatchNorm2d"))
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(helper.get_attribute_value(result.attribute[0]), 1e-3)

    def test_rejects_training_contract(self) -> None:
        node = helper.make_node(
            "BatchNormalization", ["x", "s", "b", "m", "v"], ["y"],
            training_mode=1,
        )
        model = helper.make_model(helper.make_graph([node], "g", [], []))
        with self.assertRaisesRegex(ValueError, "unsupported BatchNormalization contract"):
            rewrite_model(model)


if __name__ == "__main__":
    unittest.main()
