import unittest

import numpy as np

from .prepare_depthart_layernorm_canary import SHAPE, make_model, reference


class PrepareDepthArtLayerNormCanaryTest(unittest.TestCase):
    def test_model_contract(self) -> None:
        model = make_model()
        node = model.graph.node[0]
        self.assertEqual((node.domain, node.op_type), ("com.depthart", "DepthArtLayerNorm"))
        self.assertEqual(list(node.input), ["x", "weight", "bias"])
        self.assertEqual(list(node.output), ["y"])

    def test_reference_normalizes_last_axis_and_applies_affine(self) -> None:
        ramp = np.arange(np.prod(SHAPE), dtype=np.float32).reshape(SHAPE) / np.float32(100.0)
        weight = np.linspace(0.5, 1.5, SHAPE[-1], dtype=np.float32)
        bias = np.linspace(-0.2, 0.2, SHAPE[-1], dtype=np.float32)
        result = reference(ramp, weight, bias)
        self.assertEqual(result.shape, SHAPE)
        self.assertTrue(np.isfinite(result).all())
        normalized = (result - bias) / weight
        self.assertTrue(np.allclose(normalized.mean(axis=-1), 0.0, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
