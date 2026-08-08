import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from .lower_depthart_selective_scan_onnx import lower_model


def _make_model(
    length: int = 5,
    *,
    batch: int = 2,
    channels: int = 8,
    groups: int = 2,
    state_dim: int = 4,
    out_float: int = 0,
) -> onnx.ModelProto:
    shapes = {
        "u": [batch, channels, length],
        "delta": [batch, channels, length],
        "A": [channels, state_dim],
        "B": [batch, groups, state_dim, length],
        "C": [batch, groups, state_dim, length],
        "D": [channels],
        "delta_bias": [channels],
    }
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in shapes.items()]
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, shapes["u"])
    node = helper.make_node(
        "SelectiveScan",
        list(shapes),
        ["y"],
        name="SelectiveScanCanary",
        domain="com.depthart",
        delta_softplus=1,
        out_float=out_float,
    )
    graph = helper.make_graph([node], "selective_scan_canary", inputs, [output])
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.depthart", 1)],
        ir_version=10,
    )


def _reference(values: dict[str, np.ndarray]) -> np.ndarray:
    u = values["u"]
    delta = np.logaddexp(0.0, values["delta"] + values["delta_bias"][None, :, None])
    matrix_a = values["A"]
    matrix_b = np.repeat(values["B"], u.shape[1] // values["B"].shape[1], axis=1)
    matrix_c = np.repeat(values["C"], u.shape[1] // values["C"].shape[1], axis=1)
    state = np.zeros((u.shape[0], u.shape[1], matrix_a.shape[1]), dtype=np.float32)
    output = []
    for step in range(u.shape[2]):
        dt = delta[:, :, step]
        state = (
            np.exp(dt[:, :, None] * matrix_a[None]) * state
            + dt[:, :, None] * matrix_b[:, :, :, step] * u[:, :, step, None]
        )
        value = np.sum(state * matrix_c[:, :, :, step], axis=-1)
        output.append(value + values["D"][None] * u[:, :, step])
    return np.stack(output, axis=-1)


class LowerDepthArtSelectiveScanOnnxTest(unittest.TestCase):
    def test_lowered_graph_matches_reference(self) -> None:
        rng = np.random.default_rng(17)
        model = _make_model()
        lowered, receipt = lower_model(model)
        onnx.checker.check_model(lowered)
        values = {
            "u": (rng.standard_normal((2, 8, 5)) * 0.1).astype(np.float32),
            "delta": (rng.standard_normal((2, 8, 5)) * 0.1).astype(np.float32),
            "A": -rng.random((8, 4), dtype=np.float32),
            "B": (rng.standard_normal((2, 2, 4, 5)) * 0.1).astype(np.float32),
            "C": (rng.standard_normal((2, 2, 4, 5)) * 0.1).astype(np.float32),
            "D": rng.standard_normal(8).astype(np.float32),
            "delta_bias": rng.standard_normal(8).astype(np.float32),
        }
        session = ort.InferenceSession(lowered.SerializeToString(), providers=["CPUExecutionProvider"])
        actual = session.run(["y"], values)[0]
        np.testing.assert_allclose(actual, _reference(values), rtol=2e-5, atol=2e-6)
        self.assertEqual(receipt["selective_scan_nodes_lowered"], 1)
        self.assertEqual(receipt["remaining_custom_selective_scan_nodes"], 0)
        self.assertEqual(receipt["output_node_count"], 101)

    def test_frozen_s448_contract_shapes_match_reference(self) -> None:
        rng = np.random.default_rng(23)
        for channels in (48, 128, 336, 672):
            with self.subTest(channels=channels):
                model = _make_model(length=196, batch=1, channels=channels, groups=4, state_dim=8)
                lowered, receipt = lower_model(model)
                values = {
                    "u": (rng.standard_normal((1, channels, 196)) * 0.05).astype(np.float32),
                    "delta": (rng.standard_normal((1, channels, 196)) * 0.05).astype(np.float32),
                    "A": -rng.random((channels, 8), dtype=np.float32),
                    "B": (rng.standard_normal((1, 4, 8, 196)) * 0.05).astype(np.float32),
                    "C": (rng.standard_normal((1, 4, 8, 196)) * 0.05).astype(np.float32),
                    "D": rng.standard_normal(channels).astype(np.float32),
                    "delta_bias": rng.standard_normal(channels).astype(np.float32),
                }
                session = ort.InferenceSession(
                    lowered.SerializeToString(), providers=["CPUExecutionProvider"]
                )
                actual = session.run(["y"], values)[0]
                np.testing.assert_allclose(actual, _reference(values), rtol=3e-5, atol=3e-6)
                self.assertEqual(receipt["output_node_count"], 3730)

    def test_rejects_unfrozen_output_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "only delta_softplus=1/out_float=0"):
            lower_model(_make_model(out_float=1))


if __name__ == "__main__":
    unittest.main()
