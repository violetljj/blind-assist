#!/usr/bin/env python3
"""Synthetic focused tests for sealed TARO DepthART candidate inference."""

from __future__ import annotations

import copy
import hashlib
import math
import unittest

import numpy as np

from scripts.research.taro_o0r_factor_headroom_runtime import depthart_runner as runtime
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


def _assets(token: str) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for index, role in enumerate(adapter.DECODED_PAYLOAD_KINDS):
        payload = f"synthetic:{token}:{role}".encode()
        output[role] = {
            "container_id": "synthetic-container",
            "member_path": f"{role}/{token}.bin" if role != "trajectory" else "trajectory/lowres_wide.traj",
            "exact_timestamp_stem": None if role == "trajectory" else token,
            "bytes": index + 1,
            "sha256": hashlib.sha256(payload).hexdigest().upper(),
            "crc32": hashlib.sha256(b"crc" + payload).hexdigest()[:8].upper(),
        }
    return output


def _source_receipt(color: np.ndarray) -> dict[str, object]:
    visit_id, video_id = adapter.O0R_EVAL_CANDIDATE_ROSTER[0]
    token = "1.100000000"
    trajectory = [
        {"timestamp_token": "1.090000000", "rotation_vector": [math.pi / 2.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
        {"timestamp_token": "1.110000000", "rotation_vector": [math.pi / 2.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
    ]
    lowres = {"width": 256, "height": 192, "fx": 200.0, "fy": 200.0, "cx": 127.5, "cy": 95.5}
    assets = _assets(token)
    decoded_values: dict[str, object] = {
        "color": color,
        "highres_depth": {"identity": "synthetic-faro"},
        "lowres_depth": {"identity": "synthetic-apple"},
        "confidence": {"identity": "synthetic-confidence"},
        "intrinsics": lowres,
        "trajectory": trajectory,
    }
    decoded = {
        role: {
            "asset_role": role,
            "member_sha256": assets[role]["sha256"],
            "member_crc32": assets[role]["crc32"],
            "decoded_kind": adapter.DECODED_PAYLOAD_KINDS[role],
            "decoded_content_sha256": adapter.canonical_sha256(decoded_values[role]),
        }
        for role in adapter.DECODED_PAYLOAD_KINDS
    }
    return adapter.build_source_frame_receipt(
        source_role="O0R_EVAL_CANDIDATE",
        visit_id=visit_id,
        video_id=video_id,
        frame_timestamp_token=token,
        lowres_intrinsics=lowres,
        trajectory_rows=trajectory,
        asset_bindings=assets,
        decoded_payload_bindings=decoded,
    )


class _ConstantDepthModel:
    def __call__(self, image, intrinsics):
        import torch

        del intrinsics
        return torch.full((image.shape[0], image.shape[2], image.shape[3]), 2.0, dtype=torch.float32, device=image.device)


class DepthARTRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rows = np.arange(runtime.HIGHRES_SHAPE_HW[0], dtype=np.uint16)[:, None]
        cols = np.arange(runtime.HIGHRES_SHAPE_HW[1], dtype=np.uint16)[None, :]
        cls.color = np.stack(
            (
                np.broadcast_to((cols % 256).astype(np.uint8), runtime.HIGHRES_SHAPE_HW),
                np.broadcast_to((rows % 256).astype(np.uint8), runtime.HIGHRES_SHAPE_HW),
                ((rows + cols) % 256).astype(np.uint8),
            ),
            axis=2,
        )
        cls.source = _source_receipt(cls.color)
        cls.candidate_input = runtime.candidate_input_from_bound_source(cls.source, cls.color)

    def test_frozen_landscape_shape_and_k_scaling(self) -> None:
        self.assertEqual((608, 448), runtime.lower_bound_size(1920, 1440))
        matrix = np.asarray(self.source["intrinsics_highres"]["matrix_3x3"], dtype=np.float32)
        tensor, resized_k = runtime.preprocess_depthart_input(self.color, matrix)
        self.assertEqual((1, 3, 448, 608), tensor.shape)
        self.assertEqual((1, 3, 3), resized_k.shape)
        self.assertAlmostEqual(float(resized_k[0, 0, 0]), float(matrix[0, 0]) * 608.0 / 1920.0, places=5)
        self.assertAlmostEqual(float(resized_k[0, 1, 1]), float(matrix[1, 1]) * 448.0 / 1440.0, places=4)

    def test_native_blob_is_deterministic_and_roundtrips(self) -> None:
        native = np.linspace(0.5, 5.0, np.prod(runtime.NATIVE_SHAPE_HW), dtype=np.float32).reshape(runtime.NATIVE_SHAPE_HW)
        first = runtime.deterministic_npy_gzip_bytes(native)
        second = runtime.deterministic_npy_gzip_bytes(native)
        self.assertEqual(first, second)
        np.testing.assert_array_equal(native, runtime.decode_npy_gzip_bytes(first))

    def test_candidate_inference_is_truth_blind_and_hash_bound(self) -> None:
        output = runtime.infer_depthart_candidate(
            _ConstantDepthModel(),
            candidate_input_receipt=self.candidate_input,
            color_rgb_u8=self.color,
            runtime_identity={"backend": "synthetic-cpu-fp32"},
            device="cpu",
        )
        self.assertEqual(runtime.NATIVE_SHAPE_HW, output["native_depth_m"].shape)
        self.assertEqual(runtime.HIGHRES_SHAPE_HW, output["highres_depth_m"].shape)
        receipt = runtime.validate_depthart_inference_receipt(output["inference_receipt"])
        self.assertFalse(receipt["truth_payload_read"])
        self.assertFalse(receipt["truth_alignment_used"])
        highres, candidate_output_receipt = runtime.bind_sealed_candidate_to_source(
            inference_receipt=receipt,
            native_depth_m=output["native_depth_m"],
            source_frame_receipt=self.source,
        )
        self.assertEqual(runtime.HIGHRES_SHAPE_HW, highres.shape)
        self.assertEqual(receipt["highres_output_array_sha256"], candidate_output_receipt["output_array_sha256"])
        tampered = copy.deepcopy(receipt)
        tampered["truth_payload_read"] = True
        with self.assertRaises(runtime.DepthARTRuntimeError):
            runtime.validate_depthart_inference_receipt(tampered)

    def test_upsample_rule_is_unique_and_finite(self) -> None:
        import torch
        import torch.nn.functional as torch_functional

        native = np.linspace(0.25, 4.25, np.prod(runtime.NATIVE_SHAPE_HW), dtype=np.float32).reshape(runtime.NATIVE_SHAPE_HW)
        highres = runtime.upsample_native_depth(native)
        expected = torch_functional.interpolate(
            torch.from_numpy(native)[None, None],
            runtime.HIGHRES_SHAPE_HW,
            mode="bilinear",
            align_corners=True,
        )[0, 0].numpy()
        self.assertEqual(runtime.HIGHRES_SHAPE_HW, highres.shape)
        self.assertEqual(np.float32, highres.dtype)
        np.testing.assert_array_equal(expected, highres)
        self.assertEqual(float(native[0, 0]), float(highres[0, 0]))
        self.assertEqual(float(native[-1, -1]), float(highres[-1, -1]))


if __name__ == "__main__":
    unittest.main()
