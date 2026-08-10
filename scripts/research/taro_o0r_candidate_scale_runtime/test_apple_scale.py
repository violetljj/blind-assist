#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from scripts.research.taro_o0r_candidate_scale_runtime import apple_scale
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


def _png_bytes(array: np.ndarray) -> bytes:
    import io

    output = io.BytesIO()
    Image.fromarray(array).save(output, format="PNG")
    return output.getvalue()


class AppleScaleTest(unittest.TestCase):
    def test_registered_sampling_uses_frozen_pixel_centers(self) -> None:
        highres = np.zeros(adapter.HIGHRES_SHAPE_HW, dtype=np.float32)
        rows, columns = np.mgrid[0 : adapter.APPLE_SHAPE_HW[0], 0 : adapter.APPLE_SHAPE_HW[1]]
        x = np.rint((columns + 0.5) * 7.5 - 0.5).astype(np.int64)
        y = np.rint((rows + 0.5) * 7.5 - 0.5).astype(np.int64)
        highres[y, x] = rows * 1000 + columns
        sampled = apple_scale.sample_candidate_at_apple_centers(highres)
        self.assertEqual(sampled.shape, adapter.APPLE_SHAPE_HW)
        self.assertEqual(sampled[0, 0], 0.0)
        self.assertEqual(sampled[-1, -1], 191255.0)
        np.testing.assert_array_equal(sampled, rows * 1000 + columns)

    def test_estimator_sign_and_even_median_are_frozen(self) -> None:
        apple = np.zeros(adapter.APPLE_SHAPE_HW, dtype=np.uint16)
        confidence = np.zeros(adapter.APPLE_SHAPE_HW, dtype=np.uint8)
        candidate = np.ones(adapter.APPLE_SHAPE_HW, dtype=np.float64)
        apple.flat[: apple_scale.MINIMUM_PAIR_COUNT] = 2000
        confidence.flat[: apple_scale.MINIMUM_PAIR_COUNT] = 2
        result = apple_scale.estimate_source_metric_scale(apple, confidence, candidate)
        self.assertTrue(result["evaluable"])
        self.assertAlmostEqual(result["log_metric_scale"], np.log(2.0), places=11)
        self.assertAlmostEqual(result["metric_scale"], 2.0, places=11)

        apple.flat[: apple_scale.MINIMUM_PAIR_COUNT] = 1000
        candidate.flat[: apple_scale.MINIMUM_PAIR_COUNT // 2] = np.e
        candidate.flat[apple_scale.MINIMUM_PAIR_COUNT // 2 : apple_scale.MINIMUM_PAIR_COUNT] = np.exp(-1.0)
        even = apple_scale.estimate_source_metric_scale(apple, confidence, candidate)
        self.assertEqual(even["log_metric_scale"], 0.0)
        self.assertEqual(even["metric_scale"], 1.0)

    def test_estimator_returns_unknown_below_fixed_support(self) -> None:
        apple = np.full(adapter.APPLE_SHAPE_HW, 1000, dtype=np.uint16)
        confidence = np.zeros(adapter.APPLE_SHAPE_HW, dtype=np.uint8)
        confidence.flat[: apple_scale.MINIMUM_PAIR_COUNT - 1] = 2
        candidate = np.ones(adapter.APPLE_SHAPE_HW, dtype=np.float64)
        result = apple_scale.estimate_source_metric_scale(apple, confidence, candidate)
        self.assertFalse(result["evaluable"])
        self.assertEqual(result["reason_codes"], ["APPLE_SCALE_COMMON_SUPPORT_INSUFFICIENT"])
        self.assertIsNone(result["log_metric_scale"])

    def test_narrow_decoder_never_opens_faro_or_rgb_member(self) -> None:
        video_id, token = "123", "4.500"
        apple = np.full(adapter.APPLE_SHAPE_HW, 1200, dtype=np.uint16)
        confidence = np.full(adapter.APPLE_SHAPE_HW, 2, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "source.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr(f"{video_id}/lowres_depth/{video_id}_{token}.png", _png_bytes(apple))
                bundle.writestr(f"{video_id}/confidence/{video_id}_{token}.png", _png_bytes(confidence))
                bundle.writestr(f"{video_id}/highres_depth/{video_id}_{token}.png", b"FARO_TRAP")
                bundle.writestr(f"{video_id}/wide/{video_id}_{token}.png", b"RGB_TRAP")
            payload = archive.read_bytes()
            container = {
                "asset": "upsampling.zip",
                "relative_path": "upsampling/Training/123.zip",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest().upper(),
            }
            opened: list[str] = []
            original = zipfile.ZipFile.read

            def spy(bundle: zipfile.ZipFile, name, *args, **kwargs):
                path = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
                opened.append(path)
                if "highres_depth" in path or "/wide/" in path:
                    raise AssertionError("truth/RGB payload was opened")
                return original(bundle, name, *args, **kwargs)

            with mock.patch.object(zipfile.ZipFile, "read", new=spy):
                decoded = apple_scale.decode_apple_scale_source(
                    archive,
                    container,
                    parent_id="7",
                    video_id=video_id,
                    timestamp_token=token,
                    physical_frame_id=f"{video_id}:{token}",
                    frame_plan_sha256="A" * 64,
                    candidate_phase_completion_sha256="B" * 64,
                )
            self.assertEqual(len(opened), 2)
            self.assertTrue(all("lowres_depth" in path or "confidence" in path for path in opened))
            self.assertFalse(decoded["source_receipt"]["faro_member_opened"])
            np.testing.assert_array_equal(decoded["apple_depth_mm"], apple)
            np.testing.assert_array_equal(decoded["confidence"], confidence)

            tampered = decoded["apple_depth_mm"].copy()
            tampered[0, 0] += 1
            with self.assertRaisesRegex(apple_scale.AppleScaleError, "decoded source arrays differ"):
                apple_scale.validate_apple_scale_source_receipt(decoded["source_receipt"], tampered, confidence)

    def test_receipt_tampering_is_rejected(self) -> None:
        value = apple_scale._seal(
            {
                "schema": apple_scale.CANDIDATE_REPLAY_BINDING_SCHEMA,
                "parent_id": "7",
                "video_id": "123",
                "timestamp_token": "4.500",
                "physical_frame_id": "123:4.500",
                "candidate_frame_record_sha256": "A" * 64,
                "inference_receipt_sha256": "B" * 64,
                "native_depth_array_sha256": "C" * 64,
                "highres_depth_array_sha256": "D" * 64,
                "candidate_truth_payload_read": False,
                "candidate_truth_alignment_used": False,
            }
        )
        apple_scale.validate_candidate_replay_binding(value)
        tampered = copy.deepcopy(value)
        tampered["candidate_truth_payload_read"] = True
        with self.assertRaises(apple_scale.AppleScaleError):
            apple_scale.validate_candidate_replay_binding(tampered)


if __name__ == "__main__":
    unittest.main()
