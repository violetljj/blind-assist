from __future__ import annotations

import copy
import unittest

from analyze_parity import index_unique_frames, validate_device_binding


def fixtures() -> tuple[dict, dict, dict]:
    config = {
        "_config_sha256": "config",
        "parent": {"windows_sha256": "windows", "frame_count": 1, "source_frame_counts": {"source": 1}},
        "detector": {"model_sha256": "model", "labels_sha256": "labels", "input_shape": [1, 320, 320, 3], "output_shape": [1, 84, 2100]},
    }
    manifest = {
        "schema": "blindassist_ustrf_detector_taxonomy_device_input_v1",
        "config_sha256": "config", "windows_sha256": "windows", "frame_count": 1,
        "model_sha256": "model", "labels_sha256": "labels",
        "input_shape": [1, 320, 320, 3], "output_shape": [1, 84, 2100],
        "frames": [{"source_name": "name", "source_id": "source", "frame_id": "000001", "image_sha256": "image", "host_input_tensor_sha256": "input", "host_raw_output_sha256": "raw"}],
    }
    device = {
        "schema": "blindassist_ustrf_detector_taxonomy_device_output_v1",
        "input_manifest_sha256": "manifest", "frame_count": 1, "failure_count": 0,
        "frames": [{"source_name": "name", "frame_id": "000001", "image_sha256": "image", "host_input_tensor_sha256": "input", "host_raw_output_sha256": "raw"}],
    }
    return config, manifest, device


class EvidenceBindingsTest(unittest.TestCase):
    def test_complete_binding_passes(self) -> None:
        config, manifest, device = fixtures()
        validate_device_binding(config, manifest, "manifest", device)

    def test_rejects_stale_manifest(self) -> None:
        config, manifest, device = fixtures()
        device["input_manifest_sha256"] = "stale"
        with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
            validate_device_binding(config, manifest, "manifest", device)

    def test_rejects_per_frame_host_hash_drift(self) -> None:
        config, manifest, device = fixtures()
        device["frames"][0]["host_raw_output_sha256"] = "other"
        with self.assertRaisesRegex(ValueError, "per-frame binding mismatch"):
            validate_device_binding(config, manifest, "manifest", device)

    def test_rejects_duplicate_frame_identity(self) -> None:
        row = {"source_name": "name", "frame_id": "000001"}
        with self.assertRaisesRegex(ValueError, "duplicate"):
            index_unique_frames([row, copy.deepcopy(row)], "test")


if __name__ == "__main__":
    unittest.main()
