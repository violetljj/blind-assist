#!/usr/bin/env python3

import copy
import json
import unittest

import numpy as np

from run_development import run_development
from validate_protocol import DEFAULT_PROTOCOL


class DevelopmentRunnerTest(unittest.TestCase):
    def test_four_folds_and_validation_execute_without_sealed_rows(self) -> None:
        protocol = json.loads(DEFAULT_PROTOCOL.read_text(encoding="utf-8"))
        protocol = copy.deepcopy(protocol)
        protocol["training"].update({"epochs": 3, "batch_size_frames": 128})
        rng = np.random.default_rng(12)
        records = []
        for parent_index in range(20):
            role = "train" if parent_index < 16 else "validation"
            for frame_index in range(3):
                records.append({
                    "parent_id": f"p{parent_index:02d}",
                    "video_id": f"v{parent_index:02d}",
                    "timestamp": float(frame_index),
                    "role": role,
                    "cv_fold": parent_index // 4 if role == "train" else None,
                })
        count = len(records)
        raw = rng.uniform(0.8, 2.5, size=(count, 3)).astype(np.float32)
        affine = np.column_stack((np.full(count, 1.02), np.full(count, 0.02))).astype(np.float32)
        truth = (affine[:, :1] * raw + affine[:, 1:]).astype(np.float32)
        arrays = {
            "region_inputs": rng.normal(size=(count, 3, 781)).astype(np.float32),
            "raw_clearance": raw,
            "truth_clearance": truth,
            "truth_valid": np.ones((count, 3), dtype=bool),
            "cls_features": rng.normal(size=(count, 384)).astype(np.float32),
            "affine_targets": affine,
            "affine_valid": np.ones(count, dtype=bool),
        }
        manifest = {"protocol_sha256": "test", "records": records, "sealed_truth_included": False}
        result, receipt = run_development(protocol, manifest, arrays)
        self.assertEqual(len(result["folds"]), 4)
        self.assertFalse(result["sealed_truth_opened"])
        self.assertEqual(set(result["fixed_validation"]["arms"]), set(protocol["arms"]["primary"]))
        self.assertEqual(receipt["spatial_trainable_parameters"], 9423)
        self.assertIn("global_model", receipt)


if __name__ == "__main__":
    unittest.main()
