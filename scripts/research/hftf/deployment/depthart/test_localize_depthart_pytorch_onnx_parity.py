from __future__ import annotations

import unittest

import numpy as np

from scripts.research.hftf.deployment.depthart.localize_depthart_pytorch_onnx_parity import (
    compare,
    first_failure,
)


class DepthArtPytorchOnnxLocalizationTest(unittest.TestCase):
    def test_compare_and_first_failure_use_frozen_tolerance(self) -> None:
        exact = compare(np.asarray([1.0], np.float32), np.asarray([1.0], np.float32))
        changed = compare(np.asarray([1.0], np.float32), np.asarray([1.01], np.float32))
        comparisons = {
            "patch_conv1": exact,
            "patch_bn1": exact,
            "patch_gelu1": exact,
            "patch_conv2": exact,
            "patch_bn2": exact,
            "patch_embed": exact,
            **{
                name: exact
                for name in (
                    "daa1_cam_dw", "daa1_cam_pw", "daa1_norm_x", "daa1_norm_ctx",
                    "daa1_kv", "daa1_q", "daa1_out", "daa1_ls1", "daa1_mlp_norm",
                    "daa1_mlp_proj1", "daa1_mlp_act", "daa1_mlp_proj2", "daa1_ls2",
                    "daa1_attention",
                )
            },
            "daa1": changed,
            **{
                name: changed
                for name in (
                    "stage1", "daa2", "stage2", "daa3", "stage3", "daa4",
                    "stage4", "depth_head", "scale_head", "depth",
                )
            },
        }
        self.assertTrue(exact["allclose"])
        self.assertFalse(changed["allclose"])
        self.assertEqual(first_failure(comparisons), "daa1")


if __name__ == "__main__":
    unittest.main()
