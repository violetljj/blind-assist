from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sanpo_depth_anything_linear_probe as probe
import smoke_depth_anything_v2_pytorch as depth_smoke


class TokensToFeatureMapTests(unittest.TestCase):
    def test_reshapes_patch_tokens_in_row_major_order(self) -> None:
        tokens = np.arange(2 * 3 * 4, dtype=np.float32).reshape(1, 6, 4)
        result = probe.tokens_to_feature_map(tokens, patch_height=2, patch_width=3)
        self.assertEqual((2, 3, 4), result.shape)
        np.testing.assert_array_equal(tokens[0, 4], result[1, 1])

    def test_rejects_wrong_token_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 2×3 patch tokens"):
            probe.tokens_to_feature_map(np.zeros((1, 5, 8), dtype=np.float32), patch_height=2, patch_width=3)

    def test_resizes_feature_map_to_requested_grid(self) -> None:
        feature_map = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
        resized = probe.resize_feature_map(feature_map, height=4, width=6)
        self.assertEqual((4, 6, 4), resized.shape)
        self.assertEqual(np.float32, resized.dtype)

    def test_parser_rejects_non_patch_multiple_input(self) -> None:
        with self.assertRaises(SystemExit):
            probe.parse_args(["--input-size", "225"])

    def test_depth_model_device_matches_image_tensor_policy(self) -> None:
        class Torch:
            class cuda:
                @staticmethod
                def is_available() -> bool:
                    return True
            class backends:
                class mps:
                    @staticmethod
                    def is_available() -> bool:
                        return False
        self.assertEqual("cuda", depth_smoke.inference_device(Torch))


if __name__ == "__main__":
    unittest.main()
