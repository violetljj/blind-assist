from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    pdm_official_runtime_adapter as sut,
)


class PdmOfficialRuntimeAdapterTest(unittest.TestCase):
    def test_crop_id_is_order_stable_and_sensitive_to_bounds(self) -> None:
        first = {"image_sha256": "a" * 64, "crop_bbox_xyxy_normalized": [0.1, 0.2, 0.8, 0.9]}
        reordered = {"crop_bbox_xyxy_normalized": [0.1, 0.2, 0.8, 0.9], "image_sha256": "a" * 64}
        changed = {**first, "crop_bbox_xyxy_normalized": [0.0, 0.2, 0.8, 0.9]}
        self.assertEqual(sut._crop_id(first), sut._crop_id(reordered))
        self.assertNotEqual(sut._crop_id(first), sut._crop_id(changed))

    def test_reference_mask_uses_same_crop_and_shared_64_grid(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image_path = root / "image.png"
            mask_path = root / "mask.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[25:75, 25:75] = 255
            Image.fromarray(mask).save(mask_path)
            item = {
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "crop_bbox_xyxy_normalized": [0.0, 0.0, 1.0, 1.0],
            }
            result = sut._reference_mask64(item)
            self.assertEqual((64, 64), result.shape)
            self.assertTrue(result[32, 32])
            self.assertFalse(result[0, 0])

    def test_selective_store_keeps_only_frozen_qk_at_timestamp(self) -> None:
        import torch

        store = sut._SelectiveAttentionStore()
        store.cur_step = sut.ATTENTION_TIMESTAMP
        place = f"Q_{sut.ATTENTION_PLACE_STEM}"
        key = f"Q_{sut.ATTENTION_KEY_FORMAT}"
        value = torch.ones(1, 2, 3)
        self.assertIs(value, store(value, False, place))
        self.assertIn(key, store.step_store)
        store(value, False, "Q_other")
        self.assertNotIn("Q_other_self", store.step_store)


if __name__ == "__main__":
    unittest.main()
