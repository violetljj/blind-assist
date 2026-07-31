from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from .canonicalizer import (
    CanonicalizationError,
    canonicalize_array,
    load_contract,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = REPO_ROOT / "configs/dual_loop_segmentation_r2_p0/canonicalization_contract.json"


class CanonicalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = load_contract(CONTRACT)

    def test_mapping_is_complete_and_exact(self) -> None:
        mapping = self.contract["_parsed_mapping"]
        self.assertEqual(set(mapping), set(range(31)))
        self.assertEqual(
            mapping,
            {
                0: 3, 1: 0, 2: 1, 3: 0, 4: 2, 5: 0, 6: 0, 7: 3,
                8: 2, 9: 2, 10: 2, 11: 2, 12: 2, 13: 2, 14: 2, 15: 1,
                16: 2, 17: 0, 18: 2, 19: 2, 20: 2, 21: 2, 22: 2, 23: 2,
                24: 2, 25: 2, 26: 2, 27: 3, 28: 2, 29: 3, 30: 3,
            },
        )

    def test_rgb_decoder_uses_red_channel_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            value = np.asarray([[[2, 255, 255], [4, 1, 99]]], dtype=np.uint8)
            Image.fromarray(value, mode="RGB").save(path)
            result = canonicalize_array(path, "source_native", self.contract)
            self.assertEqual(result.shape, (256, 256))
            self.assertEqual(set(np.unique(result)), {1, 2})

    def test_palette_decoder_uses_indices_not_colors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            image = Image.fromarray(np.asarray([[1, 2]], dtype=np.uint8), mode="P")
            palette = [0] * 768
            palette[3:6] = [255, 255, 255]
            palette[6:9] = [0, 0, 0]
            image.putpalette(palette)
            image.save(path)
            result = canonicalize_array(path, "source_native", self.contract)
            self.assertEqual(set(np.unique(result)), {0, 1})

    def test_unknown_native_id_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.fromarray(np.asarray([[31]], dtype=np.uint8), mode="L").save(path)
            with self.assertRaises(CanonicalizationError):
                canonicalize_array(path, "source_native", self.contract)

    def test_canonical_invalid_id_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.fromarray(np.asarray([[4]], dtype=np.uint8), mode="L").save(path)
            with self.assertRaises(CanonicalizationError):
                canonicalize_array(path, "canonical_passthrough", self.contract)

    def test_resize_is_nearest_neighbor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.png"
            Image.fromarray(np.asarray([[0, 1], [2, 3]], dtype=np.uint8), mode="L").save(path)
            result = canonicalize_array(path, "canonical_passthrough", self.contract)
            self.assertEqual(int(result[63, 63]), 0)
            self.assertEqual(int(result[63, 192]), 1)
            self.assertEqual(int(result[192, 63]), 2)
            self.assertEqual(int(result[192, 192]), 3)

    def test_contract_is_json_object(self) -> None:
        value = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertIsInstance(value, dict)


if __name__ == "__main__":
    unittest.main()
