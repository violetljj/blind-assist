from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

import build_public_v3_canonical_dataset as builder


class PublicV3CanonicalBuilderTest(unittest.TestCase):
    def test_sanpo_mapper_covers_native_taxonomy_and_four_target_classes(self) -> None:
        self.assertEqual(set(range(31)), set(builder.SANPO_MAP))
        self.assertEqual({0, 1, 2, 3}, set(builder.SANPO_MAP.values()))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.png"
            output = root / "mapped.png"
            Image.new("RGB", (31, 1)).save(source)
            pixels = Image.open(source)
            pixels.putdata([(value, 0, 0) for value in range(31)])
            pixels.save(source)
            builder.remap_mask(source, output, "sanpo_v0")
            with Image.open(output) as mapped:
                self.assertEqual([builder.SANPO_MAP[value] for value in range(31)], list(mapped.getdata()))

    def test_unknown_native_class_and_unimplemented_adapter_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.png"
            Image.new("RGB", (1, 1), (31, 0, 0)).save(source)
            with self.assertRaisesRegex(ValueError, "unmapped class IDs"):
                builder.remap_mask(source, root / "mapped.png", "sanpo_v0")
            with self.assertRaisesRegex(ValueError, "no full-mask mapper"):
                builder.remap_mask(source, root / "mapped.png", "guidetwsi_v1")


if __name__ == "__main__":
    unittest.main()
