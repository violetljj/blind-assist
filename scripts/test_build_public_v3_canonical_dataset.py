from __future__ import annotations

import tempfile
import unittest
import hashlib
import json
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

    def test_procedural_adapter_copies_and_rebinds_all_sha_bound_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package, staging = root / "package", root / "staging"
            files = {
                "images/frame.png": b"image", "semantic/frame.png": b"mask",
                "evidence/generator.py": b"code", "evidence/config.json": b"config",
                "evidence/tactile.png": b"tactile", "evidence/obstacle.png": b"obstacle",
            }
            for relative, content in files.items():
                path = package / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            digest = lambda value: hashlib.sha256(files[value]).hexdigest()
            matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            row = {
                "image_path": "images/frame.png", "image_sha256": digest("images/frame.png"),
                "semantic_mask_path": "semantic/frame.png", "semantic_mask_sha256": digest("semantic/frame.png"),
                "label_authority": "procedural_ground_truth",
                "label_provenance": {
                    "schema": "blindassist_procedural_ground_truth_v1", "generator_id": "tactile_occupied_compositor_v1",
                    "generator_code_path": "evidence/generator.py", "generator_code_sha256": digest("evidence/generator.py"),
                    "generator_config_path": "evidence/config.json", "generator_config_sha256": digest("evidence/config.json"),
                    "seed": 7, "transform_matrix": matrix,
                    "transform_sha256": hashlib.sha256(json.dumps(matrix, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                    "source_masks": [
                        {"role": "tactile_ground_truth", "source_id": "guide", "path": "evidence/tactile.png", "sha256": digest("evidence/tactile.png")},
                        {"role": "obstacle_ground_truth", "source_id": "sanpo", "path": "evidence/obstacle.png", "sha256": digest("evidence/obstacle.png")},
                    ],
                    "output_mask_sha256": digest("semantic/frame.png"),
                },
            }
            image, mask, provenance, image_sha, mask_sha = builder.copy_procedural_assets(
                package, staging, row, "sample", "dev",
            )
            self.assertEqual(digest("images/frame.png"), image_sha)
            self.assertEqual(digest("semantic/frame.png"), mask_sha)
            for relative in (image, mask, provenance["generator_code_path"], provenance["generator_config_path"]):
                self.assertTrue((staging / relative).is_file())
            for item in provenance["source_masks"]:
                self.assertTrue((staging / item["path"]).is_file())
                self.assertEqual(item["sha256"], builder.sha256_file(staging / item["path"]))
            self.assertEqual(row["label_provenance"]["generator_code_path"], "evidence/generator.py")

    def test_procedural_adapter_rejects_tampered_package_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "asset.bin"
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "source SHA256 mismatch"):
                builder.copy_sha_bound_file(root, root / "staging", "asset.bin", "0" * 64, Path("copy.bin"))


if __name__ == "__main__":
    unittest.main()
