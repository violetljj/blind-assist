from __future__ import annotations

import tempfile
import unittest
import hashlib
import json
from pathlib import Path

from PIL import Image

import build_public_v3_canonical_dataset as builder


class PublicV3CanonicalBuilderTest(unittest.TestCase):
    def test_source_sequence_is_frame_count_and_official_split_bound(self) -> None:
        rows = [
            {"frame_index": index, "source": {"session_id": "session-a", "official_split": "train"}}
            for index in range(3)
        ]
        selected = builder.select_source_sequence(rows, "session-a", 3, "train")
        self.assertEqual([0, 1, 2], [item["frame_index"] for item in selected])
        with self.assertRaisesRegex(ValueError, "official split"):
            builder.select_source_sequence(rows, "session-a", 3, "test")
        with self.assertRaisesRegex(ValueError, "contiguous 4-frame"):
            builder.select_source_sequence(rows, "session-a", 4, "train")

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
                "evidence/guide.png": b"guide", "evidence/guide.txt": b"polygon",
                "evidence/sanpo.png": b"sanpo", "evidence/sanpo-mask.png": b"raw-mask",
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
                    "source_assets": [
                        {"role": "guide_rgb", "source_id": "guide", "path": "evidence/guide.png", "sha256": digest("evidence/guide.png")},
                        {"role": "guide_polygon", "source_id": "guide", "path": "evidence/guide.txt", "sha256": digest("evidence/guide.txt")},
                        {"role": "sanpo_rgb", "source_id": "sanpo", "path": "evidence/sanpo.png", "sha256": digest("evidence/sanpo.png")},
                        {"role": "sanpo_raw_mask", "source_id": "sanpo", "path": "evidence/sanpo-mask.png", "sha256": digest("evidence/sanpo-mask.png")},
                    ],
                    "output_mask_sha256": digest("semantic/frame.png"),
                },
            }
            image, mask, provenance, image_sha, mask_sha, raw_assets = builder.copy_procedural_assets(
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
            self.assertEqual(4, len(raw_assets))
            for item in raw_assets:
                self.assertTrue((staging / item["path"]).is_file())
                self.assertEqual(item["sha256"], builder.sha256_file(staging / item["path"]))

    def test_procedural_adapter_rejects_tampered_package_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "asset.bin"
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "source SHA256 mismatch"):
                builder.copy_sha_bound_file(root, root / "staging", "asset.bin", "0" * 64, Path("copy.bin"))

    def test_gate_rejects_guide_remote_receipt_that_differs_from_attested_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            files = {}
            for role in ("guide_rgb", "guide_polygon", "sanpo_rgb", "sanpo_raw_mask"):
                path = raw / f"{role}.bin"
                path.write_bytes(role.encode())
                files[role] = path
            remote = root / "guide-inventory.json"
            remote.write_text(json.dumps({
                "source": {"etag": "etag-1", "generation": "7", "md5_base64": "archive-md5"},
                "members": [
                    {"path": f"Guide/{role}.bin", "size": files[role].stat().st_size,
                     "crc32": builder.gate.crc32_file(files[role])}
                    for role in ("guide_rgb", "guide_polygon")
                ],
            }), encoding="utf-8")
            sample_id = "procedural_sample"
            assets = []
            declared = []
            ids = []
            for index, role in enumerate(files):
                source_id = "guide" if role.startswith("guide_") else "sanpo"
                relative = files[role].relative_to(root).as_posix()
                item = {"role": role, "source_id": source_id, "path": relative,
                        "sha256": builder.sha256_file(files[role])}
                if source_id == "guide":
                    item["remote_receipt"] = {
                        "origin_member_path": f"Guide/{role}.bin", "size": files[role].stat().st_size,
                        "crc32": builder.gate.crc32_file(files[role]),
                        "archive": {"etag": "tampered", "generation": "7", "md5_base64": "archive-md5"},
                    }
                declared.append(item)
                entry_id = f"{sample_id}:{role}:{index}"
                ids.append(entry_id)
                assets.append({"entry_id": entry_id, "sample_id": sample_id, "session_id": "session",
                               "frame_index": 0, **item})
            row = {"id": sample_id, "split": "dev", "session_id": "session", "frame_index": 0,
                   "label_authority": "procedural_ground_truth", "source_asset_ids": ids,
                   "label_provenance": {"source_assets": declared}}
            attestation = {"sources": [
                {"source_id": "guide", "inventory_path": remote.relative_to(root).as_posix()},
                {"source_id": "sanpo", "inventory_path": "missing.json"},
            ]}
            errors = builder.gate.source_asset_inventory_errors(
                root, [row], {"schema": builder.gate.ASSET_INVENTORY_SCHEMA, "assets": assets}, attestation,
            )
            self.assertTrue(any("archive etag differs" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
