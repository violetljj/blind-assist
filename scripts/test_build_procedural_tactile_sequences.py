from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPTS = Path(__file__).resolve().parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("build_procedural_tactile_sequences")
validator = load("validate_sanpo_v3_dataset")


class ProceduralTactileSequenceTest(unittest.TestCase):
    def fixture(self, root: Path, output: Path, seed: int = 17) -> Namespace:
        guide = root / "guide.png"
        label = root / "guide.txt"
        Image.new("RGB", (64, 48), (80, 80, 80)).save(guide)
        label.write_text("0 0.1 0.65 0.9 0.65 0.9 0.95 0.1 0.95\n", encoding="utf-8")
        inventory = root / "guide-inventory.json"
        inventory.write_text(json.dumps({
            "schema": "blindassist.remote_zip_inventory.v1",
            "source": {"etag": "fixture-etag", "generation": "1", "md5_base64": "fixture-md5"},
            "members": [
                {"path": "Guide/guide.png", "size": guide.stat().st_size, "crc32": builder.crc32_file(guide)},
                {"path": "Guide/guide.txt", "size": label.stat().st_size, "crc32": builder.crc32_file(label)},
            ],
        }), encoding="utf-8")
        sanpo = root / "sanpo"
        rows = []
        for index in range(3):
            sample = f"source_{index}"
            image = sanpo / "images" / "test" / f"{sample}.png"
            mask = sanpo / "source_masks" / "test" / f"{sample}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            mask.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 24), (150 + index, 20, 20)).save(image)
            array = np.zeros((24, 32, 3), dtype=np.uint8)
            array[6:18, 8 + index:20 + index, 0] = 20
            Image.fromarray(array, "RGB").save(mask)
            rows.append({"id": sample, "image_path": f"images/test/{sample}.png"})
        manifest = sanpo / "manifest.draft.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return Namespace(
            guide_image=guide, guide_label=label, guide_inventory=inventory,
            guide_image_member="Guide/guide.png", guide_label_member="Guide/guide.txt",
            sanpo_manifest=manifest, sanpo_root=sanpo,
            split="dev", session_id="fixture_session", scene_bucket="tactile_paving_occupied",
            frame_count=3, seed=seed, output=output,
        )

    def test_builds_validator_accepted_hash_bound_sequence_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, second = root / "first", root / "second"
            first_rows = builder.build(self.fixture(root, first))
            second_rows = builder.build(self.fixture(root, second))
            self.assertEqual(3, len(first_rows))
            self.assertEqual(
                [row["semantic_mask_sha256"] for row in first_rows],
                [row["semantic_mask_sha256"] for row in second_rows],
            )
            errors, summary = validator.validate_rows(first_rows, first, {"dev"})
            self.assertEqual([], errors)
            self.assertEqual(3, summary["row_count"])
            matrices = [row["label_provenance"]["transform_matrix"] for row in first_rows]
            self.assertNotEqual(matrices[0], matrices[-1])
            for row in first_rows:
                raw_assets = row["label_provenance"]["source_assets"]
                self.assertEqual(
                    {"guide_rgb", "guide_polygon", "sanpo_rgb", "sanpo_raw_mask"},
                    {item["role"] for item in raw_assets},
                )
                for item in raw_assets:
                    self.assertEqual(item["sha256"], builder.sha256_file(first / item["path"]))
            tactile = np.asarray(Image.open(first / "procedural_evidence" / "tactile_ground_truth.png")) > 0
            for row in first_rows:
                semantic = np.asarray(Image.open(first / row["semantic_mask_path"]))
                values = set(np.unique(semantic).tolist())
                self.assertTrue({0, 2, 3}.issubset(values))
                self.assertGreater(np.count_nonzero((semantic == 2) & tactile), 0)

    def test_rejects_guide_asset_without_matching_remote_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = self.fixture(root, root / "output")
            inventory = json.loads(args.guide_inventory.read_text(encoding="utf-8"))
            inventory["members"][0]["crc32"] = "00000000"
            args.guide_inventory.write_text(json.dumps(inventory), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "match exactly one remote inventory member"):
                builder.build(args)


if __name__ == "__main__":
    unittest.main()
