from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from .canonicalizer import CanonicalizationError, sha256_file
from .materialize_canonical_view import materialize


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT = REPO_ROOT / "configs/dual_loop_segmentation_r2_p0/canonicalization_contract.json"
SCHEMA = REPO_ROOT / "configs/dual_loop_segmentation_r2_p0/materialized_canonical_view.schema.json"


class MaterializationTests(unittest.TestCase):
    def _fixture(self, root: Path, *, rows: int = 1) -> Path:
        image = root / "image.png"
        mask = root / "mask.png"
        Image.new("RGB", (2, 2), color=(0, 0, 0)).save(image)
        Image.fromarray(np.asarray([[1, 2], [3, 4]], dtype=np.uint8), mode="L").save(mask)
        manifest = root / "manifest.jsonl"
        values = [
            {
                "id": f"row-{index}",
                "split": "dev",
                "source_id": "fixture",
                "session_id": "fixture",
                "frame_id": index,
                "image_path": image.name,
                "image_sha256": sha256_file(image),
                "semantic_mask_path": mask.name,
                "semantic_mask_sha256": sha256_file(mask),
            }
            for index in range(rows)
        ]
        manifest.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in values),
            encoding="utf-8",
            newline="\n",
        )
        config = root / "sources.json"
        config.write_text(
            json.dumps(
                {
                    "protocol_id": "DUAL_LOOP_SEGMENTATION_R2_P0",
                    "sources": [
                        {
                            "role": "dev",
                            "manifest": str(manifest.relative_to(REPO_ROOT)),
                            "manifest_sha256": sha256_file(manifest),
                            "dataset_root": str(root.relative_to(REPO_ROOT)),
                            "split": "dev",
                            "decoder": "source_native",
                            "expected_rows": rows,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return config

    def test_interruption_resume_and_atomic_publish(self) -> None:
        temp_parent = REPO_ROOT / "artifacts.local/tmp"
        temp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_parent) as directory:
            root = Path(directory)
            config = self._fixture(root)
            output = root / "view"
            from . import materialize_canonical_view as module

            real_write = module.write_canonical_png
            calls = 0

            def interrupted(*args, **kwargs):
                nonlocal calls
                calls += 1
                real_write(*args, **kwargs)
                if calls == 1:
                    raise RuntimeError("simulated interruption")

            with mock.patch.object(module, "write_canonical_png", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    materialize(
                        repo_root=REPO_ROOT,
                        contract_path=CONTRACT,
                        schema_path=SCHEMA,
                        source_config_paths=[config],
                        output_root=output,
                    )
            self.assertFalse(output.exists())
            self.assertTrue(output.with_name("view.staging").is_dir())
            receipt = materialize(
                repo_root=REPO_ROOT,
                contract_path=CONTRACT,
                schema_path=SCHEMA,
                source_config_paths=[config],
                output_root=output,
            )
            self.assertEqual(receipt["status"], "CANONICAL_VIEW_MATERIALIZED")
            self.assertTrue(output.is_dir())
            self.assertFalse(output.with_name("view.staging").exists())

    def test_zero_rows_fail_closed(self) -> None:
        temp_parent = REPO_ROOT / "artifacts.local/tmp"
        temp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_parent) as directory:
            root = Path(directory)
            config = self._fixture(root, rows=0)
            with self.assertRaisesRegex(CanonicalizationError, "zero rows"):
                materialize(
                    repo_root=REPO_ROOT,
                    contract_path=CONTRACT,
                    schema_path=SCHEMA,
                    source_config_paths=[config],
                    output_root=root / "empty-view",
                )


if __name__ == "__main__":
    unittest.main()
