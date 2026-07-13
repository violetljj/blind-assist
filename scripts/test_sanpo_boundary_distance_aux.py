from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPTS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "sanpo_boundary_distance_aux", SCRIPTS / "sanpo_boundary_distance_aux.py"
)
assert SPEC and SPEC.loader
aux = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aux)


class SanpoBoundaryDistanceAuxTest(unittest.TestCase):
    def test_unsigned_and_signed_distance_contract(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        unsigned = aux.boundary_distance_target(mask, truncate=2, signed=False)
        signed = aux.boundary_distance_target(mask, truncate=2, signed=True)
        self.assertEqual(0.0, unsigned[2, 2])
        self.assertEqual(-0.5, signed[2, 2])
        self.assertEqual(0.5, signed[2, 3])
        self.assertEqual(1.0, unsigned[0, 0])
        self.assertEqual(np.float32, unsigned.dtype)

    def test_empty_and_full_masks_have_explicit_sentinels(self) -> None:
        empty = np.zeros((3, 4), dtype=bool)
        full = np.ones((3, 4), dtype=bool)
        np.testing.assert_array_equal(1, aux.boundary_distance_target(empty, signed=False))
        np.testing.assert_array_equal(1, aux.boundary_distance_target(empty, signed=True))
        np.testing.assert_array_equal(0, aux.boundary_distance_target(full, signed=False))
        np.testing.assert_array_equal(-1, aux.boundary_distance_target(full, signed=True))

    def test_smooth_l1_weights_focus_on_boundary_and_retain_absence(self) -> None:
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        target, weight = aux.smooth_l1_target_and_weight(
            mask, truncate=2, min_weight=0.1, boundary_weight=3,
        )
        self.assertEqual(0.0, target[2, 2])
        self.assertAlmostEqual(3.0, float(weight[2, 2]))
        self.assertAlmostEqual(0.1, float(weight[0, 0]))
        _, empty_weight = aux.smooth_l1_target_and_weight(
            np.zeros((2, 2)), empty_weight=0.07,
        )
        np.testing.assert_allclose(0.07, empty_weight)

    def test_invalid_contract_arguments_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "2-D"):
            aux.boundary_distance_target(np.zeros((1, 2, 3)))
        with self.assertRaisesRegex(ValueError, "truncate"):
            aux.boundary_distance_target(np.zeros((2, 2)), truncate=0)
        with self.assertRaisesRegex(ValueError, "boundary_weight"):
            aux.smooth_l1_target_and_weight(np.zeros((2, 2)), boundary_weight=0.5)

    def test_loader_and_diagnostic_refuse_blind_rows_and_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "canonical"
            root.mkdir()
            self._write_policy(root)
            self._write_row(root, split="blind", mask_path="semantic_masks/blind/x.png")
            with self.assertRaisesRegex(ValueError, "refusing line 1 split 'blind'"):
                aux.load_training_rows(root)

            self._write_row(root, split="train", mask_path="blind_holdout/x.png")
            with self.assertRaisesRegex(ValueError, "refusing semantic mask path"):
                aux.load_training_rows(root)

    def test_real_only_train_dev_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "canonical"
            root.mkdir()
            self._write_policy(root)
            rows = []
            for split in ("train", "dev"):
                relative = Path("semantic_masks") / split / f"{split}.png"
                (root / relative).parent.mkdir(parents=True, exist_ok=True)
                semantic = np.zeros((4, 4), dtype=np.uint8)
                semantic[1, 1] = aux.BOUNDARY_CLASS_ID
                Image.fromarray(semantic).save(root / relative)
                rows.append({
                    "id": split, "split": split, "session_id": f"real-{split}",
                    "scene_bucket": "step_curb", "semantic_mask_path": relative.as_posix(),
                    "source": {"source_id": "sanpo_real_v0"},
                })
            (root / "training_manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = aux.diagnose_dataset(
                root, truncate=2, real_only=True, analysis_size=2,
            )
            self.assertEqual(1, report["splits"]["train"]["frame_count"])
            self.assertEqual(1, report["splits"]["dev"]["frames_with_boundary"])
            self.assertEqual(2, report["contract"]["analysis_size"])
            self.assertEqual("refused", report["contract"]["blind_access"])

            with self.assertRaisesRegex(ValueError, "analysis_size"):
                aux.diagnose_dataset(root, analysis_size=0)

    @staticmethod
    def _write_policy(root: Path) -> None:
        (root / "access_policy.json").write_text(json.dumps({
            "training_manifest": "training_manifest.jsonl",
            "forbidden_training_sessions": ["held-out"],
        }), encoding="utf-8")

    @staticmethod
    def _write_row(root: Path, *, split: str, mask_path: str) -> None:
        row = {
            "id": "x", "split": split, "session_id": "ordinary",
            "semantic_mask_path": mask_path,
        }
        (root / "training_manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
