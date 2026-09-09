from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.research.egomotion_compensated_looming.rcle_synthetic_scene_r0.dataset import (
    build_dataset,
)
from scripts.research.egomotion_compensated_looming.rcle_synthetic_scene_r0.evaluator import (
    evaluate_algorithm_ledger,
    evaluate_dataset,
)
from scripts.research.egomotion_compensated_looming.rcle_synthetic_scene_r0.io_utils import (
    read_jsonl,
    write_jsonl,
)


SPEC = (
    Path(__file__).resolve().parents[1]
    / "rcle_synthetic_scene_r0"
    / "dataset_spec.json"
)


def _tiny_spec(root: Path) -> Path:
    value = json.loads(SPEC.read_text(encoding="utf-8"))
    value["rendering"].update(
        {
            "width": 96,
            "height": 72,
            "fps": 2,
            "duration_seconds": 1,
            "frame_count": 3,
            "pair_count_per_sequence": 2,
        }
    )
    value["rendering"]["intrinsics"] = {
        "fx_pixels": 84.0,
        "fy_pixels": 84.0,
        "cx_pixels": 47.5,
        "cy_pixels": 35.5,
    }
    value["splits"]["development"]["scene_families"] = [
        value["splits"]["development"]["scene_families"][0]
    ]
    value["motions"] = value["motions"][:1]
    path = root / "tiny_spec.json"
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


class EvaluatorFirewallTest(unittest.TestCase):
    def _dataset(self, root: Path) -> Path:
        dataset = root / "dataset"
        build_dataset(_tiny_spec(root), dataset)
        return dataset

    def _assert_truth_only_mutation_isolated(self, mutation) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._dataset(Path(temporary))
            before = evaluate_algorithm_ledger(dataset)
            mutation(dataset)
            after = evaluate_algorithm_ledger(dataset)
            self.assertEqual(before, after)

    def test_rgb_byte_poison_fails_before_algorithm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._dataset(Path(temporary))
            target = next((dataset / "algorithm_staging").rglob("rgb/*.png"))
            payload = bytearray(target.read_bytes())
            payload[-1] ^= 1
            target.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "RGB_HASH_MISMATCH"):
                evaluate_dataset(dataset)

    def test_valid_mask_byte_poison_fails_before_algorithm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self._dataset(Path(temporary))
            target = next(
                (dataset / "algorithm_staging").rglob("valid_mask/*.png")
            )
            payload = bytearray(target.read_bytes())
            payload[-1] ^= 1
            target.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "VALID_MASK_HASH_MISMATCH"):
                evaluate_dataset(dataset)

    def test_depth_mutation_does_not_change_algorithm_ledger(self) -> None:
        def mutate(dataset: Path) -> None:
            row = read_jsonl(dataset / "frame_manifest.jsonl")[0]
            path = dataset / row["depth_npy_path"]
            depth = np.load(path, allow_pickle=False)
            np.save(path, depth + np.float32(0.125), allow_pickle=False)

        self._assert_truth_only_mutation_isolated(mutate)

    def test_full_pose_translation_mutation_does_not_change_algorithm_ledger(
        self,
    ) -> None:
        def mutate(dataset: Path) -> None:
            path = dataset / "frame_manifest.jsonl"
            rows = read_jsonl(path)
            rows[0]["t_world_camera"][0][3] += 7.0
            rows[0]["t_camera_world"][0][3] -= 7.0
            write_jsonl(path, rows)

        self._assert_truth_only_mutation_isolated(mutate)

    def test_role_mutation_does_not_change_algorithm_ledger(self) -> None:
        def mutate(dataset: Path) -> None:
            path = dataset / "frame_manifest.jsonl"
            rows = read_jsonl(path)
            rows[0]["role"] = "POISON_TRUTH_ONLY_ROLE"
            write_jsonl(path, rows)

        self._assert_truth_only_mutation_isolated(mutate)

    def test_surface_id_mutation_does_not_change_algorithm_ledger(self) -> None:
        def mutate(dataset: Path) -> None:
            row = read_jsonl(dataset / "frame_manifest.jsonl")[0]
            path = dataset / row["surface_id_path"]
            surface = np.load(path, allow_pickle=False)
            surface.flat[0] = np.int16(surface.flat[0] + 1)
            np.save(path, surface, allow_pickle=False)

        self._assert_truth_only_mutation_isolated(mutate)


if __name__ == "__main__":
    unittest.main()
