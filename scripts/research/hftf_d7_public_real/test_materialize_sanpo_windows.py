from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from materialize_sanpo_windows import ContractError, materialize_windows


def _inventory(*, include_depth: bool = True) -> dict[str, object]:
    objects: list[dict[str, object]] = [
        {
            "kind": "rgb",
            "name": "sanpo_dataset/v0/sanpo-real/session-a/camera_chest/left/video_frames/000030.png",
            "generation": "rgb-rev-1",
            "sha256": "rgb-hash-30",
        },
        {
            "kind": "segmentation",
            "name": "sanpo_dataset/v0/sanpo-real/session-a/camera_chest/left/segmentation_masks/000030.png",
            "generation": "seg-rev-1",
            "sha256": "seg-hash-30",
        },
        {
            "kind": "intrinsics",
            "name": "sanpo_dataset/v0/sanpo-real/session-a/description.json",
            "generation": "description-rev-1",
            "sha256": "description-hash",
        },
        {
            "kind": "pose",
            "name": "sanpo_dataset/v0/sanpo-real/session-a/camera_chest/camera_poses.csv",
            "generation": "pose-rev-1",
            "sha256": "pose-hash",
        },
    ]
    if include_depth:
        objects.append(
            {
                "kind": "depth",
                "name": "sanpo_dataset/v0/sanpo-real/session-a/camera_chest/left/depth_maps/000030.float16.gz",
                "generation": "depth-rev-1",
                "sha256": "depth-hash-30",
            }
        )
    return {
        "schema": "hftf_d7_public_real_sanpo_public_gcs_inventory_v1",
        "dataset_id": "SANPO-Real",
        "run_id": "inventory-r1",
        "official_url": "https://google-research-datasets.github.io/sanpo_dataset/",
        "license": "CC-BY-4.0",
        "records": [
            {
                "source_session_id": "session-a",
                "ancestry_group": "session-a",
                "objects": objects,
            }
        ],
    }


def _selection(*, duplicate: bool = False) -> list[dict[str, object]]:
    row: dict[str, object] = {
        "session": "session-a",
        "camera": "chest",
        "view": "left",
        "frames": [30],
    }
    return [row, dict(row)] if duplicate else [row]


class MaterializeSanpoWindowsTest(unittest.TestCase):
    def _write_inputs(
        self,
        root: Path,
        *,
        include_depth: bool = True,
        duplicate: bool = False,
    ) -> tuple[Path, Path, Path]:
        inventory = root / "sanpo-gcs-inventory.json"
        selection = root / "session-camera-view.json"
        output = root / "candidate-evidence.jsonl"
        inventory.write_text(
            json.dumps(_inventory(include_depth=include_depth)),
            encoding="utf-8",
        )
        selection.write_text(json.dumps(_selection(duplicate=duplicate)), encoding="utf-8")
        return inventory, selection, output

    def test_missing_fps_is_rejected_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inventory, selection, output = self._write_inputs(Path(directory))
            with self.assertRaisesRegex(ContractError, "fps.*required"):
                materialize_windows(inventory, selection, output, fps=None)
            self.assertFalse(output.exists())

    def test_missing_object_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inventory, selection, output = self._write_inputs(
                Path(directory), include_depth=False
            )
            with self.assertRaisesRegex(ContractError, "missing depth"):
                materialize_windows(inventory, selection, output, fps=15)
            self.assertFalse(output.exists())

    def test_nominal_time_is_explicitly_non_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inventory, selection, output = self._write_inputs(Path(directory))
            result = materialize_windows(inventory, selection, output, fps=15)
            self.assertEqual(result["rows_appended"], 1)
            row = json.loads(output.read_text(encoding="utf-8").strip())
            self.assertEqual(row["frame"], 30)
            self.assertEqual(row["timestamp"]["value"], 2.0)
            self.assertIsNone(row["timestamp_ns"])
            self.assertEqual(row["nominal_time_ns"], 2_000_000_000)
            self.assertEqual(row["time_semantics"], "DERIVED_RELATIVE_NOMINAL")
            self.assertFalse(row["capture_timestamp_authoritative"])
            self.assertEqual(row["pose_row_binding"], "NOT_EVALUABLE")
            self.assertEqual(row["source"]["fps"], 15.0)
            self.assertEqual(row["dataset"], "SANPO-Real")
            self.assertEqual(row["ancestry"], "session-a")
            for field in (
                "rgb",
                "intrinsics",
                "pose",
                "depth",
                "segmentation",
                "source",
                "license",
                "revision",
                "hash",
            ):
                self.assertIn(field, row)
            serialized = json.dumps(row).lower()
            for forbidden in ("review", "label", "event_bucket", "admission", "model_output"):
                self.assertNotIn(forbidden, serialized)

    def test_duplicate_candidate_is_rejected_before_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            inventory, selection, output = self._write_inputs(
                Path(directory), duplicate=True
            )
            with self.assertRaisesRegex(ContractError, "duplicate candidate"):
                materialize_windows(inventory, selection, output, fps=15)
            self.assertFalse(output.exists())

    def test_existing_candidate_is_rejected_incrementally(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory, selection, output = self._write_inputs(root)
            materialize_windows(inventory, selection, output, fps=15)
            before = output.read_bytes()
            with self.assertRaisesRegex(ContractError, "duplicate candidate"):
                materialize_windows(inventory, selection, output, fps=15)
            self.assertEqual(output.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
